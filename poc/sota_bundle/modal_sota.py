"""Run SOTA CPS anomaly detectors (USAD / TranAD / GDN) on WADI via the official
TranAD harness (imperial-qore/TranAD), and recompute RAW point-wise F1 (their eval
is point-adjusted). WADI arrays (MinMax-normalised, x10 downsample, 123 channels) are
uploaded from the local bundle. First run is diagnostic-heavy: it prints the repo's
data-loading contract so we can fix the .npy layout if needed, and attempts USAD.

Invoke:  python <skill>/modal_runner.py run --script modal_sota.py --args "--models USAD"
Outputs: printed JSON + /results score arrays on the results volume.
"""
from __future__ import annotations
import json
from pathlib import Path
import modal

HERE = Path(__file__).parent.resolve()
NPY = ["wadi_train.npy", "wadi_test.npy", "wadi_labels.npy", "wadi_triv_test.npy", "wadi_triv_thr.npy",
       "wadi_testfull.npy", "wadi_labelsfull.npy",
       "HAI_train.npy", "HAI_test.npy", "HAI_labels.npy", "SKAB_train.npy", "SKAB_test.npy", "SKAB_labels.npy"]

image = (
    modal.Image.from_registry("pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel", add_python="3.11")
    .apt_install("git")
    .pip_install("numpy<2", "pandas", "scikit-learn", "scipy", "tqdm", "matplotlib")
)
for f in NPY:
    lp = HERE / f
    if lp.exists():
        image = image.add_local_file(str(lp), f"/app/{f}")

app = modal.App("latad-sota", image=image)
results_vol = modal.Volume.from_name("latad-sota-results", create_if_missing=True)


@app.function(gpu="A10G", timeout=2 * 60 * 60, memory=16384, volumes={"/results": results_vol})
def run_sota(models: str = "USAD", epochs: int = 5, recon: bool = True, ds: str = "WADI") -> dict:
    import subprocess, os, sys, shutil, numpy as np
    os.environ["MKL_THREADING_LAYER"] = "GNU"            # dgl(MKL) vs libgomp conflict fix (GDN)
    os.environ["MKL_SERVICE_FORCE_INTEL"] = "0"
    os.environ["NEPOCHS"] = str(epochs)                  # epoch override for fast GDN feedback
    os.environ["USE_CUDA"] = "1" if "GDN" in models else "0"   # GPU only for GDN (heavy graph attn)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"   # reduce OOM fragmentation
    def sh(cmd, **kw):
        print(f"[sota] $ {cmd}", flush=True)
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)
        return r.returncode, r.stdout, r.stderr

    os.chdir("/root")
    sh("git clone --depth 1 https://github.com/imperial-qore/TranAD.git")
    repo = "/root/TranAD"; os.chdir(repo)
    out = {"recon": {}, "results": {}}

    # --- patches: make dgl optional (only GDN needs it); dump raw per-timestep score+labels ---
    mm = Path("src/models.py").read_text()
    mm = mm.replace("import dgl\n", "try:\n\timport dgl\nexcept Exception:\n\tdgl = None\n")
    mm = mm.replace("from dgl.nn import GATConv\n",
                    "try:\n\tfrom dgl.nn import GATConv\nexcept Exception:\n\tGATConv = None\n")
    Path("src/models.py").write_text(mm)
    pp = Path("src/plotting.py").read_text()
    pp = pp.replace("plt.style.use(['science', 'ieee'])", "pass  # SciencePlots style removed")
    Path("src/plotting.py").write_text(pp)
    mp = Path("main.py").read_text()
    # pandas 3.0 removed DataFrame.append -> use concat
    mp = mp.replace("df = df.append(result, ignore_index=True)",
                    "df = pd.concat([df, pd.DataFrame([result])], ignore_index=True)")
    # torch 2.4: reimplement TransformerEncoder/Decoder.forward WITHOUT is_causal so
    # TranAD's custom (old-torch) layers are called with only the args they accept.
    mpatch = (
        "\nimport torch as _t\n"
        "def _enc_fwd(self, src, mask=None, src_key_padding_mask=None, **kw):\n"
        "\tout = src\n"
        "\tfor mod in self.layers: out = mod(out, src_mask=mask, src_key_padding_mask=src_key_padding_mask)\n"
        "\treturn self.norm(out) if self.norm is not None else out\n"
        "def _dec_fwd(self, tgt, memory, tgt_mask=None, memory_mask=None, "
        "tgt_key_padding_mask=None, memory_key_padding_mask=None, **kw):\n"
        "\tout = tgt\n"
        "\tfor mod in self.layers: out = mod(out, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, "
        "tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)\n"
        "\treturn self.norm(out) if self.norm is not None else out\n"
        "_t.nn.TransformerEncoder.forward = _enc_fwd\n"
        "_t.nn.TransformerDecoder.forward = _dec_fwd\n"
        # let .numpy() work on GPU tensors (auto-move to CPU) so scoring can stay on GPU
        "_np_orig = _t.Tensor.numpy\n"
        "def _np_cpu(self, *a, **k): return _np_orig(self.detach().cpu(), *a, **k)\n"
        "try:\n\t_t.Tensor.numpy = _np_cpu\nexcept Exception as _e:\n\tprint('[patch] numpy patch failed', _e, flush=True)\n")
    mp = mp.replace("from pprint import pprint\n", "from pprint import pprint\n" + mpatch, 1)
    # allow epoch override for fast GDN feedback (harmless no-op if literal differs)
    mp = mp.replace("num_epochs = 5", "num_epochs = int(os.environ.get('NEPOCHS', '5'))")
    # --- GDN DEVICE PATCH: harness is CPU-only (.double(), no .cuda()). Force CUDA when
    #     USE_CUDA=1: move model + its dgl graph to GPU, and move batch tensors at backprop entry.
    if os.environ.get("USE_CUDA") == "1":
        mp = mp.replace(
            "model = model_class(dims).double()",
            "model = model_class(dims).double()\n"
            "\tif _t.cuda.is_available():\n"
            "\t\tmodel = model.cuda()\n"
            "\t\tfor _a in ('g','graph','edge_index'):\n"
            "\t\t\tif hasattr(model,_a) and hasattr(getattr(model,_a),'to'):\n"
            "\t\t\t\ttry: setattr(model,_a, getattr(model,_a).to('cuda'))\n"
            "\t\t\t\texcept Exception as _e: print('[gdn] graph move fail', _e, flush=True)\n"
            "\t\tprint('[gdn] model on cuda:', next(model.parameters()).is_cuda, flush=True)")
        mp = mp.replace(
            "def backprop(epoch, model, data, dataO, optimizer, scheduler, training = True):",
            "def backprop(epoch, model, data, dataO, optimizer, scheduler, training = True):\n"
            "\t_t.set_grad_enabled(bool(training))   # eval: no autograd graph -> no memory blowup\n"
            "\tif _t.cuda.is_available() and next(model.parameters()).is_cuda:   # keep GPU for train AND score\n"
            "\t\ttry: data, dataO = data.cuda(), dataO.cuda()\n"
            "\t\texcept Exception: pass")
    # dump the aggregate per-timestep score + labels for RAW-F1 recompute
    save = ("\tnp.save(f'/results/score_{args.model}.npy', lossFinal)\n"
            "\tnp.save(f'/results/labels_{args.model}.npy', labelsFinal)\n")
    tgt = "\tresult, _ = pot_eval(lossTfinal, lossFinal, labelsFinal)"
    mp = mp.replace(tgt, save + tgt, 1)
    Path("main.py").write_text(mp)
    print(f"[sota] patched: dgl-optional + score dump ({'save inserted' if save in mp else 'INSERT FAILED'})", flush=True)

    # place the selected dataset's arrays as processed/WADI/{train,test,labels}.npy (name is just a slot)
    proc = f"{repo}/processed/WADI"; os.makedirs(proc, exist_ok=True)
    pfx = "wadi" if ds == "WADI" else ds
    tr = np.load(f"/app/{pfx}_train.npy"); te = np.load(f"/app/{pfx}_test.npy"); lb = np.load(f"/app/{pfx}_labels.npy")
    feats = tr.shape[1]
    np.save(f"{proc}/train.npy", tr.astype(np.float64))
    np.save(f"{proc}/test.npy", te.astype(np.float64))
    np.save(f"{proc}/labels.npy", np.tile(lb.reshape(-1, 1), (1, feats)).astype(np.float64))  # (T,feats)
    print(f"[sota] wrote processed/WADI train{tr.shape} test{te.shape} labels(T,{feats})", flush=True)

    def raw_f1(score, y):
        """best raw point-wise F1 over thresholds + F1@5%FPR; score,y are 1-D per-timestep."""
        from sklearn.metrics import f1_score
        y = y.astype(int)
        qs = np.quantile(score, np.linspace(0.80, 0.999, 60))
        best = max(f1_score(y, score > t) for t in qs)
        thr = np.quantile(score[y == 0], 0.95); tpr = float((score[y == 1] > thr).mean())
        f5 = f1_score(y, score > thr)
        return dict(bestF1=round(float(best), 3), F1_at5pctFPR=round(float(f5), 3),
                    TPR_at5pctFPR=round(tpr, 3))

    triv = np.load("/app/wadi_triv_test.npy") if ds == "WADI" and os.path.exists("/app/wadi_triv_test.npy") else None
    thr_triv = float(np.load("/app/wadi_triv_thr.npy")) if triv is not None else 0.0
    mdls = [m.strip() for m in models.split(",") if m.strip()]
    if "GDN" in mdls:                                    # GDN needs a real dgl (graph attention)
        smoke = "python -c \"import dgl,torch; from dgl.nn import GATConv; print('DGL_OK',dgl.__version__)\""
        ok, so, se = sh(smoke)
        if ok != 0:
            for cmd in ["pip install -q dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html",
                        "pip install -q dgl -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html",
                        "pip install -q dgl -f https://data.dgl.ai/wheels/torch-2.4/repo.html",
                        "pip install -q 'dgl' -f https://data.dgl.ai/wheels/repo.html",
                        "pip install -q dglgo dgl"]:
                rc, io, ie = sh(cmd)
                ok, so, se = sh(smoke)
                print(f"[sota] dgl try: {cmd[:60]}... install_rc={rc} smoke_ok={ok==0} :: {(so or se).strip()[-160:]}", flush=True)
                if ok == 0:
                    break
        out["recon"]["dgl_smoke"] = (so or "") + " || " + (se or "")
        print(f"[sota] FINAL dgl smoke: {'OK ' + so.strip() if ok==0 else 'FAILED :: ' + (se or '')[-300:]}", flush=True)
    import threading, time
    def gpu_note(tag):
        try:
            import torch
            print(f"[gpu] {tag} alloc={torch.cuda.memory_allocated()/1e9:.2f}G "
                  f"reserved={torch.cuda.memory_reserved()/1e9:.2f}G", flush=True)
        except Exception as e:
            print(f"[gpu] {tag} err {e}", flush=True)
    try:
        import torch
        print(f"[gpu] device={torch.cuda.get_device_name(0)} "
              f"total={torch.cuda.get_device_properties(0).total_memory/1e9:.1f}G", flush=True)
    except Exception as e:
        print(f"[gpu] device probe err {e}", flush=True)

    def stream_run(cmd, env, max_s=1800):
        """Stream child stdout live (NOT captured) + 30-min watchdog + GPU heartbeat."""
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1, env=env)
        killed = {"v": False}
        def watchdog():
            t = 0
            while p.poll() is None and t < max_s:
                time.sleep(10); t += 10
                if t % 30 == 0: gpu_note(f"heartbeat t+{t}s")
            if p.poll() is None:
                killed["v"] = True; print(f"[sota] WATCHDOG kill after {max_s}s", flush=True); p.terminate()
        threading.Thread(target=watchdog, daemon=True).start()
        buf = []
        for line in iter(p.stdout.readline, ''):
            print(line.rstrip(), flush=True); buf.append(line)
        p.wait()
        return (124 if killed["v"] else p.returncode), "".join(buf)

    for model in mdls:
        print(f"[sota] === training {model} (STREAMED, NEPOCHS={os.environ.get('NEPOCHS','5')}) ===", flush=True)
        gpu_note("pre-train")
        try:
            rc, log = stream_run(f"python -u main.py --model {model} --dataset WADI --retrain",
                                 {**os.environ, "MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1"})
        except Exception:
            import traceback; rc, log = 99, traceback.format_exc(); print(log, flush=True)
        gpu_note("post-train")
        res = {"rc": rc, "log_tail": log[-3000:]}
        sp, lp = f"/results/score_{model}.npy", f"/results/labels_{model}.npy"
        if os.path.exists(sp) and os.path.exists(lp):
            s = np.load(sp); y = np.load(lp).astype(int)
            s = s.mean(1) if s.ndim > 1 else s
            res["raw_ALL"] = raw_f1(s, y)
            if triv is not None:                            # WADI-only easy/hard raw split
                t = triv[-len(s):]; easy = (y == 1) & (t > thr_triv); hard = (y == 1) & ~easy
                res["n_easy_hard"] = [int(easy.sum()), int(hard.sum())]
                if hard.sum() > 3:
                    res["raw_HARD"] = raw_f1(s[(y == 0) | hard], y[(y == 0) | hard])
                if easy.sum() > 3:
                    res["raw_EASY"] = raw_f1(s[(y == 0) | easy], y[(y == 0) | easy])
            # tag the per-timestep arrays with the dataset so the local comparison can align them
            for a, b in [(sp, f"/results/score_{model}_{ds}.npy"), (lp, f"/results/labels_{model}_{ds}.npy")]:
                shutil.copy(a, b)
        # --- FULL-TEST scoring: score all 10 phase-offsets of the full-res WADI test with NO
        #     retraining (reuse the checkpoint the --retrain run just saved). Same model, all anomalies. ---
        if ds == "WADI" and model != "GDN" and rc == 0 and os.path.exists("/app/wadi_testfull.npy"):
            tef = np.load("/app/wadi_testfull.npy"); lbf = np.load("/app/wadi_labelsfull.npy"); feats = tef.shape[1]
            offs, lens = [], []
            for o in range(10):
                te_o, lb_o = tef[o::10], lbf[o::10]
                np.save(f"{proc}/test.npy", te_o.astype(np.float64))
                np.save(f"{proc}/labels.npy", np.tile(lb_o.reshape(-1, 1), (1, feats)).astype(np.float64))
                r2 = subprocess.run(f"python -u main.py --model {model} --dataset WADI",   # NO --retrain
                                    shell=True, capture_output=True, text=True,
                                    env={**os.environ, "MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1"})
                if os.path.exists(sp):
                    ss = np.load(sp); ss = ss.mean(1) if ss.ndim > 1 else ss; offs.append(ss); lens.append(len(ss))
                print(f"[sota] {model} full-offset {o} rc={r2.returncode} len={lens[-1] if lens else 0}", flush=True)
            if offs:
                np.save(f"/results/scorefull_{model}_WADI.npy", np.concatenate(offs))
                np.save(f"/results/scorefull_{model}_WADI_lens.npy", np.array(lens))
                results_vol.commit()
                print(f"[sota] {model} FULL-TEST saved: {len(offs)} offsets, total {sum(lens)} pts", flush=True)
        out["results"][model] = res
        Path("/results/sota_out.json").write_text(json.dumps(out, indent=2)); results_vol.commit()  # incremental
        print(f"[sota] {model} rc={rc} raw={res.get('raw_ALL')}", flush=True)

    Path("/results").mkdir(exist_ok=True)
    Path("/results/sota_out.json").write_text(json.dumps(out, indent=2))
    results_vol.commit()
    print("=== DONE ===", flush=True)
    return out


@app.local_entrypoint()
def main(models: str = "USAD,TranAD", epochs: int = 5, ds: str = "WADI"):
    r = run_sota.remote(models=models, epochs=epochs, recon=False, ds=ds)
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "sota_out.json").write_text(json.dumps(r, indent=2))
    # print recon + result tails for quick inspection
    for k, v in r.get("recon", {}).items():
        print(f"\n===== RECON: {k} =====\n{v[:1500]}")
    if "dgl_smoke" in r.get("recon", {}):
        print(f"\n===== dgl smoke =====\n{r['recon']['dgl_smoke'][:400]}")
    for m, res in r.get("results", {}).items():
        print(f"\n===== {m} rc={res['rc']} =====")
        for k in ("raw_ALL", "raw_EASY", "raw_HARD", "n_easy_hard"):
            if k in res:
                print(f"  {k}: {res[k]}")
        if res.get("rc") != 0 and res.get("log_tail"):
            print(res["log_tail"][-1800:])
