"""Run SOTA CPS anomaly detectors (USAD / TranAD / GDN) on WADI/HAI/SWaT via the official
TranAD harness (imperial-qore/TranAD), recomputing RAW point-wise metrics (their eval is
point-adjusted). Each (dataset, model) trains in its OWN Modal container, so the full
3x3 matrix runs in parallel; wall-clock = slowest single job, not the sum.

Speed: the harness defaults to float64 (.double()), which an A10 runs ~1/32 speed. We patch
to float32 and keep model+data on CUDA -> large per-epoch speedup. Arrays are standardize+clip
preprocessed to match our detector (prep_sota_general.py).

Invoke (all 9):   modal run modal_sota.py --datasets "WADI,HAI,SWaT" --models "USAD,TranAD,GDN" --epochs 30
Smoke (one job):  modal run modal_sota.py --smoke "WADI:USAD" --epochs 5
Outputs: printed JSON + /results score arrays on the results volume.
"""
from __future__ import annotations
import json
from pathlib import Path
import modal

HERE = Path(__file__).parent.resolve()
# per-dataset array sets (train/test/labels + trivial-split arrays for easy/difficult)
NPY = []
for pfx in ["wadi", "HAI", "SWaT"]:
    NPY += [f"{pfx}_train.npy", f"{pfx}_test.npy", f"{pfx}_labels.npy",
            f"{pfx}_triv_test.npy", f"{pfx}_triv_thr.npy"]

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

PFX = {"WADI": "wadi", "HAI": "HAI", "SWaT": "SWaT"}


def _patch_harness(repo, fp32=True):
    """Apply all harness patches: dgl-optional, torch-2.4 transformer fwd, numpy->cpu,
    float32 (speed), CUDA device move, and raw per-timestep score dump."""
    import os
    from pathlib import Path as P
    mm = P(f"{repo}/src/models.py").read_text()
    mm = mm.replace("import dgl\n", "try:\n\timport dgl\nexcept Exception:\n\tdgl = None\n")
    mm = mm.replace("from dgl.nn import GATConv\n",
                    "try:\n\tfrom dgl.nn import GATConv\nexcept Exception:\n\tGATConv = None\n")
    # USAD forward: accept a BATCH of windows (2-D) for fast minibatch training, while keeping
    # the exact per-window (1-D) path for scoring. Batched output keeps the leading dim.
    mm = mm.replace("z = self.encoder(g.view(1,-1))",
                    "z = self.encoder(g.view(g.size(0), -1) if g.dim() > 1 else g.view(1,-1))")
    mm = mm.replace("return ae1.view(-1), ae2.view(-1), ae2ae1.view(-1)",
                    "return (ae1, ae2, ae2ae1) if g.dim() > 1 else (ae1.view(-1), ae2.view(-1), ae2ae1.view(-1))")
    P(f"{repo}/src/models.py").write_text(mm)
    pp = P(f"{repo}/src/plotting.py").read_text()
    pp = pp.replace("plt.style.use(['science', 'ieee'])", "pass  # SciencePlots removed")
    P(f"{repo}/src/plotting.py").write_text(pp)
    mp = P(f"{repo}/main.py").read_text()
    mp = mp.replace("df = df.append(result, ignore_index=True)",
                    "df = pd.concat([df, pd.DataFrame([result])], ignore_index=True)")
    # torch 2.4: reimplement Transformer{Encoder,Decoder}.forward WITHOUT is_causal
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
        "_np_orig = _t.Tensor.numpy\n"
        "def _np_cpu(self, *a, **k): return _np_orig(self.detach().cpu(), *a, **k)\n"
        "try:\n\t_t.Tensor.numpy = _np_cpu\nexcept Exception as _e:\n\tprint('[patch] numpy patch failed', _e, flush=True)\n")
    mp = mp.replace("from pprint import pprint\n", "from pprint import pprint\n" + mpatch, 1)
    mp = mp.replace("num_epochs = 5", "num_epochs = int(os.environ.get('NEPOCHS', '5'))")
    # USAD training: replace per-window SGD (one optimizer.step per window, ~N iterations/epoch)
    # with minibatch SGD (batch SOTA_BS). ~N/BS iterations -> removes the Python-loop bottleneck.
    # Scoring (else-branch, plural->singular names) is untouched. USAD-specific by the plural names.
    mp = mp.replace(
        "\t\tif training:\n\t\t\tfor d in data:\n\t\t\t\tae1s, ae2s, ae2ae1s = model(d)",
        "\t\tif training:\n\t\t\t_BS = int(os.environ.get('SOTA_BS','128'))\n"
        "\t\t\tfor _i in range(0, len(data), _BS):\n\t\t\t\td = data[_i:_i+_BS]\n"
        "\t\t\t\tae1s, ae2s, ae2ae1s = model(d)")
    # FLOAT32 speedup: harness is float64 by default (slow on A10). Cast model + all loaded
    # tensors to float32. pot_eval math is numpy, unaffected.
    if fp32:
        mp = mp.replace(".double()", ".float()")
        mp = mp.replace("torch.DoubleTensor", "torch.FloatTensor")
        mp = mp.replace("torch.float64", "torch.float32").replace("torch.double", "torch.float32")
    # CUDA device move: model + dgl graph to GPU; batches to GPU at backprop entry.
    mp = mp.replace(
        "model = model_class(dims).double()" if not fp32 else "model = model_class(dims).float()",
        ("model = model_class(dims).float()\n" if fp32 else "model = model_class(dims).double()\n") +
        "\tif _t.cuda.is_available():\n"
        "\t\tmodel = model.cuda()\n"
        "\t\tfor _a in ('g','graph','edge_index'):\n"
        "\t\t\tif hasattr(model,_a) and hasattr(getattr(model,_a),'to'):\n"
        "\t\t\t\ttry: setattr(model,_a, getattr(model,_a).to('cuda'))\n"
        "\t\t\t\texcept Exception as _e: print('[dev] graph move fail', _e, flush=True)\n"
        "\t\tprint('[dev] model on cuda:', next(model.parameters()).is_cuda, flush=True)")
    mp = mp.replace(
        "def backprop(epoch, model, data, dataO, optimizer, scheduler, training = True):",
        "def backprop(epoch, model, data, dataO, optimizer, scheduler, training = True):\n"
        "\t_t.set_grad_enabled(bool(training))\n"
        "\tif _t.cuda.is_available() and next(model.parameters()).is_cuda:\n"
        "\t\ttry: data, dataO = data.cuda().float(), dataO.cuda().float()\n"
        "\t\texcept Exception: pass")
    # dump aggregate per-timestep score + labels; guard pot_eval so an fp32/POT hiccup
    # cannot lose an already-dumped score array.
    save = ("\tnp.save(f'/results/score_{args.model}_{args.dataset}.npy', lossFinal)\n"
            "\tnp.save(f'/results/labels_{args.model}_{args.dataset}.npy', labelsFinal)\n")
    tgt = "\tresult, _ = pot_eval(lossTfinal, lossFinal, labelsFinal)"
    guarded = save + "\ttry:\n\t\tresult, _ = pot_eval(lossTfinal, lossFinal, labelsFinal)\n\texcept Exception as _e:\n\t\tprint('[pot] skipped', _e, flush=True); result = {}\n"
    mp = mp.replace(tgt, guarded, 1)
    P(f"{repo}/main.py").write_text(mp)
    return save in mp


@app.function(gpu="A10G", timeout=2 * 60 * 60, memory=16384, volumes={"/results": results_vol})
def run_one(ds: str, model: str, epochs: int = 5, fp32: bool = True) -> dict:
    import subprocess, os, sys, time, threading, numpy as np
    from sklearn.metrics import f1_score, roc_auc_score
    os.environ["MKL_THREADING_LAYER"] = "GNU"
    os.environ["MKL_SERVICE_FORCE_INTEL"] = "0"
    os.environ["NEPOCHS"] = str(epochs)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    def sh(cmd, **kw):
        print(f"[sota] $ {cmd}", flush=True)
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)
        return r.returncode, r.stdout, r.stderr

    os.chdir("/root")
    sh("git clone --depth 1 https://github.com/imperial-qore/TranAD.git")
    repo = "/root/TranAD"; os.chdir(repo)
    ok_dump = _patch_harness(repo, fp32=fp32)
    print(f"[sota] patched ds={ds} model={model} fp32={fp32} dump={'ok' if ok_dump else 'FAIL'}", flush=True)

    for stale in (f"/results/score_{model}_{ds}.npy", f"/results/labels_{model}_{ds}.npy"):
        try: os.remove(stale)                       # avoid a crashed run reporting a prior run's scores
        except OSError: pass
    pfx = PFX[ds]
    proc = f"{repo}/processed/WADI"; os.makedirs(proc, exist_ok=True)   # 'WADI' is just the slot name
    tr = np.load(f"/app/{pfx}_train.npy"); te = np.load(f"/app/{pfx}_test.npy"); lb = np.load(f"/app/{pfx}_labels.npy")
    feats = tr.shape[1]
    np.save(f"{proc}/train.npy", tr.astype(np.float32))          # float32 end-to-end (fp32 harness)
    np.save(f"{proc}/test.npy", te.astype(np.float32))
    np.save(f"{proc}/labels.npy", np.tile(lb.reshape(-1, 1), (1, feats)).astype(np.float32))
    print(f"[sota] {ds} train{tr.shape} test{te.shape} anom={lb.mean():.3f}", flush=True)

    def metrics(score, y):
        """AUROC + best raw point-wise F1 + F1/TPR at 5% train-normal FPR. 1-D per-timestep."""
        y = y.astype(int)
        out = {}
        try: out["AUROC"] = round(float(roc_auc_score(y, score)), 3)
        except Exception: out["AUROC"] = None
        qs = np.quantile(score, np.linspace(0.80, 0.999, 60))
        out["bestF1"] = round(float(max(f1_score(y, score > t) for t in qs)), 3)
        thr = np.quantile(score[y == 0], 0.95)
        out["F1_at5pctFPR"] = round(float(f1_score(y, score > thr)), 3)
        out["TPR_at5pctFPR"] = round(float((score[y == 1] > thr).mean()), 3)
        return out

    triv = np.load(f"/app/{pfx}_triv_test.npy")
    thr_triv = float(np.load(f"/app/{pfx}_triv_thr.npy"))

    # GDN needs a real dgl (graph attention); install into this container only when needed.
    if model == "GDN":
        smoke = "python -c \"import dgl,torch; from dgl.nn import GATConv; print('DGL_OK',dgl.__version__)\""
        okc, so, se = sh(smoke)
        if okc != 0:
            for cmd in ["pip install -q dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html",
                        "pip install -q dgl -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html",
                        "pip install -q dgl -f https://data.dgl.ai/wheels/torch-2.4/repo.html"]:
                sh(cmd); okc, so, se = sh(smoke)
                if okc == 0: break
        print(f"[sota] dgl smoke ds={ds}: {'OK '+so.strip() if okc==0 else 'FAIL '+(se or '')[-200:]}", flush=True)

    def gpu_note(tag):
        try:
            import torch
            print(f"[gpu] {ds}:{model} {tag} alloc={torch.cuda.memory_allocated()/1e9:.2f}G", flush=True)
        except Exception as e:
            print(f"[gpu] {tag} err {e}", flush=True)

    def stream_run(cmd, env, max_s=6600):
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1, env=env)
        killed = {"v": False}
        def watchdog():
            t = 0
            while p.poll() is None and t < max_s:
                time.sleep(15); t += 15
                if t % 60 == 0: gpu_note(f"heartbeat t+{t}s")
            if p.poll() is None:
                killed["v"] = True; print(f"[sota] {ds}:{model} WATCHDOG kill {max_s}s", flush=True); p.terminate()
        threading.Thread(target=watchdog, daemon=True).start()
        buf = []
        for line in iter(p.stdout.readline, ''):
            print(f"[{ds}:{model}] {line.rstrip()}", flush=True); buf.append(line)
        p.wait()
        return (124 if killed["v"] else p.returncode), "".join(buf)

    print(f"[sota] === training {ds}:{model} (NEPOCHS={epochs}, fp32={fp32}) ===", flush=True)
    t0 = time.time()
    try:
        rc, log = stream_run(f"python -u main.py --model {model} --dataset WADI --retrain",
                             {**os.environ, "MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1"})
    except Exception:
        import traceback; rc, log = 99, traceback.format_exc(); print(log, flush=True)
    train_s = round(time.time() - t0, 1)

    res = {"dataset": ds, "model": model, "rc": rc, "train_s": train_s, "epochs": epochs, "fp32": fp32}
    sp = f"/results/score_{model}_{ds}.npy"; lp = f"/results/labels_{model}_{ds}.npy"
    if os.path.exists(sp) and os.path.exists(lp):
        s = np.load(sp); y = np.load(lp).astype(int)
        s = s.mean(1) if s.ndim > 1 else s
        t = triv[-len(s):]                                # align trivial score to dumped length
        yy = y[-len(s):] if len(y) != len(s) else y
        res["n"] = int(len(s)); res["n_anom"] = int(yy.sum())
        res["raw_ALL"] = metrics(s, yy)
        easy = (yy == 1) & (t > thr_triv); hard = (yy == 1) & ~easy
        res["n_easy_hard"] = [int(easy.sum()), int(hard.sum())]
        if hard.sum() > 3:
            m = (yy == 0) | hard; res["raw_HARD"] = metrics(s[m], yy[m])
        if easy.sum() > 3:
            m = (yy == 0) | easy; res["raw_EASY"] = metrics(s[m], yy[m])
    else:
        res["error"] = "no score dumped"; res["log_tail"] = log[-2000:]
    Path(f"/results/one_{ds}_{model}.json").write_text(json.dumps(res, indent=2)); results_vol.commit()
    print(f"[sota] DONE {ds}:{model} rc={rc} {train_s}s raw_ALL={res.get('raw_ALL')} "
          f"raw_HARD={res.get('raw_HARD')}", flush=True)
    return res


# per-model epochs: USAD is minibatched here (cheap) -> train to convergence; TranAD already
# minibatches (harness default 5); GDN is per-window (slow) -> keep the standard 5.
EPO = {"USAD": 30, "TranAD": 5, "GDN": 5}


@app.local_entrypoint()
def main(datasets: str = "WADI,HAI,SWaT", models: str = "USAD,TranAD,GDN",
         epochs: int = 0, smoke: str = "", fp32: bool = True):
    ep = lambda m: (epochs if epochs else EPO.get(m, 5))
    if smoke:                                            # e.g. "WADI:USAD"
        d, m = smoke.split(":"); matrix = [(d, m, ep(m), fp32)]
    else:
        matrix = [(d, m, ep(m), fp32) for d in datasets.split(",") for m in models.split(",")]
    print(f"[sota] launching {len(matrix)} parallel jobs: {[(d,m) for d,m,_,_ in matrix]}", flush=True)
    results = list(run_one.starmap(matrix))             # parallel: one container per (ds,model)
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "sota_matrix.json").write_text(json.dumps(results, indent=2))
    print("\n==================== SUMMARY ====================")
    for r in sorted(results, key=lambda x: (x.get("dataset", ""), x.get("model", ""))):
        a = r.get("raw_ALL", {}); h = r.get("raw_HARD", {})
        print(f"{r.get('dataset'):5} {r.get('model'):7} rc={r.get('rc')} {r.get('train_s')}s "
              f"| ALL auroc={a.get('AUROC')} f1={a.get('bestF1')} "
              f"| HARD auroc={h.get('AUROC')} f1={h.get('bestF1')} | n_eh={r.get('n_easy_hard')}")
