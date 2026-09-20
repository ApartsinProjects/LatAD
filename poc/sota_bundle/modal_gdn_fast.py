"""FAST canonical GDN: batches the per-window DGL graph-attention forward (root cause of the
~80-90 min stride-only run: no batching path exists for GDN in the TranAD harness, unlike USAD which
modal_sota.py already batches) so training uses minibatch SGD (batch=SOTA_BS) instead of one
optimizer.step() per window, and runs on a GPU now that there is real batched compute to amortize
kernel-launch cost. TEST/scoring stays on the EXACT original per-window path (unchanged) - only the
TRAINING forward is batched, so any discrepancy vs the safety-net stride run can only come from
optimizer dynamics (batch-averaged vs per-window gradients), never from a scoring-path bug.

Correctness invariant (REQUIRED before trusting this path): forward_batch(data) called with batch
size 1 must reproduce model.forward(data[0]) exactly (same weights, same output, float32 tolerance).
This is checked by `--verify` before any real training runs. If it fails, this script refuses to
train and the safety-net stride run (modal_sota.py, already completed/running) is the numbers to use.

Resumable: each dataset's result is checkpointed to the `latad-sota-results` volume as
gdn_fast_<ds>.done + gdn_fast_scores_<ds>.npz the moment it finishes; a re-run skips any dataset
whose .done marker already exists on the volume, so relaunching after a crash/OOM only redoes the
missing dataset(s). Outputs are written under distinct `gdn_fast_*` names - they never overwrite the
stride safety-net run's `score_GDN_<ds>_s0.npy` files.

Invoke:
  modal run modal_gdn_fast.py --verify                      # forward-batch exactness check only, no training
  modal run modal_gdn_fast.py --datasets "WADI_clean,HAI,SWaT_canon" --epochs 5 --batch-size 256
"""
from __future__ import annotations
import json
from pathlib import Path
import modal

HERE = Path(__file__).parent.resolve()
NPY = []
for pfx in ["wadi_clean", "HAI", "SWaT_canon"]:
    NPY += [f"{pfx}_train.npy", f"{pfx}_test.npy", f"{pfx}_labels.npy",
            f"{pfx}_triv_test.npy", f"{pfx}_triv_thr.npy"]

image = (
    modal.Image.from_registry("pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel", add_python="3.11")
    .apt_install("git")
    .pip_install("numpy<2", "pandas", "scikit-learn", "scipy", "tqdm", "matplotlib")
    .pip_install("dgl", find_links="https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html")
)
for f in NPY:
    lp = HERE / f
    if lp.exists():
        image = image.add_local_file(str(lp), f"/app/{f}")

app = modal.App("latad-gdn-fast", image=image)
results_vol = modal.Volume.from_name("latad-sota-results", create_if_missing=True)

PFX = {"WADI_clean": "wadi_clean", "HAI": "HAI", "SWaT_canon": "SWaT_canon"}


def _patch_harness(repo, fp32=True):
    """Same base patches as modal_sota.py's _patch_harness (dgl-optional import, torch-2.4
    transformer forward, float32, per-timestep score dump) PLUS the GDN batched-forward model
    patch and the GDN minibatch-training patch. Kept as a near-duplicate (not imported from
    modal_sota) so this file is self-contained inside the Modal container."""
    from pathlib import Path as P
    mm = P(f"{repo}/src/models.py").read_text()
    mm = mm.replace("import dgl\n", "try:\n\timport dgl\nexcept Exception:\n\tdgl = None\n")
    mm = mm.replace("from dgl.nn import GATConv\n",
                    "try:\n\tfrom dgl.nn import GATConv\nexcept Exception:\n\tGATConv = None\n")
    # --- GDN batched forward: add forward_batch(data) alongside the untouched forward(data). ---
    # CAUGHT BY INSPECTION (not by a failed run): self.attention ends in nn.Softmax(dim=0), which
    # is only correct for an UNBATCHED (n_window,) input (dim=0 = the window axis). Naively calling
    # self.attention(batched_data) on a (B, n_window*n_feats) input would softmax ACROSS THE BATCH
    # instead of across each sample's 5 window positions - silently wrong. forward_batch below
    # re-applies the same Linear/LeakyReLU layers by index and does the softmax explicitly on
    # dim=1, then batches ONLY the expensive GATConv step via dgl.batch (graph topology is fixed -
    # channels never change - only node features vary per window, so B disjoint copies of the same
    # graph is an exact vectorization of calling forward() B times, not an approximation).
    gdn_anchor = (
        "\tdef forward(self, data):\n"
        "\t\t# Bahdanau style attention\n"
        "\t\tatt_score = self.attention(data).view(self.n_window, 1)\n"
        "\t\tdata = data.view(self.n_window, self.n_feats)\n"
        "\t\tdata_r = torch.matmul(data.permute(1, 0), att_score)\n"
        "\t\t# GAT convolution on complete graph\n"
        "\t\tfeat_r = self.feature_gat(self.g, data_r)\n"
        "\t\tfeat_r = feat_r.view(self.n_feats, self.n_feats)\n"
        "\t\t# Pass through a FCN\n"
        "\t\tx = self.fcn(feat_r)\n"
        "\t\treturn x.view(-1)\n"
    )
    gdn_batched = gdn_anchor + (
        "\n\tdef forward_batch(self, data):\n"
        "\t\t# data: (B, n_window*n_feats). Exact vectorization of forward() over B windows -\n"
        "\t\t# graph topology (self.g) is identical for every window, only node features change.\n"
        "\t\tB = data.shape[0]\n"
        "\t\traw = data\n"
        "\t\tfor _layer in self.attention[:-1]:\n"
        "\t\t\traw = _layer(raw)\n"
        "\t\tatt_score = torch.softmax(raw, dim=1).view(B, self.n_window, 1)\n"
        "\t\td = data.view(B, self.n_window, self.n_feats)\n"
        "\t\tdata_r = torch.matmul(d.permute(0, 2, 1), att_score).reshape(B * self.n_feats, 1)\n"
        "\t\tif getattr(self, '_bg_B', None) != B:\n"
        "\t\t\tself._bg = dgl.batch([self.g] * B).to(data.device)\n"
        "\t\t\tself._bg_B = B\n"
        "\t\tfeat_r = self.feature_gat(self._bg, data_r)\n"
        "\t\tfeat_r = feat_r.reshape(B, self.n_feats, self.n_feats)\n"
        "\t\tx = self.fcn(feat_r)\n"
        "\t\treturn x.reshape(B, -1)\n"
    )
    mm = mm.replace(gdn_anchor, gdn_batched, 1)
    # USAD batching (unchanged from modal_sota.py)
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
        "try:\n\t_t.Tensor.numpy = _np_cpu\nexcept Exception as _e:\n\tprint('[patch] numpy patch failed', _e, flush=True)\n"
        "import os as _os3, random as _r3, numpy as _np3\n"
        "_SD = int(_os3.environ.get('SOTA_SEED', '0'))\n"
        "_t.manual_seed(_SD); _np3.random.seed(_SD); _r3.seed(_SD)\n"
        "try:\n\t_t.cuda.manual_seed_all(_SD)\nexcept Exception:\n\tpass\n")
    mp = mp.replace("from pprint import pprint\n", "from pprint import pprint\n" + mpatch, 1)
    mp = mp.replace("num_epochs = 5", "num_epochs = int(os.environ.get('NEPOCHS', '5'))")
    if fp32:
        mp = mp.replace(".double()", ".float()")
        mp = mp.replace("torch.DoubleTensor", "torch.FloatTensor")
        mp = mp.replace("torch.float64", "torch.float32").replace("torch.double", "torch.float32")
    mp = mp.replace(
        "def backprop(epoch, model, data, dataO, optimizer, scheduler, training = True):",
        "def backprop(epoch, model, data, dataO, optimizer, scheduler, training = True):\n"
        "\t_t.set_grad_enabled(bool(training))")
    # GDN: minibatch the TRAINING loop only via forward_batch; eval/scoring (else branch) is
    # BYTE-FOR-BYTE the original per-window loop - untouched, so scores come from the exact same
    # code path as the stride safety-net run and every other baseline.
    gdn_train_anchor = (
        "\t\tif training:\n"
        "\t\t\tfor i, d in enumerate(data):\n"
        "\t\t\t\tif 'MTAD_GAT' in model.name: \n"
        "\t\t\t\t\tx, h = model(d, h if i else None)\n"
        "\t\t\t\telse:\n"
        "\t\t\t\t\tx = model(d)\n"
        "\t\t\t\tloss = torch.mean(l(x, d))\n"
        "\t\t\t\tl1s.append(torch.mean(loss).item())\n"
        "\t\t\t\toptimizer.zero_grad()\n"
        "\t\t\t\tloss.backward()\n"
        "\t\t\t\toptimizer.step()\n"
    )
    # NOTE: this nests the batched/per-window CHOICE one level INSIDE the original `if training:`
    # block (as an inner if/else at the same depth the original `for` loop had), rather than
    # restructuring `if training: / else:` into `if training and X: / elif training:` - the first
    # attempt at this patch did the latter and orphaned the trailing `tqdm.write(...)` / `return`
    # (they stayed lexically attached to only the last-written branch, so the OTHER branch's
    # backprop() call implicitly returned None -> `TypeError: cannot unpack non-iterable
    # NoneType object` at the `lossT, lr = backprop(...)` call site). Caught on the first live
    # GPU run (WADI_clean/HAI/SWaT_canon all crashed identically at epoch 0, ~17-56s each, no
    # real cost) - keeping this as a recorded lesson, not just a silent fix.
    gdn_train_batched = (
        "\t\tif training:\n"
        "\t\t\tif model.name == 'GDN' and hasattr(model, 'forward_batch'):\n"
        "\t\t\t\t_BS = int(os.environ.get('SOTA_BS', '256'))\n"
        "\t\t\t\tfor _i in range(0, data.shape[0], _BS):\n"
        "\t\t\t\t\td = data[_i:_i + _BS]\n"
        "\t\t\t\t\tx = model.forward_batch(d)\n"
        "\t\t\t\t\tloss = torch.mean(l(x, d))\n"
        "\t\t\t\t\tl1s.append(loss.item())\n"
        "\t\t\t\t\toptimizer.zero_grad()\n"
        "\t\t\t\t\tloss.backward()\n"
        "\t\t\t\t\toptimizer.step()\n"
        "\t\t\telse:\n"
        "\t\t\t\tfor i, d in enumerate(data):\n"
        "\t\t\t\t\tif 'MTAD_GAT' in model.name: \n"
        "\t\t\t\t\t\tx, h = model(d, h if i else None)\n"
        "\t\t\t\t\telse:\n"
        "\t\t\t\t\t\tx = model(d)\n"
        "\t\t\t\t\tloss = torch.mean(l(x, d))\n"
        "\t\t\t\t\tl1s.append(torch.mean(loss).item())\n"
        "\t\t\t\t\toptimizer.zero_grad()\n"
        "\t\t\t\t\tloss.backward()\n"
        "\t\t\t\t\toptimizer.step()\n"
    )
    assert gdn_train_anchor in mp, "GDN training-loop anchor not found - harness source changed"
    mp = mp.replace(gdn_train_anchor, gdn_train_batched, 1)
    # Move model + GDN graph + WINDOWED train/test data onto CUDA when GDN_GPU=1 (set only for
    # GDN jobs). trainO/testO are deliberately left UNTOUCHED (still the original, pre-window,
    # CPU arrays): GDN's backprop() never reads the dataO parameter at all (confirmed by reading
    # the harness - the GDN elif branch only uses `data`), and plotter() (called right after the
    # test backprop, BEFORE any score is saved) indexes testO/labels by the ORIGINAL channel count
    # - feeding it the windowed (n_window*feats)-wide tensor crashes with an IndexError one column
    # past the true channel count. First version of this patch reassigned trainO=trainD/testO=testD
    # (windowed) "to be safe" and broke exactly this - caught on the first live run: all three
    # datasets crashed identically in plotter() before saving any score, cheaply (~30-230s each).
    mp = mp.replace(
        "\tif model.name in ['Attention', 'DAGMM', 'USAD', 'MSCRED', 'CAE_M', 'GDN', 'MTAD_GAT', 'MAD_GAN'] or 'TranAD' in model.name: \n"
        "\t\ttrainD, testD = convert_to_windows(trainD, model), convert_to_windows(testD, model)",
        "\tif model.name in ['Attention', 'DAGMM', 'USAD', 'MSCRED', 'CAE_M', 'GDN', 'MTAD_GAT', 'MAD_GAN'] or 'TranAD' in model.name: \n"
        "\t\ttrainD, testD = convert_to_windows(trainD, model), convert_to_windows(testD, model)\n"
        "\tif model.name == 'GDN' and os.environ.get('GDN_GPU', '0') == '1' and _t.cuda.is_available():\n"
        "\t\t_dev = _t.device('cuda')\n"
        "\t\tmodel = model.to(_dev); model.g = model.g.to(_dev)\n"
        "\t\ttrainD = trainD.to(_dev); testD = testD.to(_dev)\n",
        1)
    _sfx = "os.environ.get('REAL_DS','WADI')+'_s'+os.environ.get('SOTA_SEED','0')"
    save = ("\tnp.save('/results/gdn_fast_score_'+" + _sfx + "+'.npy', lossFinal)\n"
            "\tnp.save('/results/gdn_fast_labels_'+" + _sfx + "+'.npy', labelsFinal)\n")
    tgt = "\tresult, _ = pot_eval(lossTfinal, lossFinal, labelsFinal)"
    guarded = save + "\ttry:\n\t\tresult, _ = pot_eval(lossTfinal, lossFinal, labelsFinal)\n\texcept Exception as _e:\n\t\tprint('[pot] skipped', _e, flush=True); result = {}\n"
    mp = mp.replace(tgt, guarded, 1)
    P(f"{repo}/main.py").write_text(mp)
    return save in mp


@app.function(cpu=4.0, gpu="A10G", timeout=2 * 60 * 60, memory=24576, volumes={"/results": results_vol})
def verify_forward_batch() -> dict:
    """Correctness invariant: forward_batch(data, B=1) must reproduce forward(data[0]) exactly
    (same random init weights, same input, float32). No training, no data files needed - this only
    checks that the batched code path is a correct vectorization of the per-window path, on a tiny
    synthetic model+graph so it is cheap and runs before any real GPU minutes are spent training."""
    import subprocess, os, torch
    os.chdir("/root")
    subprocess.run("git clone --depth 1 https://github.com/imperial-qore/TranAD.git", shell=True)
    repo = "/root/TranAD"; os.chdir(repo)
    ok_dump = _patch_harness(repo, fp32=True)
    import sys; sys.path.insert(0, repo)
    from src.models import GDN
    torch.manual_seed(0)
    feats = 17
    m = GDN(feats).float()
    m.eval()  # no dropout/batchnorm in this model, but be explicit
    torch.manual_seed(1)
    n_window = m.n_window
    B = 6
    data = torch.randn(B, n_window * feats, dtype=torch.float32)
    with torch.no_grad():
        out_loop = torch.stack([m(data[i]) for i in range(B)])
        out_batch = m.forward_batch(data)
    diff = (out_loop - out_batch).abs()
    max_abs = float(diff.max()); mean_abs = float(diff.mean())
    corr = float(torch.corrcoef(torch.stack([out_loop.flatten(), out_batch.flatten()]))[0, 1])
    # also check batch=1 exactness explicitly (isolates any batch-mixing bug)
    d1 = data[:1]
    with torch.no_grad():
        o1_loop = m(d1[0]); o1_batch = m.forward_batch(d1)[0]
    max_abs_b1 = float((o1_loop - o1_batch).abs().max())
    res = {"ok_dump_patch": ok_dump, "max_abs_diff_B6": max_abs, "mean_abs_diff_B6": mean_abs,
           "corr_B6": corr, "max_abs_diff_B1": max_abs_b1,
           "PASS": bool(max_abs_b1 < 1e-4 and max_abs < 1e-3 and corr > 0.999)}
    print(f"[verify] {res}", flush=True)
    Path("/results/gdn_fast_verify.json").write_text(json.dumps(res, indent=2)); results_vol.commit()
    return res


@app.function(cpu=4.0, gpu="A10G", timeout=2 * 60 * 60, memory=24576, volumes={"/results": results_vol})
def run_one_fast(ds: str, epochs: int = 5, batch_size: int = 256, seed: int = 0,
                 max_s: int = 6000, force: bool = False) -> dict:
    import subprocess, os, time, threading, numpy as np
    from sklearn.metrics import f1_score, roc_auc_score
    os.environ["MKL_THREADING_LAYER"] = "GNU"; os.environ["NEPOCHS"] = str(epochs)
    os.environ["REAL_DS"] = ds; os.environ["SOTA_SEED"] = str(seed)
    os.environ["SOTA_BS"] = str(batch_size); os.environ["GDN_GPU"] = "1"
    os.environ["OMP_NUM_THREADS"] = "4"; os.environ["MKL_NUM_THREADS"] = "4"
    # MUST match the save patch's filename exactly: it saves '/results/gdn_fast_score_'+REAL_DS+
    # '_s'+SOTA_SEED+'.npy' (no model prefix - this script only ever runs GDN). A first version of
    # this file used _tag="GDN_{ds}_s{seed}" here (copying modal_sota.py's convention, where the
    # save path DOES include args.model), which doesn't match what THIS patch actually writes -
    # caught on the first live run: WADI_clean finished rc=0 with a fully successful POT/pot_eval
    # printout but raw_ALL=None, because run_one_fast was checking for a file
    # (gdn_fast_score_GDN_WADI_clean_s0.npy) that was never created; the real file
    # (gdn_fast_score_WADI_clean_s0.npy) sat there unread.
    _tag = f"{ds}_s{seed}"

    results_vol.reload()
    done_marker = f"/results/gdn_fast_{ds}.done"
    if os.path.exists(done_marker) and not force:
        prior = json.loads(Path(f"/results/gdn_fast_one_{_tag}.json").read_text())
        print(f"[fast] SKIP {ds}: already done ({done_marker} exists). Use force=True to redo.", flush=True)
        prior["skipped"] = True
        return prior

    def sh(cmd, **kw):
        print(f"[fast] $ {cmd}", flush=True)
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)
        return r.returncode, r.stdout, r.stderr

    os.chdir("/root")
    sh("git clone --depth 1 https://github.com/imperial-qore/TranAD.git")
    repo = "/root/TranAD"; os.chdir(repo)
    ok_dump = _patch_harness(repo, fp32=True)
    print(f"[fast] patched ds={ds} dump={'ok' if ok_dump else 'FAIL'}", flush=True)

    for stale in (f"/results/gdn_fast_score_{_tag}.npy", f"/results/gdn_fast_labels_{_tag}.npy"):
        try: os.remove(stale)
        except OSError: pass
    pfx = PFX[ds]
    proc = f"{repo}/processed/WADI"; os.makedirs(proc, exist_ok=True)
    tr = np.load(f"/app/{pfx}_train.npy"); te = np.load(f"/app/{pfx}_test.npy"); lb = np.load(f"/app/{pfx}_labels.npy")
    feats = tr.shape[1]
    np.save(f"{proc}/train.npy", tr.astype(np.float32))
    np.save(f"{proc}/test.npy", te.astype(np.float32))
    np.save(f"{proc}/labels.npy", np.tile(lb.reshape(-1, 1), (1, feats)).astype(np.float32))
    print(f"[fast] {ds} train{tr.shape} test{te.shape} anom={lb.mean():.3f}", flush=True)

    def metrics(score, y):
        y = y.astype(int); out = {}
        try: out["AUROC"] = round(float(roc_auc_score(y, score)), 3)
        except Exception: out["AUROC"] = None
        qs = np.quantile(score, np.linspace(0.80, 0.999, 60))
        out["bestF1"] = round(float(max(f1_score(y, score > t) for t in qs)), 3)
        thr = np.quantile(score[y == 0], 0.95)
        out["F1_at5pctFPR"] = round(float(f1_score(y, score > thr)), 3)
        out["TPR_at5pctFPR"] = round(float((score[y == 1] > thr).mean()), 3)
        return out

    triv = np.load(f"/app/{pfx}_triv_test.npy"); thr_triv = float(np.load(f"/app/{pfx}_triv_thr.npy"))

    def gpu_note(tag):
        try:
            import torch
            print(f"[gpu] {ds}:GDN-fast {tag} alloc={torch.cuda.memory_allocated()/1e9:.2f}G "
                  f"reserved={torch.cuda.memory_reserved()/1e9:.2f}G", flush=True)
        except Exception as e:
            print(f"[gpu] {tag} err {e}", flush=True)

    def stream_run(cmd, env, max_s=max_s):
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1, env=env)
        killed = {"v": False}
        def watchdog():
            t = 0
            while p.poll() is None and t < max_s:
                time.sleep(10); t += 10
                if t % 30 == 0: gpu_note(f"heartbeat t+{t}s")
            if p.poll() is None:
                killed["v"] = True; print(f"[fast] {ds} WATCHDOG kill {max_s}s", flush=True); p.terminate()
        threading.Thread(target=watchdog, daemon=True).start()
        buf = []
        for line in iter(p.stdout.readline, ''):
            print(f"[{ds}:GDN-fast] {line.rstrip()}", flush=True); buf.append(line)
        p.wait()
        return (124 if killed["v"] else p.returncode), "".join(buf)

    print(f"[fast] === training {ds}:GDN-fast (NEPOCHS={epochs}, BS={batch_size}, GPU=A10G) ===", flush=True)
    t0 = time.time()
    try:
        rc, log = stream_run(f"python -u main.py --model GDN --dataset WADI --retrain",
                             {**os.environ, "MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1"})
    except Exception:
        import traceback; rc, log = 99, traceback.format_exc(); print(log, flush=True)
    train_s = round(time.time() - t0, 1)

    res = {"dataset": ds, "model": "GDN-fast", "seed": seed, "rc": rc, "train_s": train_s,
           "epochs": epochs, "batch_size": batch_size}
    sp = f"/results/gdn_fast_score_{_tag}.npy"; lp = f"/results/gdn_fast_labels_{_tag}.npy"
    if os.path.exists(sp) and os.path.exists(lp):
        s = np.load(sp); y = np.load(lp).astype(int)
        s = s.mean(1) if s.ndim > 1 else s
        t = triv[-len(s):]; yy = y[-len(s):] if len(y) != len(s) else y
        res["n"] = int(len(s)); res["n_anom"] = int(yy.sum())
        res["raw_ALL"] = metrics(s, yy)
        easy = (yy == 1) & (t > thr_triv); hard = (yy == 1) & ~easy
        res["n_easy_hard"] = [int(easy.sum()), int(hard.sum())]
        if hard.sum() > 3:
            m = (yy == 0) | hard; res["raw_HARD"] = metrics(s[m], yy[m])
        if easy.sum() > 3:
            m = (yy == 0) | easy; res["raw_EASY"] = metrics(s[m], yy[m])
        # per-window npz aligned to the window grid, re-scorable against a revised difficulty mask
        np.savez(f"/results/gdn_scores_{ds}.npz", score=s, labels=yy, n=len(s))
        res["npz"] = f"gdn_scores_{ds}.npz"
        Path(done_marker).write_text(json.dumps({"ds": ds, "train_s": train_s}))
    else:
        res["error"] = "no score dumped"; res["log_tail"] = log[-2000:]
    Path(f"/results/gdn_fast_one_{_tag}.json").write_text(json.dumps(res, indent=2)); results_vol.commit()
    print(f"[fast] DONE {_tag} rc={rc} {train_s}s raw_ALL={res.get('raw_ALL')} "
          f"raw_HARD={res.get('raw_HARD')}", flush=True)
    return res


@app.function(cpu=2.0, timeout=600, memory=8192, volumes={"/results": results_vol})
def recover_scores(ds: str, seed: int = 0) -> dict:
    """Compute metrics from an ALREADY-TRAINED run's score/labels npy sitting on the volume under
    the correct (post-fix) filename, without retraining. Recovers a run that trained/scored fine
    but hit the run_one_fast filename-lookup bug (fixed above) before this call existed."""
    import os, numpy as np
    from sklearn.metrics import f1_score, roc_auc_score
    results_vol.reload()
    _tag = f"{ds}_s{seed}"
    sp = f"/results/gdn_fast_score_{_tag}.npy"; lp = f"/results/gdn_fast_labels_{_tag}.npy"
    if not (os.path.exists(sp) and os.path.exists(lp)):
        return {"dataset": ds, "error": f"no score file at {sp}"}
    pfx = PFX[ds]
    triv = np.load(f"/app/{pfx}_triv_test.npy"); thr_triv = float(np.load(f"/app/{pfx}_triv_thr.npy"))

    def metrics(score, y):
        y = y.astype(int); out = {}
        try: out["AUROC"] = round(float(roc_auc_score(y, score)), 3)
        except Exception: out["AUROC"] = None
        qs = np.quantile(score, np.linspace(0.80, 0.999, 60))
        out["bestF1"] = round(float(max(f1_score(y, score > t) for t in qs)), 3)
        thr = np.quantile(score[y == 0], 0.95)
        out["F1_at5pctFPR"] = round(float(f1_score(y, score > thr)), 3)
        out["TPR_at5pctFPR"] = round(float((score[y == 1] > thr).mean()), 3)
        return out

    s = np.load(sp); y = np.load(lp).astype(int)
    s = s.mean(1) if s.ndim > 1 else s
    t = triv[-len(s):]; yy = y[-len(s):] if len(y) != len(s) else y
    res = {"dataset": ds, "model": "GDN-fast-recovered", "n": int(len(s)), "n_anom": int(yy.sum())}
    res["raw_ALL"] = metrics(s, yy)
    easy = (yy == 1) & (t > thr_triv); hard = (yy == 1) & ~easy
    res["n_easy_hard"] = [int(easy.sum()), int(hard.sum())]
    if hard.sum() > 3:
        m = (yy == 0) | hard; res["raw_HARD"] = metrics(s[m], yy[m])
    if easy.sum() > 3:
        m = (yy == 0) | easy; res["raw_EASY"] = metrics(s[m], yy[m])
    np.savez(f"/results/gdn_scores_{ds}.npz", score=s, labels=yy, n=len(s))
    Path(f"/results/gdn_fast_one_{_tag}.json").write_text(json.dumps(res, indent=2)); results_vol.commit()
    print(f"[recover] {ds}: {res}", flush=True)
    return res


@app.local_entrypoint()
def main(datasets: str = "WADI_clean,HAI,SWaT_canon", epochs: int = 5, batch_size: int = 256,
         verify: bool = False, force: bool = False, max_s: int = 6000, recover: str = ""):
    if recover:
        for d in recover.split(","):
            res = recover_scores.remote(d)
            print(f"\n==================== RECOVER {d} ====================\n{json.dumps(res, indent=2)}")
        return
    if verify:
        res = verify_forward_batch.remote()
        print(f"\n==================== VERIFY ====================\n{json.dumps(res, indent=2)}")
        if not res.get("PASS"):
            print("[fast] VERIFY FAILED - do not trust the batched path. Use the stride safety-net run.")
        return
    ds_list = [d for d in datasets.split(",") if d]
    print(f"[fast] launching {len(ds_list)} GPU jobs: {ds_list} (skip-if-done unless force)", flush=True)
    results = list(run_one_fast.starmap([(d, epochs, batch_size, 0, max_s, force) for d in ds_list]))
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "gdn_fast_matrix.json").write_text(json.dumps(results, indent=2))
    print("\n==================== SUMMARY ====================")
    for r in results:
        a = r.get("raw_ALL", {}); h = r.get("raw_HARD", {})
        print(f"{r.get('dataset'):12} GDN-fast rc={r.get('rc')} {r.get('train_s')}s skipped={r.get('skipped', False)} "
              f"| ALL auroc={a.get('AUROC')} | HARD auroc={h.get('AUROC')} | n_eh={r.get('n_easy_hard')}")
