"""Local CPU simulation of modal_e3_cost._bench_deep for the cell sequence TranAD -> USAD in ONE
process (as _bench_all does in one Modal container). FIX=0 reproduces the bug (patch applied to an
already-patched clone); FIX=1 applies the proposed quick fix (fresh clone per cell + drop cached
src.* modules + parse 'Training time' from harness stdout)."""
import os, sys, re, time, json, shutil, subprocess, numpy as np, torch
ROOT = r"E:\tmp\claude\usad"; SB = r"E:\Projects\Backlog\LatAD\poc\sota_bundle"
FIX = int(os.environ.get("FIX", "1")); NTR = int(os.environ.get("NTR", "3000")); NTE = int(os.environ.get("NTE", "2000"))
src = open(os.path.join(SB, "modal_sota.py"), encoding="utf-8").read()
s = src.index("def _patch_harness"); e = src.index("@app.function", s); ns = {}; exec(src[s:e], ns)
_patch_harness = ns["_patch_harness"]
os.makedirs("/results", exist_ok=True)   # harness dumps scores to /results (drive root on Windows)

def _median_latency(fn, n_calls=30, warmup=5):
    import statistics
    for _ in range(warmup): fn()
    ts = []
    for _ in range(n_calls):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1000)
    return round(statistics.median(ts), 3)

def bench_deep(model, out, epochs):
    feats = 123
    os.chdir(ROOT)
    repo = os.path.join(ROOT, "TranAD_sim" + (f"_{model}" if FIX else ""))
    if FIX:
        shutil.rmtree(repo, ignore_errors=True)                       # FIX: fresh clone every cell
    if not os.path.isdir(repo):
        # local stand-in for `git clone`: copy the pristine clone
        shutil.copytree(os.path.join(ROOT, "TranAD"), repo, ignore=shutil.ignore_patterns("processed", "checkpoints", "__pycache__"))
        shutil.copy(os.path.join(repo, "main.py.orig"), os.path.join(repo, "main.py"))
        shutil.copy(os.path.join(repo, "src", "models.py.orig"), os.path.join(repo, "src", "models.py"))
    os.chdir(repo)
    _patch_harness(repo, fp32=True)
    proc = os.path.join(repo, "processed", "WADI"); os.makedirs(proc, exist_ok=True)
    tr = np.load(f"{SB}/wadi_train.npy")[:NTR]; te = np.load(f"{SB}/wadi_test.npy")[:NTE]; lb = np.load(f"{SB}/wadi_labels.npy")[:NTE]
    np.save(f"{proc}/train.npy", tr.astype(np.float32)); np.save(f"{proc}/test.npy", te.astype(np.float32))
    np.save(f"{proc}/labels.npy", np.tile(lb.reshape(-1, 1), (1, feats)).astype(np.float32))
    env = {**os.environ, "NEPOCHS": str(epochs), "REAL_DS": "WADI", "SOTA_SEED": "0", "MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1"}
    for f in (f"/results/score_{model}_WADI_s0.npy", f"/results/labels_{model}_WADI_s0.npy"):
        try: os.remove(f)
        except OSError: pass
    t0 = time.time()
    r = subprocess.run([sys.executable, "-u", "main.py", "--model", model, "--dataset", "WADI", "--retrain"],
                       env=env, capture_output=True, text=True)
    out["train_s"] = round(time.time() - t0, 1); out["rc"] = r.returncode
    m = re.search(r"Training time:\s*([\d.]+) s", r.stdout)
    out["train_only_s"] = float(m.group(1)) if m else None
    if r.returncode != 0: out["train_err"] = (r.stderr or r.stdout)[-400:]
    sp = f"/results/score_{model}_WADI_s0.npy"
    if os.path.exists(sp):
        sc = np.load(sp); out["score_shape"] = list(sc.shape); out["score_finite"] = bool(np.isfinite(sc).all())
        out["score_std"] = float(sc.std())
    try:
        if FIX:
            for k in [k for k in sys.modules if k == "src" or k.startswith("src.")]: del sys.modules[k]  # FIX: no stale module
        sys.path.insert(0, repo)
        if FIX:  # FIX: same torch>=2 Transformer forward shim that _patch_harness injects into main.py
            def _enc_fwd(self, src, mask=None, src_key_padding_mask=None, **kw):
                out = src
                for mod in self.layers: out = mod(out, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
                return self.norm(out) if self.norm is not None else out
            def _dec_fwd(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, **kw):
                out = tgt
                for mod in self.layers: out = mod(out, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
                return self.norm(out) if self.norm is not None else out
            torch.nn.TransformerEncoder.forward = _enc_fwd; torch.nn.TransformerDecoder.forward = _dec_fwd
        import src.models as M
        net = getattr(M, model)(feats).float()
        out["params"] = int(sum(p.numel() for p in net.parameters())); net.eval()
        nw = int(getattr(net, "n_window", 10)); BATCH = 256
        if model == "USAD":
            in_dim = next((m.in_features for m in net.modules() if isinstance(m, torch.nn.Linear)), nw * feats)
            out["in_dim"] = in_dim; out["n_window"] = nw
            x1 = torch.zeros(1, in_dim); xB = torch.zeros(BATCH, in_dim)
            def fwd1(): net(x1)
            def fwd256(): net(xB)
        else:
            w1 = torch.zeros(nw, 1, feats); wB = torch.zeros(nw, BATCH, feats)
            def fwd1(): net(w1, w1)
            def fwd256(): net(wB, wB)
        with torch.no_grad():
            out["lat_b1_ms"] = _median_latency(fwd1); out["lat_b256_ms"] = _median_latency(fwd256)
    except Exception as ex:
        out["infer_error"] = f"{type(ex).__name__}: {ex}"

rows = []
for model, ep in [("TranAD", 1), ("USAD", 2)]:
    out = {"method": model, "FIX": FIX}
    bench_deep(model, out, ep); rows.append(out)
    print(json.dumps(out), flush=True)
