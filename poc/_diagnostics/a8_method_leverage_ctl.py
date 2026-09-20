"""Persistence controls for the self-gated context head (a8_method_leverage.py), all datasets, 5 seeds.

The `selfgate` rule holds the expected regime at the last window the detector itself scored normal (LatAD <=
train q95), so during an alarm run every window is scored against the PRE-alarm regime. That is regime context
AND alarm persistence at once. Controls that have the persistence but no regime geometry, label-free, unit weight:
  rmax_m   : rolling max of robust-z LatAD over the last m windows (m = 2, 3, 5)
  hold     : running max of robust-z LatAD since the last self-normal window (exactly the selfgate timing)
  ewma     : exponential smoothing of robust-z LatAD (alpha 0.5)
  hl_rmax3 : rolling max of the headline itself (smoothing the fused score)
Each is fused into the headline (HCcoh+LatAD) and into HC_coh like the heads, evaluated on the clean Difficult /
DoubleHard subsets, with the paired episode bootstrap vs the base. Also: per-episode breakdown of the selfgate
lift on HAI, and the share of difficult windows on which the selfgate expectation differs from prev1 (frozen).
Writes a8_method_leverage_ctl.json.
"""
from __future__ import annotations
import os, sys, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); POC = os.path.dirname(HERE); sys.path.insert(0, POC); sys.path.insert(0, HERE)
from sklearn.metrics import roc_auc_score
import ensemble_final as EF
import eda_real as E
from a8_method_leverage import rz, _js

def main(names, reps=500):
    jf = os.path.join(HERE, "a8_method_leverage_ctl.json")
    OUT = json.load(open(jf)) if os.path.exists(jf) else {}
    for name in names:
        d = np.load(os.path.join(HERE, f"scores_{name}.npz"))
        Ex = np.load(os.path.join(POC, "sota_bundle", "experts_full", f"expert_{name}.npz"), allow_pickle=True)
        y = d["label"].astype(int); nseed = 5
        Cal, Tst = Ex["calib_surprise"], Ex["test_surprise"]; S = Tst.shape[1]
        w = Ex["comm_cohesion"].astype(float) * np.sqrt(Ex["comm_size"].astype(float))
        lat, lat_tr = d["LatAD"], d["LatAD_train"]
        sb = np.load(os.path.join(HERE, f"a8_leverage_subsets_{name}.npz")); diff, dhard = sb["diff"], sb["dhard"]
        zh, zl, hcc = [], [], []
        for sd in range(nseed):
            P = np.stack([EF.pval(Cal[sd, g], Tst[sd, g]) for g in range(S)]); hc_coh = EF.HC(P, wt=w)
            Pc = np.stack([EF.pval(Cal[sd, g], Cal[sd, g]) for g in range(S)]); hccoh_c = EF.HC(Pc, wt=w)
            z = lambda s, r: (s - r.mean()) / (r.std() + 1e-9)
            nulltail = EF.surv(lat_tr[sd], lat[sd]); zl.append(z(nulltail, EF.surv(lat_tr[sd], lat_tr[sd])))
            zh.append(z(hc_coh, hccoh_c)); hcc.append(hc_coh)
        zh, zl, hcc = np.stack(zh), np.stack(zl), np.stack(hcc); headline = zh + zl
        # controls per seed, from the refit npz (latad_te/tr pair with the refit's own selfgate threshold)
        ctl = {}; selfg = []; prev1 = []
        for sd in range(nseed):
            a = np.load(os.path.join(HERE, f"a8_leverage_{name}_seed{sd}.npz"))
            lz = rz(lat[sd], lat_tr[sd]); thr = np.quantile(a["latad_tr"], 0.95); alarm = a["latad_te"] > thr
            n = len(lz)
            for m in (2, 3, 5):
                r = np.array([lz[max(0, t - m + 1):t + 1].max() for t in range(n)]); ctl.setdefault(f"rmax{m}", []).append(r)
            hold = np.empty(n); cur = -np.inf
            for t in range(n):
                cur = max(cur, lz[t]) if (t > 0 and alarm[t - 1]) else lz[t]; hold[t] = cur
            ctl.setdefault("hold", []).append(hold)
            ew = np.empty(n); ew[0] = lz[0]
            for t in range(1, n): ew[t] = 0.5 * lz[t] + 0.5 * ew[t - 1]
            ctl.setdefault("ewma", []).append(ew)
            hl = headline[sd]; ctl.setdefault("hl_rmax3", []).append(np.array([hl[max(0, t - 2):t + 1].max() for t in range(n)]))
            selfg.append(rz(a["vade_exp_selfgate_te"], a["vade_exp_selfgate_tr"])); prev1.append(rz(a["vade_exp_prev1_te"], a["vade_exp_prev1_tr"]))
            ctl.setdefault("frozen_share_difficult", []).append(float((a["vade_exp_selfgate_te"] != a["vade_exp_prev1_te"])[diff].mean()))
            ctl.setdefault("frozen_share_normal", []).append(float((a["vade_exp_selfgate_te"] != a["vade_exp_prev1_te"])[y == 0].mean()))
        selfg, prev1 = np.stack(selfg), np.stack(prev1)
        fn, W, stride = E.RAW[name]; L = int(np.ceil(W / stride)) + 1
        def au(arr, mk):
            keep = np.where((y == 0) | mk)[0]; v = [roc_auc_score(y[keep], arr[i][keep]) for i in range(nseed)]
            return round(float(np.mean(v)), 4), round(float(np.std(v)), 4)
        res = {"frozen_share_difficult": float(np.mean(ctl.pop("frozen_share_difficult"))), "frozen_share_normal": float(np.mean(ctl.pop("frozen_share_normal"))), "rows": {}, "boots": {}}
        cands = {"selfgate": selfg, **{k: np.stack(v) for k, v in ctl.items()}}
        for k, arr in cands.items():
            if k == "hl_rmax3":
                fused_hl, fused_hc = arr, None
            else:
                fused_hl, fused_hc = headline + arr, zh + arr
            res["rows"][f"HCcoh+LatAD+{k}"] = {s: au(fused_hl, mk) for s, mk in (("Difficult", diff), ("DoubleHard", dhard))}
            res["boots"][f"HCcoh+LatAD+{k} vs headline"] = {s: EF.boot(y, fused_hl, headline, mk, L, reps=reps) for s, mk in (("Difficult", diff), ("DoubleHard", dhard))}
            if fused_hc is not None:
                res["rows"][f"HCcoh+{k}"] = {s: au(fused_hc, mk) for s, mk in (("Difficult", diff), ("DoubleHard", dhard))}
            print(name, k, res["rows"][f"HCcoh+LatAD+{k}"], json.dumps(res["boots"][f"HCcoh+LatAD+{k} vs headline"], default=_js)[:200], flush=True)
        # selfgate vs the strongest persistence control head-to-head (paired bootstrap)
        res["boots"]["HCcoh+LatAD+selfgate vs HCcoh+LatAD+hold"] = {s: EF.boot(y, headline + selfg, headline + cands["hold"], mk, L, reps=reps) for s, mk in (("Difficult", diff), ("DoubleHard", dhard))}
        res["boots"]["HCcoh+LatAD+selfgate vs HCcoh+LatAD+rmax3"] = {s: EF.boot(y, headline + selfg, headline + cands["rmax3"], mk, L, reps=reps) for s, mk in (("Difficult", diff), ("DoubleHard", dhard))}
        # per-episode breakdown of the selfgate lift (seed-mean test-normal percentile of fused minus headline)
        nrm = y == 0
        pN = lambda arr: np.mean([np.array([(arr[i][nrm] < v).mean() for v in arr[i]]) for i in range(nseed)], 0)
        p_head, p_self, p_hold = pN(headline), pN(headline + selfg), pN(headline + cands["hold"])
        eps = []
        for e in EF.episodes(y):
            m = e[diff[e]]
            if len(m) == 0: continue
            eps.append(dict(start=int(e[0]), n_diff=int(len(m)), n_dh=int(dhard[m].sum()), head=round(float(p_head[m].mean()), 3),
                            selfgate_lift=round(float((p_self[m] - p_head[m]).mean()), 3), hold_lift=round(float((p_hold[m] - p_head[m]).mean()), 3)))
        res["episodes"] = eps
        res["episodes_selfgate_up_down"] = [int(sum(e["selfgate_lift"] > 0.01 for e in eps)), int(sum(e["selfgate_lift"] < -0.01 for e in eps))]
        res["episodes_hold_up_down"] = [int(sum(e["hold_lift"] > 0.01 for e in eps)), int(sum(e["hold_lift"] < -0.01 for e in eps))]
        OUT[name] = res
        json.dump(OUT, open(os.path.join(HERE, "a8_method_leverage_ctl.json"), "w"), indent=1, default=_js)
    return OUT

if __name__ == "__main__":
    main(sys.argv[1:] or ["HAI", "SWaT_canon", "WADI_clean"], reps=int(os.environ.get("BOOT_REPS", "500")))
