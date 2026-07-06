"""
LDT Model B - the trainable trajectory encoder (queryable history).

B watches A's responsibility stream {gamma_t} and keeps a running, queryable
memory of "which modes, in what order, for how long". It NEVER forecasts and
NEVER judges anomalies - it only summarises the determined past.

Pipeline per the spec:
  - step embedding  e_t = M gamma_t     (M init from A's centroids; learnable) -
    the anchored linear form, so a between-modes window sits on the segment
    between anchors (betweenness) and rare modes are not starved (rare-robustness).
  - encoder         a long-range sequence model over {e_t}. We use a GRU as the
    Mamba/SSM stand-in (dilated-TCN is the alternative). Its recurrent state is
    the memory; the per-step hidden state h_t is the trajectory context c_t.
  - two query heads, both reading c_t and both rare/common-balanced:
      (a) instance head  - recover gamma_{t-k} at offsets k in {1,2,4,...,512}
                           via cross-entropy, rare-weighted (lambda_k prop 1/pi).
      (b) containment head - BCE "was mode a present in period P", positive/negative
                           balanced and rare-up-weighted (w_a prop 1/pi_a).

After training B is frozen and emits c_t. A memory-horizon probe (query accuracy
vs offset) verifies the effective memory horizon.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# offsets the instance head is quizzed at (spec: 1,2,4,...,512)
INSTANCE_OFFSETS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


class TrajectoryEncoderB(nn.Module):
    def __init__(self, K, emb_dim, ctx_dim, centroids=None, backbone="gru",
                 n_layers=1):
        super().__init__()
        self.K = K
        self.emb_dim = emb_dim
        self.ctx_dim = ctx_dim
        # anchored step embedding e_t = M gamma_t : a K->emb_dim linear map, rows
        # init from A's centroids (projected to emb_dim), regularised toward them.
        self.M = nn.Linear(K, emb_dim, bias=False)
        if centroids is not None:
            with torch.no_grad():
                C = torch.as_tensor(centroids, dtype=torch.float32)  # (K, d_lat)
                if C.shape[1] >= emb_dim:
                    init = C[:, :emb_dim]
                else:
                    pad = torch.zeros(K, emb_dim - C.shape[1])
                    init = torch.cat([C, pad], dim=1)
                self.M.weight.copy_(init.t())                # weight is (emb, K)
        self.register_buffer("M_init", self.M.weight.detach().clone())

        self.backbone = backbone
        if backbone == "gru":
            self.rnn = nn.GRU(emb_dim, ctx_dim, num_layers=n_layers,
                              batch_first=True)
        else:  # dilated-TCN stand-in
            self.tcn = _DilatedTCN(emb_dim, ctx_dim)

        # instance head: r_theta(c_t, psi(k)) -> logits over K modes
        n_off = len(INSTANCE_OFFSETS)
        self.off_emb = nn.Embedding(n_off, 16)               # psi(k) offset code
        self.inst_head = nn.Sequential(
            nn.Linear(ctx_dim + 16, ctx_dim), nn.ReLU(),
            nn.Linear(ctx_dim, K))
        # containment head: q_theta(c_t, P, a) -> P(a present in P) as one logit.
        # P is encoded by its (log start, log length); a by its mode embedding.
        self.cont_head = nn.Sequential(
            nn.Linear(ctx_dim + 2 + 16, ctx_dim), nn.ReLU(),
            nn.Linear(ctx_dim, 1))
        self.mode_a_emb = nn.Embedding(K, 16)

    # --------------------------------------------------------------------- #
    def embed(self, gamma):
        """gamma: (B, T, K) -> e: (B, T, emb_dim)."""
        return self.M(gamma)

    def context(self, gamma):
        """Full-sequence context c_t for every step. gamma (B,T,K) -> (B,T,ctx)."""
        e = self.embed(gamma)
        if self.backbone == "gru":
            c, _ = self.rnn(e)
        else:
            c = self.tcn(e)
        return c

    # ---- head forward passes (used both in training and the probe) ------- #
    def instance_logits(self, c_t, off_idx):
        """c_t (B, ctx), off_idx (B,) long -> logits (B, K)."""
        psi = self.off_emb(off_idx)
        return self.inst_head(torch.cat([c_t, psi], dim=1))

    def containment_logit(self, c_t, P_feat, a_idx):
        """c_t (B,ctx), P_feat (B,2), a_idx (B,) -> logit (B,)."""
        a_e = self.mode_a_emb(a_idx)
        return self.cont_head(torch.cat([c_t, P_feat, a_e], dim=1)).squeeze(1)


class _DilatedTCN(nn.Module):
    """Causal dilated 1-D conv stack (Mamba stand-in alternative to the GRU)."""

    def __init__(self, in_dim, out_dim, dilations=(1, 2, 4, 8, 16, 32)):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(out_dim, out_dim, 3, padding=0, dilation=dl)
            for dl in dilations])
        self.dilations = dilations

    def forward(self, e):                     # e: (B, T, in)
        h = self.proj(e).transpose(1, 2)      # (B, out, T)
        for conv, dl in zip(self.convs, self.dilations):
            pad = 2 * dl                       # causal left pad
            hp = F.pad(h, (pad, 0))
            h = h + torch.relu(conv(hp))
        return h.transpose(1, 2)               # (B, T, out)


# ------------------------------------------------------------------------- #
#  Training
# ------------------------------------------------------------------------- #
def _chunk_sequence(gamma_seq, seg_len, stride, device):
    """Cut one long gamma stream (T, K) into overlapping (seg_len, K) segments
    so the GRU sees a manageable context per step."""
    T = len(gamma_seq)
    starts = list(range(0, max(1, T - seg_len + 1), stride))
    if not starts:
        starts = [0]
    g = torch.as_tensor(gamma_seq, dtype=torch.float32, device=device)
    segs = [g[s:s + seg_len] for s in starts if s + 2 <= T]
    return segs


def train_B(model, gamma_seq, pi, epochs=8, seg_len=512, stride=256,
            batch_segs=8, lr=1e-3, device="cpu", anchor_reg=1e-3,
            n_cont_queries=8, verbose=False, seed=0):
    """Train B on one normal gamma stream with the two query heads.

    gamma_seq : (T, K) responsibilities from frozen A, in time order.
    pi        : (K,) mode prior (rarity weights lambda_k, w_a prop 1/pi).
    """
    torch.manual_seed(seed)
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    pi_t = torch.as_tensor(np.maximum(pi, 1e-4), dtype=torch.float32, device=device)
    inv_pi = 1.0 / pi_t
    lam = (inv_pi / inv_pi.mean())                        # lambda_k prop 1/pi, mean 1
    segs = _chunk_sequence(gamma_seq, seg_len, stride, device)
    K = model.K
    offs = torch.tensor(INSTANCE_OFFSETS, device=device)
    rng = np.random.default_rng(seed)

    for ep in range(epochs):
        order = rng.permutation(len(segs))
        tot_i = tot_c = 0.0
        for b0 in range(0, len(order), batch_segs):
            idx = order[b0:b0 + batch_segs]
            batch = torch.stack([segs[i] for i in idx])   # (B, L, K)
            B, L, _ = batch.shape
            c = model.context(batch)                      # (B, L, ctx)
            hard = torch.argmax(batch, dim=2)             # (B, L) mode id per step

            # ---- (a) instance head: recover gamma_{t-k} at each offset -----
            loss_i = 0.0
            n_i = 0
            for oi, k in enumerate(INSTANCE_OFFSETS):
                if k >= L:
                    break
                t = torch.arange(k, L, device=device)     # queryable positions
                if len(t) == 0:
                    continue
                c_t = c[:, t, :].reshape(-1, model.ctx_dim)
                tgt = hard[:, t - k].reshape(-1)          # true mode k steps back
                oidx = torch.full((c_t.shape[0],), oi, device=device,
                                  dtype=torch.long)
                logits = model.instance_logits(c_t, oidx)
                # rare-weighted CE (lambda_k prop 1/pi on the TARGET class)
                w = lam[tgt]
                ce = F.cross_entropy(logits, tgt, reduction="none")
                loss_i = loss_i + (w * ce).mean()
                n_i += 1
            loss_i = loss_i / max(1, n_i)

            # ---- (b) containment head: was mode a present in period P ------
            # sample balanced (present/absent) queries, rare-up-weighted.
            loss_c = 0.0
            n_c = 0
            for _ in range(n_cont_queries):
                # random period P = [t-b, t-a] within the segment
                t = int(rng.integers(max(8, L // 4), L))
                a_off = int(rng.integers(1, max(2, t // 2)))
                b_off = int(rng.integers(a_off + 1, t + 1))
                lo, hi = t - b_off, t - a_off              # window [lo, hi)
                present = hard[:, lo:hi]                   # (B, len)
                # per sequence, decide query mode: half positives, half negatives
                c_t = c[:, t, :]                           # (B, ctx)
                # positive candidate: a mode actually present; negative: absent.
                a_ids = torch.empty(B, dtype=torch.long, device=device)
                labels = torch.empty(B, device=device)
                for bi in range(B):
                    modes_in = torch.unique(present[bi])
                    if rng.random() < 0.5 and len(modes_in) > 0:
                        a_ids[bi] = modes_in[rng.integers(len(modes_in))]
                        labels[bi] = 1.0
                    else:
                        # plausible-but-absent negative: a rare mode not in P
                        allm = torch.arange(K, device=device)
                        mask = torch.ones(K, dtype=torch.bool, device=device)
                        mask[modes_in] = False
                        absent = allm[mask]
                        if len(absent) == 0:
                            a_ids[bi] = modes_in[rng.integers(len(modes_in))]
                            labels[bi] = 1.0
                        else:
                            a_ids[bi] = absent[rng.integers(len(absent))]
                            labels[bi] = 0.0
                P_feat = torch.tensor(
                    [[np.log1p(a_off), np.log1p(b_off - a_off)]] * B,
                    dtype=torch.float32, device=device)
                logit = model.containment_logit(c_t, P_feat, a_ids)
                w = lam[a_ids]                             # rare-up-weight w_a
                bce = F.binary_cross_entropy_with_logits(
                    logit, labels, reduction="none")
                loss_c = loss_c + (w * bce).mean()
                n_c += 1
            loss_c = loss_c / max(1, n_c)

            # anchor regularisation: keep M near A's centroid init
            anch = anchor_reg * ((model.M.weight - model.M_init) ** 2).sum()

            loss = loss_i + loss_c + anch
            opt.zero_grad(); loss.backward(); opt.step()
            tot_i += float(loss_i.detach()) if torch.is_tensor(loss_i) else float(loss_i)
            tot_c += float(loss_c.detach()) if torch.is_tensor(loss_c) else float(loss_c)
        if verbose:
            print(f"  [B] epoch {ep:2d}  inst {tot_i:.3f}  cont {tot_c:.3f}")
    model.eval()
    return model


@torch.no_grad()
def emit_context(model, gamma_seq, seg_len=512, device="cpu"):
    """Emit c_t for every step of a gamma stream, streaming so long sequences
    fit. Returns (T, ctx). Uses a stateful pass for the GRU (true long memory)."""
    model.eval().to(device)
    g = torch.as_tensor(gamma_seq, dtype=torch.float32, device=device).unsqueeze(0)
    if model.backbone == "gru":
        e = model.embed(g)
        c, _ = model.rnn(e)                                # full stateful pass
        return c.squeeze(0).cpu().numpy()
    # TCN: process in one shot (causal conv is already streaming-consistent)
    c = model.context(g)
    return c.squeeze(0).cpu().numpy()


@torch.no_grad()
def memory_horizon_probe(model, gamma_seq, device="cpu", seg_len=512):
    """Acceptance test: instance-head query accuracy vs offset = memory horizon.
    Returns {offset: accuracy} on a held stream."""
    model.eval().to(device)
    segs = _chunk_sequence(gamma_seq, seg_len, seg_len, device)
    if not segs:
        return {}
    batch = torch.stack(segs[: min(len(segs), 16)])        # (B, L, K)
    c = model.context(batch)
    hard = torch.argmax(batch, dim=2)
    B, L, _ = batch.shape
    out = {}
    for oi, k in enumerate(INSTANCE_OFFSETS):
        if k >= L:
            break
        t = torch.arange(k, L, device=device)
        c_t = c[:, t, :].reshape(-1, model.ctx_dim)
        tgt = hard[:, t - k].reshape(-1)
        oidx = torch.full((c_t.shape[0],), oi, device=device, dtype=torch.long)
        pred = model.instance_logits(c_t, oidx).argmax(1)
        out[k] = float((pred == tgt).float().mean())
    return out
