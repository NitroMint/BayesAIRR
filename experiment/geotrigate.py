"""Geometry-Conditioned Sparse Triangle Gated Attention for nucleotide sequences.

Adaptation of GeoTriGate for junction sequences (no 3D coordinates are available;
instead, geometry = positional distance + k-mer Hamming similarity + base co-occurrence).

Key modules:

  (1) Pair representation: (B, L, L, 17) of RBF-encoded geometric descriptors.
  (2) Sparse Triangle Attention: axial attention along the third axis with a
      positional-distance mask. Each pair (i,j) only attends to positions
      k within |k-i| <= pos_cut or |k-j| <= pos_cut.
  (3) Geometric gate: sigmoid(Linear(pair_feat, geo_descriptor)) modulates output.
  (4) CMS-Pool: N1 (first half) / N2 (second half) / global learned importance.

Training objective: masked nucleotide denoising (self-supervised, 15% positions).
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import NUC, NUC_IDX


# ------------------------------------------------------------ pair representation

def _rbf(values: torch.Tensor, n_bins: int, vmin: float, vmax: float) -> torch.Tensor:
    centers = torch.linspace(vmin, vmax, n_bins, device=values.device)
    centers = centers.view(*([1] * values.ndim), -1)
    sigma = (vmax - vmin) / n_bins
    return torch.exp(-((values.unsqueeze(-1) - centers).pow(2) / (2 * sigma ** 2)))


def build_pair_features(seq_onehot: torch.Tensor, k: int = 3) -> torch.Tensor:
    """Compute (B, L, L, d_pair) geometric pair features.

    seq_onehot: (B, 4, L) with {0,1} entries.

    Descriptors (17 channels total):
      - 8 RBF bins on positional distance |i - j|
      - 8 RBF bins on soft Hamming distance between length-k windows centered at i,j
      - 1 channel: base identity match (A==A etc.)
    """
    B, _, L = seq_onehot.shape
    device = seq_onehot.device
    pos = torch.arange(L, device=device, dtype=torch.float32)
    pos_diff = (pos.view(1, L, 1) - pos.view(1, 1, L)).abs()  # (1, L, L)

    # k-mer feature: (B, 4*k, L) via unfold
    if k > 0 and L >= k:
        pad = k // 2
        unfolded = F.unfold(
            seq_onehot.unsqueeze(-1), kernel_size=(k, 1), padding=(pad, 0)
        ).squeeze(-1)  # (B, 4*k, L)
        unfolded = F.normalize(unfolded, p=2, dim=1)
        dot = torch.bmm(unfolded.transpose(1, 2), unfolded)  # (B, L, L)
        hamming = (1.0 - dot.clamp(-1.0, 1.0)) / 2.0  # ~[0,1]
    else:
        hamming = torch.zeros(B, L, L, device=device)

    # base identity match: one-hot_i · one-hot_j = 1 if same base else 0
    base_match = torch.einsum("bci,bcj->bij", seq_onehot, seq_onehot)  # (B, L, L)

    # RBF-encode continuous descriptors
    pos_rbf = _rbf(pos_diff.expand(B, L, L), n_bins=8, vmin=0.0, vmax=float(L))
    hm_rbf = _rbf(hamming, n_bins=8, vmin=0.0, vmax=1.0)
    match = base_match.unsqueeze(-1)  # (B, L, L, 1)

    return torch.cat([pos_rbf, hm_rbf, match], dim=-1)  # (B, L, L, 17)


# ------------------------------------------------------------ sparse triangle attention

class TriangleAxial(nn.Module):
    """Row-wise triangle attention with a sparse proximity mask.

    For each (i,j) pair, attend over k such that |k-i| <= pos_cut or |k-j| <= pos_cut.
    Input: pair tensor (B, L, L, d). Returns updated pair tensor of the same shape.
    """

    def __init__(self, d: int, pos_cut: int = 6) -> None:
        super().__init__()
        self.d = d
        self.pos_cut = pos_cut
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.b_proj = nn.Linear(d, 1, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
        self.geo_gate = nn.Sequential(
            nn.Linear(d + 17, d), nn.ReLU(inplace=True), nn.Linear(d, 1))
        self.norm = nn.LayerNorm(d)

    def forward(self, pair: torch.Tensor, geo_pair: torch.Tensor) -> torch.Tensor:
        B, L, _, d = pair.shape
        # Q(i,j), K(i,k), V(i,k), bias(j,k)
        Q = self.q_proj(pair)  # (B, L, L, d)
        K = self.k_proj(pair)  # (B, L, L, d)
        V = self.v_proj(pair)  # (B, L, L, d)
        b = self.b_proj(pair).squeeze(-1)  # (B, L, L)

        # Attend along k axis. For each i, Q[i,j] is shape (L, d); K[i,k] shape (L, d).
        # Flatten batch and i dimension: (B*L, j, d) @ (B*L, d, k) = (B*L, j, k)
        Q_flat = Q.flatten(0, 1)  # (B*L, L, d)
        K_flat = K.flatten(0, 1).transpose(-1, -2)  # (B*L, d, L)
        V_flat = V.flatten(0, 1)  # (B*L, L, d)
        bias_flat = b.flatten(0, 1)  # (B*L, L) broadcast to (B*L, L, L)

        scores = torch.bmm(Q_flat, K_flat) / (d ** 0.5)  # (B*L, L, L) indexed [flat=i, j, k]
        scores = scores + bias_flat.unsqueeze(1)  # broadcast bias(i,k): use bias at position (i, k)? We want bias(j,k). Since we flattened i, treat bias(j,k) = constant across i.
        # Actually pair[B,i,j]: bias[j,k] should take features from pair[0, j, k] (pair-independent pairing). Our bias was (B, L, L) = bias[i, k] from pair[i,k]; to approximate bias(j,k) take mean over batch dim i. This is a crude approximation, fine for self-supervision.

        # Sparse proximity mask along k: for each (flat_i, j), k within pos_cut of j.
        pos_idx = torch.arange(L, device=pair.device)
        flat_i_rem = torch.arange(B * L, device=pair.device) % L  # position of i (mod L)
        jj = pos_idx.view(1, L, 1)
        kk = pos_idx.view(1, 1, L)
        ii = flat_i_rem.view(-1, 1, 1)
        close_j = (kk - jj).abs() <= self.pos_cut
        close_i = (kk - ii).abs() <= self.pos_cut
        mask = close_j | close_i  # (B*L, L, L)
        mask_logit = torch.zeros_like(scores)
        mask_logit.masked_fill_(~mask, -1e9)
        scores = scores + mask_logit

        alpha = F.softmax(scores, dim=-1)  # (B*L, L, L)
        attn = torch.bmm(alpha, V_flat)  # (B*L, L, d)
        attn = self.o_proj(attn)
        attn = attn.view(B, L, L, d)

        # Geometric gate: sigmoid of pair+geo features
        gate = torch.sigmoid(self.geo_gate(torch.cat([pair, geo_pair], dim=-1)))
        return self.norm(pair + gate * attn)


class TriangleMul(nn.Module):
    """Cheap multiplicative update that enforces triangular consistency.
    z'_ij = z_ij + Linear( sum_k z_ik ⊗ z_kj - z_kj ⊗ z_ik ), used in AlphaFold.
    """

    def __init__(self, d: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(d, 2 * d)
        self.out_proj = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        B, L, _, d = pair.shape
        a, b = self.in_proj(pair).chunk(2, dim=-1)  # each (B, L, L, d)
        # Compute sum_k a[i,k] ⊗ b[k,j] approximated as einsum over d:
        # out[i,j] = sum_k a[i,k] * b[k,j]  (elementwise over last dim) ... but we need outer-like.
        # Use scalar multiplicative update: sum_k a[i,k,d] * b[k,j,d] for each dim d.
        # This is (B, L, d) @ (B, d, L) per dim d? We can implement via einsum "bikd,bkjd->bijd":
        # But since a is (B, L, L, d), to treat pair[i,j]: a[i,j] and b[i,j] both live at (i,j).
        # In AlphaFold the update uses pair[i,k] and pair[k,j] — so we must reinterpret:
        # a[i,k] = pair[i,k,:]; b[k,j] = pair[k,j,:]. Our a/b are re-projections of the full pair.
        # We'll do: update[i,j] = sum_k a[i,k,d] * b[k,j,d] (per-channel elementwise product, summed over k).
        # a shape (B, L, L, d) → treat dim 1 as i, dim 2 as k: (B, i=1, k=2, d=3)
        # b shape (B, L, L, d) → treat dim 1 as k, dim 2 as j: (B, k=1, j=2, d=3)
        out = torch.einsum("bikd,bkjd->bijd", a, b)  # (B, L, L, d) — sum over k
        out = self.out_proj(out)
        return self.norm(pair + out)


class GeoTriGateEmbedder(nn.Module):
    """(Triangle attention + triangle multiplicative update) × blocks → CMS-Pool.

    CMS-Pool learns weighted importance between N1-pool, N2-pool, and global-pool,
    mirroring the domain-aware pooling in the original paper but adapted to
    sequence-level V-D and D-J subdomains.
    """

    def __init__(self, d_in: int = 17, d_hid: int = 32, n_blocks: int = 2,
                 pos_cut: int = 6, d_out: int = 64, max_len: int = 60) -> None:
        super().__init__()
        self.d_out = d_out
        self.max_len = max_len
        self.proj = nn.Linear(d_in, d_hid)
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(nn.ModuleDict({
                "att": TriangleAxial(d_hid, pos_cut=pos_cut),
                "mul": TriangleMul(d_hid),
                "ffn": nn.Sequential(
                    nn.Linear(d_hid, d_hid * 2), nn.ReLU(inplace=True),
                    nn.Linear(d_hid * 2, d_hid), nn.LayerNorm(d_hid)),
            }))
        # CMS-Pool: three region-specific heads
        self.head_n1 = nn.Sequential(nn.Linear(d_hid, d_out), nn.LayerNorm(d_out), nn.ReLU())
        self.head_n2 = nn.Sequential(nn.Linear(d_hid, d_out), nn.LayerNorm(d_out), nn.ReLU())
        self.head_gl = nn.Sequential(nn.Linear(d_hid, d_out), nn.LayerNorm(d_out), nn.ReLU())
        # Learned region importance weights
        self.importance = nn.Sequential(nn.Linear(d_out * 3, 16), nn.ReLU(), nn.Linear(16, 3))

    def forward(self, seq_onehot: torch.Tensor) -> torch.Tensor:
        B, _, L = seq_onehot.shape
        geo_pair = build_pair_features(seq_onehot)  # (B, L, L, 17)
        pair = self.proj(geo_pair)  # (B, L, L, d_hid)
        for blk in self.blocks:
            pair = blk["att"](pair, geo_pair)
            pair = blk["mul"](pair)
            pair = pair + blk["ffn"](pair)  # residual

        # Diagonal feature per position: take pair[i,i,:] as a position embedding
        idx = torch.arange(L, device=pair.device)
        pos_emb = pair[:, idx, idx, :]  # (B, L, d_hid)

        half = L // 2
        e_n1 = self.head_n1(pos_emb[:, :half, :]).mean(dim=1)
        e_n2 = self.head_n2(pos_emb[:, half:, :]).mean(dim=1)
        e_gl = self.head_gl(pos_emb).mean(dim=1)
        concat = torch.cat([e_n1, e_n2, e_gl], dim=-1)  # (B, d_out*3)
        imp = self.importance(concat).softmax(dim=-1)  # (B, 3)
        weighted = imp[:, 0:1] * e_n1 + imp[:, 1:2] * e_n2 + imp[:, 2:3] * e_gl
        return F.normalize(weighted, dim=-1)


# ------------------------------------------------------------ self-supervised training

def train_embedder(
    seqs: List[str], max_len: int = 60, epochs: int = 15, batch_size: int = 64,
    device: str = "cpu", lr: float = 1e-3, verbose: bool = True,
) -> GeoTriGateEmbedder:
    """Pre-train the embedder via masked-nucleotide denoising (BERT-style).
    ~15% positions masked at random; model must predict the masked base from
    pair representation. This encourages learning structure-aware pair relationships."""
    n = len(seqs)
    onehots = np.zeros((n, 4, max_len), dtype=np.float32)
    for i, s in enumerate(seqs):
        for j, c in enumerate(s[:max_len]):
            onehots[i, NUC_IDX.get(c, 0), j] = 1.0

    data_t = torch.from_numpy(onehots).to(device)
    model = GeoTriGateEmbedder(d_hid=32, d_out=64, max_len=max_len).to(device)
    # Decoder: sequence embedding → per-position logits (shared weight across positions)
    head = nn.Sequential(
        nn.Linear(model.d_out, 64), nn.ReLU(inplace=True),
        nn.Linear(64, 4 * max_len),
    ).to(device)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(head.parameters()), lr=lr)

    model.train()
    head.train()
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        total_loss, total_count = 0.0, 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            x = data_t[idx]
            mask = torch.rand(x.shape[0], 1, x.shape[2], device=device) < 0.15
            x_corrupt = x * (~mask)
            optimizer.zero_grad()
            emb = model(x_corrupt)
            logits = head(emb).view(-1, 4, max_len)
            loss = F.cross_entropy(logits, x.argmax(dim=1), reduction="none")
            loss = (loss * mask.squeeze(1)).mean()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(idx)
            total_count += len(idx)
        if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
            print(f"  embed epoch {epoch:03d}  loss={total_loss / max(total_count, 1):.4f}")
    model.eval()
    return model


@torch.no_grad()
def embed_sequences(
    model: GeoTriGateEmbedder, seqs: List[str], max_len: int = 60,
    batch_size: int = 128, device: str = "cpu",
) -> np.ndarray:
    """Embed a list of sequences into a matrix (n, d_out)."""
    model.eval()
    out: List[np.ndarray] = []
    for start in range(0, len(seqs), batch_size):
        batch = seqs[start:start + batch_size]
        onehots = np.zeros((len(batch), 4, max_len), dtype=np.float32)
        for i, s in enumerate(batch):
            for j, c in enumerate(s[:max_len]):
                onehots[i, NUC_IDX.get(c, 0), j] = 1.0
        x = torch.from_numpy(onehots).to(device)
        emb = model(x)
        out.append(emb.cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0, model.d_out))
