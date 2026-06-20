"""WGAN-GP-style sequence generator for junction strings.

This generator operates in logit-space: for each position it emits a softmax
over {A,C,G,T}. We train via straight-through Gumbel-softmax relaxation so that
discrete sampling remains differentiable for backprop through the critic.

Architecture:
    cond(z, feat) → 2-layer MLP → position-wise logits (n_batch, 4, max_len)
    critic: conv-net over one-hot sequence → scalar Wasserstein score.

Crucially, this is a deliberately *naive* generator — it does NOT use BayesAIRR
internally. The two-stage pipeline (BayesAIRR scoring + GeoTriGate manifold
pruning) is meant to *reject* its bad samples, demonstrating that post-hoc
filtering recovers biological fidelity even from a weak generator.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .data import NUC, NUC_IDX, JunctionRecord, conditioning_to_features


class Generator(nn.Module):
    def __init__(self, cond_dim: int, latent_dim: int = 64, hidden_dim: int = 128,
                 max_len: int = 60) -> None:
        super().__init__()
        self.max_len = max_len
        self.cond_dim = cond_dim
        self.latent_dim = latent_dim
        self.fc = nn.Sequential(
            nn.Linear(cond_dim + latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(inplace=True),
        )
        # Per-position logits: (n, hidden*2) → (n, 4*max_len)
        self.head = nn.Linear(hidden_dim * 2, 4 * max_len)
        # Length predictor: (n, hidden*2) → (n, max_len+1) softmax
        self.len_head = nn.Linear(hidden_dim * 2, max_len + 1)

    def forward(self, cond: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.fc(torch.cat([cond, z], dim=-1))
        logits = self.head(x).view(-1, 4, self.max_len)  # (B,4,max_len)
        len_logits = self.len_head(x)  # (B, max_len+1)
        return logits, len_logits


class Critic(nn.Module):
    """1D-convolutional critic over one-hot sequences; outputs a scalar."""

    def __init__(self, hidden_dim: int = 64, max_len: int = 60) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(4, hidden_dim, kernel_size=5, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, seq_onehot: torch.Tensor) -> torch.Tensor:
        h = self.conv(seq_onehot)  # (B, hidden, max_len)
        h = self.pool(h).squeeze(-1)
        return self.fc(h).squeeze(-1)  # (B,)


def gumbel_softmax_sample(logits: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """Differentiable relaxation of discrete sampling along the 4-dim axis.
    logits shape (B, 4, L) → return shape (B, 4, L) — soft one-hot."""
    g = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
    y = F.softmax((logits + g) / tau, dim=1)
    # straight-through estimator: binarize in forward pass, keep soft in backward.
    y_hard = torch.zeros_like(y).scatter_(
        1, y.argmax(dim=1, keepdim=True), 1.0
    )
    return (y_hard - y).detach() + y


def sample_discrete(logits: torch.Tensor, len_logits: torch.Tensor) -> List[str]:
    """Argmax-sample from generator outputs → nucleotide strings with learned lengths."""
    len_probs = F.softmax(len_logits, dim=-1).cpu().numpy()
    base_probs = F.softmax(logits, dim=1).cpu().numpy()  # (B,4,L)
    out: List[str] = []
    B = logits.shape[0]
    for i in range(B):
        L = int(np.argmax(len_probs[i]))
        L = max(1, min(L, logits.shape[-1]))
        bases = base_probs[i, :, :L].argmax(axis=0)
        out.append("".join(NUC[b] for b in bases))
    return out


def train_gan(
    records_train: List[JunctionRecord],
    n_epochs: int = 60,
    batch_size: int = 128,
    lr: float = 1e-3,
    latent_dim: int = 64,
    max_len: int = 60,
    device: str = "cpu",
    tau_start: float = 2.0,
    tau_end: float = 0.5,
    verbose: bool = True,
) -> Tuple[Generator, np.ndarray]:
    """Train a conditional WGAN-GP generator. Returns the trained generator and
    the fixed conditioning features (for eval-time sampling consistency)."""
    feats = conditioning_to_features(records_train)
    cond_dim = feats.shape[1]

    # Build training one-hot tensors
    seq_onehots = np.zeros((len(records_train), 4, max_len), dtype=np.float32)
    for i, r in enumerate(records_train):
        for j, c in enumerate(r.junction[:max_len]):
            seq_onehots[i, NUC_IDX.get(c, 0), j] = 1.0

    feats_t = torch.from_numpy(feats).float().to(device)
    seqs_t = torch.from_numpy(seq_onehots).float().to(device)
    ds = TensorDataset(feats_t, seqs_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    gen = Generator(cond_dim=cond_dim, latent_dim=latent_dim, max_len=max_len).to(device)
    critic = Critic(max_len=max_len).to(device)
    opt_g = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.9))
    opt_c = torch.optim.Adam(critic.parameters(), lr=lr * 2, betas=(0.5, 0.9))

    n_critic = 5
    gp_lambda = 10.0

    gen.train()
    critic.train()
    for epoch in range(n_epochs):
        tau = tau_start - (tau_start - tau_end) * (epoch / max(n_epochs - 1, 1))
        c_losses, g_losses = [], []
        for it, (cond_b, real_b) in enumerate(loader):
            B = cond_b.shape[0]
            # --- Critic step ---
            for _ in range(n_critic):
                opt_c.zero_grad()
                z = torch.randn(B, latent_dim, device=device)
                with torch.no_grad():
                    fake_logits, _ = gen(cond_b, z)
                fake_soft = gumbel_softmax_sample(fake_logits, tau=tau)
                real_score = critic(real_b)
                fake_score = critic(fake_soft)
                # Gradient penalty
                eps = torch.rand(B, 1, 1, device=device)
                x_hat = eps * real_b + (1 - eps) * fake_soft
                x_hat.requires_grad_(True)
                d_xhat = critic(x_hat)
                grads = torch.autograd.grad(
                    d_xhat.sum(), x_hat, create_graph=True, retain_graph=True
                )[0]
                gp = ((grads.flatten(1).norm(2, dim=1) - 1.0) ** 2).mean()
                loss_c = -(real_score.mean() - fake_score.mean()) + gp_lambda * gp
                loss_c.backward()
                opt_c.step()
                c_losses.append(loss_c.item())

            # --- Generator step ---
            opt_g.zero_grad()
            z = torch.randn(B, latent_dim, device=device)
            fake_logits, fake_len = gen(cond_b, z)
            fake_soft = gumbel_softmax_sample(fake_logits, tau=tau)
            g_score = critic(fake_soft)
            # Length-matching auxiliary loss: encourage generator to emit
            # length-logits matching the empirical distribution.
            real_lengths = (real_b.sum(dim=1) > 0).long().sum(dim=1)  # (B,)
            len_target = F.one_hot(real_lengths.clip(max=max_len), max_len + 1).float()
            loss_len = F.cross_entropy(fake_len, len_target)
            loss_g = -g_score.mean() + 0.5 * loss_len
            loss_g.backward()
            opt_g.step()
            g_losses.append(loss_g.item())

        if verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
            print(f"  epoch {epoch:03d}  c={np.mean(c_losses):+.3f}  g={np.mean(g_losses):+.3f}  tau={tau:.2f}")

    gen.eval()
    return gen, feats


def sample_with_gan(
    gen: Generator,
    cond_feats: np.ndarray,
    n_samples: int,
    device: str = "cpu",
    latent_dim: int = 64,
    seed: int = 7,
) -> List[str]:
    """Sample `n_samples` sequences from the generator. Conditioning features are
    drawn with replacement from `cond_feats` to mimic amplifying a small sample."""
    gen.eval()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, cond_feats.shape[0], size=n_samples)
    cond_t = torch.from_numpy(cond_feats[idx]).float().to(device)
    z = torch.randn(n_samples, latent_dim, device=device)
    with torch.no_grad():
        logits, len_logits = gen(cond_t, z)
    return sample_discrete(logits, len_logits)


if __name__ == "__main__":  # smoke test
    print("GAN smoke test disabled — run via run_all.py.")
