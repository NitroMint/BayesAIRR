"""Data pipeline: uses the pretrained BayesAIRR generator to produce a corpus of
IGH junction sequences with known VDJ + deletion + flank conditioning.
These act as our "real" data gold-standard on which we train our GAN and baselines.
Also provides: holdout reference distribution (evaluation) and feature encoding.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

NUC = ["A", "C", "G", "T"]
NUC_IDX = {c: i for i, c in enumerate(NUC)}

# A small pool of biologically plausible IGH V/D/J genes and flanks (3-nt trimmed).
# These resemble real IMGT alleles; used only to drive BayesAIRR's feature encoder.
V_GENES = [
    "IGHV3-23*01", "IGHV3-30*01", "IGHV3-33*01", "IGHV4-34*01",
    "IGHV1-2*01", "IGHV1-69*01", "IGHV2-5*01", "IGHV5-51*01",
    "IGHV6-1*01", "IGHV7-4-1*01",
]
D_GENES = [
    "IGHD1-1*01", "IGHD2-2*01", "IGHD3-3*01", "IGHD3-10*01",
    "IGHD4-17*01", "IGHD5-12*01", "IGHD6-6*01", "IGHD7-27*01",
]
J_GENES = ["IGHJ1*01", "IGHJ2*01", "IGHJ3*01", "IGHJ4*01", "IGHJ5*01", "IGHJ6*01"]

# A pool of plausible 3-nt flanks — these are short enough to be generic.
FLANK_POOL = ["TGC", "GGT", "ACA", "TGG", "GGA", "CCT", "GCA", "TAC", "ATC", "GCC"]


@dataclass
class JunctionRecord:
    v_gene: str
    d_gene: str
    j_gene: str
    v_flank: str
    d5_flank: str
    d3_flank: str
    j_flank: str
    deletions: Tuple[int, int, int, int]  # v3, d5, d3, j5
    n1: str
    n2: str
    log_p: float

    @property
    def junction(self) -> str:
        """N1 + D-stub-less reconstruction: we treat N1+N2 as the junction string.
        For clustering/metrics we operate on N1 concatenated with N2."""
        return self.n1 + self.n2


def sample_conditioning(n: int, seed: int = 1234) -> dict:
    """Sample VDJ + deletion + flank conditioning uniformly from pools."""
    rng = random.Random(seed)
    torch_gen = torch.Generator().manual_seed(seed)

    v_genes = [rng.choice(V_GENES) for _ in range(n)]
    d_genes = [rng.choice(D_GENES) for _ in range(n)]
    j_genes = [rng.choice(J_GENES) for _ in range(n)]
    v_flanks = [rng.choice(FLANK_POOL) for _ in range(n)]
    d5_flanks = [rng.choice(FLANK_POOL) for _ in range(n)]
    d3_flanks = [rng.choice(FLANK_POOL) for _ in range(n)]
    j_flanks = [rng.choice(FLANK_POOL) for _ in range(n)]

    # Plausible deletion lengths: roughly Poisson-ish integers.
    # Clip to [0, 15] for sanity with BayesAIRR encoder.
    del_tensor = torch.randint(0, 11, (n, 4), generator=torch_gen).to(torch.float32)
    return dict(
        v_genes=v_genes, d_genes=d_genes, j_genes=j_genes,
        v_flanks=v_flanks, d5_flanks=d5_flanks, d3_flanks=d3_flanks, j_flanks=j_flanks,
        deletions=del_tensor,
    )


def generate_corpus(
    generator,
    n_train: int = 6000,
    n_eval: int = 2000,
    sigma: float = 1.0,
    seed: int = 1234,
    batch_size: int = 1000,
) -> Tuple[List[JunctionRecord], List[JunctionRecord]]:
    """Generate train/eval corpora through the pretrained BayesAIRR generator.
    The "real" distribution is whatever BayesAIRR outputs at sigma=1.0; this is a
    controlled surrogate because we lack true OAS sequencing data here.
    """
    def _gen(n: int, seed_offset: int) -> List[JunctionRecord]:
        cond = sample_conditioning(n, seed=seed + seed_offset)
        records: List[JunctionRecord] = []
        for i in range(0, n, batch_size):
            j = min(i + batch_size, n)
            n1_seqs, n1_lens, n2_seqs, n2_lens, log_ps = generator.generate(
                v_genes=cond["v_genes"][i:j],
                d_genes=cond["d_genes"][i:j],
                j_genes=cond["j_genes"][i:j],
                v_flanks=cond["v_flanks"][i:j],
                d5_flanks=cond["d5_flanks"][i:j],
                d3_flanks=cond["d3_flanks"][i:j],
                j_flanks=cond["j_flanks"][i:j],
                deletions=cond["deletions"][i:j],
                sigma_thresh=sigma,
            )
            dels = cond["deletions"][i:j].tolist()
            for k in range(j - i):
                records.append(JunctionRecord(
                    v_gene=cond["v_genes"][i + k],
                    d_gene=cond["d_genes"][i + k],
                    j_gene=cond["j_genes"][i:j][k],
                    v_flank=cond["v_flanks"][i + k],
                    d5_flank=cond["d5_flanks"][i + k],
                    d3_flank=cond["d3_flanks"][i + k],
                    j_flank=cond["j_flanks"][i + k],
                    deletions=tuple(int(x) for x in dels[k]),
                    n1=n1_seqs[k], n2=n2_seqs[k], log_p=float(log_ps[k]),
                ))
        return records

    train = _gen(n_train, 0)
    eval_ = _gen(n_eval, 9999)
    return train, eval_


def conditioning_to_features(records: List[JunctionRecord]) -> np.ndarray:
    """Encode VDJ family one-hot + flanks (6-channel physchem) + deletions into a
    dense real vector suitable for feeding to our neural generator.
    Returns array (n_records, d_feat)."""
    # family one-hot
    def family(allele: str) -> str:
        return allele.split("*")[0]

    v_fams = sorted({family(g) for g in V_GENES})
    d_fams = sorted({family(g) for g in D_GENES})
    j_fams = sorted({family(g) for g in J_GENES})
    v_map = {f: i for i, f in enumerate(v_fams)}
    d_map = {f: i for i, f in enumerate(d_fams)}
    j_map = {f: i for i, f in enumerate(j_fams)}

    feats = []
    for r in records:
        v_onehot = np.zeros(len(v_fams), dtype=np.float32)
        d_onehot = np.zeros(len(d_fams), dtype=np.float32)
        j_onehot = np.zeros(len(j_fams), dtype=np.float32)
        v_onehot[v_map[family(r.v_gene)]] = 1.0
        d_onehot[d_map[family(r.d_gene)]] = 1.0
        j_onehot[j_map[family(r.j_gene)]] = 1.0

        def flank_physchem(seq: str) -> np.ndarray:
            # 6-channel physicochemical encoding over up to 6 nt (pad to 6).
            out = np.zeros((6, 6), dtype=np.float32)
            for i, c in enumerate(seq[:6]):
                idx = NUC_IDX.get(c, -1)
                if 0 <= idx < 4:
                    out[idx, i] = 1.0
                # GC channel
                out[4, i] = 1.0 if c in ("G", "C") else 0.0
                # purine/pyrimidine
                out[5, i] = 1.0 if c in ("A", "G") else 0.0
            return out.flatten()

        flank_feat = np.concatenate([
            flank_physchem(r.v_flank), flank_physchem(r.d5_flank),
            flank_physchem(r.d3_flank), flank_physchem(r.j_flank),
        ])
        del_feat = np.array(r.deletions, dtype=np.float32) / 10.0  # normalize
        feat = np.concatenate([v_onehot, d_onehot, j_onehot, flank_feat, del_feat])
        feats.append(feat)
    return np.stack(feats, axis=0)


def sequence_to_onehot(seqs: List[str], max_len: int = 60) -> np.ndarray:
    """N1+N2 sequences → (n, 4, max_len) one-hot with positional masking."""
    out = np.zeros((len(seqs), 4, max_len), dtype=np.float32)
    mask = np.zeros((len(seqs), max_len), dtype=np.float32)
    for i, s in enumerate(seqs):
        for j, c in enumerate(s[:max_len]):
            idx = NUC_IDX.get(c, -1)
            if idx >= 0:
                out[i, idx, j] = 1.0
                mask[i, j] = 1.0
    return out, mask


if __name__ == "__main__":  # smoke test
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from bayes_airr import load_checkpoint

    gen = load_checkpoint("model.pt", device="cpu")
    gen.eval()
    train_rec, eval_rec = generate_corpus(gen, n_train=200, n_eval=80)
    print(f"Generated {len(train_rec)} train + {len(eval_rec)} eval records.")
    print("Sample junction:", train_rec[0].junction)
    feats = conditioning_to_features(train_rec)
    print("Feature shape:", feats.shape)
