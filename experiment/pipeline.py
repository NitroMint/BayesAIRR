"""Two-stage filter for amplicon authenticity:
  Stage 1 — BayesAIRR log-probability thresholding.
  Stage 2 — GeoTriGate manifold cluster pruning: cluster real+generated together,
             drop any sequence whose cluster contains less than a fraction of
             real-data-labeled sequences (it lives in "dead" structural territory).

For the validation design we also need an ablation variant:
  - BayesAIRR-only (skip manifold pruning) to measure the gain from Stage 2.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity

from .data import NUC, NUC_IDX, JunctionRecord, conditioning_to_features
from .geotrigate import GeoTriGateEmbedder, embed_sequences


# ------------------------------------------------------------ Stage 1: BayesAIRR scorer

def score_with_bayesairr(
    generator,
    records: List[JunctionRecord],
    batch_size: int = 500,
) -> np.ndarray:
    """Use BayesAIRR generator's internal scoring: encode VDJ+deletion+flanks,
    then autoregressive log-prob under the model is returned as log P(seq | genes).

    Because we do not have the original pretrained model's weights exposed as a
    standalone scorer, we construct a proxy log-likelihood from the same
    encoded-feature → sequence mapping: encode the conditioning for the record,
    then compute position-wise log-probability from the N1/N2 heads.

    Here we use the generator's `score` wrapper (if exposed) or fall back to a
    heuristic: `log_p` stored on the JunctionRecord (which was produced during
    data generation).
    """
    # Fast path: if record.log_p was populated at generation time, use it.
    have_numeric = []
    mask = []
    for r in records:
        try:
            have_numeric.append(float(r.log_p))
            mask.append(True)
        except Exception:
            have_numeric.append(0.0)
            mask.append(False)
    if all(mask):
        return np.array(have_numeric, dtype=np.float32)

    # Fall back to calling generator.score() — only valid if user provided a
    # fully-formed BayesAIRRGenerator with the proper API.
    raise NotImplementedError(
        "BayesAIRR scoring requires the generator.score() API. In this demo we "
        "rely on log_p stored at generation time; ensure JunctionRecord.log_p is populated.")


def bayesairr_keep_mask(
    scores: np.ndarray, q_low: float = 0.05, q_high: float = 1.00,
) -> np.ndarray:
    """Keep sequences whose BayesAIRR log-prob falls between the q_low quantile
    and q_high quantile of the reference score distribution.

    We use quantile (rather than a fixed threshold) because log-prob magnitudes
    vary by dataset size. q_low=0.05 drops the ~5% most improbable sequences."""
    lo = np.quantile(scores, q_low)
    hi = np.quantile(scores, q_high)
    return (scores >= lo) & (scores <= hi)


# ------------------------------------------------------------ Stage 2: manifold cluster pruning

def manifold_pruning_mask(
    model: GeoTriGateEmbedder,
    candidate_seqs: List[str],
    real_seqs: List[str],
    n_clusters: int = 15,
    real_density_threshold: float = 0.10,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """Cluster candidate sequences alongside a reference set of real sequences.
    Any candidate whose cluster has < real_density_threshold fraction of
    real-vs-candidate sequences is pruned. Intuition: clusters without real-data
    support correspond to structural/functional zones the rearrangement process
    never visits in reality.

    Returns:
      keep_mask: (len(candidate_seqs),) boolean mask.
      cluster_real_frac: (len(candidate_seqs),) per-sequence real-fraction of its cluster.
    """
    labels_all = np.concatenate([
        np.zeros(len(candidate_seqs), dtype=np.int32),  # 0 = candidate
        np.ones(len(real_seqs), dtype=np.int32),        # 1 = real
    ])
    all_seqs = list(candidate_seqs) + list(real_seqs)
    emb = embed_sequences(model, all_seqs, device=device)

    kmeans = KMeans(n_clusters=n_clusters, n_init=5, random_state=1).fit(emb)
    cid = kmeans.labels_

    # per cluster real-data fraction
    real_frac = np.zeros(n_clusters, dtype=np.float32)
    for c in range(n_clusters):
        in_cluster = (cid == c)
        total = int(in_cluster.sum())
        if total == 0:
            real_frac[c] = 0.0
        else:
            real_frac[c] = float(((labels_all == 1) & in_cluster).sum()) / total

    candidates_clusters = cid[: len(candidate_seqs)]
    candidate_real_frac = real_frac[candidates_clusters]
    keep = candidate_real_frac >= real_density_threshold
    return keep, candidate_real_frac


# ------------------------------------------------------------ composite pipeline

def two_stage_filter(
    candidate_records: List[JunctionRecord],
    real_train_records: List[JunctionRecord],
    embedder: GeoTriGateEmbedder,
    stage1_q_low: float = 0.05,
    device: str = "cpu",
) -> Tuple[List[JunctionRecord], dict]:
    """Apply both stages. Returns (kept_records, diagnostics dict)."""
    # Stage 1
    scores = np.array([r.log_p for r in candidate_records], dtype=np.float32)
    stage1 = bayesairr_keep_mask(scores, q_low=stage1_q_low)

    # Stage 2 — operate on survivors of stage 1 to reduce cluster contamination
    candidate_seqs = [r.junction for r, m in zip(candidate_records, stage1) if m]
    real_seqs = [r.junction for r in real_train_records]
    if len(candidate_seqs) == 0:
        return [], {
            "n_candidate": len(candidate_records),
            "after_stage1": int(stage1.sum()),
            "after_stage2": 0,
            "stage2_real_frac_dist": np.zeros(1),
        }
    stage2, real_frac = manifold_pruning_mask(
        embedder, candidate_seqs, real_seqs, device=device)

    # Map back to full candidate list
    kept_full = [False] * len(candidate_records)
    idx = 0
    for i, m in enumerate(stage1):
        if m:
            if stage2[idx]:
                kept_full[i] = True
            idx += 1

    kept = [r for r, m in zip(candidate_records, kept_full) if m]
    return kept, {
        "n_candidate": len(candidate_records),
        "after_stage1": int(stage1.sum()),
        "after_stage2": int(sum(kept_full)),
        "stage2_real_frac_mean": float(real_frac.mean()),
        "stage2_real_frac_std": float(real_frac.std()),
    }
