"""Evaluation metrics. We compare a generated set (candidates) against a held-out
reference (real). Key metrics:

  - GC content L1 distance to real distribution
  - 2-mer / 3-mer frequency Jensen-Shannon divergence vs. real data
  - Novelty: fraction of candidate sequences not appearing in training data

These are intentionally "physics-aware" (GC/oligo-mer) rather than perplexity-based:
the whole point of the two-stage filter is to *recover* realistic local biochemical
statistics from a weak generator, not to match sequence-level likelihoods.
"""
from __future__ import annotations

from typing import Iterable, List, Sequence, Set, Tuple

import numpy as np

from .data import NUC


def _kmer_counts(seqs: Sequence[str], k: int, min_len: int = 5) -> dict:
    counts: dict = {}
    total = 0
    for s in seqs:
        if len(s) < min_len:
            continue
        for i in range(len(s) - k + 1):
            mer = s[i:i + k]
            if any(c not in NUC for c in mer):
                continue
            counts[mer] = counts.get(mer, 0) + 1
            total += 1
    return counts, total


def js_divergence(p_counts: dict, q_counts: dict, total_p: int, total_q: int,
                  eps: float = 1e-8) -> float:
    """Jensen-Shannon divergence between two k-mer count distributions.
    JSD = 0.5 * (KL(p || m) + KL(q || m)). Returns value in [0, log 2]."""
    all_keys = set(p_counts) | set(q_counts)
    m = {k: 0.5 * (p_counts.get(k, 0) / max(total_p, 1) +
                   q_counts.get(k, 0) / max(total_q, 1)) for k in all_keys}
    def kl(a_counts, total_a):
        out = 0.0
        for k, c in a_counts.items():
            pk = c / max(total_a, 1)
            out += pk * np.log2((pk + eps) / (m[k] + eps))
        return out
    return 0.5 * (kl(p_counts, total_p) + kl(q_counts, total_q))


def gc_content(seqs: Sequence[str]) -> np.ndarray:
    gc = []
    for s in seqs:
        if len(s) == 0:
            continue
        gc.append(sum(1 for c in s if c in "GC") / len(s))
    return np.array(gc)


def novelty(candidate_seqs: Sequence[str], train_seqs: Set[str]) -> float:
    if len(candidate_seqs) == 0:
        return 0.0
    novel = sum(1 for s in candidate_seqs if s not in train_seqs)
    return novel / len(candidate_seqs)


def evaluate(candidate_seqs: List[str], real_eval_seqs: List[str],
             train_seqs: Set[str], label: str = "") -> dict:
    if len(candidate_seqs) == 0:
        return {
            "label": label, "n": 0, "gc_dist_l1": float("nan"),
            "jsd_2mer": float("nan"), "jsd_3mer": float("nan"),
            "novelty": float("nan"),
        }
    gc_real = gc_content(real_eval_seqs)
    gc_gen = gc_content(candidate_seqs)
    gc_dist = float(np.abs(gc_gen.mean() - gc_real.mean()) +
                    0.5 * np.abs(gc_gen.std() - gc_real.std()))

    p2, tot_p2 = _kmer_counts(candidate_seqs, 2)
    q2, tot_q2 = _kmer_counts(real_eval_seqs, 2)
    jsd2 = js_divergence(p2, q2, tot_p2, tot_q2)

    p3, tot_p3 = _kmer_counts(candidate_seqs, 3)
    q3, tot_q3 = _kmer_counts(real_eval_seqs, 3)
    jsd3 = js_divergence(p3, q3, tot_p3, tot_q3)

    nov = novelty(candidate_seqs, train_seqs)

    return {
        "label": label, "n": len(candidate_seqs),
        "gc_dist_l1": gc_dist, "jsd_2mer": jsd2, "jsd_3mer": jsd3,
        "novelty": nov, "gc_mean": float(gc_gen.mean()),
    }
