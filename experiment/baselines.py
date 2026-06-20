"""Baselines: k-th order Markov chain (standard IG/TCR junction generation)
and uniform random sampling. These act as the lower-bound generators that
BayesAIRR and our proposed two-stage pipeline must outperform."""
from __future__ import annotations

import random
from typing import Dict, List, Tuple

import numpy as np

from .data import NUC, NUC_IDX, JunctionRecord


class MarkovBaseline:
    """k-th order Markov model over nucleotides.
    Trained on N1+N2 concatenated junction strings (with implicit independence
    between N1 and N2 we handle by training on the concatenation; this matches
    how the original IGoR-like models factor junction generation)."""

    def __init__(self, k: int = 2, pseudo: float = 0.1) -> None:
        self.k = k
        self.pseudo = pseudo
        # transition counts: context_tuple -> dict(base -> count)
        self.trans: Dict[Tuple, Dict[str, float]] = {}
        self.length_dist: Dict[int, float] = {}

    def fit(self, seqs: List[str]) -> "MarkovBaseline":
        # length distribution
        for s in seqs:
            self.length_dist[len(s)] = self.length_dist.get(len(s), 0.0) + 1.0
        # k-mer transition counts
        for s in seqs:
            padded = "^" * self.k + s
            for i in range(len(padded) - self.k):
                ctx = tuple(padded[i : i + self.k])
                nxt = padded[i + self.k]
                self.trans.setdefault(ctx, {})
                self.trans[ctx][nxt] = self.trans[ctx].get(nxt, 0.0) + 1.0
        return self

    def _sample_length(self, rng: random.Random) -> int:
        total = sum(self.length_dist.values())
        r = rng.uniform(0, total)
        acc = 0.0
        for L, w in self.length_dist.items():
            acc += w
            if acc >= r:
                return L
        return max(self.length_dist.keys())

    def sample(self, n: int, seed: int = 0) -> List[str]:
        rng = random.Random(seed)
        out: List[str] = []
        uniform = {c: 1.0 / len(NUC) for c in NUC}
        for _ in range(n):
            L = self._sample_length(rng)
            ctx = tuple("^" * self.k)
            seq_chars: List[str] = []
            for _ in range(L):
                counts = self.trans.get(ctx, uniform)
                total = sum(counts.values()) + self.pseudo * len(NUC)
                # add pseudo-count smoothing over all 4 nucleotides
                probs = [(counts.get(c, 0.0) + self.pseudo) / total for c in NUC]
                r = rng.uniform(0, 1.0)
                acc = 0.0
                chosen = NUC[-1]
                for c, p in zip(NUC, probs):
                    acc += p
                    if acc >= r:
                        chosen = c
                        break
                seq_chars.append(chosen)
                ctx = tuple((*ctx[1:], chosen))
            out.append("".join(seq_chars))
        return out


class RandomBaseline:
    """Uniform nucleotide + length matched to training set."""

    def __init__(self) -> None:
        self.lengths: List[int] = []

    def fit(self, seqs: List[str]) -> "RandomBaseline":
        self.lengths = [len(s) for s in seqs]
        return self

    def sample(self, n: int, seed: int = 0) -> List[str]:
        rng = random.Random(seed)
        out: List[str] = []
        for _ in range(n):
            L = rng.choice(self.lengths)
            out.append("".join(rng.choice(NUC) for _ in range(L)))
        return out


def records_from_seqs(
    seqs: List[str], template_records: List[JunctionRecord]
) -> List[JunctionRecord]:
    """Wrap generator output strings into JunctionRecords by pairing them with
    conditioning sampled from template records (cycle through)."""
    out: List[JunctionRecord] = []
    n = len(template_records)
    for i, s in enumerate(seqs):
        tmpl = template_records[i % n]
        # simple split: first half → N1, second half → N2
        mid = len(s) // 2
        out.append(JunctionRecord(
            v_gene=tmpl.v_gene, d_gene=tmpl.d_gene, j_gene=tmpl.j_gene,
            v_flank=tmpl.v_flank, d5_flank=tmpl.d5_flank,
            d3_flank=tmpl.d3_flank, j_flank=tmpl.j_flank,
            deletions=tmpl.deletions, n1=s[:mid], n2=s[mid:],
            log_p=float("nan"),
        ))
    return out
