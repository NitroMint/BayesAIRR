#!/usr/bin/env python3
"""Example: load BayesAIRR pretrained model and generate N-region sequences."""
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bayes_airr import load_checkpoint


def main():
    print("Loading model...")
    gen = load_checkpoint("model.pt", device="cuda")
    gen.eval()
    print(f"  Model loaded. Device: {gen.device}")

    v_gene = "IGHV3-23*01"
    d_gene = "IGHD1-1*01"
    j_gene = "IGHJ4*01"

    print(f"\nGenerating 5 sequences for {v_gene} x {d_gene} x {j_gene}:")
    print("-" * 60)

    for i in range(5):
        n1, n1_len, n2, n2_len, log_p = gen.generate(
            v_genes=[v_gene],
            d_genes=[d_gene],
            j_genes=[j_gene],
            v_flanks=["TGC"],
            d5_flanks=["GGT"],
            d3_flanks=["ACA"],
            j_flanks=["TGG"],
            deletions=torch.tensor([[5, 2, 3, 4]], dtype=torch.float32),
            sigma_thresh=1.0,
        )
        print(f"  Seq {i+1}: N1={n1[0] or '(empty)'} (len={n1_len[0]}), "
              f"N2={n2[0] or '(empty)'} (len={n2_len[0]}), logP={log_p[0]:.2f}")

    print(f"\nDiversity sweep (sigma_thresh = 0.5, 1.0, 2.0, 5.0):")
    print("-" * 60)
    for sigma in [0.5, 1.0, 2.0, 5.0]:
        n1, n1_len, n2, n2_len, log_p = gen.generate(
            v_genes=[v_gene] * 100,
            d_genes=[d_gene] * 100,
            j_genes=[j_gene] * 100,
            v_flanks=["TGC"] * 100,
            d5_flanks=["GGT"] * 100,
            d3_flanks=["ACA"] * 100,
            j_flanks=["TGG"] * 100,
            deletions=torch.tensor([[5, 2, 3, 4]] * 100, dtype=torch.float32),
            sigma_thresh=sigma,
        )
        unique = len(set(zip(n1, n2)))
        print(f"  sigma={sigma:.1f}: {unique}/100 unique junctions")


if __name__ == "__main__":
    main()
