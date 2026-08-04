# BayesAIRR

**Bayesian deep generative model of B-cell receptor N-region nucleotide insertion**

Given V/D/J gene segments, flanking sequences, and deletion lengths, BayesAIRR generates realistic N1 and N2 nucleotide sequences using a Bayesian neural network with ~800K parameters.

## Quick Start

```bash
pip install -e .
```

```python
import torch
from bayes_airr import load_checkpoint

# Load pretrained model (GPU recommended)
gen = load_checkpoint("model.pt", device="cuda")
gen.eval()

# Generate N-region sequences
n1, n1_len, n2, n2_len, log_p = gen.generate(
    v_genes=["IGHV3-23*01"],
    d_genes=["IGHD1-1*01"],
    j_genes=["IGHJ4*01"],
    v_flanks=["TGC"],
    d5_flanks=["GGT"],
    d3_flanks=["ACA"],
    j_flanks=["TGG"],
    deletions=torch.tensor([[5, 2, 3, 4]], dtype=torch.float32),
    sigma_thresh=1.0,  # 1.0=realistic, >1.0=more diverse
)

print(f"N1: {n1[0]} (len={n1_len[0]})")
print(f"N2: {n2[0]} (len={n2_len[0]})")
```

## Model Architecture

```
V/D/J genes → GeneEmbedding (family+allele, 64d×3)
Flanks      → FlankPhysChemEncoder (6-channel CNN, 64d×4)
Deletions   → DeletionEncoder (Linear 4→16)
    ↓ Concat → 464d → LayerNorm
    ↓ 3-layer Bayesian MLP (512→512→256)
    ├── N1 head (length + sequence)
    └── N2 head (length + sequence)
```

**Key features:**
- **Condition-aware**: Gene context determines global N-region properties (GC bias)
- **Bayesian uncertainty**: KL regularization protects rare gene combinations from overfitting
- **Temperature control**: `sigma_thresh` tunes generation diversity
- **Local context bias**: 2-order Markov correction for base-level accuracy

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0
- NumPy


## License

See LICENSE file.


## E-mail
2210240103@csu.edu.cn

