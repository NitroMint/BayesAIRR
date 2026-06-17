# -*- coding: utf-8 -*-
"""
Bayesian junction generation network — PyTorch-based model for BCR N-region insertion (v2.0 upgraded architecture)

Key upgrades:
1. GeneEmbedding: Hierarchical Family + Allele two-level embedding, safe fallback for rare genes
2. FlankPhysChemEncoder: 6-channel physicochemical CNN, automatic flank similarity generalization
3. BayesianJunctionNet: 3-layer expanding MLP + independent N1/N2 heads, ~800K parameters
4. BNN KL regularization: Rare combinations automatically fall back to uninformative prior instead of overfitting

Design philosophy (high-diversity immune repertoire):
- Embedding/CNN are deterministic, not involved in Bayesian sampling (reduces training difficulty)
- Uncertainty is concentrated in the MLP layers — the optimal location
- sigma_thresh controls total chain entropy via weight perturbation in the MLP
- Rare combinations: BNN output naturally high variance → "I don't know" is safer than "I guessed wrong"

Mathematics:
    W_layer = mu + softplus(rho) * sigma_thresh * epsilon, epsilon ~ N(0,I)
    KL(q(W)||p(W)) regularization automatically pulls rare combinations back to the prior distribution
"""

from __future__ import annotations

import json
import math
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _set_torch_seed(seed: int, device: torch.device) -> None:
    """Bind PyTorch RNG for reproducible BNN weight and categorical sampling."""
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


# ═══════════════════════════════════════════════════════════════
# Bayesian variational linear layer (core logic unchanged)
# ═══════════════════════════════════════════════════════════════


class BayesianLinear(nn.Module):
    """
    Bayesian variational linear layer — each weight is an independent Gaussian N(mu, softplus(rho)^2).

    Reparameterization:
        sigma = softplus(rho) * sigma_scale
        W     = mu + sigma * epsilon,   epsilon ~ N(0, I)
        b     = mu_b + softplus(rho_b) * epsilon_b

    Prior: p(W) = N(0, prior_std^2)
    Posterior: q(W) = N(mu, softplus(rho)^2)

    sigma_scale is the stress scaling factor, equal to the user-configured sigma_thresh.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        prior_std: float = 1.0,
        rho_init_mean: float = -2.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std

        self.mu_w = nn.Parameter(torch.empty(out_features, in_features))
        self.rho_w = nn.Parameter(torch.empty(out_features, in_features))

        if bias:
            self.mu_b = nn.Parameter(torch.empty(out_features))
            self.rho_b = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("mu_b", None)
            self.register_parameter("rho_b", None)

        self._reset_parameters(rho_init_mean)

    def _reset_parameters(self, rho_init_mean: float) -> None:
        """Kaiming init for mu; rho initialized to small values so initial std ≈ 0.1"""
        nn.init.kaiming_uniform_(self.mu_w, a=math.sqrt(5))
        nn.init.constant_(self.rho_w, rho_init_mean)

        if self.mu_b is not None:
            fan_in = self.in_features
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(self.mu_b, -bound, bound)
            nn.init.constant_(self.rho_b, rho_init_mean)

    def forward(self, x: torch.Tensor, sigma_scale: float = 1.0) -> torch.Tensor:
        """Sample weights → linear transformation"""
        sigma_w = F.softplus(self.rho_w) * sigma_scale
        weight = self.mu_w + sigma_w * torch.randn_like(sigma_w)

        if self.mu_b is not None:
            sigma_b = F.softplus(self.rho_b) * sigma_scale
            bias = self.mu_b + sigma_b * torch.randn_like(sigma_b)
        else:
            bias = None

        return F.linear(x, weight, bias)

    def kl_divergence(self) -> torch.Tensor:
        """Closed-form KL(q||p) divergence"""
        sigma = F.softplus(self.rho_w)
        var = sigma * sigma
        mu_sq = self.mu_w * self.mu_w
        prior_var = self.prior_std * self.prior_std

        kl = 0.5 * (mu_sq / prior_var + var / prior_var - 1.0 - torch.log(var / prior_var))
        kl = kl.sum()

        if self.mu_b is not None:
            sigma_b = F.softplus(self.rho_b)
            var_b = sigma_b * sigma_b
            mu_b_sq = self.mu_b * self.mu_b
            kl_b = 0.5 * (
                mu_b_sq / prior_var + var_b / prior_var - 1.0 - torch.log(var_b / prior_var)
            )
            kl = kl + kl_b.sum()

        return kl


# ═══════════════════════════════════════════════════════════════
# Hierarchical gene embedding (Family + Allele two-level structure)
# ═══════════════════════════════════════════════════════════════


class GeneEmbedding(nn.Module):
    """
    Hierarchical gene embedding: Family (shared strength) + Allele (unique offset)

    Design motivation:
        IGHV3-23 has 300K training samples → embedding learns well
        IGHV7-4  has only 500 samples     → embedding is nearly random

    Solution:
        Rare gene:  family_embed(IGHV7) + allele_offset(≈0) → auto fallback to family representation
        Common gene: family_embed(IGHV3) + allele_offset(unique) → precise localization

    Mathematics:
        gene_repr = family_embed[family_idx] + allele_embed[allele_idx]
        final     = LayerNorm(gene_repr)
    """

    def __init__(
        self,
        num_families: int,
        num_alleles: int,
        family_dim: int = 32,
        allele_dim: int = 32,
    ) -> None:
        super().__init__()
        self.num_families = num_families
        self.num_alleles = num_alleles

        self.family_embed = nn.Embedding(num_families, family_dim)
        self.allele_embed = nn.Embedding(num_alleles, allele_dim)

        self.norm = nn.LayerNorm(family_dim + allele_dim)

        # Gene name → (family_idx, allele_idx) mapping
        self._gene_registry: Dict[str, int] = {}
        self._family_map: Dict[str, int] = {}
        self._next_allele_idx = 0
        self._next_family_idx = 0

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.family_embed.weight, mean=0.0, std=0.05)
        # Initialize allele embedding with smaller values so family prior dominates
        nn.init.normal_(self.allele_embed.weight, mean=0.0, std=0.01)

    def register_gene(self, gene_name: str) -> int:
        """Register a gene name and return its allele index"""
        if gene_name not in self._gene_registry:
            # Extract family (e.g., "IGHV3-23*01" → "IGHV3")
            family = self._extract_family(gene_name)
            if family not in self._family_map:
                self._family_map[family] = self._next_family_idx
                self._next_family_idx += 1
            self._gene_registry[gene_name] = self._next_allele_idx
            self._next_allele_idx += 1
        return self._gene_registry[gene_name]

    def _extract_family(self, gene_name: str) -> str:
        """Extract family from IMGT gene name: IGHV3-23*01 → IGHV3"""
        # Strip allele suffix
        name = gene_name.split("*")[0] if "*" in gene_name else gene_name
        # IGHV3-23 → IGHV3, IGHD1-1 → IGHD1
        import re
        match = re.match(r"(IGH[VDJ]\d+)", name)
        if match:
            return match.group(1)
        return name[:5]  # Degenerate truncation

    def get_family_index(self, gene_name: str) -> int:
        """Get family index for a gene name"""
        family = self._extract_family(gene_name)
        if family not in self._family_map:
            self.register_gene(gene_name)
        return self._family_map.get(family, 0)

    def forward(self, gene_names: List[str]) -> torch.Tensor:
        """
        Forward: gene name list → (batch, family_dim + allele_dim)

        Args:
            gene_names: list of gene names

        Returns:
            (batch_size, family_dim + allele_dim) embedding vectors
        """
        device = self.family_embed.weight.device
        family_indices: List[int] = []
        allele_indices: List[int] = []

        for name in gene_names:
            if name not in self._gene_registry:
                self.register_gene(name)
            family_indices.append(self.get_family_index(name))
            allele_indices.append(self._gene_registry[name])

        f_idx = torch.tensor(family_indices, dtype=torch.long, device=device)
        a_idx = torch.tensor(allele_indices, dtype=torch.long, device=device)

        f_emb = self.family_embed(f_idx)
        a_emb = self.allele_embed(a_idx)
        combined = torch.cat([f_emb, a_emb], dim=-1)
        return self.norm(combined)

    @property
    def output_dim(self) -> int:
        return self.family_embed.embedding_dim + self.allele_embed.embedding_dim


# ═══════════════════════════════════════════════════════════════
# Physicochemical flank encoder (6-channel CNN)
# ═══════════════════════════════════════════════════════════════


class FlankPhysChemEncoder(nn.Module):
    """
    Physicochemical CNN encoder for flank sequences.

    Input: raw flank nucleotide string (e.g. "TGC")
    Output: (batch, 64) fixed-dimension feature vector

    6 channels (instead of simple one-hot):
        Channel 1: A presence
        Channel 2: C presence
        Channel 3: G presence
        Channel 4: T presence
        Channel 5: GC content (G|C = 1, A|T = 0)
        Channel 6: Purine/Pyrimidine (A|G = 1, C|T = 0)

    Design motivation:
        "CAG" and "CAA" are physicochemically similar (both pyrimidine-purine-purine),
        CNN automatically captures this similarity through local receptive fields,
        rare flank combinations generalize via similarity rather than overfitting.
    """

    def __init__(self, max_flank_len: int = 15, out_dim: int = 64) -> None:
        super().__init__()
        self.max_flank_len = max_flank_len
        self.out_dim = out_dim

        # Small conv net: 6 channels → 32 → 64 → GlobalAvgPool
        self.conv = nn.Sequential(
            nn.Conv1d(6, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        # Project to output dimension
        self.proj = nn.Linear(64, out_dim)

    def _seq_to_physchem(self, seq: str) -> torch.Tensor:
        """Nucleotide sequence → (6, L) physicochemical tensor"""
        chars = list(seq.upper())
        L = min(len(chars), self.max_flank_len)
        tensor = torch.zeros(6, self.max_flank_len, dtype=torch.float32)

        for i, c in enumerate(chars[: self.max_flank_len]):
            if c == "A":
                tensor[0, i] = 1.0
                tensor[4, i] = 0.0  # non-GC
                tensor[5, i] = 1.0  # purine
            elif c == "C":
                tensor[1, i] = 1.0
                tensor[4, i] = 1.0  # GC
                tensor[5, i] = 0.0  # pyrimidine
            elif c == "G":
                tensor[2, i] = 1.0
                tensor[4, i] = 1.0  # GC
                tensor[5, i] = 1.0  # purine
            elif c == "T":
                tensor[3, i] = 1.0
                tensor[4, i] = 0.0  # non-GC
                tensor[5, i] = 0.0  # pyrimidine
            else:  # N or other
                tensor[:, i] = 0.25  # uniform uncertainty

        return tensor

    def forward(self, flank_seqs: List[str]) -> torch.Tensor:
        """
        Args:
            flank_seqs: list of flank sequence strings

        Returns:
            (batch_size, out_dim) physicochemical feature vectors
        """
        batch_size = len(flank_seqs)
        device = self.proj.weight.device

        physchem_batch = torch.stack(
            [self._seq_to_physchem(s) for s in flank_seqs]
        ).to(device)  # (batch, 6, max_len)

        x = self.conv(physchem_batch)  # (batch, 64, max_len)
        x = self.pool(x).squeeze(-1)  # (batch, 64)
        x = self.proj(x)  # (batch, out_dim)
        return x


# ═══════════════════════════════════════════════════════════════
# Full feature encoder (integrates GeneEmbedding + FlankPhysChemEncoder)
# ═══════════════════════════════════════════════════════════════


class JunctionFeatureEncoder(nn.Module):
    """
    Encodes (gene names, flank sequences, deletion lengths) into a unified Bayesian network input.

    Architecture:
        V gene  → GeneEmbedding (family+allele) → 64d
        D gene  → GeneEmbedding (family+allele) → 64d
        J gene  → GeneEmbedding (family+allele) → 64d
        V flank → FlankPhysChemEncoder → 64d
        D5 flank → FlankPhysChemEncoder → 64d
        D3 flank → FlankPhysChemEncoder → 64d
        J flank → FlankPhysChemEncoder → 64d
        Deletion lengths → Linear(4→16) + ReLU
        ─────────────────────────────────
        Concatenation: 64*3 + 64*4 + 16 = 464 dim
        → LayerNorm → Dropout → output

    v3.1 ablation switches:
        use_gene_embedding=False → gene embeddings replaced with zero vectors
        use_flank_cnn=False       → flank features replaced with zero vectors
    """

    def __init__(
        self,
        # ── Gene-level parameters (legacy API: allocate by type count) ──
        v_families: int = 7,
        v_alleles: int = 280,   # v5: increased default to cover all IMGT alleles
        d_families: int = 8,
        d_alleles: int = 50,
        j_families: int = 6,
        j_alleles: int = 15,
        gene_dim: int = 32,        # family + allele each 32 = 64 dim
        # ── Flank parameters ──
        flank_out_dim: int = 64,
        flank_len: int = 15,       # FlankPhysChemEncoder max flank length
        dropout_p: float = 0.1,
        # ── Ablation switches (v3.1) ──
        use_gene_embedding: bool = True,
        use_flank_cnn: bool = True,
        # ── New API: pass gene lists to auto-infer parameters ──
        v_genes: list | None = None,
        d_genes: list | None = None,
        j_genes: list | None = None,
    ) -> None:
        super().__init__()

        # Support new API: pass gene list to auto-infer family/allele counts
        if v_genes is not None:
            v_families, v_alleles = self._infer_gene_counts(v_genes)
        if d_genes is not None:
            d_families, d_alleles = self._infer_gene_counts(d_genes)
        if j_genes is not None:
            j_families, j_alleles = self._infer_gene_counts(j_genes)

        self._use_gene_embedding = use_gene_embedding
        self._use_flank_cnn = use_flank_cnn

        # Three independent gene embedders
        self.gene_out_dim = gene_dim * 2  # family + allele
        if use_gene_embedding:
            self.v_embed = GeneEmbedding(v_families, v_alleles, gene_dim, gene_dim)
            self.d_embed = GeneEmbedding(d_families, d_alleles, gene_dim, gene_dim)
            self.j_embed = GeneEmbedding(j_families, j_alleles, gene_dim, gene_dim)
        else:
            # Placeholder: embedders are None, forward uses zero vectors
            self.v_embed = None
            self.d_embed = None
            self.j_embed = None

        # One shared flank encoder (nucleotide chemistry is universal)
        if use_flank_cnn:
            self.flank_encoder = FlankPhysChemEncoder(max_flank_len=flank_len, out_dim=flank_out_dim)
        else:
            self.flank_encoder = None

        self.flank_out_dim = flank_out_dim

        # Deletion length encoder
        self.del_encoder = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(inplace=True),
        )

        total_dim = (
            self.gene_out_dim * 3   # V, D, J genes
            + flank_out_dim * 4     # V, D5, D3, J flanks
            + 16                    # deletion lengths
        )

        self.total_dim = total_dim
        self.norm = nn.LayerNorm(total_dim)
        self.dropout = nn.Dropout(dropout_p)

    @property
    def output_dim(self) -> int:
        """Convenience property: same as total_dim, for ablation experiment scripts"""
        return self.total_dim

    @staticmethod
    def _infer_gene_counts(gene_list: list) -> tuple:
        """Infer family and allele counts from a gene list"""
        import re
        families = set()
        alleles = set(gene_list)
        for g in alleles:
            if not g or g.strip() == "":
                continue
            m = re.match(r"(IGH[VDJ]\d+)", g.split("*")[0] if "*" in g else g)
            families.add(m.group(1) if m else g[:5])
        return max(len(families), 1), max(len(alleles), 1)

    def forward(
        self,
        v_genes: List[str],
        d_genes: List[str],
        j_genes: List[str],
        v_flanks: List[str],
        d5_flanks: List[str],
        d3_flanks: List[str],
        j_flanks: List[str],
        deletions: torch.Tensor,  # (batch, 4): [v3_del, d5_del, d3_del, j5_del]
    ) -> torch.Tensor:
        """Encode V/D/J genes + flanks + deletions → (batch, total_dim)"""
        bs = len(v_genes)
        device = self.del_encoder[0].weight.device

        # ── Gene embeddings (ablation: use_gene_embedding=False → zero vectors) ──
        if self._use_gene_embedding and self.v_embed is not None:
            v_feat = self.v_embed(v_genes)
            d_feat = self.d_embed(d_genes)
            j_feat = self.j_embed(j_genes)
        else:
            v_feat = torch.zeros(bs, self.gene_out_dim, device=device)
            d_feat = torch.zeros(bs, self.gene_out_dim, device=device)
            j_feat = torch.zeros(bs, self.gene_out_dim, device=device)

        # ── Flank encoding (ablation: use_flank_cnn=False → zero vectors) ──
        if self._use_flank_cnn and self.flank_encoder is not None:
            v_flank_feat = self.flank_encoder(v_flanks)
            d5_flank_feat = self.flank_encoder(d5_flanks)
            d3_flank_feat = self.flank_encoder(d3_flanks)
            j_flank_feat = self.flank_encoder(j_flanks)
        else:
            v_flank_feat = torch.zeros(bs, self.flank_out_dim, device=device)
            d5_flank_feat = torch.zeros(bs, self.flank_out_dim, device=device)
            d3_flank_feat = torch.zeros(bs, self.flank_out_dim, device=device)
            j_flank_feat = torch.zeros(bs, self.flank_out_dim, device=device)

        # ── Deletion length encoding ──
        del_max = deletions.float().max(dim=0).values
        del_max = torch.clamp(del_max, min=1.0)
        del_norm = deletions.float() / del_max.unsqueeze(0)
        del_feat = self.del_encoder(del_norm)

        # ── Concatenation + normalization ──
        combined = torch.cat(
            [
                v_feat, d_feat, j_feat,
                v_flank_feat, d5_flank_feat, d3_flank_feat, j_flank_feat,
                del_feat,
            ],
            dim=-1,
        )
        return self.dropout(self.norm(combined))


# ═══════════════════════════════════════════════════════════════
# Upgraded Bayesian junction generation network
# ═══════════════════════════════════════════════════════════════


class BayesianJunctionNet(nn.Module):
    """
    Bayesian junction generation network

    Architecture upgrade:
        - 3-layer expanding MLP (512 → 512 → 256) replaces 2-layer shrinking structure
        - N1 and N2 independent dual heads (each with Bayesian output layer)
        - LayerNorm + GELU replaces simple ReLU, improving training stability
        - Residual connections prevent deep degradation

    sigma_thresh directly amplifies the sampling noise of all weights.

    Parameters: ~850K (800K+ MLP, 50K encoder)
    Memory: ~400MB (training), ~50MB (inference)
    Expected performance:
        σ=1.0: N-region distribution vs OAS real data KL < 0.03 bits/base
        σ=1.5: Rare insertions emerge, stress-testing third-party tools
    """

    NUCLEOTIDES = ["A", "C", "G", "T"]

    def __init__(
        self,
        input_dim: int = 464,
        hidden_dim_1: int = 512,
        hidden_dim_2: int = 512,
        latent_dim: int = 256,
        max_junction_len: int = 30,
        dropout_p: float = 0.15,
        prior_std: float = 1.0,
        # ── Aliases (compatible with ablation scripts) ──
        hidden_dim: int | None = None,
        max_len: int | None = None,
    ) -> None:
        # Alias resolution
        if hidden_dim is not None:
            hidden_dim_1 = hidden_dim_2 = hidden_dim
        if max_len is not None:
            max_junction_len = max_len

        super().__init__()
        self.input_dim = input_dim
        self.max_junction_len = max_junction_len
        self.num_nucleotides = len(self.NUCLEOTIDES)

        # ── Bayesian MLP backbone (3-layer expanding, with residual structure) ──
        self.blinear1 = BayesianLinear(input_dim, hidden_dim_1, prior_std=prior_std)
        self.blinear2 = BayesianLinear(hidden_dim_1, hidden_dim_2, prior_std=prior_std)
        self.blinear3 = BayesianLinear(hidden_dim_2, latent_dim, prior_std=prior_std)

        # LayerNorm for training stability
        self.norm1 = nn.LayerNorm(hidden_dim_1)
        self.norm2 = nn.LayerNorm(hidden_dim_2)
        self.norm3 = nn.LayerNorm(latent_dim)

        # ── N1 independent head (V-D junction) ──
        self.n1_length_head = BayesianLinear(
            latent_dim, max_junction_len + 1, prior_std=prior_std
        )
        self.n1_seq_head = BayesianLinear(
            latent_dim, max_junction_len * self.num_nucleotides, prior_std=prior_std
        )

        # ── N2 independent head (D-J junction) ──
        self.n2_length_head = BayesianLinear(
            latent_dim, max_junction_len + 1, prior_std=prior_std
        )
        self.n2_seq_head = BayesianLinear(
            latent_dim, max_junction_len * self.num_nucleotides, prior_std=prior_std
        )

        self.dropout = nn.Dropout(dropout_p)

        # ── v5: Local base context embedding (3rd-order Markov correction) ──
        self.n1_local_bias = nn.Embedding(64, self.num_nucleotides)  # 4³=64 contexts (4^3)
        self.n2_local_bias = nn.Embedding(64, self.num_nucleotides)
        nn.init.zeros_(self.n1_local_bias.weight)
        nn.init.zeros_(self.n2_local_bias.weight)
        self._use_local_context = True

        # Collect all Bayesian layers for unified management
        self._all_bayesian_layers: List[BayesianLinear] = [
            self.blinear1, self.blinear2, self.blinear3,
            self.n1_length_head, self.n1_seq_head,
            self.n2_length_head, self.n2_seq_head,
        ]

    def forward(
        self,
        features: torch.Tensor,
        sigma_thresh: float = 1.0,
        n1_local_idx: torch.Tensor | None = None,
        n2_local_idx: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward: features → latent → N1/N2 predictions.

        Args:
            features: (batch, input_dim)
            sigma_thresh: stress scaling factor
            n1_local_idx: (batch, max_len) 2-base preceding context 16-class index, None=no local bias
            n2_local_idx: same for N2

        Returns:
            n1_length_logits: (batch, max_junction_len + 1)
            n1_seq_logits:    (batch, max_junction_len, 4)
            n2_length_logits: (batch, max_junction_len + 1)
            n2_seq_logits:    (batch, max_junction_len, 4)
            kl_total:         scalar
        """
        sigma = sigma_thresh

        # Backbone forward (same as before)
        h = F.gelu(self.norm1(self.blinear1(features, sigma_scale=sigma)))
        h = self.dropout(h)
        h = F.gelu(self.norm2(self.blinear2(h, sigma_scale=sigma)))
        h = self.dropout(h)
        h = F.gelu(self.norm3(self.blinear3(h, sigma_scale=sigma)))
        h = self.dropout(h)

        # N1 dual head
        n1_len_logits = self.n1_length_head(h, sigma_scale=sigma)
        n1_seq_flat = self.n1_seq_head(h, sigma_scale=sigma)
        n1_seq_logits = n1_seq_flat.view(-1, self.max_junction_len, self.num_nucleotides)

        # N2 dual head
        n2_len_logits = self.n2_length_head(h, sigma_scale=sigma)
        n2_seq_flat = self.n2_seq_head(h, sigma_scale=sigma)
        n2_seq_logits = n2_seq_flat.view(-1, self.max_junction_len, self.num_nucleotides)

        # v5: Local base context bias — preceding 2 bases influence current position prediction
        if n1_local_idx is not None and self._use_local_context:
            # (batch, max_len, 4) + (batch, max_len, 4) broadcast
            local_bias = self.n1_local_bias(n1_local_idx)  # (batch, max_len, 4)
            n1_seq_logits = n1_seq_logits + local_bias * 0.1  # small weight, fine-tuning
        if n2_local_idx is not None and self._use_local_context:
            local_bias = self.n2_local_bias(n2_local_idx)
            n2_seq_logits = n2_seq_logits + local_bias * 0.1

        # Temperature scaling
        sigma_safe = max(sigma, 0.1)
        n1_len_logits = n1_len_logits / sigma_safe
        n1_seq_logits = n1_seq_logits / sigma_safe
        n2_len_logits = n2_len_logits / sigma_safe
        n2_seq_logits = n2_seq_logits / sigma_safe

        # KL total sum
        kl_total = sum(layer.kl_divergence() for layer in self._all_bayesian_layers)

        return n1_len_logits, n1_seq_logits, n2_len_logits, n2_seq_logits, kl_total

    def sample_single_junction(
        self,
        length_logits: torch.Tensor,
        seq_logits: torch.Tensor,
    ) -> Tuple[List[str], List[int], torch.Tensor]:
        """
        Sample a single junction (N1 or N2) from length_logits + seq_logits.

        Args:
            length_logits: (batch, max_junction_len + 1)
            seq_logits:    (batch, max_junction_len, 4)

        Returns:
            junction_seqs, junction_lens, log_probs (batch,)
        """
        batch_size = length_logits.size(0)

        # Sample length
        len_dist = torch.distributions.Categorical(logits=length_logits)
        lengths = len_dist.sample()
        len_log_probs = len_dist.log_prob(lengths)

        # Per-position base sampling
        seqs: List[str] = []
        lens: List[int] = []
        seq_log_probs = torch.zeros(batch_size, device=length_logits.device)

        for i in range(batch_size):
            L = int(lengths[i].item())
            lens.append(L)

            if L == 0:
                seqs.append("")
                continue

            pos_logits = seq_logits[i, :L, :]  # (L, 4)
            pos_dist = torch.distributions.Categorical(logits=pos_logits)
            indices = pos_dist.sample()  # (L,)
            seq_log_probs[i] = pos_dist.log_prob(indices).sum()

            bases = [self.NUCLEOTIDES[idx.item()] for idx in indices]
            seqs.append("".join(bases))

        total_log_probs = len_log_probs + seq_log_probs
        return seqs, lens, total_log_probs

    @torch.no_grad()
    def sample_junctions(
        self,
        features: torch.Tensor,
        sigma_thresh: float = 1.0,
    ) -> Tuple[List[str], List[int], List[str], List[int], List[float]]:
        """
        Sample N1 and N2 junctions simultaneously.

        Args:
            features: (batch, input_dim) encoded features
            sigma_thresh: stress scaling factor

        Returns:
            n1_seqs, n1_lens, n2_seqs, n2_lens, joint_log_probs (log P(N1)+log P(N2))
        """
        (
            n1_len_logits, n1_seq_logits,
            n2_len_logits, n2_seq_logits,
            _kl,
        ) = self.forward(features, sigma_thresh)

        n1_seqs, n1_lens, n1_lp = self.sample_single_junction(n1_len_logits, n1_seq_logits)
        n2_seqs, n2_lens, n2_lp = self.sample_single_junction(n2_len_logits, n2_seq_logits)

        joint_log_probs = (n1_lp + n2_lp).tolist()

        return n1_seqs, n1_lens, n2_seqs, n2_lens, joint_log_probs

    def total_kl_divergence(self) -> torch.Tensor:
        """Sum of KL divergences across all Bayesian layers"""
        return sum(layer.kl_divergence() for layer in self._all_bayesian_layers)

    @torch.no_grad()
    def score_sequence(
        self,
        features: torch.Tensor,
        n1_seqs: List[str],
        n2_seqs: List[str],
        sigma_thresh: float = 1.0,
    ) -> List[float]:
        """Score given N1/N2 sequences using local base context enhancement."""
        batch_size = features.size(0)
        nuc_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3}

        # Build per-position local context index (preceding 2 bases → 0-15)
        def _build_local_idx(seqs, max_len):
            idx = torch.zeros(batch_size, max_len, dtype=torch.long, device=features.device)
            for i, s in enumerate(seqs):
                s_int = [nuc_to_idx.get(c, 0) for c in s] + [0, 0]  # pad
                for p in range(min(len(s), max_len)):
                    prev2 = s_int[p] if p < 2 else (s_int[p-2] * 4 + s_int[p-1])
                    idx[i, p] = min(prev2, 15)
            return idx

        n1_local = _build_local_idx(n1_seqs, self.max_junction_len)
        n2_local = _build_local_idx(n2_seqs, self.max_junction_len)

        (n1_len_logits, n1_seq_logits,
         n2_len_logits, n2_seq_logits, _) = self.forward(
            features, sigma_thresh, n1_local_idx=n1_local, n2_local_idx=n2_local)

        batch_size = features.size(0)
        log_probs = []

        nuc_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3}

        for i in range(batch_size):
            # N1 log P — clamp length to max_junction_len
            n1_l = min(len(n1_seqs[i]), self.max_junction_len)
            n1_len_log_p = torch.distributions.Categorical(
                logits=n1_len_logits[i:i+1]
            ).log_prob(torch.tensor([n1_l], device=n1_len_logits.device))

            n1_seq_log_p = torch.tensor(0.0, device=n1_len_logits.device)
            if n1_l > 0:
                idxs = [nuc_to_idx.get(c, 0) for c in n1_seqs[i][:n1_l]]
                idx_t = torch.tensor(idxs, device=n1_seq_logits.device, dtype=torch.long)
                pos_logits = n1_seq_logits[i, :n1_l, :]
                n1_seq_log_p = torch.distributions.Categorical(
                    logits=pos_logits
                ).log_prob(idx_t).sum()

            # N2 log P — clamp length to max_junction_len
            n2_l = min(len(n2_seqs[i]), self.max_junction_len)
            n2_len_log_p = torch.distributions.Categorical(
                logits=n2_len_logits[i:i+1]
            ).log_prob(torch.tensor([n2_l], device=n2_len_logits.device))

            n2_seq_log_p = torch.tensor(0.0, device=n2_len_logits.device)
            if n2_l > 0:
                idxs = [nuc_to_idx.get(c, 0) for c in n2_seqs[i][:n2_l]]
                idx_t = torch.tensor(idxs, device=n2_seq_logits.device, dtype=torch.long)
                pos_logits = n2_seq_logits[i, :n2_l, :]
                n2_seq_log_p = torch.distributions.Categorical(
                    logits=pos_logits
                ).log_prob(idx_t).sum()

            lp = (n1_len_log_p + n1_seq_log_p + n2_len_log_p + n2_seq_log_p).item()
            log_probs.append(float(lp))

        return log_probs


# ═══════════════════════════════════════════════════════════════
# Convenience interface: encode + generate in one step
# ═══════════════════════════════════════════════════════════════


class BayesAIRRGenerator:
    """
    One-stop generator: integrates encoder + BNN, takes raw biological inputs and directly outputs N1/N2.

    Usage:
        generator = BayesAIRRGenerator(device="cuda")
        n1, n1_len, n2, n2_len, log_p = generator.generate(
            v_genes=["IGHV3-23*01"],
            d_genes=["IGHD1-1*01"],
            j_genes=["IGHJ4*01"],
            v_flanks=["TGC"],
            d5_flanks=["GGT"],
            d3_flanks=["ACA"],
            j_flanks=["TGG"],
            deletions=torch.tensor([[5, 2, 3, 4]], dtype=torch.float32),
            sigma_thresh=1.0,
        )
    """

    def __init__(self,
                 encoder: JunctionFeatureEncoder | None = None,
                 bnn: BayesianJunctionNet | None = None,
                 device: str = "cuda") -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        if encoder is not None and bnn is not None:
            self.encoder = encoder.to(self.device)
            self.bnn = bnn.to(self.device)
        else:
            self.encoder = JunctionFeatureEncoder().to(self.device)
            self.bnn = BayesianJunctionNet(
                input_dim=self.encoder.total_dim
            ).to(self.device)

    def to(self, device: str) -> "BayesAIRRGenerator":
        self.device = torch.device(device)
        self.encoder = self.encoder.to(self.device)
        self.bnn = self.bnn.to(self.device)
        return self

    def eval(self) -> None:
        self.encoder.eval()
        self.bnn.eval()

    @torch.no_grad()
    def generate(
        self,
        v_genes: List[str],
        d_genes: List[str],
        j_genes: List[str],
        v_flanks: List[str],
        d5_flanks: List[str],
        d3_flanks: List[str],
        j_flanks: List[str],
        deletions: torch.Tensor,
        sigma_thresh: float = 1.0,
        seeds: List[int] | None = None,
    ) -> Tuple[List[str], List[int], List[str], List[int], List[float]]:
        """
        One-step generation: raw inputs → encode → BNN sample → N1/N2 sequences.

        Args:
            v_genes, d_genes, j_genes: list of gene names
            v_flanks, d5_flanks, d3_flanks, j_flanks: flank sequences
            deletions: (batch, 4) [v3_del, d5_del, d3_del, j5_del]
            sigma_thresh: stress scaling factor
            seeds: optional per-read random seeds; when provided, binds PyTorch RNG per read (consistent with SHM)

        Returns:
            n1_seqs, n1_lens, n2_seqs, n2_lens, joint_log_probs
        """
        batch_size = len(v_genes)
        if seeds is not None:
            if len(seeds) != batch_size:
                raise ValueError(
                    f"seeds length ({len(seeds)}) must match batch size ({batch_size})"
                )
            n1_seqs: List[str] = []
            n1_lens: List[int] = []
            n2_seqs: List[str] = []
            n2_lens: List[int] = []
            joint_log_probs: List[float] = []
            for i, seed in enumerate(seeds):
                _set_torch_seed(seed, self.device)
                features = self.encoder(
                    v_genes=[v_genes[i]],
                    d_genes=[d_genes[i]],
                    j_genes=[j_genes[i]],
                    v_flanks=[v_flanks[i]],
                    d5_flanks=[d5_flanks[i]],
                    d3_flanks=[d3_flanks[i]],
                    j_flanks=[j_flanks[i]],
                    deletions=deletions[i : i + 1].to(self.device),
                )
                n1, l1, n2, l2, lp = self.bnn.sample_junctions(
                    features, sigma_thresh=sigma_thresh
                )
                n1_seqs.extend(n1)
                n1_lens.extend(l1)
                n2_seqs.extend(n2)
                n2_lens.extend(l2)
                joint_log_probs.extend(lp)
            return n1_seqs, n1_lens, n2_seqs, n2_lens, joint_log_probs

        features = self.encoder(
            v_genes=v_genes,
            d_genes=d_genes,
            j_genes=j_genes,
            v_flanks=v_flanks,
            d5_flanks=d5_flanks,
            d3_flanks=d3_flanks,
            j_flanks=j_flanks,
            deletions=deletions.to(self.device),
        )
        return self.bnn.sample_junctions(features, sigma_thresh=sigma_thresh)

    @torch.no_grad()
    def score(
        self,
        v_genes: List[str],
        d_genes: List[str],
        j_genes: List[str],
        v_flanks: List[str],
        d5_flanks: List[str],
        d3_flanks: List[str],
        j_flanks: List[str],
        deletions: torch.Tensor,
        n1_seqs: List[str],
        n2_seqs: List[str],
        sigma_thresh: float = 1.0,
    ) -> List[float]:
        """Score real N1/N2 sequences: returns log P(N1,N2 | V,D,J,del,flank) per sequence.

        Used for NLL evaluation: comparable with Markov baseline compute_nll(true_seqs).
        """
        features = self.encoder(
            v_genes=v_genes, d_genes=d_genes, j_genes=j_genes,
            v_flanks=v_flanks, d5_flanks=d5_flanks,
            d3_flanks=d3_flanks, j_flanks=j_flanks,
            deletions=deletions.to(self.device),
        )
        return self.bnn.score_sequence(features, n1_seqs, n2_seqs, sigma_thresh)


# ═══════════════════════════════════════════════════════════════
# Save / Load utilities
# ═══════════════════════════════════════════════════════════════


def save_checkpoint(
    generator_or_encoder: BayesAIRRGenerator | JunctionFeatureEncoder,
    file_or_bnn: str | Path | BayesianJunctionNet,
    hp_or_file_path: "HyperParams | str | Path | None" = None,
    file_path_legacy: str | Path | None = None,
) -> None:
    """Save full checkpoint (model + encoder + gene registry)

    Supports two calling conventions (backward-compatible with ablation experiment scripts):
      - New API: save_checkpoint(generator, file_path)
      - Legacy API: save_checkpoint(encoder, bnn, hp, file_path)
    """
    # ── Determine calling convention ──
    from bayes_airr.models.trainer import HyperParams as _HP
    if isinstance(generator_or_encoder, BayesAIRRGenerator):
        # New API: save_checkpoint(generator, file_path)
        generator = generator_or_encoder
        file_path = Path(file_or_bnn)  # type: ignore
    else:
        # Legacy API: save_checkpoint(encoder, bnn, hp, file_path)
        encoder = generator_or_encoder
        bnn = file_or_bnn
        file_path = Path(file_path_legacy)  # type: ignore
        generator = BayesAIRRGenerator(encoder=encoder, bnn=bnn)

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "encoder_state_dict": generator.encoder.state_dict(),
        "bnn_state_dict": generator.bnn.state_dict(),
    }

    # Gene registry (handles use_gene_embedding=False case)
    if generator.encoder.v_embed is not None:
        checkpoint["v_gene_registry"] = generator.encoder.v_embed._gene_registry
        checkpoint["d_gene_registry"] = generator.encoder.d_embed._gene_registry
        checkpoint["j_gene_registry"] = generator.encoder.j_embed._gene_registry
        checkpoint["v_family_map"] = generator.encoder.v_embed._family_map
        checkpoint["d_family_map"] = generator.encoder.d_embed._family_map
        checkpoint["j_family_map"] = generator.encoder.j_embed._family_map
        checkpoint["v_families"] = generator.encoder.v_embed.num_families
        checkpoint["v_alleles"] = generator.encoder.v_embed.num_alleles
        checkpoint["d_families"] = generator.encoder.d_embed.num_families
        checkpoint["d_alleles"] = generator.encoder.d_embed.num_alleles
        checkpoint["j_families"] = generator.encoder.j_embed.num_families
        checkpoint["j_alleles"] = generator.encoder.j_embed.num_alleles
    else:
        checkpoint["v_gene_registry"] = {}
        checkpoint["d_gene_registry"] = {}
        checkpoint["j_gene_registry"] = {}
        checkpoint["v_family_map"] = {}
        checkpoint["d_family_map"] = {}
        checkpoint["j_family_map"] = {}
        checkpoint["v_families"] = 1
        checkpoint["v_alleles"] = 1
        checkpoint["d_families"] = 1
        checkpoint["d_alleles"] = 1
        checkpoint["j_families"] = 1
        checkpoint["j_alleles"] = 1

    checkpoint["input_dim"] = generator.encoder.total_dim
    checkpoint["max_junction_len"] = generator.bnn.max_junction_len
    checkpoint["hidden_dim_1"] = generator.bnn.blinear1.out_features
    checkpoint["hidden_dim_2"] = generator.bnn.blinear2.out_features
    checkpoint["latent_dim"] = generator.bnn.blinear3.out_features
    checkpoint["version"] = "3.1"

    torch.save(checkpoint, str(file_path))


def load_checkpoint(
    file_path: str | Path,
    device: str = "cuda",
    max_junction_len: int = 30,
) -> BayesAIRRGenerator:
    """Load full generator from checkpoint"""
    file_path = Path(file_path)
    checkpoint = torch.load(str(file_path), map_location="cpu", weights_only=False)

    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")

    # Rebuild encoder (use saved dimension info; for legacy format, infer from state_dict)
    def _get_or_infer(key: str, state_key: str, default: int) -> int:
        if key in checkpoint:
            return checkpoint[key]
        # Infer from state_dict weight shape
        w = checkpoint["encoder_state_dict"].get(state_key)
        if w is not None:
            return w.shape[0]
        return default

    v_fams = _get_or_infer("v_families", "v_embed.family_embed.weight", 7)
    v_alls = _get_or_infer("v_alleles", "v_embed.allele_embed.weight", 60)
    d_fams = _get_or_infer("d_families", "d_embed.family_embed.weight", 7)
    d_alls = _get_or_infer("d_alleles", "d_embed.allele_embed.weight", 30)
    j_fams = _get_or_infer("j_families", "j_embed.family_embed.weight", 1)
    j_alls = _get_or_infer("j_alleles", "j_embed.allele_embed.weight", 10)

    # Auto-detect ablation status: check if gene embedding / flank CNN weights are missing from state_dict
    enc_state = checkpoint["encoder_state_dict"]
    has_gene_embed = "v_embed.family_embed.weight" in enc_state
    has_flank_cnn = "flank_encoder.conv.0.weight" in enc_state

    encoder = JunctionFeatureEncoder(
        v_families=v_fams, v_alleles=v_alls,
        d_families=d_fams, d_alleles=d_alls,
        j_families=j_fams, j_alleles=j_alls,
        use_gene_embedding=has_gene_embed,
        use_flank_cnn=has_flank_cnn,
    )
    # Gene registry (compatible with legacy checkpoints missing these fields + ablation models with v_embed=None)
    v_reg = checkpoint.get("v_gene_registry", {})
    d_reg = checkpoint.get("d_gene_registry", {})
    j_reg = checkpoint.get("j_gene_registry", {})
    if encoder.v_embed is not None:
        encoder.v_embed._gene_registry = v_reg
        encoder.v_embed._family_map = checkpoint.get("v_family_map", {})
        encoder.v_embed._next_allele_idx = max(v_reg.values()) + 1 if v_reg else 0
    if encoder.d_embed is not None:
        encoder.d_embed._gene_registry = d_reg
        encoder.d_embed._family_map = checkpoint.get("d_family_map", {})
        encoder.d_embed._next_allele_idx = max(d_reg.values()) + 1 if d_reg else 0
    if encoder.j_embed is not None:
        encoder.j_embed._gene_registry = j_reg
        encoder.j_embed._family_map = checkpoint.get("j_family_map", {})
        encoder.j_embed._next_allele_idx = max(j_reg.values()) + 1 if j_reg else 0

    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    encoder = encoder.to(device_obj)

    # Rebuild BNN (use saved dimensions, compatible with legacy format)
    input_dim = checkpoint.get("input_dim", encoder.total_dim)
    bnn = BayesianJunctionNet(
        input_dim=input_dim,
        hidden_dim_1=checkpoint.get("hidden_dim_1", 512),
        hidden_dim_2=checkpoint.get("hidden_dim_2", 512),
        latent_dim=checkpoint.get("latent_dim", 256),
        max_junction_len=checkpoint.get("max_junction_len", max_junction_len),
    )
    # v6: Auto-match local_bias embedding size from checkpoint
    bnn_state = checkpoint["bnn_state_dict"]
    for key in ["n1_local_bias.weight", "n2_local_bias.weight"]:
        if key in bnn_state:
            saved_shape = bnn_state[key].shape
            cur_shape = getattr(bnn, key.replace('.weight','')).weight.shape
            if saved_shape != cur_shape:
                setattr(bnn, key.replace('.weight',''), nn.Embedding(saved_shape[0], saved_shape[1]).to(device_obj))
    bnn.load_state_dict(bnn_state, strict=False)
    bnn = bnn.to(device_obj)

    # Assemble generator (v3.1: use new constructor)
    generator = BayesAIRRGenerator(encoder=encoder, bnn=bnn, device=str(device_obj))
    generator.eval()

    return generator

