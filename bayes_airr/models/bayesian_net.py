# -*- coding: utf-8 -*-
"""
深度概率编程网络 — 基于 PyTorch 的贝叶斯连接区生成模型 (v2.0 升级架构)

升级要点:
1. GeneEmbedding: 层级化 Family + Allele 两级嵌入，罕见基因安全回退
2. FlankPhysChemEncoder: 6通道物理化学 CNN，侧翼相似性自动泛化
3. BayesianJunctionNet: 3层扩张 MLP + N1/N2 独立双头，~800K 参数
4. BNN KL 正则化: 罕见组合自动回退到无信息先验，而非过拟合

设计哲学 (免疫组库高多样性):
- Embedding/CNN 是确定性的，不参与贝叶斯采样(降低训练难度)
- 不确定性集中在 MLP 层——这是最优位置
- sigma_thresh 通过 MLP 的权重扰动控制全链熵
- 罕见组合: BNN 输出天然高方差 → "我不知道"比"我猜错了"安全

数学:
    W_layer = mu + softplus(rho) * sigma_thresh * epsilon, epsilon ~ N(0,I)
    KL(q(W)||p(W)) 正则化自动将罕见组合拉回先验分布
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
# 贝叶斯变分线性层 (保持核心逻辑不变)
# ═══════════════════════════════════════════════════════════════


class BayesianLinear(nn.Module):
    """
    贝叶斯变分线性层 — 每个权重是独立高斯分布 N(mu, softplus(rho)^2)。

    重参数化:
        sigma = softplus(rho) * sigma_scale
        W     = mu + sigma * epsilon,   epsilon ~ N(0, I)
        b     = mu_b + softplus(rho_b) * epsilon_b

    先验: p(W) = N(0, prior_std^2)
    后验: q(W) = N(mu, softplus(rho)^2)

    sigma_scale 是应力放大系数，等于用户配置的 sigma_thresh。
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
        """Kaiming 初始化 mu，rho 初始化为小值使初始 std ≈ 0.1"""
        nn.init.kaiming_uniform_(self.mu_w, a=math.sqrt(5))
        nn.init.constant_(self.rho_w, rho_init_mean)

        if self.mu_b is not None:
            fan_in = self.in_features
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(self.mu_b, -bound, bound)
            nn.init.constant_(self.rho_b, rho_init_mean)

    def forward(self, x: torch.Tensor, sigma_scale: float = 1.0) -> torch.Tensor:
        """采样权重 → 线性变换"""
        sigma_w = F.softplus(self.rho_w) * sigma_scale
        weight = self.mu_w + sigma_w * torch.randn_like(sigma_w)

        if self.mu_b is not None:
            sigma_b = F.softplus(self.rho_b) * sigma_scale
            bias = self.mu_b + sigma_b * torch.randn_like(sigma_b)
        else:
            bias = None

        return F.linear(x, weight, bias)

    def kl_divergence(self) -> torch.Tensor:
        """KL(q||p) 闭式解"""
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
# 层级化基因嵌入 (Family + Allele 两级结构)
# ═══════════════════════════════════════════════════════════════


class GeneEmbedding(nn.Module):
    """
    层级化基因嵌入: Family(共享强度) + Allele(独有偏移)

    设计动机:
        IGHV3-23 有 30万条训练数据 → embedding 学得很好
        IGHV7-4  只有 500条       → embedding 几乎随机

    解决方案:
        罕见基因: family_embed(IGHV7) + allele_offset(≈0) → 自动回退到家族表示
        常见基因: family_embed(IGHV3) + allele_offset(独有) → 精确定位

    数学:
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

        # 基因名 → (family_idx, allele_idx) 映射表
        self._gene_registry: Dict[str, int] = {}
        self._family_map: Dict[str, int] = {}
        self._next_allele_idx = 0
        self._next_family_idx = 0

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.family_embed.weight, mean=0.0, std=0.05)
        # allele embedding 初始化为更小的值，让家族先验主导
        nn.init.normal_(self.allele_embed.weight, mean=0.0, std=0.01)

    def register_gene(self, gene_name: str) -> int:
        """注册基因名，返回 allele 索引"""
        if gene_name not in self._gene_registry:
            # 提取家族 (如 "IGHV3-23*01" → "IGHV3")
            family = self._extract_family(gene_name)
            if family not in self._family_map:
                self._family_map[family] = self._next_family_idx
                self._next_family_idx += 1
            self._gene_registry[gene_name] = self._next_allele_idx
            self._next_allele_idx += 1
        return self._gene_registry[gene_name]

    def _extract_family(self, gene_name: str) -> str:
        """从 IMGT 基因名提取家族: IGHV3-23*01 → IGHV3"""
        # 去掉等位基因后缀
        name = gene_name.split("*")[0] if "*" in gene_name else gene_name
        # IGHV3-23 → IGHV3, IGHD1-1 → IGHD1
        import re
        match = re.match(r"(IGH[VDJ]\d+)", name)
        if match:
            return match.group(1)
        return name[:5]  # 降级截断

    def get_family_index(self, gene_name: str) -> int:
        """获取家族索引"""
        family = self._extract_family(gene_name)
        if family not in self._family_map:
            self.register_gene(gene_name)
        return self._family_map.get(family, 0)

    def forward(self, gene_names: List[str]) -> torch.Tensor:
        """
        前向: 基因名列表 → (batch, family_dim + allele_dim)

        Args:
            gene_names: 基因名列表

        Returns:
            (batch_size, family_dim + allele_dim) 嵌入向量
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
# 物理化学侧翼编码器 (6通道 CNN)
# ═══════════════════════════════════════════════════════════════


class FlankPhysChemEncoder(nn.Module):
    """
    侧翼序列的物理化学 CNN 编码器。

    输入: 原始侧翼核苷酸字符串 (如 "TGC")
    输出: (batch, 64) 固定维度特征向量

    6 个通道 (替代简单的 one-hot):
        通道 1: A 位
        通道 2: C 位
        通道 3: G 位
        通道 4: T 位
        通道 5: GC 含量 (G|C = 1, A|T = 0)
        通道 6: 嘌呤/嘧啶 (A|G = 1, C|T = 0)

    设计动机:
        "CAG" 和 "CAA" 在物理化学上很接近(都是嘧啶-嘌呤-嘌呤)，
        CNN 通过局部感受野自动捕获这种相似性，
        罕见侧翼组合通过相似性泛化而非过拟合。
    """

    def __init__(self, max_flank_len: int = 15, out_dim: int = 64) -> None:
        super().__init__()
        self.max_flank_len = max_flank_len
        self.out_dim = out_dim

        # 小卷积网络: 6通道 → 32 → 64 → GlobalAvgPool
        self.conv = nn.Sequential(
            nn.Conv1d(6, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        # 投影到输出维度
        self.proj = nn.Linear(64, out_dim)

    def _seq_to_physchem(self, seq: str) -> torch.Tensor:
        """核苷酸序列 → (6, L) 物理化学张量"""
        chars = list(seq.upper())
        L = min(len(chars), self.max_flank_len)
        tensor = torch.zeros(6, self.max_flank_len, dtype=torch.float32)

        for i, c in enumerate(chars[: self.max_flank_len]):
            if c == "A":
                tensor[0, i] = 1.0
                tensor[4, i] = 0.0  # 非GC
                tensor[5, i] = 1.0  # 嘌呤
            elif c == "C":
                tensor[1, i] = 1.0
                tensor[4, i] = 1.0  # GC
                tensor[5, i] = 0.0  # 嘧啶
            elif c == "G":
                tensor[2, i] = 1.0
                tensor[4, i] = 1.0  # GC
                tensor[5, i] = 1.0  # 嘌呤
            elif c == "T":
                tensor[3, i] = 1.0
                tensor[4, i] = 0.0  # 非GC
                tensor[5, i] = 0.0  # 嘧啶
            else:  # N 或其他
                tensor[:, i] = 0.25  # 均匀不确定

        return tensor

    def forward(self, flank_seqs: List[str]) -> torch.Tensor:
        """
        Args:
            flank_seqs: 侧翼序列字符串列表

        Returns:
            (batch_size, out_dim) 物理化学特征向量
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
# 完整特征编码器 (整合 GeneEmbedding + FlankPhysChemEncoder)
# ═══════════════════════════════════════════════════════════════


class JunctionFeatureEncoder(nn.Module):
    """
    将 (基因名, 侧翼序列, 删除长度) 编码为统一的贝叶斯网络输入。

    架构:
        V 基因 → GeneEmbedding (family+allele) → 64d
        D 基因 → GeneEmbedding (family+allele) → 64d
        J 基因 → GeneEmbedding (family+allele) → 64d
        V侧翼  → FlankPhysChemEncoder → 64d
        D5侧翼 → FlankPhysChemEncoder → 64d
        D3侧翼 → FlankPhysChemEncoder → 64d
        J侧翼  → FlankPhysChemEncoder → 64d
        删除长度 → Linear(4→16) + ReLU
        ─────────────────────────────────
        拼接: 64*3 + 64*4 + 16 = 464 维
        → LayerNorm → Dropout → 输出

    v3.1 新增消融开关:
        use_gene_embedding=False → 基因嵌入替换为简单零向量
        use_flank_cnn=False       → 侧翼特征替换为零向量
    """

    def __init__(
        self,
        # ── 基因级别参数 (旧 API: 按类型数量分配) ──
        v_families: int = 7,
        v_alleles: int = 280,   # v5: 增大默认值覆盖所有IMGT等位基因
        d_families: int = 8,
        d_alleles: int = 50,
        j_families: int = 6,
        j_alleles: int = 15,
        gene_dim: int = 32,        # family + allele 各 32 = 64 维
        # ── 侧翼参数 ──
        flank_out_dim: int = 64,
        flank_len: int = 15,       # FlankPhysChemEncoder 最大侧翼长度
        dropout_p: float = 0.1,
        # ── 消融开关 (v3.1) ──
        use_gene_embedding: bool = True,
        use_flank_cnn: bool = True,
        # ── 新 API: 直接传基因列表自动推断参数 ──
        v_genes: list | None = None,
        d_genes: list | None = None,
        j_genes: list | None = None,
    ) -> None:
        super().__init__()

        # 支持新 API: 传基因列表自动推断 family/allele 数量
        if v_genes is not None:
            v_families, v_alleles = self._infer_gene_counts(v_genes)
        if d_genes is not None:
            d_families, d_alleles = self._infer_gene_counts(d_genes)
        if j_genes is not None:
            j_families, j_alleles = self._infer_gene_counts(j_genes)

        self._use_gene_embedding = use_gene_embedding
        self._use_flank_cnn = use_flank_cnn

        # 三个独立的基因嵌入器
        self.gene_out_dim = gene_dim * 2  # family + allele
        if use_gene_embedding:
            self.v_embed = GeneEmbedding(v_families, v_alleles, gene_dim, gene_dim)
            self.d_embed = GeneEmbedding(d_families, d_alleles, gene_dim, gene_dim)
            self.j_embed = GeneEmbedding(j_families, j_alleles, gene_dim, gene_dim)
        else:
            # 占位: 嵌入器为 None，forward 时用零向量
            self.v_embed = None
            self.d_embed = None
            self.j_embed = None

        # 一个共享的侧翼编码器 (核苷酸化学是通用的)
        if use_flank_cnn:
            self.flank_encoder = FlankPhysChemEncoder(max_flank_len=flank_len, out_dim=flank_out_dim)
        else:
            self.flank_encoder = None

        self.flank_out_dim = flank_out_dim

        # 删除长度编码
        self.del_encoder = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(inplace=True),
        )

        total_dim = (
            self.gene_out_dim * 3   # V, D, J 基因
            + flank_out_dim * 4     # V, D5, D3, J 侧翼
            + 16                    # 删除长度
        )

        self.total_dim = total_dim
        self.norm = nn.LayerNorm(total_dim)
        self.dropout = nn.Dropout(dropout_p)

    @property
    def output_dim(self) -> int:
        """便捷属性: 与 total_dim 同义，供消融实验脚本使用"""
        return self.total_dim

    @staticmethod
    def _infer_gene_counts(gene_list: list) -> tuple:
        """从基因列表推断 family 和 allele 数量"""
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
        """编码 V/D/J 基因 + 侧翼 + 删除 → (batch, total_dim)"""
        bs = len(v_genes)
        device = self.del_encoder[0].weight.device

        # ── 基因嵌入 (支持消融: use_gene_embedding=False → 零向量) ──
        if self._use_gene_embedding and self.v_embed is not None:
            v_feat = self.v_embed(v_genes)
            d_feat = self.d_embed(d_genes)
            j_feat = self.j_embed(j_genes)
        else:
            v_feat = torch.zeros(bs, self.gene_out_dim, device=device)
            d_feat = torch.zeros(bs, self.gene_out_dim, device=device)
            j_feat = torch.zeros(bs, self.gene_out_dim, device=device)

        # ── 侧翼编码 (支持消融: use_flank_cnn=False → 零向量) ──
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

        # ── 删除长度编码 ──
        del_max = deletions.float().max(dim=0).values
        del_max = torch.clamp(del_max, min=1.0)
        del_norm = deletions.float() / del_max.unsqueeze(0)
        del_feat = self.del_encoder(del_norm)

        # ── 拼接 + 归一化 ──
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
# 升级版贝叶斯连接区生成网络
# ═══════════════════════════════════════════════════════════════


class BayesianJunctionNet(nn.Module):
    """
    贝叶斯连接区生成网络

    架构升级:
        - 3 层扩张 MLP (512 → 512 → 256) 替代 2 层收缩结构
        - N1 和 N2 独立双头 (各自 Bayesian 输出层)
        - LayerNorm + GELU 替代简单 ReLU, 提高训练稳定性
        - Residual 连接防止深度退化

    sigma_thresh 直接放大所有权重的采样噪声。

    参数量: ~850K (800K+ MLP, 50K 编码器)
    显存: ~400MB (训练), ~50MB (推理)
    预期效果:
        σ=1.0: N区分布与 OAS 真实数据 KL < 0.03 bits/base
        σ=1.5: 罕见插入涌现, 压测第三方工具
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
        # ── 别名 (兼容消融脚本) ──
        hidden_dim: int | None = None,
        max_len: int | None = None,
    ) -> None:
        # 别名解析
        if hidden_dim is not None:
            hidden_dim_1 = hidden_dim_2 = hidden_dim
        if max_len is not None:
            max_junction_len = max_len

        super().__init__()
        self.input_dim = input_dim
        self.max_junction_len = max_junction_len
        self.num_nucleotides = len(self.NUCLEOTIDES)

        # ── 贝叶斯 MLP 骨架 (3层扩张, 带残差结构) ──
        self.blinear1 = BayesianLinear(input_dim, hidden_dim_1, prior_std=prior_std)
        self.blinear2 = BayesianLinear(hidden_dim_1, hidden_dim_2, prior_std=prior_std)
        self.blinear3 = BayesianLinear(hidden_dim_2, latent_dim, prior_std=prior_std)

        # LayerNorm 用于稳定训练
        self.norm1 = nn.LayerNorm(hidden_dim_1)
        self.norm2 = nn.LayerNorm(hidden_dim_2)
        self.norm3 = nn.LayerNorm(latent_dim)

        # ── N1 独立头 (V-D 连接区) ──
        self.n1_length_head = BayesianLinear(
            latent_dim, max_junction_len + 1, prior_std=prior_std
        )
        self.n1_seq_head = BayesianLinear(
            latent_dim, max_junction_len * self.num_nucleotides, prior_std=prior_std
        )

        # ── N2 独立头 (D-J 连接区) ──
        self.n2_length_head = BayesianLinear(
            latent_dim, max_junction_len + 1, prior_std=prior_std
        )
        self.n2_seq_head = BayesianLinear(
            latent_dim, max_junction_len * self.num_nucleotides, prior_std=prior_std
        )

        self.dropout = nn.Dropout(dropout_p)

        # ── v5: 局部碱基上下文嵌入 (3阶 Markov 矫正) ──
        self.n1_local_bias = nn.Embedding(64, self.num_nucleotides)  # 4³=64 contexts
        self.n2_local_bias = nn.Embedding(64, self.num_nucleotides)
        nn.init.zeros_(self.n1_local_bias.weight)
        nn.init.zeros_(self.n2_local_bias.weight)
        self._use_local_context = True

        # 收集所有贝叶斯层以便统一管理
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
        前向传播: 特征 → 潜在表示 → N1/N2 预测。

        Args:
            features: (batch, input_dim)
            sigma_thresh: 应力放大系数
            n1_local_idx: (batch, max_len) 前2碱基的16-class索引, None=不用局部bias
            n2_local_idx: 同上用于N2

        Returns:
            n1_length_logits: (batch, max_junction_len + 1)
            n1_seq_logits:    (batch, max_junction_len, 4)
            n2_length_logits: (batch, max_junction_len + 1)
            n2_seq_logits:    (batch, max_junction_len, 4)
            kl_total:         标量
        """
        sigma = sigma_thresh

        # 骨架前向 (与之前相同)
        h = F.gelu(self.norm1(self.blinear1(features, sigma_scale=sigma)))
        h = self.dropout(h)
        h = F.gelu(self.norm2(self.blinear2(h, sigma_scale=sigma)))
        h = self.dropout(h)
        h = F.gelu(self.norm3(self.blinear3(h, sigma_scale=sigma)))
        h = self.dropout(h)

        # N1 双头
        n1_len_logits = self.n1_length_head(h, sigma_scale=sigma)
        n1_seq_flat = self.n1_seq_head(h, sigma_scale=sigma)
        n1_seq_logits = n1_seq_flat.view(-1, self.max_junction_len, self.num_nucleotides)

        # N2 双头
        n2_len_logits = self.n2_length_head(h, sigma_scale=sigma)
        n2_seq_flat = self.n2_seq_head(h, sigma_scale=sigma)
        n2_seq_logits = n2_seq_flat.view(-1, self.max_junction_len, self.num_nucleotides)

        # v5: 局部碱基 context bias — 前2个碱基影响当前位置预测
        if n1_local_idx is not None and self._use_local_context:
            # (batch, max_len, 4) + (batch, max_len, 4) broadcast
            local_bias = self.n1_local_bias(n1_local_idx)  # (batch, max_len, 4)
            n1_seq_logits = n1_seq_logits + local_bias * 0.1  # 小权重, 微调
        if n2_local_idx is not None and self._use_local_context:
            local_bias = self.n2_local_bias(n2_local_idx)
            n2_seq_logits = n2_seq_logits + local_bias * 0.1

        # 温度缩放
        sigma_safe = max(sigma, 0.1)
        n1_len_logits = n1_len_logits / sigma_safe
        n1_seq_logits = n1_seq_logits / sigma_safe
        n2_len_logits = n2_len_logits / sigma_safe
        n2_seq_logits = n2_seq_logits / sigma_safe

        # KL 总和
        kl_total = sum(layer.kl_divergence() for layer in self._all_bayesian_layers)

        return n1_len_logits, n1_seq_logits, n2_len_logits, n2_seq_logits, kl_total

    def sample_single_junction(
        self,
        length_logits: torch.Tensor,
        seq_logits: torch.Tensor,
    ) -> Tuple[List[str], List[int], torch.Tensor]:
        """
        从 length_logits + seq_logits 采样单个连接区 (N1 或 N2)。

        Args:
            length_logits: (batch, max_junction_len + 1)
            seq_logits:    (batch, max_junction_len, 4)

        Returns:
            junction_seqs, junction_lens, log_probs (batch,)
        """
        batch_size = length_logits.size(0)

        # 长度采样
        len_dist = torch.distributions.Categorical(logits=length_logits)
        lengths = len_dist.sample()
        len_log_probs = len_dist.log_prob(lengths)

        # 逐位置碱基采样
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
        同时采样 N1 和 N2 连接区。

        Args:
            features: (batch, input_dim) 编码特征
            sigma_thresh: 应力放大系数

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
        """所有贝叶斯层的 KL 总和"""
        return sum(layer.kl_divergence() for layer in self._all_bayesian_layers)

    @torch.no_grad()
    def score_sequence(
        self,
        features: torch.Tensor,
        n1_seqs: List[str],
        n2_seqs: List[str],
        sigma_thresh: float = 1.0,
    ) -> List[float]:
        """对给定 N1/N2 序列打分，使用局部碱基 context 增强。"""
        batch_size = features.size(0)
        nuc_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3}

        # 构建每位置的 local context index (前2个碱基 → 0-15)
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
# 便捷接口: 编码+生成一步完成
# ═══════════════════════════════════════════════════════════════


class BayesAIRRGenerator:
    """
    一站式生成器: 整合编码器 + BNN, 输入原始生物学信息直接输出 N1/N2。

    用法:
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
        一步生成: 原始信息 → 编码 → BNN采样 → N1/N2 序列。

        Args:
            v_genes, d_genes, j_genes: 基因名列表
            v_flanks, d5_flanks, d3_flanks, j_flanks: 侧翼序列
            deletions: (batch, 4) [v3_del, d5_del, d3_del, j5_del]
            sigma_thresh: 应力放大系数
            seeds: 可选 per-read 随机种子；提供时逐条绑定 PyTorch RNG（与 SHM 一致）

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
        """对真实 N1/N2 序列打分：返回每条序列的 log P(N1,N2 | V,D,J,del,flank)。

        用于 NLL 评估：与 Markov 基线的 compute_nll(true_seqs) 同口径对比。
        """
        features = self.encoder(
            v_genes=v_genes, d_genes=d_genes, j_genes=j_genes,
            v_flanks=v_flanks, d5_flanks=d5_flanks,
            d3_flanks=d3_flanks, j_flanks=j_flanks,
            deletions=deletions.to(self.device),
        )
        return self.bnn.score_sequence(features, n1_seqs, n2_seqs, sigma_thresh)


# ═══════════════════════════════════════════════════════════════
# 保存/加载工具
# ═══════════════════════════════════════════════════════════════


def save_checkpoint(
    generator_or_encoder: BayesAIRRGenerator | JunctionFeatureEncoder,
    file_or_bnn: str | Path | BayesianJunctionNet,
    hp_or_file_path: "HyperParams | str | Path | None" = None,
    file_path_legacy: str | Path | None = None,
) -> None:
    """保存完整检查点 (模型+编码器+基因注册表)

    支持两种调用方式（向后兼容消融实验脚本）：
      - 新 API: save_checkpoint(generator, file_path)
      - 旧 API: save_checkpoint(encoder, bnn, hp, file_path)
    """
    # ── 判断调用方式 ──
    from bayes_airr.models.trainer import HyperParams as _HP
    if isinstance(generator_or_encoder, BayesAIRRGenerator):
        # 新 API: save_checkpoint(generator, file_path)
        generator = generator_or_encoder
        file_path = Path(file_or_bnn)  # type: ignore
    else:
        # 旧 API: save_checkpoint(encoder, bnn, hp, file_path)
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

    # 基因注册表 (支持 use_gene_embedding=False 的情况)
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
    """从检查点加载完整生成器"""
    file_path = Path(file_path)
    checkpoint = torch.load(str(file_path), map_location="cpu", weights_only=False)

    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")

    # 重建编码器 (使用保存的维度信息，旧格式则从 state_dict 反推)
    def _get_or_infer(key: str, state_key: str, default: int) -> int:
        if key in checkpoint:
            return checkpoint[key]
        # 从 state_dict 权重形状反推
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

    # 自动检测消融状态：检查 state_dict 中是否缺少基因嵌入/侧翼 CNN 权重
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
    # 基因注册表 (兼容旧 checkpoint 不含这些字段 + 消融模型 v_embed=None 的情况)
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

    # 重建 BNN (使用保存的维度，兼容旧格式)
    input_dim = checkpoint.get("input_dim", encoder.total_dim)
    bnn = BayesianJunctionNet(
        input_dim=input_dim,
        hidden_dim_1=checkpoint.get("hidden_dim_1", 512),
        hidden_dim_2=checkpoint.get("hidden_dim_2", 512),
        latent_dim=checkpoint.get("latent_dim", 256),
        max_junction_len=checkpoint.get("max_junction_len", max_junction_len),
    )
    # v6: 自动匹配 checkpoint 中 local_bias embedding 的大小
    bnn_state = checkpoint["bnn_state_dict"]
    for key in ["n1_local_bias.weight", "n2_local_bias.weight"]:
        if key in bnn_state:
            saved_shape = bnn_state[key].shape
            cur_shape = getattr(bnn, key.replace('.weight','')).weight.shape
            if saved_shape != cur_shape:
                setattr(bnn, key.replace('.weight',''), nn.Embedding(saved_shape[0], saved_shape[1]).to(device_obj))
    bnn.load_state_dict(bnn_state, strict=False)
    bnn = bnn.to(device_obj)

    # 组装生成器 (v3.1: 使用新构造函数)
    generator = BayesAIRRGenerator(encoder=encoder, bnn=bnn, device=str(device_obj))
    generator.eval()

    return generator

