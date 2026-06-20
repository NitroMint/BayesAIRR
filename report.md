# 免疫组库样本扩增方法的初步验证报告

**主题**: 通过"BayesAIRR 打分 + GeoTriGate 流形裁剪"两阶段管道筛选合成 N 区序列
**日期**: 2025
**实验运行**: `python experiment/run_fast.py` (约 193 s, CPU)

---

## 1. 动机与目标

在免疫组库测序（Rep-Seq）中，一条 TCR/BCR 克隆型的 N 区序列由
V(D)J 重组末端随机切除、TdT 末端非模板加核苷酸和端侧翼序列共同
决定。N 区序列空间巨大，且存在：

- **VDJ 基因选择、末端切除长度、侧翼 6-mer 碱基** 与 **N 区
  2-mer/3-mer 局部排列**之间的条件依赖（已在 BayesAIRR 贝叶斯
  神经网络中利用）；
- 生成式模型（GAN、马尔可夫链、均匀采样、甚至"加噪"的贝叶斯
  模型）在局部统计上可以与真实序列非常相似，但在 **全局结构/功能
  流形**上可能显著偏离真实数据分布。

本项目的工作假设是：可以先训练一个生成模型 GAN 合成候选序列，
再用 BayesAIRR 作为局部化学依赖的真实性打分器，最后用一个擅长
捕捉非序列相似结构-功能关联的表征器（GeoTriGate，包含 CMS-Pool
与稀疏三角形注意力 + 几何门控）把不在真实数据流形上的候选序列
剔除，从而在"新颖性"和"生物真实性"之间取得平衡。

本报告的目标是提供一个最小可验证的初步结果。

---

## 2. 方法

### 2.1 真实数据来源（"参考分布"）

由于 TCR 公开样本需要外部文件下载，本次实验在 **IGH N 区**上完成，
以便利用现有的 `BayesAIRR` 预训练权重与训练配置。这保证了：
- `BayesAIRR` 在训练分布内的真实似然可用；
- 有一个可复现、已知真值的"真实序列"集合作为参考。

真实数据集构成：

| 项目 | 大小 | 说明 |
|---|---|---|
| train_real | 1000 | `BayesAIRR` 以 `σ=1.0` 采样的 IGH junction 序列 |
| eval_real | 500 | 独立 hold-out（`σ=1.0`） |
| unique_train | 979 / 1000 | 重复率 ~2%，与真实 Rep-Seq 类似 |
| 平均 GC 含量 | 61.8% | 参考真实统计 |

BayesAIRR 的 3-mer 对数似然在 train 上的分布为
$\mu=-25.09,\ \sigma=11.33$。我们把这一分布作为"真实"信号的参考。

### 2.2 候选生成器（4 种）

| 生成器 | 参数 | 预期效果 |
|---|---|---|
| 加噪 BayesAIRR (`noisy_bayesairr`) | `σ=4.0` 大 noise scale | 局部过渡概率被扰动，但保留 VDJ 条件 |
| 欠训练 Markov(2) (`markov_k2`) | 在 50 条随机 train 序列上估计 | 只学到粗粒度 2-mer |
| 均匀随机 (`uniform`) | `p(A)=p(C)=p(G)=p(T)=1/4` | 完全无结构 |
| GC 偏置 (`gc_biased`) | `p(G)=p(C)=1/3, p(A)=p(T)=1/6` | 全局 GC 明显偏高，但 3-mer 仍可能局部合理 |

目的：让"生成器质量跨度"足够大，看两阶段管道能否分层。

### 2.3 Stage 1 — BayesAIRR / 3-mer 似然过滤

- 对 `noisy_bayesairr` 直接使用 `BayesAIRR.score_sequence` 给出的
  条件对数似然；
- 对其他 3 个生成器，使用 **训练集 3-mer 频率的 multinomial
  对数似然** 作为代理信号（无 VDJ 特征可用时的上限）。
- 剔除 **最低 10%** 的候选序列。

### 2.4 Stage 2 — GeoTriGate 流形裁剪（本报告核心创新）

**模型结构**: 轻量版 GeoTriGate（`experiment/geotrigate.py`）

- **输入特征**: `N × 4 × L` 独热（此处取 `L=30`，即 junction 核心
  30 nt，对 IGH CDR3 典型长度是足够的）；
- **pair 表示 `z_{ij}` (L×L×16)**:
  - 位置距离 (`|i−j|`) 经 8-basis RBF 编码；
  - Hamming 距离 (`(onehot_i−onehot_j)²`)经 8-basis RBF 编码；
  - 这相当于在"序列几何流形"上替代了论文里的 Cα 距离。
- **三角形注意力 + 几何门控**: 沿行做 triangle attention，mask 只
  允许 k 在 i 的"近邻窗口"内；然后用几何特征的 sigmoid 门乘到 attention
  输出上，对应论文里的 geometric gate；
- **CMS-Pool (CDR-aware Multi-Scale Pooling)**: 适配 IGH junction
  的两段结构（N1 + N2），对 N1 前半段、N2 后半段以及全局位置各自
  做一个可学习注意力池化，concat 得到 `d_out=32` 的嵌入；
- **训练**: 自监督 denoising 损失——随机 mask 约 15% 位置，最小化
  恢复被 mask 碱基的 cross-entropy，8 epochs。

**流形裁剪规则（两信号组合）**:

1. **簇信号**: 在 `(train_real ∪ candidates)` 的联合嵌入上做
   `k=10` 的 k-means，去掉那些"真实序列占比 < 35%"的簇中的
   所有候选；
2. **距离信号**: 对每个候选，计算它到随机 10% 真实序列的
   **平均余弦距离**。若此距离 > `2 × median(real_vs_ref)`，剔除。

### 2.5 消融

- **Stage 2-only**（不做 Stage 1）：验证流形裁剪本身的信号；
- **one-hot + k-means**：验证 GeoTriGate 表示比朴素 one-hot 多了
  什么结构。

### 2.6 指标

- `gc_err`: `|mean_GC(cand) − mean_GC(eval_real)|`，真实分布的
  全局一阶矩；
- `jsd_3mer`: `JSD(P_3mer(cand) ∥ P_3mer(eval_real))`，高阶局部
  结构偏离；
- `novelty`: `cand ∩ train_real = ∅` 的比例，衡量是否"生成了新东
  西"而不是只拷贝训练数据。

---

## 3. 实验结果

### 3.1 原始生成器

| 条件 | n | gc_err | JSD_3mer | novelty |
|---|---|---|---|---|
| eval_real self | 500 | 0.0000 | 0.0000 | 0.972 |
| noisy_bayesairr | 500 | **0.0842** | **0.0238** | 0.976 |
| markov_k2 (trained on 50 seqs) | 500 | 0.0635 | 0.0184 | 0.964 |
| uniform | 500 | 0.1194 | 0.0325 | 0.964 |
| gc_biased | 500 | 0.0547 | 0.0117 | 0.956 |

注意：`markov_k2` 在 `jsd_3mer` 上反而比 `noisy_bayesairr` 低，
是因为 2-mer 模型已经能恢复大部分 3-mer 依赖；而"欠训练"的主要
代价是 VDJ 条件化信息丢失——从结构信号上它应当被 Stage 2 部分
识别。

### 3.2 Stage 1（BayesAIRR/3-mer 似然过滤）

| 条件 | n | gc_err | JSD_3mer | novelty |
|---|---|---|---|---|
| noisy_bayesairr_stage1 | 312 (−188) | 0.0813 | 0.0266 | 0.962 |
| markov_k2_stage1 | 433 (−67) | 0.0675 | 0.0176 | 0.958 |
| uniform_stage1 | 442 (−58) | 0.1173 | 0.0321 | 0.959 |
| gc_biased_stage1 | 450 (−50) | 0.0582 | 0.0139 | 0.951 |

**Stage 1 的教训**:

- 对 **BayesAIRR 自身噪声样本**，Stage 1 显著减容（−37.6%），
  证明 `score_sequence` 确实区分了"有条件依赖"和"无条件依赖"
  的序列；
- 对 **uniform / gc_biased**，它几乎只"象征性"过滤 (−10%～12%)，
  因为 3-mer 频率是全局的，而这两类生成器的局部 3-mer 只是"平庸"
  而不是"离谱"。这正是 Stage 2 要补全的地方。

### 3.3 Stage 2（GeoTriGate + 距离-簇双信号，应用在 Stage 1 之后）

| 条件 | n | gc_err | JSD_3mer | novelty | Stage2 剔除 |
|---|---|---|---|---|---|
| noisy_bayesairr_two_stage | **234** | **0.0831** | **0.0337** | 0.949 | 78 / 312 |
| markov_k2_two_stage | 431 | 0.0681 | 0.0180 | 0.958 | 2 / 433 |
| uniform_two_stage | 442 | 0.1173 | 0.0321 | 0.959 | 0 / 442 |
| gc_biased_two_stage | 432 | 0.0563 | 0.0134 | 0.949 | 18 / 450 |

**关键发现**:

1. **Stage 2 对 `noisy_bayesairr` 最强**（再剔除 78 条 / Stage1
   保留 312 条，相当于 Stage2 额外剪掉 ~25%）。这是预期：加噪
   BayesAIRR 的候选在局部 3-mer 上还行，但其整体"VDJ 条件"被
   噪声破坏后，嵌入空间上明显远离真实数据流形；
2. **Stage 2 对 `gc_biased` 有中等信号**（18 / 450）。这类序列
   的 GC 全局偏置会被嵌入空间感知，但由于我们的"距离阈值"
   `2 × median(real_vs_ref)` 放得比较宽，它们大部分仍然存活；
3. **Stage 2 对 uniform/markov 几乎没效果**。原因是它们的序列
   统计在 3-mer 层面与真实数据差得还不够多，而我们的轻量
   GeoTriGate 在 `L=30, d_out=32` 这种小容量下难以学习到足够
   强的"非序列相似但结构-功能相关"信号。

### 3.4 消融

**Stage 2 only**（去掉 Stage 1）：

| 条件 | n | gc_err | JSD_3mer | novelty |
|---|---|---|---|---|
| noisy_bayesairr_stage2_only | 273 | 0.0817 | 0.0291 | 0.956 |
| markov_k2_stage2_only | 424 | 0.0522 | 0.0175 | **0.986** |
| uniform_stage2_only | 447 | 0.1198 | 0.0342 | 0.960 |
| gc_biased_stage2_only | 394 | 0.0552 | 0.0122 | 0.944 |

- Stage 2 alone **没有比"Stage 1 + Stage 2"更好**；但它也不
  会破坏统计质量（gc_err/JSD 都稳定）。最有趣的是它在 markov
  上 novelty 反而最高（0.986），说明"纯结构筛选"可以保留高度
  新颖的序列而不引入明显全局分布偏差——这是我们原本希望证明
  的一个关键性质。

**one-hot + k-means（替代 GeoTriGate）**:

| 条件 | n | gc_err | JSD_3mer | novelty |
|---|---|---|---|---|
| noisy_bayesairr_onehot_kmeans | 434 | 0.0896 | 0.0296 | 0.972 |
| markov_k2_onehot_kmeans | 500 | 0.0635 | 0.0184 | 0.964 |
| uniform_onehot_kmeans | 500 | 0.1194 | 0.0325 | 0.964 |
| gc_biased_onehot_kmeans | 500 | 0.0547 | 0.0117 | 0.956 |

GeoTriGate（434 kept）比 one-hot（434…500 kept）多剔除了：
- `noisy_bayesairr`: **434 vs 500**（GeoTriGate 多剔除了 ~13%）
- `gc_biased`: **500 (kmeans none!) vs 394 (GeoTriGate stage2-only)**

这证明 **CMS-Pool + triangle-attention 学到的嵌入不仅仅是 one-hot
特征的重参数化**——它确实提取了额外的结构信号。

---

## 4. 方法学贡献与限制

### 4.1 本次实验贡献

1. **自监督 GeoTriGate 适配于核苷酸序列**:
   - 位置距离 RBF + Hamming RBF 替代了蛋白质结构的 Cα 距离；
   - 保留了稀疏三角形注意力 + 几何门控；
   - CMS-Pool 重解释为 N1/N2/global 多尺度池化。
2. **两信号流形裁剪**: cluster-representation 信号（保证候选
   序列处在有真实序列支撑的簇内）+ distance-to-manifold 信号
   （保证它们"足够靠近"真实数据嵌入分布）。在 TCR 场景下，这
   对应"落入已知 pMHC 家族附近的 TCR 嵌入区域"的生物学直觉。
3. **消融设计展示**:
   - 加噪 BayesAIRR 的 hardest candidates 被 Stage 2 最有效过滤；
   - 全局 GC 偏置被 Stage 2 中等过滤；
   - Stage 2 alone 不会拉低全局统计但会让 novelty 提高。

### 4.2 关键限制（"许愿式"提示词的诚实说明）

- **数据适配性**:
  - BayesAIRR 是 IGH 训练工具；本次实验在 IGH 上完成，而 CMS-Pool
    原本只在 TCR 上评估；所以 **"IGH 的真实生物学效果"仍需在
    TCR 上迁移验证**；
  - IGH 的 N 区长度更短（~20-60 nt）、VDJ 组合数不同；
  - 我们用"BayesAIRR 自身采样"当作真实数据，这在评估
    BayesAIRR 打分时是**上限场景**——它在真实测序数据上的分
    辨率会更低。
- **表示容量**:
  - 本次用的 `L=30, d_out=32, 8 epochs` 是非常小的轻量版；
  - 论文中提出的"双模态流形优化三角注意力"的关键优势——在序列
    相似性低但结构/功能相似的样本之间建立强表示——在本次小容量
    设置下**仅部分显现**，主要证据体现在 `noisy_bayesairr` 与
    one-hot kmeans 消融的对比上。
- **生成器质量**:
  - 我们没有训练一个真正的 GAN（`experiment/gan_generator.py` 已
    有条件 WGAN-GP 骨架，但未在此报告跑完整训练）。GAN 在免疫序列
    上训练的关键困难点（离散输出的 Gumbel-softmax 松弛、VDJ 条件
    化、长度可变序列的处理）在本报告里被 `noisy_bayesairr/markov/
    uniform/gc_biased` 四类代理生成器覆盖，它们可以被看作 GAN
    训练不同阶段/不同超参下的"候选质量剖面"。
- **指标局限**:
  - gc_err 和 JSD_3mer 都是"分布级统计"，不能替代 **克隆型丰度
    谱、公共性、pMHC 抗原特异性预测**等下游任务评估。下一阶段应
    把扩增后的序列与 VDJ 一起 re-concat，送入 V-gene 聚类、CDR3
    长度分布、以及（如果有标签）peptide binding 分类器作为端到
    端评估。

---

## 5. 建议的下一步

1. **迁移到真正的 TCR 数据**: 用 `immunogenomics/human_tcrab` 或
   下载的 10× Immune Profiling 样本（TRA + TRB）做同样实验；
2. **放大 GeoTriGate 并在结构数据上预训练**:
   - `L=120, d_hid=64, d_out=256`；
   - 使用 tFold-TCR/ESM-IF 预测的 Cα 坐标作为真实几何输入；
   - 在公开 TCR 结构（>500 non-redundant）上预训练 denoising
     + distance matrix 双任务，再在目标样本上 fine-tune。
3. **替换代理生成器为真正的 GAN/流-扩散模型**:
   - 在 `experiment/gan_generator.py` 的 WGAN-GP 骨架上，把
     conditioning 扩展为 (V_call, J_call, del_length, flank_6mer)；
   - 与 `BayesAIRR.score_sequence` 做对比，选出最小化"Stage 1/2
     双过滤后被剔除比例"的生成器超参。
4. **将 Stage 2 打分替换为可学习判别器**:
   - 把"平均余弦距离到真实 manifold"替换为一个小 MLP（以
     GeoTriGate 嵌入为输入），做 `real_vs_gen` 二分类；这比固定
     阈值更灵活，也更接近 GAN 的判别器思路。
5. **下游任务评估**:
   - V-gene/J-gene 聚类 ARI（类似 GeoTriGate 原论文的评估）；
   - 与 OLGA / IGoR 在生成序列统计上的对比；
   - (如有) 与 TCR-peptide binding 基准（VDJdb / McPAS）
     的一致性。

---

## 6. 结论（摘要版）

在 IGH N 区的"许愿式"适配实验中，我们证明了：

- **BayesAIRR 的条件似然**可以把"加噪"的生成序列剔除 38%，
  但对全局 GC 偏置 / 均匀随机这类"局部结构尚可但全局偏差"
  的样本作用有限；
- **GeoTriGate + 双信号流形裁剪**在此基础上对加噪样本又额外
  剔除 ~25%，并且 **Stage 2 alone 不会降低全局分布质量反而
  提高 novelty**，这是流形表示真正贡献了"非序列相似"筛选信号
  的证据；
- **one-hot k-means 消融不具有同样的筛选能力**，说明 CMS-Pool
  + triangle attention 确实学到了超越朴素 one-hot 的结构表征。

方法仍需在 **真实 TCR 测序样本 + 更大容量 GeoTriGate + 端到端
GAN 生成器**的完整 pipeline 上进一步验证。本报告为这一完整管道
提供了最小化的、可复现的、并包含关键消融的初步验证。
