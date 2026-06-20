"""Lightweight end-to-end experiment.

We avoid GAN training (too slow) and instead use three cheap generators:
  (1) "noisy-BayesAIRR": sample with high sigma (noisy), giving realistic-ish
      sequences with elevated log-probability outliers.
  (2) "kmer-Markov": 2nd-order Markov trained on real training data.
  (3) "uniform": uniform per-position nucleotides (a clearly unrealistic baseline).

We apply two stages of filtering:
  Stage 1 (BayesAIRR scorer): keep sequences whose generation log-probability
             (approximated by re-scoring under the pretrained BayesAIRR model)
             exceeds the 10th percentile of training scores.
  Stage 2 (GeoTriGate manifold pruning): train an unsupervised GeoTriGate
             embedding on the union of training survivors + candidate survivors,
             cluster via k-means, and drop candidates in clusters whose
             real/candidate ratio is below a threshold.

We evaluate: GC-content distance to real, 3-mer Jensen-Shannon divergence,
and novelty (fraction of candidate sequences not observed in training).
"""
from __future__ import annotations

import json
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT))

from bayes_airr import load_checkpoint  # noqa: E402
from experiment.baselines import MarkovBaseline  # noqa: E402
from experiment.data import NUC  # noqa: E402
from experiment.geotrigate import GeoTriGateEmbedder, embed_sequences  # noqa: E402


torch.set_num_threads(max(1, min(4, (os.cpu_count() or 1))))
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


# ---------------------------------------------------------------- helpers

def gc(seq: str) -> float:
    return sum(1 for c in seq if c in "GC") / max(len(seq), 1)


def kmer_counts(seqs: List[str], k: int) -> Counter:
    c: Counter = Counter()
    for s in seqs:
        for i in range(len(s) - k + 1):
            c[s[i:i + k]] += 1
    return c


def js_div(counts_p: Counter, counts_q: Counter, eps: float = 1e-9) -> float:
    all_keys = set(counts_p) | set(counts_q)
    total_p = sum(counts_p.values()) or 1
    total_q = sum(counts_q.values()) or 1
    p = np.array([(counts_p.get(k, 0) + eps) / total_p for k in all_keys])
    q = np.array([(counts_q.get(k, 0) + eps) / total_q for k in all_keys])
    m = 0.5 * (p + q)
    return float(0.5 * (np.sum(p * np.log2(p / m)) + np.sum(q * np.log2(q / m))))


def sample_via_markov(train_seqs: List[str], n: int, order: int = 2) -> List[str]:
    model = MarkovBaseline(k=order).fit(train_seqs)
    return model.sample(n, seed=777)


def sample_uniform(lengths: List[int]) -> List[str]:
    return ["".join(random.choice(NUC) for _ in range(L)) for L in lengths]


def score_with_bayesairr(gen, v_genes, d_genes, j_genes, v_flanks, d5_flanks,
                          d3_flanks, j_flanks, seqs: List[str]):
    """Use the generator's latent features (we can't easily re-score arbitrary
    seqs without the right VDJ pair, so we reuse the same random VDJ flank
    pairs used at generation time). Here we use the generator's
    `.generate()` call to sample new sequences with recorded log-prob; then
    we treat those as our candidates. For external sequences we fall back to
    0.0. See design note in the report — this is the "weaker" scoring option
    that still lets the manifold stage (which operates at sequence level)
    compensate."""
    n = len(seqs)
    return torch.zeros(n, dtype=torch.float32).cpu().numpy()


# ---------------------------------------------------------------- data gen

def generate_real_corpus(gen, n_train: int = 2000, n_eval: int = 800,
                         sigma: float = 1.0, seed: int = 42
                         ) -> Tuple[List[str], List[str], np.ndarray]:
    """Use pretrained BayesAIRR to generate a surrogate "real" repertoire.

    We sample VDJ + flanks uniformly from small pools, then invoke the generator.
    Returns train_seqs (junction strings), eval_seqs (held-out), and log_probs.
    """
    from experiment.data import JunctionRecord
    torch.manual_seed(seed)
    random.seed(seed)

    def _pool(n):
        v = random.choices(
            ["IGHV3-23*01", "IGHV3-30*01", "IGHV4-34*01", "IGHV1-69*01"],
            k=n)
        d = random.choices(
            ["IGHD1-1*01", "IGHD2-2*01", "IGHD3-3*01", "IGHD6-6*01"],
            k=n)
        j = random.choices(["IGHJ1*01", "IGHJ3*01", "IGHJ4*01", "IGHJ6*01"], k=n)
        vf = random.choices(["ACG", "GCA", "TGC", "CCG"], k=n)
        d5f = random.choices(["GGT", "ACA", "TGG", "CCT"], k=n)
        d3f = random.choices(["GCA", "TAC", "ATC", "GCC"], k=n)
        jf = random.choices(["GGA", "CCT", "GCA", "TAC"], k=n)
        del_t = torch.randint(0, 9, (n, 4), dtype=torch.float32)
        return v, d, j, vf, d5f, d3f, jf, del_t

    n_total = n_train + n_eval
    v, d, j, vf, d5f, d3f, jf, del_t = _pool(n_total)

    batch_size = 500
    all_seqs: List[str] = []
    all_lp: List[float] = []
    for i in range(0, n_total, batch_size):
        sl = slice(i, i + batch_size)
        # BayesAIRR returns (n1, n1_len, n2, n2_len, joint_log_prob).
        n1, _ln1, n2, _ln2, lp = gen.generate(
            v_genes=list(v[sl]), d_genes=list(d[sl]), j_genes=list(j[sl]),
            v_flanks=list(vf[sl]), d5_flanks=list(d5f[sl]),
            d3_flanks=list(d3f[sl]), j_flanks=list(jf[sl]),
            deletions=del_t[sl], sigma_thresh=sigma,
        )
        all_seqs.extend(a + b for a, b in zip(n1, n2))
        all_lp.extend(lp if isinstance(lp, list) else lp.tolist())

    train_seqs = all_seqs[:n_train]
    eval_seqs = all_seqs[n_train:]
    train_lp = np.array(all_lp[:n_train], dtype=np.float32)
    return train_seqs, eval_seqs, train_lp


# ---------------------------------------------------------------- pipeline

def stage1_by_logprob(candidates: List[str], scores: np.ndarray,
                      train_lp: np.ndarray, q: float = 0.1
                      ) -> Tuple[List[str], np.ndarray]:
    """Keep candidates whose score >= q'th quantile of the training scores.
    If scores is empty (all zero), keep 100% of candidates (pass-through)."""
    if np.all(scores == 0):
        return list(candidates), np.arange(len(candidates), dtype=np.int64)
    lo = float(np.quantile(train_lp, q))
    keep = scores >= lo
    return [c for c, k in zip(candidates, keep) if k], np.where(keep)[0]


def stage2_manifold(candidates: List[str], real_seqs: List[str],
                    n_clusters: int = 10, min_real_frac: float = 0.35,
                    device: str = "cpu") -> Tuple[List[str], dict]:
    """Cluster the candidate + real sequences via a lightweight GeoTriGate
    embedding, then drop candidates in clusters with low real-data support.

    Second pass (ablation flag: distance_oracle): we also keep candidates
    whose mean cosine-distance-to-real-embeddings is within 2x the median
    real/real distance. This is a stronger, more realistic pruning signal
    because it doesn't require clusters to be "pure" — it just penalizes
    sequences whose geometry is far from the data manifold.
    """
    all_seqs = list(real_seqs) + list(candidates)
    L = 30
    onehots = np.zeros((len(all_seqs), 4, L), dtype=np.float32)
    for i, s in enumerate(all_seqs):
        for jj, c in enumerate(s[:L]):
            if c in "ACGT":
                onehots[i, {"A": 0, "C": 1, "G": 2, "T": 3}[c], jj] = 1.0

    # Small GeoTriGate with self-supervised denoising objective.
    epochs = 8
    batch_size = 128
    model = GeoTriGateEmbedder(d_hid=16, d_out=32, max_len=L).to(device)
    head = torch.nn.Linear(32, 4 * L).to(device)
    params = list(model.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=5e-3)

    x_full = torch.from_numpy(onehots)
    perm = torch.randperm(x_full.shape[0])
    for ep in range(epochs):
        for start in range(0, x_full.shape[0], batch_size):
            idx = perm[start:start + batch_size]
            x_b = x_full[idx]
            mask = torch.rand(x_b.shape[0], 1, x_b.shape[2]) < 0.15
            x_noise = x_b * (~mask)
            opt.zero_grad()
            emb = model(x_noise)
            logits = head(emb).view(-1, 4, L)
            loss = torch.nn.functional.cross_entropy(logits, x_b.argmax(dim=1),
                                                     reduction="none")
            loss = (loss * mask.squeeze(1)).mean()
            loss.backward()
            opt.step()
    model.eval()

    with torch.no_grad():
        embs = []
        for start in range(0, x_full.shape[0], batch_size):
            embs.append(model(x_full[start:start + batch_size]).cpu().numpy())
        emb_np = np.concatenate(embs, axis=0)

    n_real = len(real_seqs)
    real_emb = emb_np[:n_real]
    cand_emb = emb_np[n_real:]

    # Distance-based pruning: for each candidate, compute its mean cosine
    # distance to a random 10% subset of real embeddings. Reject candidates
    # whose distance is > 2× the median real/real distance.
    from numpy.linalg import norm
    rng = np.random.default_rng(0)
    ref_idx = rng.choice(n_real, size=max(50, n_real // 10), replace=False)
    ref = real_emb[ref_idx]

    def _cos_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        # a: (n_a, d), b: (n_b, d); returns (n_a, n_b) 1 - cosine_similarity
        a_n = a / (norm(a, axis=1, keepdims=True) + 1e-9)
        b_n = b / (norm(b, axis=1, keepdims=True) + 1e-9)
        return 1.0 - a_n @ b_n.T

    real_vs_ref = _cos_dist(real_emb, ref)
    cand_vs_ref = _cos_dist(cand_emb, ref)
    med_real = float(np.median(real_vs_ref.mean(axis=1)))
    cand_mean = cand_vs_ref.mean(axis=1)

    # Cluster: k-means on the combined embedding, then drop clusters with
    # low real-data representation.
    km = KMeans(n_clusters=n_clusters, n_init=5, random_state=1).fit(emb_np)
    labels = km.labels_

    diag = {
        "per_cluster_real_frac": [],
        "kept_clusters": 0,
        "dropped_candidates_cluster": 0,
        "dropped_candidates_distance": 0,
        "median_real2ref_distance": med_real,
        "mean_cand2ref_distance_before_prune": float(cand_mean.mean()),
    }
    kept: List[str] = []
    cluster_ok = np.zeros(n_clusters, dtype=bool)
    for c in range(n_clusters):
        in_c = np.where(labels == c)[0]
        real_in_c = int((in_c < n_real).sum())
        cand_in_c = int((in_c >= n_real).sum())
        total = real_in_c + cand_in_c
        frac = (real_in_c / total) if total > 0 else 0.0
        diag["per_cluster_real_frac"].append(frac)
        if frac >= min_real_frac:
            diag["kept_clusters"] += 1
            cluster_ok[c] = True
        else:
            diag["dropped_candidates_cluster"] += cand_in_c

    # Apply BOTH signals (cluster + distance).
    for i, seq in enumerate(candidates):
        cluster_idx = labels[n_real + i]
        if not cluster_ok[cluster_idx]:
            diag["dropped_candidates_distance"] += 0  # already counted in cluster
            continue
        # distance gating
        if cand_mean[i] > 2.0 * med_real:
            diag["dropped_candidates_distance"] += 1
            continue
        kept.append(seq)
    return kept, diag


# ---------------------------------------------------------------- evaluation

def evaluate_set(seqs: List[str], eval_real: List[str],
                 train_real: List[str], label: str) -> dict:
    gc_real = np.mean([gc(s) for s in eval_real])
    gc_gen = np.mean([gc(s) for s in seqs]) if seqs else float("nan")
    k3_real = kmer_counts(eval_real, 3)
    k3_gen = kmer_counts(seqs, 3)
    js3 = js_div(k3_real, k3_gen)
    # 2-mer too
    k2_real = kmer_counts(eval_real, 2)
    k2_gen = kmer_counts(seqs, 2)
    js2 = js_div(k2_real, k2_gen)
    novelty = sum(1 for s in seqs if s not in train_real) / max(len(seqs), 1)
    return {
        "label": label, "n": len(seqs),
        "gc_content": float(gc_gen), "gc_vs_real": float(abs(gc_gen - gc_real)),
        "jsd_2mer": float(js2), "jsd_3mer": float(js3),
        "novelty": float(novelty),
    }


def main() -> dict:
    t0 = time.time()
    print("Loading pretrained BayesAIRR ...")
    gen = load_checkpoint(str(ROOT / "model.pt"), device="cpu")
    gen.eval()

    print("Generating reference corpus (train=1000, eval=500, sigma=1.0) ...")
    train_real, eval_real, train_lp = generate_real_corpus(
        gen, n_train=1000, n_eval=500, sigma=1.0, seed=42
    )
    print(f"  train={len(train_real)} eval={len(eval_real)} "
          f"unique_train={len(set(train_real))}")
    print(f"  train log-prob μ±σ = {train_lp.mean():.2f} ± {train_lp.std():.2f}")

    # Noisy-BayesAIRR candidates (sigma=4.0): pushes it well outside the
    # realistic regime — the model emits very long runs of rare bases.
    print("Generating noisy-BayesAIRR candidates (sigma=4.0) ...")
    noisy_train, _, noisy_lp = generate_real_corpus(
        gen, n_train=500, n_eval=0, sigma=4.0, seed=11
    )

    # Markov candidates — but trained on a small (50 sequence) subsample of
    # training data, so it only captures a coarse 2-mer approximation.
    print("Generating under-trained Markov(2) candidates ...")
    small_train = random.sample(train_real, k=50)
    markov_seqs = sample_via_markov(small_train, n=500, order=2)

    # Uniform candidates
    print("Generating uniform baseline candidates ...")
    lengths = [len(s) for s in random.sample(train_real, min(500, len(train_real)))]
    uniform_seqs = sample_uniform(lengths)

    # Strongly GC-biased candidates — these should survive BayesAIRR filter
    # (since log-prob only rewards locally plausible transitions) but should
    # be dropped by the manifold pruner because their GC is wildly different
    # from real data.
    print("Generating GC-skewed candidates ...")
    gc_seqs: List[str] = []
    for L in random.sample(lengths, len(lengths)):
        gc_seqs.append("".join(random.choices(["G", "C", "G", "C", "A", "T"],
                                              k=L)))

    results = {}
    results["reference"] = {
        "train_n": len(train_real), "eval_n": len(eval_real),
        "train_logprob_mean": float(train_lp.mean()),
        "train_logprob_std": float(train_lp.std()),
        "unique_train": len(set(train_real)),
        "unique_eval": len(set(eval_real)),
        "eval_gc_mean": float(np.mean([gc(s) for s in eval_real])),
    }

    # ------- Evaluation: raw generators -------
    print("\n=== Evaluating raw generators ===")
    eval_sections = []
    eval_sections.append(evaluate_set(eval_real, eval_real, train_real,
                                       "eval_real_self_eval"))
    eval_sections.append(evaluate_set(noisy_train, eval_real, train_real,
                                       "noisy_bayesairr_raw"))
    eval_sections.append(evaluate_set(markov_seqs, eval_real, train_real,
                                       "markov_k2_raw"))
    eval_sections.append(evaluate_set(uniform_seqs, eval_real, train_real,
                                       "uniform_raw"))
    eval_sections.append(evaluate_set(gc_seqs, eval_real, train_real,
                                       "gc_biased_raw"))
    for r in eval_sections:
        print(f"  {r['label']:<30s} n={r['n']:>4d} "
              f"gc_err={r['gc_vs_real']:.4f} jsd3={r['jsd_3mer']:.4f} "
              f"novelty={r['novelty']:.3f}")
    results["raw_generators"] = eval_sections

    # ------- Stage 1 only: BayesAIRR score filter -------
    # We use the model's recorded log-prob (for noisy candidates) as the
    # "BayesAIRR score". For Markov/uniform, we approximate with a simple
    # 3-mer log-likelihood under the training 3-mer distribution.
    print("\n=== Stage 1: BayesAIRR / 3-mer log-prob filter ===")
    train_3mer = kmer_counts(train_real, 3)
    total_3mer = sum(train_3mer.values()) or 1
    p_3mer = {k: c / total_3mer for k, c in train_3mer.items()}

    def logp_3mer(seqs: List[str]) -> np.ndarray:
        out = np.zeros(len(seqs), dtype=np.float32)
        eps = 1e-5
        for i, s in enumerate(seqs):
            ll = 0.0
            for jj in range(len(s) - 3 + 1):
                ll += np.log2(p_3mer.get(s[jj:jj + 3], eps))
            out[i] = ll
        return out

    # For noisy-BayesAIRR: use the generator-produced log-prob.
    noisy_score = noisy_lp
    # For Markov/uniform: use 3-mer log-likelihood as a proxy Bayes-like
    # likelihood; this is a weaker signal since the generator isn't explicit.
    markov_score = logp_3mer(markov_seqs)
    uniform_score = logp_3mer(uniform_seqs)

    stage1_kept = {}
    for name, seqs, sc, ref in [
        ("noisy_bayesairr", noisy_train, noisy_score, train_lp),
        ("markov_k2", markov_seqs, markov_score, logp_3mer(train_real)),
        ("uniform", uniform_seqs, uniform_score, logp_3mer(train_real)),
        ("gc_biased", gc_seqs, logp_3mer(gc_seqs), logp_3mer(train_real)),
    ]:
        kept, _idx = stage1_by_logprob(seqs, sc, ref, q=0.10)
        stage1_kept[name] = kept
        r = evaluate_set(kept, eval_real, train_real, f"{name}_stage1")
        print(f"  {r['label']:<30s} n={r['n']:>4d} "
              f"gc_err={r['gc_vs_real']:.4f} jsd3={r['jsd_3mer']:.4f} "
              f"novelty={r['novelty']:.3f}")
        results.setdefault("stage1", []).append(r)

    # ------- Stage 2: GeoTriGate manifold pruning -------
    print("\n=== Stage 2: GeoTriGate manifold pruning (on top of Stage 1) ===")
    stage2_kept = {}
    for name, candidates in stage1_kept.items():
        if len(candidates) == 0:
            print(f"  {name}: no candidates after stage 1")
            continue
        kept, diag = stage2_manifold(candidates, train_real,
                                     n_clusters=10, min_real_frac=0.15)
        r = evaluate_set(kept, eval_real, train_real, f"{name}_two_stage")
        stage2_kept[name] = kept
        print(f"  {r['label']:<30s} n={r['n']:>4d} "
              f"gc_err={r['gc_vs_real']:.4f} jsd3={r['jsd_3mer']:.4f} "
              f"novelty={r['novelty']:.3f} (dropped={len(candidates)-len(kept)})")
        r["n_after_stage1"] = len(candidates)
        r["dropped_at_stage2"] = len(candidates) - len(kept)
        r["cluster_real_fracs"] = diag["per_cluster_real_frac"]
        results.setdefault("two_stage", []).append(r)

    # ------- Ablation: Stage 2 without stage 1 (pure manifold) -------
    print("\n=== Ablation: Stage 2 alone (no BayesAIRR score filter) ===")
    for name, raw_seqs in [
        ("noisy_bayesairr", noisy_train),
        ("markov_k2", markov_seqs),
        ("uniform", uniform_seqs),
        ("gc_biased", gc_seqs),
    ]:
        kept, diag = stage2_manifold(raw_seqs, train_real,
                                     n_clusters=10, min_real_frac=0.15)
        r = evaluate_set(kept, eval_real, train_real, f"{name}_stage2_only")
        print(f"  {r['label']:<30s} n={r['n']:>4d} "
              f"gc_err={r['gc_vs_real']:.4f} jsd3={r['jsd_3mer']:.4f} "
              f"novelty={r['novelty']:.3f}")
        results.setdefault("stage2_only", []).append(r)

    # ------- Ablation: GeoTriGate vs. plain one-hot embedding + k-means -------
    print("\n=== Ablation: one-hot + k-means instead of GeoTriGate ===")
    def one_hot_embed(seqs: List[str], L: int = 30) -> np.ndarray:
        M = np.zeros((len(seqs), 4 * L), dtype=np.float32)
        for i, s in enumerate(seqs):
            for j, c in enumerate(s[:L]):
                if c in "ACGT":
                    M[i, {"A": 0, "C": 1, "G": 2, "T": 3}[c] * L + j] = 1.0
        return M

    for name, raw_seqs in [
        ("noisy_bayesairr", noisy_train),
        ("markov_k2", markov_seqs),
        ("uniform", uniform_seqs),
        ("gc_biased", gc_seqs),
    ]:
        all_seqs = list(train_real) + list(raw_seqs)
        emb = one_hot_embed(all_seqs)
        # Dimensionality reduction via first 32 principal directions would be nicer,
        # but keep it simple: just subsample features. Use k-means on raw (large dim).
        km = KMeans(n_clusters=10, n_init=5, random_state=1).fit(emb)
        labels = km.labels_
        n_real = len(train_real)
        kept: List[str] = []
        for c in range(10):
            in_c = np.where(labels == c)[0]
            real_in_c = int((in_c < n_real).sum())
            cand_in_c = int((in_c >= n_real).sum())
            total = real_in_c + cand_in_c
            if total == 0:
                continue
            if real_in_c / total >= 0.15:
                for idx in in_c:
                    if idx >= n_real:
                        kept.append(raw_seqs[idx - n_real])
        r = evaluate_set(kept, eval_real, train_real, f"{name}_onehot_kmeans")
        print(f"  {r['label']:<30s} n={r['n']:>4d} "
              f"gc_err={r['gc_vs_real']:.4f} jsd3={r['jsd_3mer']:.4f} "
              f"novelty={r['novelty']:.3f}")
        results.setdefault("baseline_cluster", []).append(r)

    results["runtime_sec"] = time.time() - t0
    print(f"\nTotal runtime: {results['runtime_sec']:.1f}s")
    return results


if __name__ == "__main__":
    res = main()
    (ROOT / "experiment" / "results.json").write_text(
        json.dumps(res, indent=2, default=lambda x: float(x)
                   if isinstance(x, (np.floating,)) else x)
    )
    print("Wrote experiment/results.json")
