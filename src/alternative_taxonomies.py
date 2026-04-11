"""Alternative-taxonomy null test for B1.

The critical question: does the attractor project strongly onto the Schwartz
value subspace SPECIFICALLY, or onto ANY coherent 19-dim semantic subspace?

This script builds dimension-matched comparison subspaces from:
  1. Moral Foundations Theory (6 foundations from MFTC)
  2. Big Five personality traits (5 traits)
  3. Topic categories (4 topics from AG News)
  4. Random label control (19 random directions from the same corpus)

For each taxonomy, we:
  - Extract difference-in-means vectors (same method as Schwartz value vectors)
  - Project the attractor onto the resulting subspace
  - Compare the projection ratio to the Schwartz baseline

If ALL taxonomies capture ~0.99 of the attractor, the Schwartz framing is
decoration. If Schwartz captures significantly more, the claim survives.

Also computes base-vs-instruct subspace overlap (Direction 1: navigation vs creation).

Run:
  CUDA_VISIBLE_DEVICES=0,1 CONFIG=config.yaml python -m src.alternative_taxonomies
"""
from __future__ import annotations

import logging
import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset

from . import data as D
from . import models as M
from . import vectors as V
from . import stats as S
from . import utils as U
from .checkpoint import save_array_atomic, load_array_if_exists


def _extract_contrastive_vectors(
    loaded: M.LoadedModel,
    positive_texts: list[list[str]],
    negative_texts: list[list[str]],
    labels: list[str],
    cfg: dict,
    ckpt_dir: Path,
    tag: str,
) -> np.ndarray:
    """Build difference-in-means vectors for an arbitrary taxonomy.

    Args:
        positive_texts: list of K lists, each containing positive texts for label k
        negative_texts: list of K lists, each containing negative texts for label k
        labels: list of K label names

    Returns: (K, hidden_dim) array of contrastive vectors
    """
    from .models import extract_activations

    cache_path = ckpt_dir / f"vectors_{tag}.npy"
    cached = load_array_if_exists(cache_path)
    if cached is not None:
        logging.info(f"Loaded cached {tag} vectors from {cache_path}")
        return cached

    layer = cfg["extraction"]["layer"]
    agg = cfg["extraction"]["aggregation"]
    max_len = cfg["extraction"]["max_length"]
    bs = cfg["extraction"].get("batch_size", 8)
    H = loaded.hidden_size

    vectors = np.zeros((len(labels), H), dtype=np.float32)
    for i, label in enumerate(labels):
        pos = positive_texts[i]
        neg = negative_texts[i]
        if len(pos) == 0 or len(neg) == 0:
            logging.warning(f"  {tag} label '{label}': insufficient data (pos={len(pos)}, neg={len(neg)})")
            continue
        pos_acts = extract_activations(loaded, pos, layer, agg, max_len, batch_size=bs)
        neg_acts = extract_activations(loaded, neg, layer, agg, max_len, batch_size=bs)
        vectors[i] = (pos_acts.mean(axis=0) - neg_acts.mean(axis=0)).astype(np.float32)
        logging.info(f"  {tag} [{i:2d}] {label:30s} |v|={np.linalg.norm(vectors[i]):.3f} "
                     f"n_pos={len(pos)} n_neg={len(neg)}")

    save_array_atomic(cache_path, vectors)
    return vectors


# --- Dataset loaders ---------------------------------------------------------

def load_mft_texts(n_per: int = 200, rng: np.random.Generator = None):
    """Load Moral Foundations Twitter Corpus. 6 foundations."""
    ds = load_dataset("maciejskorski/morality-MFTC-onehot", split="train")
    foundations = ["care", "fairness", "loyalty", "authority", "sanctity"]
    # Add a 6th: "non-moral" as control
    labels = foundations + ["non-moral"]

    positive_texts = []
    negative_texts = []
    for label in labels:
        pos = [r["text"] for r in ds if r[label] == 1]
        neg = [r["text"] for r in ds if r[label] == 0]
        if rng is not None:
            n = min(n_per, len(pos), len(neg))
            pos = [pos[i] for i in rng.choice(len(pos), size=n, replace=False)]
            neg = [neg[i] for i in rng.choice(len(neg), size=n, replace=False)]
        positive_texts.append(pos[:n_per])
        negative_texts.append(neg[:n_per])

    return positive_texts, negative_texts, labels


def load_big5_texts(n_per: int = 200, rng: np.random.Generator = None):
    """Load Big Five personality descriptions. 5 traits.
    Dataset uses capitalized trait names and integer levels 1-5."""
    ds = load_dataset("agentlans/big-five-personality-traits", split="train")
    traits = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]

    positive_texts = []
    negative_texts = []
    for trait in traits:
        # High = levels 4,5; Low = levels 1,2
        pos = [r["description"] for r in ds if r["trait"] == trait and r["level"] >= 4]
        neg = [r["description"] for r in ds if r["trait"] == trait and r["level"] <= 2]
        if rng is not None and len(pos) > n_per:
            idx = rng.choice(len(pos), size=min(n_per, len(pos)), replace=False)
            pos = [pos[i] for i in idx]
        if rng is not None and len(neg) > n_per:
            idx = rng.choice(len(neg), size=min(n_per, len(neg)), replace=False)
            neg = [neg[i] for i in idx]
        positive_texts.append(pos[:n_per])
        negative_texts.append(neg[:n_per])

    return positive_texts, negative_texts, [t.lower() for t in traits]


def load_topic_texts(n_per: int = 200, rng: np.random.Generator = None):
    """Load AG News topics. 4 categories."""
    ds = load_dataset("fancyzhx/ag_news", split="train")
    topic_names = ["World", "Sports", "Business", "Sci/Tech"]

    positive_texts = []
    negative_texts = []
    for label_id, name in enumerate(topic_names):
        pos = [r["text"] for r in ds if r["label"] == label_id]
        neg = [r["text"] for r in ds if r["label"] != label_id]
        if rng is not None:
            n = min(n_per, len(pos), len(neg))
            pos = [pos[i] for i in rng.choice(len(pos), size=n, replace=False)]
            neg = [neg[i] for i in rng.choice(len(neg), size=n, replace=False)]
        positive_texts.append(pos[:n_per])
        negative_texts.append(neg[:n_per])

    return positive_texts, negative_texts, topic_names


def build_random_subspace(
    loaded: M.LoadedModel,
    samples: list[D.ValueEvalSample],
    cfg: dict,
    rng: np.random.Generator,
    ckpt_dir: Path,
    n_dims: int = 19,
) -> np.ndarray:
    """Build a random 19-dim subspace by shuffling value labels on the same corpus.

    Uses the same texts and same extraction pipeline as Schwartz, but randomly
    assigns which texts are "positive" and "negative" for each of 19 fake labels.
    """
    cache_path = ckpt_dir / "vectors_random_labels.npy"
    cached = load_array_if_exists(cache_path)
    if cached is not None:
        logging.info(f"Loaded cached random-label vectors from {cache_path}")
        return cached

    from .models import extract_activations
    layer = cfg["extraction"]["layer"]
    agg = cfg["extraction"]["aggregation"]
    max_len = cfg["extraction"]["max_length"]
    bs = cfg["extraction"].get("batch_size", 8)
    H = loaded.hidden_size
    n_per = cfg["extraction"]["n_per_value"]

    all_texts = [s.text for s in samples]
    vectors = np.zeros((n_dims, H), dtype=np.float32)

    for i in range(n_dims):
        # Random binary split
        idx = rng.permutation(len(all_texts))
        n = min(n_per, len(all_texts) // 2)
        pos_texts = [all_texts[j] for j in idx[:n]]
        neg_texts = [all_texts[j] for j in idx[n:2*n]]
        pos_acts = extract_activations(loaded, pos_texts, layer, agg, max_len, batch_size=bs)
        neg_acts = extract_activations(loaded, neg_texts, layer, agg, max_len, batch_size=bs)
        vectors[i] = (pos_acts.mean(axis=0) - neg_acts.mean(axis=0)).astype(np.float32)
        logging.info(f"  random [{i:2d}] |v|={np.linalg.norm(vectors[i]):.3f}")

    save_array_atomic(cache_path, vectors)
    return vectors


# --- Subspace overlap (Direction 1) -----------------------------------------

def subspace_overlap(A: np.ndarray, B: np.ndarray) -> dict:
    """Compute overlap between two subspaces via principal angles.

    A: (k, d) - basis vectors of subspace A
    B: (m, d) - basis vectors of subspace B

    Returns dict with:
      - principal_angles_deg: the principal angles
      - mean_cos: mean cosine of principal angles (1 = identical, 0 = orthogonal)
      - grassmann_distance: Grassmann distance
    """
    # Orthonormalize both
    Qa, _ = np.linalg.qr(A.T)
    Qb, _ = np.linalg.qr(B.T)
    k = min(Qa.shape[1], Qb.shape[1])
    Qa = Qa[:, :k]
    Qb = Qb[:, :k]

    # SVD of Qa^T @ Qb gives cosines of principal angles
    M = Qa.T @ Qb
    _, sigmas, _ = np.linalg.svd(M)
    sigmas = np.clip(sigmas, 0, 1)  # numerical safety
    angles_rad = np.arccos(sigmas)
    angles_deg = np.degrees(angles_rad)

    mean_cos = float(sigmas.mean())
    grassmann = float(np.sqrt(np.sum(angles_rad ** 2)))

    return {
        "principal_angles_deg": angles_deg.tolist(),
        "principal_angle_cosines": sigmas.tolist(),
        "mean_cosine": mean_cos,
        "grassmann_distance": grassmann,
        "n_dimensions": k,
    }


# --- Main -------------------------------------------------------------------

def main():
    cfg = U.load_config()
    U.set_seed(cfg["seed"])
    out_dir = U.setup_logging(cfg, "alternative_taxonomies")
    rng = np.random.default_rng(cfg["seed"])

    logging.info("Loading models...")
    base = M.load_model(
        cfg["models"]["base"]["name"], cfg["models"]["base"]["dtype"],
        gpu_ids=cfg["models"]["base"]["gpu_ids"],
        max_memory_per_gpu=cfg["models"]["max_memory_per_gpu"],
        is_instruct=False,
    )
    instruct = M.load_model(
        cfg["models"]["instruct"]["name"], cfg["models"]["instruct"]["dtype"],
        gpu_ids=cfg["models"]["instruct"]["gpu_ids"],
        max_memory_per_gpu=cfg["models"]["max_memory_per_gpu"],
        is_instruct=True,
    )

    # Load ValueEval for Schwartz baseline and random control
    samples = D.load_valueeval(cfg["data"]["valueeval_root"], cfg["data"]["split"])
    logging.info(f"Loaded {len(samples)} ValueEval sentences")

    # --- Schwartz value vectors (baseline, from B1 cache) ---
    b1_dir = out_dir.parent / "pilot_b1_attractor"
    schwartz_vectors = load_array_if_exists(b1_dir / "value_vectors_instruct.npy")
    if schwartz_vectors is None:
        logging.info("Computing Schwartz value vectors (not cached from B1)...")
        vv = V.compute_value_vectors(instruct, samples, cfg, rng, ckpt_dir=out_dir, tag="schwartz")
        schwartz_vectors = vv.vectors
    else:
        logging.info(f"Loaded Schwartz vectors from B1 cache: shape={schwartz_vectors.shape}")

    # --- Attractor (from B1 cache) ---
    attractor_data = load_array_if_exists(b1_dir / "neutral_acts_instruct.npy")
    base_acts = load_array_if_exists(b1_dir / "neutral_acts_base.npy")
    if attractor_data is not None and base_acts is not None:
        attractor = attractor_data.mean(axis=0) - base_acts.mean(axis=0)
        logging.info(f"Loaded attractor from B1 cache, norm={np.linalg.norm(attractor):.3f}")
    else:
        attractor, _, _ = V.compute_attractor(base, instruct, cfg, ckpt_dir=out_dir)

    # --- Schwartz baseline projection ---
    _, schwartz_ratio = V.projection_onto_subspace(attractor, schwartz_vectors)
    logging.info(f"\nSchwartz projection ratio: {schwartz_ratio:.4f} (19 dims)")

    # --- Alternative taxonomy 1: Moral Foundations (6 dims) ---
    logging.info("\n=== MORAL FOUNDATIONS (MFTC, 6 foundations) ===")
    mft_pos, mft_neg, mft_labels = load_mft_texts(n_per=200, rng=rng)
    mft_vectors = _extract_contrastive_vectors(
        instruct, mft_pos, mft_neg, mft_labels, cfg, out_dir, "mft"
    )
    _, mft_ratio = V.projection_onto_subspace(attractor, mft_vectors)
    logging.info(f"MFT projection ratio: {mft_ratio:.4f} ({len(mft_labels)} dims)")

    # --- Alternative taxonomy 2: Big Five (5 dims) ---
    logging.info("\n=== BIG FIVE PERSONALITY (5 traits) ===")
    b5_pos, b5_neg, b5_labels = load_big5_texts(n_per=200, rng=rng)
    b5_vectors = _extract_contrastive_vectors(
        instruct, b5_pos, b5_neg, b5_labels, cfg, out_dir, "big5"
    )
    _, b5_ratio = V.projection_onto_subspace(attractor, b5_vectors)
    logging.info(f"Big Five projection ratio: {b5_ratio:.4f} ({len(b5_labels)} dims)")

    # --- Alternative taxonomy 3: Topics (4 dims) ---
    logging.info("\n=== TOPIC CATEGORIES (AG News, 4 topics) ===")
    topic_pos, topic_neg, topic_labels = load_topic_texts(n_per=200, rng=rng)
    topic_vectors = _extract_contrastive_vectors(
        instruct, topic_pos, topic_neg, topic_labels, cfg, out_dir, "topics"
    )
    _, topic_ratio = V.projection_onto_subspace(attractor, topic_vectors)
    logging.info(f"Topic projection ratio: {topic_ratio:.4f} ({len(topic_labels)} dims)")

    # --- Random label control (19 dims, same corpus as Schwartz) ---
    logging.info("\n=== RANDOM LABEL CONTROL (19 random splits of ValueEval) ===")
    random_vectors = build_random_subspace(instruct, samples, cfg, rng, out_dir)
    _, random_ratio = V.projection_onto_subspace(attractor, random_vectors)
    logging.info(f"Random-label projection ratio: {random_ratio:.4f} (19 dims)")

    # --- Random baseline for each (what does a random vector achieve?) ---
    H = instruct.hidden_size
    n_random = 1000

    logging.info("\n=== RANDOM DIRECTION NULLS ===")
    for name, vecs in [("Schwartz", schwartz_vectors), ("MFT", mft_vectors),
                        ("Big5", b5_vectors), ("Topics", topic_vectors),
                        ("Random-labels", random_vectors)]:
        null_ratios = S.random_projection_null(H, vecs, n_random, rng)
        logging.info(f"  {name:15s}: null_mean={null_ratios.mean():.4f}, "
                     f"null_p95={np.quantile(null_ratios, 0.95):.4f}")

    # --- Direction 1: Base vs Instruct — navigation vs creation ---
    logging.info("\n=== BASE vs INSTRUCT: NAVIGATION OR CREATION? ===")
    base_schwartz = load_array_if_exists(b1_dir / "value_vectors_base.npy")
    if base_schwartz is None:
        b2_dir = out_dir.parent / "pilot_b2_steering"
        base_schwartz = load_array_if_exists(b2_dir / "value_vectors_base.npy")
    if base_schwartz is None:
        logging.info("No cached base vectors found — computing from scratch...")
        vv_base = V.compute_value_vectors(
            base, samples, cfg, np.random.default_rng(cfg["seed"]),
            ckpt_dir=out_dir, tag="base",
        )
        base_schwartz = vv_base.vectors

    logging.info(f"Loaded base Schwartz vectors, shape={base_schwartz.shape}")

    # --- Precondition: split-half reliability of BASE value vectors ---
    logging.info("\n  Split-half reliability of BASE value vectors:")
    n_splits = 20
    split_half_cosines_base = []
    for s in range(n_splits):
        split_rng = np.random.default_rng(cfg["seed"] + 1000 + s)
        vv_half1 = V.compute_value_vectors(
            base, samples, cfg, split_rng,
            ckpt_dir=None, tag="",  # no caching for splits
        )
        split_rng2 = np.random.default_rng(cfg["seed"] + 2000 + s)
        vv_half2 = V.compute_value_vectors(
            base, samples, cfg, split_rng2,
            ckpt_dir=None, tag="",
        )
        cosines = []
        for vi in range(len(D.SCHWARTZ_19)):
            n1 = np.linalg.norm(vv_half1.vectors[vi])
            n2 = np.linalg.norm(vv_half2.vectors[vi])
            if n1 > 1e-8 and n2 > 1e-8:
                c = float(np.dot(vv_half1.vectors[vi], vv_half2.vectors[vi]) / (n1 * n2))
                cosines.append(c)
        split_half_cosines_base.append(float(np.median(cosines)))
    mean_sh = float(np.mean(split_half_cosines_base))
    logging.info(f"    Mean split-half median cosine (base): {mean_sh:.3f}")
    logging.info(f"    {'RELIABLE (>= 0.7)' if mean_sh >= 0.7 else 'UNRELIABLE (< 0.7) — base subspace claims are suspect'}")
    base_reliable = mean_sh >= 0.7

    # --- Per-value paired cosine: cos(d_base[v], d_instruct[v]) ---
    logging.info("\n  Per-value paired cosine (base vs instruct):")
    paired_cosines = []
    for vi, vname in enumerate(D.SCHWARTZ_19):
        nb = np.linalg.norm(base_schwartz[vi])
        ni = np.linalg.norm(schwartz_vectors[vi])
        if nb > 1e-8 and ni > 1e-8:
            c = float(np.dot(base_schwartz[vi], schwartz_vectors[vi]) / (nb * ni))
        else:
            c = 0.0
        paired_cosines.append(c)
        logging.info(f"    v{vi:2d} {vname:35s} cos={c:+.3f}")
    median_paired = float(np.median(paired_cosines))
    mean_paired = float(np.mean(paired_cosines))
    logging.info(f"  Median paired cosine: {median_paired:.3f}")
    logging.info(f"  Mean paired cosine:   {mean_paired:.3f}")

    if median_paired >= 0.7:
        nav_verdict = "NAVIGATION: same value axes, instruction tuning translates within them"
    elif median_paired >= 0.3:
        nav_verdict = "PARTIAL ROTATION: value subspace preserved but axes partially rotated"
    else:
        nav_verdict = "CREATION: instruction tuning restructures the value geometry"

    logging.info(f"  VERDICT: {nav_verdict}")

    # --- Aggregate subspace overlap (secondary) ---
    overlap = subspace_overlap(schwartz_vectors, base_schwartz)
    logging.info(f"\n  Aggregate subspace overlap:")
    logging.info(f"    Mean cosine of principal angles: {overlap['mean_cosine']:.4f}")
    logging.info(f"    Grassmann distance: {overlap['grassmann_distance']:.4f}")

    # --- Attractor projection onto BASE value subspace ---
    _, base_ratio = V.projection_onto_subspace(attractor, base_schwartz)
    logging.info(f"  Attractor onto BASE subspace:    {base_ratio:.4f}")
    logging.info(f"  Attractor onto INSTRUCT subspace: {schwartz_ratio:.4f}")

    # --- Summary ---
    logging.info("\n" + "=" * 60)
    logging.info("ALTERNATIVE TAXONOMY NULL RESULTS")
    logging.info("=" * 60)
    results_table = [
        ("Schwartz (19 values)", schwartz_ratio, 19),
        ("Moral Foundations (6)", mft_ratio, 6),
        ("Big Five (5)", b5_ratio, 5),
        ("Topics (4)", topic_ratio, 4),
        ("Random labels (19)", random_ratio, 19),
    ]
    for name, ratio, dims in results_table:
        logging.info(f"  {name:30s}  ratio={ratio:.4f}  dims={dims}")

    if schwartz_ratio > max(mft_ratio, b5_ratio, topic_ratio, random_ratio) + 0.1:
        verdict = "SCHWARTZ WINS: value subspace captures significantly more of the attractor"
    elif schwartz_ratio > max(mft_ratio, b5_ratio, topic_ratio, random_ratio):
        verdict = "SCHWARTZ LEADS NARROWLY: may not survive peer review scrutiny"
    else:
        verdict = "SCHWARTZ DOES NOT WIN: attractor is captured by other semantic subspaces too"
    logging.info(f"\n  VERDICT: {verdict}")

    # Save results
    results = {
        "model": instruct.name,
        "layer": cfg["extraction"]["layer"],
        "attractor_norm": float(np.linalg.norm(attractor)),
        "projection_ratios": {
            "schwartz_19": float(schwartz_ratio),
            "moral_foundations_6": float(mft_ratio),
            "big_five_5": float(b5_ratio),
            "topics_4": float(topic_ratio),
            "random_labels_19": float(random_ratio),
        },
        "dimensions": {
            "schwartz": 19, "moral_foundations": 6, "big_five": 5,
            "topics": 4, "random_labels": 19,
        },
        "direction_1_navigation_vs_creation": {
            "base_split_half_reliability": mean_sh,
            "base_reliable": base_reliable,
            "per_value_paired_cosines": {
                D.SCHWARTZ_19[i]: paired_cosines[i] for i in range(len(D.SCHWARTZ_19))
            },
            "median_paired_cosine": median_paired,
            "mean_paired_cosine": mean_paired,
            "subspace_overlap": overlap,
            "attractor_on_base_subspace": float(base_ratio),
            "attractor_on_instruct_subspace": float(schwartz_ratio),
            "verdict": nav_verdict,
        },
        "taxonomy_verdict": verdict,
    }
    U.save_json(results, out_dir / "results.json")
    logging.info(f"\nWrote results to {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
