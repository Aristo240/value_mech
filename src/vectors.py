"""Value-vector construction (difference-in-means) and the geometric machinery
for the B1 attractor test, with incremental checkpointing."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .checkpoint import JSONLCheckpoint, save_array_atomic, load_array_if_exists
from .data import SCHWARTZ_19, NEUTRAL_PROMPTS, ValueEvalSample, positive_negative_for_value

if TYPE_CHECKING:
    from .models import LoadedModel


@dataclass
class ValueVectors:
    vectors: np.ndarray          # (19, hidden), float32
    n_pos: list[int]
    n_neg: list[int]
    layer: int
    aggregation: str


def compute_value_vectors(
    loaded: "LoadedModel",
    samples: list[ValueEvalSample],
    cfg: dict,
    rng: np.random.Generator,
    ckpt_dir: Path | None = None,
    tag: str = "",
) -> ValueVectors:
    """Compute a diff-in-means vector per Schwartz value. Supports resume:

    - If `<ckpt_dir>/value_vectors_<tag>.npy` already exists, load it and return.
    - Otherwise, use `<ckpt_dir>/value_vectors_<tag>.jsonl` as a per-value
      checkpoint. Values already in the JSONL are skipped on resume. The final
      dense `.npy` is written atomically at the end.
    """
    from .models import extract_activations  # lazy: keeps math testable without torch

    layer = cfg["extraction"]["layer"]
    agg = cfg["extraction"]["aggregation"]
    n_per = cfg["extraction"]["n_per_value"]
    max_len = cfg["extraction"]["max_length"]
    bs = cfg["extraction"].get("batch_size", 8)
    H = loaded.hidden_size

    # Fast-path: fully cached dense array.
    if ckpt_dir is not None:
        dense_path = ckpt_dir / f"value_vectors_{tag}.npy"
        cached = load_array_if_exists(dense_path)
        if cached is not None and cached.shape == (len(SCHWARTZ_19), H):
            logging.info(f"Loaded cached value vectors from {dense_path}")
            return ValueVectors(
                vectors=cached.astype(np.float32),
                n_pos=[-1] * len(SCHWARTZ_19),
                n_neg=[-1] * len(SCHWARTZ_19),
                layer=layer, aggregation=agg,
            )

    # Per-value checkpoint.
    ckpt: JSONLCheckpoint | None = None
    if ckpt_dir is not None:
        ckpt = JSONLCheckpoint(ckpt_dir / f"value_vectors_{tag}.jsonl")
        if len(ckpt) > 0:
            logging.info(f"Resuming value-vector extraction from {len(ckpt)} cached values")

    out = np.zeros((len(SCHWARTZ_19), H), dtype=np.float32)
    n_pos_list = [0] * len(SCHWARTZ_19)
    n_neg_list = [0] * len(SCHWARTZ_19)

    # Replay existing records so `out` matches what's on disk.
    if ckpt is not None:
        for key, val in ckpt.items():
            vi = int(key[0]) if isinstance(key, list) else int(key)
            out[vi] = np.asarray(val["vector"], dtype=np.float32)
            n_pos_list[vi] = val["n_pos"]
            n_neg_list[vi] = val["n_neg"]

    for vi, vname in enumerate(SCHWARTZ_19):
        if ckpt is not None and [vi] in ckpt:
            continue
        pos_texts, neg_texts = positive_negative_for_value(samples, vi, n_per, rng)
        n_pos_list[vi] = len(pos_texts)
        n_neg_list[vi] = len(neg_texts)
        if len(pos_texts) == 0 or len(neg_texts) == 0:
            logging.warning(
                f"Value {vi} {vname}: insufficient data "
                f"(pos={len(pos_texts)}, neg={len(neg_texts)}); zero vector."
            )
            vec = np.zeros(H, dtype=np.float32)
        else:
            pos_acts = extract_activations(loaded, pos_texts, layer, agg, max_len, batch_size=bs)
            neg_acts = extract_activations(loaded, neg_texts, layer, agg, max_len, batch_size=bs)
            vec = (pos_acts.mean(axis=0) - neg_acts.mean(axis=0)).astype(np.float32)
        out[vi] = vec
        logging.info(
            f"  v{vi:2d} {vname:32s}  |dim|={np.linalg.norm(vec):.3f}  "
            f"n_pos={len(pos_texts)} n_neg={len(neg_texts)}"
        )
        if ckpt is not None:
            ckpt.append([vi], {
                "value_name": vname,
                "vector": vec.tolist(),
                "norm": float(np.linalg.norm(vec)),
                "n_pos": len(pos_texts),
                "n_neg": len(neg_texts),
            })

    if ckpt_dir is not None:
        save_array_atomic(ckpt_dir / f"value_vectors_{tag}.npy", out)

    return ValueVectors(vectors=out, n_pos=n_pos_list, n_neg=n_neg_list,
                        layer=layer, aggregation=agg)


def compute_attractor(
    base: "LoadedModel",
    instruct: "LoadedModel",
    cfg: dict,
    ckpt_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (attractor, base_acts, instr_acts). Caches base_acts and instr_acts
    as .npy files so the bootstrap in B1 can reuse them on resume."""
    from .models import extract_activations  # lazy

    layer = cfg["extraction"]["layer"]
    agg = cfg["extraction"]["aggregation"]
    max_len = cfg["extraction"]["max_length"]
    bs = cfg["extraction"].get("batch_size", 8)
    prompts = NEUTRAL_PROMPTS[:cfg["neutral_prompts"]["n"]]

    base_acts = None
    instr_acts = None
    if ckpt_dir is not None:
        base_acts = load_array_if_exists(ckpt_dir / "neutral_acts_base.npy")
        instr_acts = load_array_if_exists(ckpt_dir / "neutral_acts_instruct.npy")

    if base_acts is None:
        logging.info("Extracting neutral-prompt activations on BASE...")
        base_acts = extract_activations(base, prompts, layer, agg, max_len, batch_size=bs)
        if ckpt_dir is not None:
            save_array_atomic(ckpt_dir / "neutral_acts_base.npy", base_acts)
    else:
        logging.info(f"Loaded cached base neutral-acts: shape={base_acts.shape}")

    if instr_acts is None:
        logging.info("Extracting neutral-prompt activations on INSTRUCT...")
        instr_acts = extract_activations(instruct, prompts, layer, agg, max_len, batch_size=bs)
        if ckpt_dir is not None:
            save_array_atomic(ckpt_dir / "neutral_acts_instruct.npy", instr_acts)
    else:
        logging.info(f"Loaded cached instruct neutral-acts: shape={instr_acts.shape}")

    attractor = instr_acts.mean(axis=0) - base_acts.mean(axis=0)
    logging.info(f"Attractor norm: {np.linalg.norm(attractor):.3f} on {len(prompts)} prompts")
    return attractor.astype(np.float32), base_acts.astype(np.float32), instr_acts.astype(np.float32)


# --- Subspace projection ---------------------------------------------------

def projection_onto_subspace(
    v: np.ndarray,
    basis_rows: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Orthogonal projection of v onto the row span of basis_rows.
    Returns (projected_vector, ||proj||/||v||)."""
    B = basis_rows
    BBt = B @ B.T
    BBt_pinv = np.linalg.pinv(BBt)
    coeffs = BBt_pinv @ (B @ v)
    proj = B.T @ coeffs
    nv = float(np.linalg.norm(v))
    if nv == 0.0:
        return proj, 0.0
    return proj, float(np.linalg.norm(proj) / nv)


def circumplex_2d(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a 2D PCA embedding of the 19 value vectors and the angles (deg)
    of each value in that plane."""
    X = vectors - vectors.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    coords = X @ Vt[:2].T
    angles = np.degrees(np.arctan2(coords[:, 1], coords[:, 0])) % 360.0
    return coords, angles


def project_into_2d(
    v: np.ndarray,
    vectors: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Project v into the same 2D plane that circumplex_2d() builds; returns (xy, angle_deg)."""
    X = vectors - vectors.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    plane = Vt[:2]
    v_centered = v - vectors.mean(axis=0)
    xy = plane @ v_centered
    angle = float(np.degrees(np.arctan2(xy[1], xy[0])) % 360.0)
    return xy, angle
