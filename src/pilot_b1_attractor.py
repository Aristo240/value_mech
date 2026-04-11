"""Pilot B1: Is the instruct-vs-base compliance attractor inside the Schwartz
value subspace, or mostly orthogonal to it?

Decision rule (pre-committed):
  PASS if projection_norm_ratio >= 0.30 with bootstrap 95% CI lower bound >= 0.20.

Resumable: value vectors and neutral-prompt activations are cached; on re-run
the expensive GPU work is skipped. Only the bootstrap and the final summary
are recomputed (they're cheap).

Run:
  python -m src.pilot_b1_attractor
"""
from __future__ import annotations

import logging

import numpy as np

from . import data as D
from . import models as M
from . import stats as S
from . import vectors as V
from . import utils as U


def main():
    cfg = U.load_config()
    U.set_seed(cfg["seed"])
    out_dir = U.setup_logging(cfg, "pilot_b1_attractor")
    rng = np.random.default_rng(cfg["seed"])

    # 1. Load both models onto their assigned GPUs.
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
    M.assert_tokenizer_match(base, instruct)
    assert base.hidden_size == instruct.hidden_size, "Hidden size mismatch"

    # 2. ValueEval samples.
    logging.info("Loading ValueEval24...")
    samples = D.load_valueeval(cfg["data"]["valueeval_root"], cfg["data"]["split"])
    logging.info(f"Loaded {len(samples)} sentences from {cfg['data']['split']}")

    # 3. Value vectors on INSTRUCT (cached via ckpt_dir).
    logging.info("Computing value vectors on INSTRUCT...")
    vv_instruct = V.compute_value_vectors(
        instruct, samples, cfg, rng, ckpt_dir=out_dir, tag="instruct",
    )

    # 4. Attractor + cached neutral activations.
    attractor, base_acts_full, instr_acts_full = V.compute_attractor(
        base, instruct, cfg, ckpt_dir=out_dir,
    )

    # 5. Project attractor onto the Schwartz subspace.
    proj, frac = V.projection_onto_subspace(attractor, vv_instruct.vectors)
    logging.info(f"Attractor norm: {np.linalg.norm(attractor):.3f}")
    logging.info(f"Projection norm: {np.linalg.norm(proj):.3f}")
    logging.info(f"Projection norm ratio ||proj||/||v||: {frac:.3f}")

    # 6. Angle in the empirical 2D circumplex plane.
    coords_19, angles_19 = V.circumplex_2d(vv_instruct.vectors)
    xy_attractor, attractor_angle = V.project_into_2d(attractor, vv_instruct.vectors)
    logging.info(f"Attractor angle in 2D circumplex plane: {attractor_angle:.1f} deg")

    diffs = np.array([D.angular_distance_deg(attractor_angle, a) for a in angles_19])
    closest_idx = int(np.argmin(diffs))
    logging.info(
        f"Closest value: {D.SCHWARTZ_19[closest_idx]} "
        f"(angular distance {diffs[closest_idx]:.1f} deg)"
    )

    # 7. Random baseline: what ratio would a random direction achieve?
    H = base.hidden_size
    n_random = cfg["stats"].get("n_random_baseline", 1000)
    logging.info(f"Computing random baseline with {n_random} random unit vectors...")
    null_ratios = S.random_projection_null(H, vv_instruct.vectors, n_random, rng)
    null_mean = float(null_ratios.mean())
    null_p95 = float(np.quantile(null_ratios, 0.95))
    null_p99 = float(np.quantile(null_ratios, 0.99))
    # p-value: fraction of random vectors with ratio >= observed
    random_p = float((np.sum(null_ratios >= frac) + 1) / (n_random + 1))
    logging.info(f"Random baseline: mean={null_mean:.4f}, 95th={null_p95:.4f}, "
                 f"99th={null_p99:.4f}, p={random_p:.4f}")

    # 8. Bootstrap over neutral-prompt indices into the cached act arrays.
    n_prompts = base_acts_full.shape[0]
    n_boot = min(cfg["stats"]["n_bootstrap"], 500)
    logging.info(f"Bootstrapping projection ratio with n={n_boot} resamples...")
    fracs = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n_prompts, size=n_prompts)
        a_boot = instr_acts_full[idx].mean(0) - base_acts_full[idx].mean(0)
        _, fracs[i] = V.projection_onto_subspace(a_boot, vv_instruct.vectors)
    lo = float(np.quantile(fracs, 0.025))
    hi = float(np.quantile(fracs, 0.975))

    PASS = (frac >= 0.30) and (lo >= 0.20)
    logging.info(f"Bootstrap 95% CI on projection ratio: [{lo:.3f}, {hi:.3f}]")
    logging.info(f"DECISION B1: {'PASS' if PASS else 'FAIL'}")

    results = {
        "models": {"base": base.name, "instruct": instruct.name},
        "layer": cfg["extraction"]["layer"],
        "aggregation": cfg["extraction"]["aggregation"],
        "n_neutral_prompts": n_prompts,
        "attractor_norm": float(np.linalg.norm(attractor)),
        "projection_norm": float(np.linalg.norm(proj)),
        "projection_norm_ratio": float(frac),
        "projection_norm_ratio_ci95": [lo, hi],
        "random_baseline": {
            "n_random": n_random,
            "mean_ratio": null_mean,
            "p95_ratio": null_p95,
            "p99_ratio": null_p99,
            "p_value": random_p,
            "interpretation": (
                f"Observed ratio {frac:.3f} vs random baseline mean {null_mean:.4f} "
                f"(99th percentile {null_p99:.4f}). "
                f"p={random_p:.4f} under random-direction null."
            ),
        },
        "attractor_angle_deg": attractor_angle,
        "closest_value": D.SCHWARTZ_19[closest_idx],
        "closest_value_angle_deg": float(angles_19[closest_idx]),
        "closest_value_angular_distance_deg": float(diffs[closest_idx]),
        "decision_rule_passed": bool(PASS),
        "value_vectors_norms": [float(np.linalg.norm(v)) for v in vv_instruct.vectors],
        "value_n_pos": vv_instruct.n_pos,
        "value_n_neg": vv_instruct.n_neg,
        "caveat": (
            "Attractor is computed without the chat template (raw text for both "
            "models) to avoid conflating chat-format direction with compliance. "
            "Chat-templated ablation is a separate run."
        ),
    }
    U.save_json(results, out_dir / "results.json")
    logging.info(f"Wrote results to {out_dir}")


if __name__ == "__main__":
    main()
