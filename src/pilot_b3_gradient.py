"""Pilot B3: Fit the inertia gradient on all 19 values.

Only run if B1 and B2 pass.

Resumable: each (model, value, alpha) score is checkpointed in sweep.jsonl.

Decision rule (pre-committed):
  PASS if cosine fit R^2 > 0 AND permutation p < 0.05.

Run:
  python -m src.pilot_b3_gradient
"""
from __future__ import annotations

import logging

import numpy as np

from . import data as D
from . import models as M
from . import vectors as V
from . import steering as ST
from . import stats as S
from . import utils as U
from .checkpoint import JSONLCheckpoint


def main():
    cfg = U.load_config()
    U.set_seed(cfg["seed"])
    out_dir = U.setup_logging(cfg, "pilot_b3_gradient")
    rng = np.random.default_rng(cfg["seed"])

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

    samples = D.load_valueeval(cfg["data"]["valueeval_root"], cfg["data"]["split"])

    logging.info("Computing value vectors on both models...")
    vv_base = V.compute_value_vectors(
        base, samples, cfg, np.random.default_rng(cfg["seed"]),
        ckpt_dir=out_dir, tag="base",
    )
    vv_instr = V.compute_value_vectors(
        instruct, samples, cfg, np.random.default_rng(cfg["seed"]),
        ckpt_dir=out_dir, tag="instruct",
    )

    attractor, _, _ = V.compute_attractor(base, instruct, cfg, ckpt_dir=out_dir)
    coords_19, value_angles_19 = V.circumplex_2d(vv_instr.vectors)
    _, attractor_angle = V.project_into_2d(attractor, vv_instr.vectors)
    angular_distances = np.array([
        D.angular_distance_deg(a, attractor_angle) for a in value_angles_19
    ])

    sweep_ckpt = JSONLCheckpoint(out_dir / "sweep.jsonl")
    if len(sweep_ckpt) > 0:
        logging.info(f"Resuming from {len(sweep_ckpt)} cached sweep points")
    # Preload all stored scores into a lookup dict for fast access.
    stored: dict[tuple, float] = {}
    for key, val in sweep_ckpt.items():
        stored[(key[0], int(key[1]), float(key[2]))] = float(val["score"])

    alphas = np.array(cfg["steering"]["alphas"], dtype=np.float64)
    inertia = np.empty(len(D.SCHWARTZ_19), dtype=np.float64)
    rows = []

    for vi, vname in enumerate(D.SCHWARTZ_19):
        scores_b = np.empty(len(alphas))
        scores_i = np.empty(len(alphas))
        for i, a in enumerate(alphas):
            key_b = ("base", vi, float(a))
            key_i = ("instruct", vi, float(a))
            if key_b in stored:
                scores_b[i] = stored[key_b]
            else:
                rng_a = np.random.default_rng(cfg["seed"] + 100 + vi)
                scores_b[i] = ST.score_value_under_steering(
                    base, vi, vv_base.vectors[vi], float(a), cfg, rng_a
                )
                sweep_ckpt.append(list(key_b), {"score": float(scores_b[i])})
                stored[key_b] = float(scores_b[i])
            if key_i in stored:
                scores_i[i] = stored[key_i]
            else:
                rng_a = np.random.default_rng(cfg["seed"] + 200 + vi)
                scores_i[i] = ST.score_value_under_steering(
                    instruct, vi, vv_instr.vectors[vi], float(a), cfg, rng_a
                )
                sweep_ckpt.append(list(key_i), {"score": float(scores_i[i])})
                stored[key_i] = float(scores_i[i])

        slope_b, _ = S.linear_slope(alphas, scores_b)
        slope_i, _ = S.linear_slope(alphas, scores_i)
        inertia[vi] = slope_b - slope_i
        logging.info(
            f"  v{vi:2d} {vname:32s}  slope_base={slope_b:+.4f}  slope_instr={slope_i:+.4f}  "
            f"inertia={inertia[vi]:+.4f}"
        )
        rows.append({
            "value_index": vi,
            "value_name": vname,
            "value_angle_deg": float(value_angles_19[vi]),
            "angular_distance_to_attractor_deg": float(angular_distances[vi]),
            "slope_base": float(slope_b),
            "slope_instruct": float(slope_i),
            "inertia": float(inertia[vi]),
            "scores_base": scores_b.tolist(),
            "scores_instruct": scores_i.tolist(),
        })

    obs_r2, A, phi = S.cosine_fit_r2(value_angles_19, inertia)
    obs_r2_perm, p_perm, _ = S.permutation_p_value(
        value_angles_19, inertia, cfg["stats"]["n_permutations"], rng
    )
    logging.info(f"\nCosine fit: R^2 = {obs_r2:.3f}, A = {A:.3f}, phi = {phi:.1f} deg")
    logging.info(f"Permutation null over {cfg['stats']['n_permutations']} shuffles: p = {p_perm:.4f}")
    logging.info(f"Attractor angle (from B1 logic): {attractor_angle:.1f} deg, fit phase: {phi:.1f} deg")

    PASS = (obs_r2 > 0) and (p_perm < 0.05)
    logging.info(f"DECISION B3: {'PASS' if PASS else 'FAIL'}")

    results = {
        "models": {"base": base.name, "instruct": instruct.name},
        "layer": cfg["steering"]["layer"],
        "alphas": alphas.tolist(),
        "value_angles_deg": value_angles_19.tolist(),
        "attractor_angle_deg": float(attractor_angle),
        "per_value": rows,
        "cosine_fit": {
            "r2": float(obs_r2),
            "amplitude": float(A),
            "phase_deg": float(phi),
            "permutation_p_value": float(p_perm),
            "n_permutations": cfg["stats"]["n_permutations"],
        },
        "decision_rule_passed": bool(PASS),
    }
    U.save_json(results, out_dir / "results.json")
    logging.info(f"Wrote results to {out_dir}")


if __name__ == "__main__":
    main()
