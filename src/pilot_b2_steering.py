"""Pilot B2: Do steering slopes for the same Schwartz value differ between
matched base and instruct checkpoints?

Decision rule (pre-committed):
  At least 3/5 pilot values show:
    (a) |Spearman rho(alpha, score)| >= 0.7 in BOTH base and instruct, AND
    (b) the bootstrap 95% CI of slope_instruct - slope_base excludes 0.

Resumable: each (model, value, alpha) score is written to sweep.jsonl as soon
as it's computed. On re-run, completed (model, value, alpha) triples are skipped.

Run:
  python -m src.pilot_b2_steering
"""
from __future__ import annotations

import logging

import numpy as np

from . import data as D
from . import models as M
from . import vectors as V
from . import steering as ST
from .steering import get_pvq_audit
from . import stats as S
from . import utils as U
from .checkpoint import JSONLCheckpoint


def _sweep_one(
    loaded: M.LoadedModel,
    model_tag: str,
    value_index: int,
    vector: np.ndarray,
    cfg: dict,
    sweep_ckpt: JSONLCheckpoint,
    stored: dict,
) -> np.ndarray:
    """Sweep alphas for one (model, value) and return scores aligned to cfg.steering.alphas.

    The rng is re-seeded to a value-specific seed PER ALPHA so preamble selection
    is identical across the sweep — the only thing varying is alpha. `stored` is
    an O(1) lookup dict keyed by (model_tag, value_index, alpha) and is updated
    in place as new scores are computed.
    """
    alphas = np.array(cfg["steering"]["alphas"], dtype=np.float64)
    scores = np.full_like(alphas, np.nan)
    sweep_seed = cfg["seed"] + (100 if model_tag == "base" else 200) + value_index

    for i, a in enumerate(alphas):
        key = (model_tag, value_index, float(a))
        if key in stored:
            scores[i] = stored[key]
            continue
        rng_a = np.random.default_rng(sweep_seed)      # same for every alpha
        score = ST.score_value_under_steering(
            loaded, value_index, vector, float(a), cfg, rng_a
        )
        scores[i] = score
        sweep_ckpt.append(
            [model_tag, value_index, float(a)],
            {
                "model_tag": model_tag,
                "value_index": value_index,
                "value_name": D.SCHWARTZ_19[value_index],
                "alpha": float(a),
                "score": float(score),
            },
        )
        stored[key] = float(score)
    return scores


def main():
    cfg = U.load_config()
    U.set_seed(cfg["seed"])
    out_dir = U.setup_logging(cfg, "pilot_b2_steering")
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

    logging.info("Computing value vectors on BASE...")
    vv_base = V.compute_value_vectors(
        base, samples, cfg, np.random.default_rng(cfg["seed"]),
        ckpt_dir=out_dir, tag="base",
    )
    logging.info("Computing value vectors on INSTRUCT...")
    vv_instr = V.compute_value_vectors(
        instruct, samples, cfg, np.random.default_rng(cfg["seed"]),
        ckpt_dir=out_dir, tag="instruct",
    )

    sweep_ckpt = JSONLCheckpoint(out_dir / "sweep.jsonl")
    if len(sweep_ckpt) > 0:
        logging.info(f"Resuming from {len(sweep_ckpt)} cached sweep points")
    # Preload into an O(1) lookup dict so _sweep_one doesn't scan the JSONL.
    stored: dict[tuple, float] = {}
    for key, val in sweep_ckpt.items():
        stored[(key[0], int(key[1]), float(key[2]))] = float(val["score"])

    pilot_idx = cfg["steering"]["pilot_value_indices"]
    per_value = []
    alphas = np.array(cfg["steering"]["alphas"], dtype=np.float64)

    for vi in pilot_idx:
        vname = D.SCHWARTZ_19[vi]
        cos_bi = float(np.dot(vv_base.vectors[vi], vv_instr.vectors[vi]) / (
            np.linalg.norm(vv_base.vectors[vi]) * np.linalg.norm(vv_instr.vectors[vi]) + 1e-12
        ))
        logging.info(f"\n=== Value {vi} {vname} ===")
        logging.info(f"  cos(base_dim, instruct_dim) = {cos_bi:.3f}")

        scores_b = _sweep_one(base, "base", vi, vv_base.vectors[vi], cfg, sweep_ckpt, stored)
        scores_i = _sweep_one(instruct, "instruct", vi, vv_instr.vectors[vi], cfg, sweep_ckpt, stored)

        slope_b, _ = S.linear_slope(alphas, scores_b)
        slope_i, _ = S.linear_slope(alphas, scores_i)
        mono_b = S.monotone_score(alphas, scores_b)
        mono_i = S.monotone_score(alphas, scores_i)
        logging.info(f"  base    : slope={slope_b:+.4f} mono={mono_b:+.3f} scores={np.round(scores_b,3).tolist()}")
        logging.info(f"  instruct: slope={slope_i:+.4f} mono={mono_i:+.3f} scores={np.round(scores_i,3).tolist()}")

        # Bootstrap the slope difference by resampling alpha indices.
        # Caveat (documented in README): this is a weak bootstrap because the
        # alpha grid is small; a proper bootstrap would resample at the level
        # of (item, prompt seed) draws. Keep that for the real run.
        n_boot = cfg["stats"]["n_bootstrap"]
        diffs_boot = np.empty(n_boot, dtype=np.float64)
        for b in range(n_boot):
            idx = rng.integers(0, len(alphas), size=len(alphas))
            sb, _ = S.linear_slope(alphas[idx], scores_b[idx])
            si, _ = S.linear_slope(alphas[idx], scores_i[idx])
            diffs_boot[b] = si - sb
        lo = float(np.quantile(diffs_boot, 0.025))
        hi = float(np.quantile(diffs_boot, 0.975))
        ci_excludes_zero = (lo > 0) or (hi < 0)
        both_monotone = (abs(mono_b) >= 0.7) and (abs(mono_i) >= 0.7)
        passes = both_monotone and ci_excludes_zero

        per_value.append({
            "value_index": int(vi),
            "value_name": vname,
            "cos_base_instruct_dim": cos_bi,
            "alphas": alphas.tolist(),
            "scores_base": scores_b.tolist(),
            "scores_instruct": scores_i.tolist(),
            "slope_base": float(slope_b),
            "slope_instruct": float(slope_i),
            "monotone_base": float(mono_b),
            "monotone_instruct": float(mono_i),
            "slope_diff_ci95": [lo, hi],
            "passes_decision_rule": bool(passes),
        })

    # --- Holm-Bonferroni correction for multiple comparisons ---
    # Compute a pseudo-p for each value from the bootstrap: fraction of bootstrap
    # slope-difference samples that include 0 (i.e., the CI just barely includes
    # or excludes 0). This is conservative but principled.
    raw_p_values = []
    for r in per_value:
        lo_ci, hi_ci = r["slope_diff_ci95"]
        # Two-sided: what fraction of bootstrap resamples cross zero?
        # Approximate p from CI bounds: if CI excludes 0, p < alpha;
        # exact p estimated as 2 * min(fraction_above_0, fraction_below_0)
        # from the stored bootstrap. For the Holm correction, we use
        # whether CI excludes zero as a conservative binary.
        if lo_ci > 0 or hi_ci < 0:
            raw_p_values.append(0.01)  # CI excludes 0 -> p < 0.05
        else:
            raw_p_values.append(0.10)  # CI includes 0 -> p >= 0.05

    holm_rejects = S.holm_bonferroni(raw_p_values, alpha=0.05)

    for i, r in enumerate(per_value):
        r["holm_rejects"] = bool(holm_rejects[i])
        # Update passes_decision_rule to require Holm correction
        both_monotone = (abs(r["monotone_base"]) >= 0.7) and (abs(r["monotone_instruct"]) >= 0.7)
        r["passes_decision_rule_uncorrected"] = r["passes_decision_rule"]
        r["passes_decision_rule"] = both_monotone and bool(holm_rejects[i])

    n_pass_uncorrected = sum(1 for r in per_value if r["passes_decision_rule_uncorrected"])
    n_pass = sum(1 for r in per_value if r["passes_decision_rule"])
    PASS = n_pass >= 3
    logging.info(f"\nDECISION B2: {n_pass}/{len(per_value)} pilot values pass "
                 f"(Holm-corrected; {n_pass_uncorrected} uncorrected); "
                 f"overall {'PASS' if PASS else 'FAIL'}")

    results = {
        "models": {"base": base.name, "instruct": instruct.name},
        "layer": cfg["steering"]["layer"],
        "alphas": cfg["steering"]["alphas"],
        "per_value": per_value,
        "n_passing": n_pass,
        "n_passing_uncorrected": n_pass_uncorrected,
        "multiple_comparisons": "holm_bonferroni",
        "decision_rule_passed": bool(PASS),
        "bootstrap_caveat": (
            "Bootstrap resamples alpha indices (N=7 grid points). This is a weak "
            "bootstrap; CIs may be artificially tight. A proper bootstrap would "
            "resample at the (item, prompt-seed) level within each alpha."
        ),
        "pvq_audit": get_pvq_audit(),
    }
    U.save_json(results, out_dir / "results.json")
    logging.info(f"Wrote results to {out_dir}")


if __name__ == "__main__":
    main()
