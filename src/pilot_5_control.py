"""Pilot 5 Control: Shuffled-speaker null for the self/other probe.

Reuses the activations already cached by pilot_5_self_other. Instead of the
true speaker labels, we randomly permute which (value, statement) pairs are
assigned to "user" vs "assistant" while keeping the actual activation vectors
fixed. This tests whether the probe's accuracy is driven by template-position
features (which survive shuffling) versus genuine value-speaker content (which
does not).

If the control probe accuracy remains high (~1.0), the original Pilot 5a result
is confounded by positional artifacts. If it drops toward 0.5, the original
signal reflects value-specific content.

The per-value cosine (Test 5b) is recomputed under the shuffle as well.

Run:
  python -m src.pilot_5_control
"""
from __future__ import annotations

import json
import logging

import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from . import data as D
from . import utils as U
from .checkpoint import JSONLCheckpoint


N_SHUFFLES = 20  # run 20 independent shuffles and report mean


def main():
    cfg = U.load_config()
    out_dir = U.setup_logging(cfg, "pilot_5_control")

    # Load cached activations from the original Pilot 5 run.
    orig_dir = out_dir.parent / "pilot_5_self_other"
    act_path = orig_dir / "activations.jsonl"
    if not act_path.exists():
        raise FileNotFoundError(
            f"Run pilot_5_self_other first to generate {act_path}"
        )

    act_ckpt = JSONLCheckpoint(act_path)
    rows = []
    for key, val in act_ckpt.items():
        rows.append({
            "value_index": int(key[0]),
            "statement_index": int(key[1]),
            "turn": str(key[2]),
            "activation": np.asarray(val["activation"], dtype=np.float32),
        })
    logging.info(f"Loaded {len(rows)} cached activations from {act_path}")

    X = np.stack([r["activation"] for r in rows], axis=0).astype(np.float32)
    y_turn_true = np.array([0 if r["turn"] == "user" else 1 for r in rows], dtype=np.int64)
    y_value = np.array([r["value_index"] for r in rows], dtype=np.int64)
    y_statement = np.array([
        f"{r['value_index']}_{r['statement_index']}" for r in rows
    ])

    # --- Control Test: shuffle speaker labels ---
    # For each shuffle, randomly reassign user/assistant labels per (value, statement)
    # pair, then run the same probe and cosine analysis.
    rng = np.random.RandomState(cfg["seed"])
    shuffle_accs = []
    shuffle_median_cos = []

    for s in range(N_SHUFFLES):
        # Shuffle: for each unique (value, statement), randomly assign one copy
        # to user and one to assistant. This preserves the 50/50 balance.
        y_turn_shuf = y_turn_true.copy()
        unique_pairs = set()
        for r in rows:
            unique_pairs.add((r["value_index"], r["statement_index"]))
        for vi, si in unique_pairs:
            mask = [(r["value_index"] == vi and r["statement_index"] == si) for r in rows]
            indices = np.where(mask)[0]
            if len(indices) == 2:
                labels = [0, 1] if rng.rand() < 0.5 else [1, 0]
                y_turn_shuf[indices[0]] = labels[0]
                y_turn_shuf[indices[1]] = labels[1]

        # Probe accuracy under shuffled labels
        unique_stmts = np.unique(y_statement)
        stmt_value = np.array([int(s.split("_")[0]) for s in unique_stmts])
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=cfg["seed"] + s)
        fold_accs = []
        for fold, (tr_idx, te_idx) in enumerate(skf.split(unique_stmts, stmt_value)):
            tr_stmts = set(unique_stmts[tr_idx])
            te_stmts = set(unique_stmts[te_idx])
            tr_mask = np.array([st in tr_stmts for st in y_statement])
            te_mask = np.array([st in te_stmts for st in y_statement])
            clf = make_pipeline(
                StandardScaler(with_mean=True, with_std=True),
                LogisticRegressionCV(
                    Cs=np.logspace(-3, 1, 5),
                    cv=3,
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=5000,
                    scoring="accuracy",
                ),
            )
            clf.fit(X[tr_mask], y_turn_shuf[tr_mask])
            acc = float(clf.score(X[te_mask], y_turn_shuf[te_mask]))
            fold_accs.append(acc)
        mean_acc = float(np.mean(fold_accs))
        shuffle_accs.append(mean_acc)

        # Per-value cosine under shuffled labels
        cos_vals = []
        for vi in range(len(D.SCHWARTZ_19)):
            user_in  = (y_value == vi) & (y_turn_shuf == 0)
            user_out = (y_value != vi) & (y_turn_shuf == 0)
            asst_in  = (y_value == vi) & (y_turn_shuf == 1)
            asst_out = (y_value != vi) & (y_turn_shuf == 1)
            if user_in.sum() == 0 or asst_in.sum() == 0:
                continue
            v_user = X[user_in].mean(axis=0) - X[user_out].mean(axis=0)
            v_asst = X[asst_in].mean(axis=0) - X[asst_out].mean(axis=0)
            denom = (np.linalg.norm(v_user) * np.linalg.norm(v_asst)) + 1e-12
            cos_vals.append(float(np.dot(v_user, v_asst) / denom))
        shuffle_median_cos.append(float(np.median(cos_vals)))

        logging.info(
            f"  shuffle {s:2d}: probe_acc={mean_acc:.3f}  "
            f"median_cos={shuffle_median_cos[-1]:.3f}"
        )

    mean_shuffle_acc = float(np.mean(shuffle_accs))
    std_shuffle_acc = float(np.std(shuffle_accs))
    mean_shuffle_cos = float(np.mean(shuffle_median_cos))

    logging.info(f"Shuffled probe accuracy: {mean_shuffle_acc:.3f} +/- {std_shuffle_acc:.3f}")
    logging.info(f"Shuffled median cosine:  {mean_shuffle_cos:.3f}")

    # Load original results for comparison
    orig_results_path = orig_dir / "results.json"
    with open(orig_results_path) as f:
        orig = json.load(f)
    orig_acc = orig["test_5a_probe_accuracy_mean"]
    orig_cos = orig["test_5b_median_cos"]

    acc_drop = orig_acc - mean_shuffle_acc
    logging.info(f"Original probe accuracy: {orig_acc:.3f}")
    logging.info(f"Accuracy drop under shuffle: {acc_drop:.3f}")
    logging.info(
        f"Interpretation: {'Probe likely detects VALUE-SPEAKER content (drop > 0.2)' if acc_drop > 0.2 else 'Probe may be detecting POSITIONAL ARTIFACTS (drop <= 0.2)'}"
    )

    results = {
        "n_shuffles": N_SHUFFLES,
        "original_probe_accuracy": orig_acc,
        "original_median_cos": orig_cos,
        "shuffled_probe_accuracy_mean": mean_shuffle_acc,
        "shuffled_probe_accuracy_std": std_shuffle_acc,
        "shuffled_probe_accuracy_all": shuffle_accs,
        "shuffled_median_cos_mean": mean_shuffle_cos,
        "shuffled_median_cos_all": shuffle_median_cos,
        "accuracy_drop": acc_drop,
        "interpretation": (
            "Large accuracy drop under speaker-label shuffle confirms the probe "
            "detects value-speaker content, not just template-position features."
            if acc_drop > 0.2
            else "Small accuracy drop suggests the probe may be detecting "
            "template-position artifacts rather than value-speaker content."
        ),
    }
    U.save_json(results, out_dir / "results.json")
    logging.info(f"Wrote results to {out_dir}")


if __name__ == "__main__":
    main()
