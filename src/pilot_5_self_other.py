"""Pilot 5: Does the residual stream linearly distinguish "value attributed to
the user" from "value performed by the assistant"?

Design: for each Schwartz value V and each of 3 first-person statements, build
two chat dialogues that contain the SAME content, differing only in which
speaker utters it. Extract residual activations at the last token of the
statement (user-turn end vs assistant-turn end).

Decision rule (pre-committed):
  PASS if Test 5a held-out probe accuracy > 0.70 AND median cos(v_user, v_asst)
  across the 19 values < 0.7.

Resumable: each (value, statement, turn) activation is saved to activations.jsonl.

Run:
  python -m src.pilot_5_self_other
"""
from __future__ import annotations

import logging

import numpy as np
import torch
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from . import data as D
from . import models as M
from . import utils as U
from .checkpoint import JSONLCheckpoint


def _build_user_dialogue(loaded: M.LoadedModel, statement: str) -> str:
    msgs = [
        {"role": "user", "content": statement},
        {"role": "assistant", "content": "I see."},
    ]
    return loaded.tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=False
    )


def _build_assistant_dialogue(loaded: M.LoadedModel, statement: str) -> str:
    msgs = [
        {"role": "user", "content": "Tell me about yourself."},
        {"role": "assistant", "content": statement},
    ]
    return loaded.tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=False
    )


def _last_token_activation(
    loaded: M.LoadedModel,
    text: str,
    target_substring: str,
    layer_idx: int,
) -> np.ndarray:
    """Extract residual activation at the LAST token of `target_substring`.

    The target must occur exactly once in `text`. The chat-templated text
    already contains special tokens, so we tokenize with add_special_tokens=False
    to avoid a double-BOS on models like Llama-3."""
    if not loaded.tokenizer.is_fast:
        raise RuntimeError(
            f"Pilot 5 requires a fast tokenizer (offset_mapping). "
            f"Tokenizer for {loaded.name} is slow."
        )
    if text.count(target_substring) != 1:
        raise RuntimeError(
            f"target_substring must occur exactly once in text "
            f"(found {text.count(target_substring)} times)"
        )
    char_start = text.index(target_substring)
    char_end = char_start + len(target_substring)

    enc = loaded.tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=512,
        add_special_tokens=False,   # chat template already includes BOS where relevant
    )
    offsets = enc.pop("offset_mapping")[0].tolist()
    enc = {k: v.to(loaded.input_device) for k, v in enc.items()}

    last_idx = -1
    for i, (s, e) in enumerate(offsets):
        if s == 0 and e == 0:                          # special tokens
            continue
        if s < char_end and e > char_start:
            last_idx = i
    if last_idx < 0:
        raise RuntimeError(
            f"Could not align substring in tokenized text:\n  text={text!r}\n  target={target_substring!r}"
        )

    with M.capture_residual(loaded, layer_idx) as cap:
        with torch.no_grad():
            _ = loaded.model(**enc, use_cache=False)
    h = cap[0]                                         # (1, S, H)
    return h[0, last_idx].to("cpu", dtype=torch.float32).numpy()


def main():
    cfg = U.load_config()
    U.set_seed(cfg["seed"])
    out_dir = U.setup_logging(cfg, "pilot_5_self_other")

    # Pilot 5 only needs the instruct model.
    instruct = M.load_model(
        cfg["models"]["instruct"]["name"], cfg["models"]["instruct"]["dtype"],
        gpu_ids=cfg["models"]["instruct"]["gpu_ids"],
        max_memory_per_gpu=cfg["models"]["max_memory_per_gpu"],
        is_instruct=True,
    )
    layer = cfg["extraction"]["layer"]

    # Incremental checkpoint of activations. Preload into a dict for O(1) lookup.
    act_ckpt = JSONLCheckpoint(out_dir / "activations.jsonl")
    if len(act_ckpt) > 0:
        logging.info(f"Resuming from {len(act_ckpt)} cached activations")
    stored_acts: dict[tuple, np.ndarray] = {}
    for key, val in act_ckpt.items():
        stored_acts[(int(key[0]), int(key[1]), str(key[2]))] = np.asarray(
            val["activation"], dtype=np.float32
        )

    # Build records (compute fresh ones, load cached ones).
    rows = []
    for vi, vname in enumerate(D.SCHWARTZ_19):
        for j, s1 in enumerate(D.PVQ_ITEMS_FIRST_PERSON[vname]):
            for turn in ("user", "assistant"):
                key = (vi, j, turn)
                if key in stored_acts:
                    act = stored_acts[key]
                else:
                    if turn == "user":
                        text = _build_user_dialogue(instruct, s1)
                    else:
                        text = _build_assistant_dialogue(instruct, s1)
                    act = _last_token_activation(instruct, text, s1, layer)
                    act_ckpt.append([vi, j, turn], {
                        "value_name": vname, "statement": s1,
                        "activation": act.tolist(),
                    })
                    stored_acts[key] = act
                rows.append({
                    "value_index": vi, "value_name": vname,
                    "statement_index": j, "statement": s1,
                    "turn": turn, "activation": act,
                })
    logging.info(f"Collected {len(rows)} activations")

    X = np.stack([r["activation"] for r in rows], axis=0).astype(np.float32)
    y_turn = np.array([0 if r["turn"] == "user" else 1 for r in rows], dtype=np.int64)
    y_value = np.array([r["value_index"] for r in rows], dtype=np.int64)
    y_statement = np.array([
        f"{r['value_index']}_{r['statement_index']}" for r in rows
    ])

    # ----- Test 5a: probe accuracy with content held out by STATEMENT -----
    # Probe uses standardization + L2 with cross-validated C (inside train
    # folds only), to avoid hand-tuning regularization.
    unique_stmts = np.unique(y_statement)
    stmt_value = np.array([int(s.split("_")[0]) for s in unique_stmts])
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=cfg["seed"])
    fold_accs = []
    for fold, (tr_stmt_idx, te_stmt_idx) in enumerate(skf.split(unique_stmts, stmt_value)):
        tr_stmts = set(unique_stmts[tr_stmt_idx])
        te_stmts = set(unique_stmts[te_stmt_idx])
        tr_mask = np.array([s in tr_stmts for s in y_statement])
        te_mask = np.array([s in te_stmts for s in y_statement])
        # LogisticRegressionCV selects C internally on the training data only.
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
        clf.fit(X[tr_mask], y_turn[tr_mask])
        acc = float(clf.score(X[te_mask], y_turn[te_mask]))
        fold_accs.append(acc)
        logging.info(f"  fold {fold}: held-out probe accuracy = {acc:.3f}")
    mean_acc = float(np.mean(fold_accs))
    logging.info(f"Mean held-out probe accuracy: {mean_acc:.3f}")

    # ----- Test 5b: per-value vector cosine across positions -----
    cos_per_value = []
    for vi, vname in enumerate(D.SCHWARTZ_19):
        user_in  = (y_value == vi) & (y_turn == 0)
        user_out = (y_value != vi) & (y_turn == 0)
        asst_in  = (y_value == vi) & (y_turn == 1)
        asst_out = (y_value != vi) & (y_turn == 1)
        v_user = X[user_in].mean(axis=0) - X[user_out].mean(axis=0)
        v_asst = X[asst_in].mean(axis=0) - X[asst_out].mean(axis=0)
        denom = (np.linalg.norm(v_user) * np.linalg.norm(v_asst)) + 1e-12
        c = float(np.dot(v_user, v_asst) / denom)
        cos_per_value.append(c)
        logging.info(f"  v{vi:2d} {vname:32s} cos(v_user, v_asst) = {c:+.3f}")
    median_cos = float(np.median(cos_per_value))
    logging.info(f"Median cos(v_user, v_asst) across 19 values: {median_cos:.3f}")

    PASS = (mean_acc > 0.70) and (median_cos < 0.7)
    logging.info(f"DECISION Pilot 5: {'PASS' if PASS else 'FAIL'}")

    results = {
        "model": instruct.name,
        "layer": layer,
        "n_activations": len(rows),
        "test_5a_probe_accuracy_per_fold": fold_accs,
        "test_5a_probe_accuracy_mean": mean_acc,
        "test_5b_cos_per_value": cos_per_value,
        "test_5b_median_cos": median_cos,
        "decision_rule_passed": bool(PASS),
        "caveats": [
            "First-person PVQ-like items are hand-written placeholders, not licensed PVQ-RR items.",
            "Synthetic dialogue templates are minimal and may not represent realistic dialogue.",
            "Test 5a (probe) may succeed by detecting chat-template position features; "
            "Test 5b (per-value cosine) is the substantive claim about value-specific divergence.",
            "With ~90 training activations per fold and ~2000-4000 features, the probe uses "
            "LogisticRegressionCV with inner-fold C selection to avoid severe overfitting.",
        ],
    }
    U.save_json(results, out_dir / "results.json")
    logging.info(f"Wrote results to {out_dir}")


if __name__ == "__main__":
    main()
