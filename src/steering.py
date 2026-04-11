"""Steering interventions and PVQ option-logit scoring.

Steering: add α·v to the residual stream at a fixed layer, at every token
position. The vector is moved to THAT LAYER'S device, which matters when the
model is sharded across GPUs.

PVQ scoring: read next-token logits over option-letter token ids, normalize via
log-softmax restricted to those ids, then take the expected score on the Likert
scale.

Chat templates: when an instruct model's chat template output already contains
special tokens (Llama-3's `<|begin_of_text|>`, Qwen's `<|im_start|>`), we
tokenize with `add_special_tokens=False` to avoid a double-BOS.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager

import numpy as np
import torch

from .data import PVQ_ITEMS, SCHWARTZ_19
from .models import LoadedModel, layer_device


ZERO_NORM_EPS = 1e-8


@contextmanager
def steer_layer(loaded: LoadedModel, layer_idx: int, vector: np.ndarray, alpha: float):
    """Install an additive steering hook on `layer_idx`. Adds alpha*vector to
    the residual stream at all token positions for the duration of the context.

    The vector is moved to the device of the hooked layer (not the model's
    first parameter device), which is required for correct behavior with
    sharded models."""
    dev = layer_device(loaded, layer_idx)
    # Query dtype from a layer parameter, not from model.parameters() — this
    # handles sharded models where layers have different dtypes in principle.
    dtype = next(loaded.layers[layer_idx].parameters()).dtype
    v = torch.from_numpy(np.ascontiguousarray(vector)).to(device=dev, dtype=dtype)
    v = v.view(1, 1, -1)

    def hook(_module, _inputs, outputs):
        if isinstance(outputs, tuple):
            h = outputs[0]
            # Defensive: if h somehow lives on a different device from v, move v.
            if h.device != v.device:
                v_local = v.to(h.device)
            else:
                v_local = v
            h_new = h + alpha * v_local
            return (h_new,) + outputs[1:]
        return outputs + alpha * v

    handle = loaded.layers[layer_idx].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


# --- PVQ option-logit scoring ---------------------------------------------

def _option_token_ids(tokenizer, options: list[str]) -> list[int]:
    """Return one token id per option label, tokenizing with a leading space to
    match the format the model will see ("Answer: 4")."""
    ids = []
    for opt in options:
        toks = tokenizer(" " + opt, add_special_tokens=False).input_ids
        if len(toks) != 1:
            logging.warning(
                f"Option '{opt}' tokenizes to {toks}; using the last token id {toks[-1]}."
            )
            ids.append(toks[-1])
        else:
            ids.append(toks[0])
    if len(set(ids)) != len(ids):
        raise RuntimeError(f"Option token ids collide: {list(zip(options, ids))}")
    return ids


PVQ_PROMPT_TEMPLATE = (
    "Read the following description and rate how much the person is like you "
    "on a scale of 1 to 6, where 1 = not like me at all and 6 = very much like me. "
    "Reply with a single digit only.\n\n"
    "Description: {item}\n\n"
    "Answer:"
)


def _build_prompt(loaded: LoadedModel, item_text: str, use_chat: bool) -> tuple[str, bool]:
    """Return (prompt_string, is_chat_templated).
    If is_chat_templated is True, caller must tokenize with add_special_tokens=False
    to avoid a double-BOS (the chat template already includes BOS for models
    like Llama-3)."""
    body = PVQ_PROMPT_TEMPLATE.format(item=item_text)
    if loaded.is_instruct and use_chat:
        msgs = [{"role": "user", "content": body}]
        rendered = loaded.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        return rendered, True
    return body, False


# Global audit counters for PVQ scoring validation.
_pvq_audit = {"total": 0, "top1_in_options": 0, "option_prob_mass": []}


def get_pvq_audit() -> dict:
    """Return PVQ scoring audit stats and reset counters."""
    total = _pvq_audit["total"]
    in_opt = _pvq_audit["top1_in_options"]
    masses = _pvq_audit["option_prob_mass"]
    result = {
        "total_scored": total,
        "top1_in_options": in_opt,
        "top1_in_options_frac": in_opt / total if total > 0 else 0.0,
        "mean_option_prob_mass": float(np.mean(masses)) if masses else 0.0,
        "min_option_prob_mass": float(np.min(masses)) if masses else 0.0,
    }
    _pvq_audit["total"] = 0
    _pvq_audit["top1_in_options"] = 0
    _pvq_audit["option_prob_mass"] = []
    return result


@torch.no_grad()
def score_pvq_item(
    loaded: LoadedModel,
    item_text: str,
    options: list[str],
    use_chat: bool,
    n_prompt_seeds: int,
    rng: np.random.Generator,
) -> float:
    """Return the expected Likert score for one PVQ item.

    Variance reduction: n_prompt_seeds surface paraphrases (neutral preambles
    that don't change semantics); mean of per-seed expected scores."""
    option_ids = _option_token_ids(loaded.tokenizer, options)
    likert_values = np.array([float(o) for o in options], dtype=np.float32)
    option_id_set = set(option_ids)

    preambles = [
        "",
        "Please answer carefully. ",
        "Take a moment to consider. ",
        "Think honestly about this. ",
        "Read this thoughtfully. ",
        "Reflect for a second. ",
        "Be candid in your reply. ",
        "Consider this question. ",
    ]
    if n_prompt_seeds > len(preambles):
        raise ValueError(
            f"n_prompt_seeds={n_prompt_seeds} exceeds available preambles "
            f"({len(preambles)}). Add more preambles or reduce config.pvq.n_prompt_seeds."
        )
    seed_idx = rng.choice(len(preambles), size=n_prompt_seeds, replace=False)

    scores = []
    for si in seed_idx:
        body, is_templated = _build_prompt(loaded, item_text, use_chat)
        prompt = preambles[int(si)] + body
        enc = loaded.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=not is_templated,  # avoid double-BOS after chat template
        )
        enc = {k: v.to(loaded.input_device) for k, v in enc.items()}
        out = loaded.model(**enc, use_cache=False)
        logits = out.logits[0, -1, :].float()

        # --- PVQ logit audit ---
        full_probs = torch.softmax(logits, dim=0)
        top1_id = int(logits.argmax())
        option_mass = float(full_probs[option_ids].sum())
        _pvq_audit["total"] += 1
        _pvq_audit["option_prob_mass"].append(option_mass)
        if top1_id in option_id_set:
            _pvq_audit["top1_in_options"] += 1
        else:
            top1_tok = loaded.tokenizer.decode([top1_id])
            logging.debug(
                f"PVQ audit: top-1 token '{top1_tok}' (id={top1_id}) is outside "
                f"option set; option mass={option_mass:.3f}"
            )

        opt_logits = logits[option_ids]
        log_probs = torch.log_softmax(opt_logits, dim=0).to("cpu").numpy()
        probs = np.exp(log_probs)
        scores.append(float(np.dot(probs, likert_values)))
    return float(np.mean(scores))


@torch.no_grad()
def score_value_under_steering(
    loaded: LoadedModel,
    value_index: int,
    steering_vector: np.ndarray | None,
    alpha: float,
    cfg: dict,
    rng: np.random.Generator,
) -> float:
    """Score one value (3 PVQ items) under a steering intervention.

    Returns the mean expected Likert score across the 3 items. If the steering
    vector has norm < ZERO_NORM_EPS OR alpha == 0, no hook is installed and the
    baseline score is returned (any non-None vector with zero norm is treated
    as "no intervention possible for this value")."""
    items = PVQ_ITEMS[SCHWARTZ_19[value_index]]
    options = cfg["pvq"]["options"]
    use_chat = cfg["pvq"]["use_chat_template_for_instruct"]
    n_seeds = cfg["pvq"]["n_prompt_seeds"]
    layer = cfg["steering"]["layer"]

    vector_norm = float(np.linalg.norm(steering_vector)) if steering_vector is not None else 0.0
    intervene = (
        steering_vector is not None
        and alpha != 0.0
        and vector_norm >= ZERO_NORM_EPS
    )

    item_scores = []
    if not intervene:
        if steering_vector is not None and vector_norm < ZERO_NORM_EPS and alpha != 0.0:
            logging.warning(
                f"Value {SCHWARTZ_19[value_index]}: steering vector has near-zero norm "
                f"({vector_norm:.2e}); skipping intervention for this value."
            )
        for item in items:
            item_scores.append(score_pvq_item(
                loaded, item, options, use_chat, n_seeds, rng
            ))
    else:
        v_unit = steering_vector / vector_norm
        with steer_layer(loaded, layer, v_unit, alpha):
            for item in items:
                item_scores.append(score_pvq_item(
                    loaded, item, options, use_chat, n_seeds, rng
                ))
    return float(np.mean(item_scores))
