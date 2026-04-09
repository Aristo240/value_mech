"""Model loading and residual-stream activation extraction.

Multi-GPU: a model can be placed on a single GPU or sharded across several via
HuggingFace's `device_map="auto"` with a `max_memory` dict restricting which
GPUs it uses. Two models loaded in the same process must use disjoint GPU sets
(or share a single GPU if both are small enough).

Hook hygiene: every hook is installed via a context manager that removes it
in `finally`. Do not install hooks any other way.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


@dataclass
class LoadedModel:
    name: str
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase | None
    is_instruct: bool
    layers: nn.ModuleList
    hidden_size: int
    input_device: torch.device  # where input_ids must be moved to


def _dtype_from_str(s: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[s]


def load_model(
    name: str,
    dtype: str,
    gpu_ids: list[int],
    max_memory_per_gpu: str,
    is_instruct: bool,
) -> LoadedModel:
    """Load a model onto an explicit set of GPUs.

    If len(gpu_ids) == 1 the model is placed on that single GPU with .to().
    Otherwise it is sharded via device_map="auto" restricted to the given GPUs.

    gpu_ids are RELATIVE to CUDA_VISIBLE_DEVICES — if you set
    `CUDA_VISIBLE_DEVICES=4,5,6,7` and pass `gpu_ids=[0,1]`, the model will land
    on physical GPUs 4 and 5.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available; multi-GPU loading requires CUDA.")
    for g in gpu_ids:
        if g >= torch.cuda.device_count():
            raise ValueError(
                f"gpu_id {g} out of range (torch sees {torch.cuda.device_count()} GPUs). "
                f"Check CUDA_VISIBLE_DEVICES."
            )

    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    torch_dtype = _dtype_from_str(dtype)
    if len(gpu_ids) == 1:
        g = gpu_ids[0]
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch_dtype)
        model = model.to(f"cuda:{g}").eval()
    else:
        max_memory = {g: max_memory_per_gpu for g in gpu_ids}
        max_memory["cpu"] = "8GiB"                  # overflow safety
        model = AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype=torch_dtype,
            device_map="auto",
            max_memory=max_memory,
            low_cpu_mem_usage=True,
        ).eval()

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    else:
        raise RuntimeError(f"Unsupported architecture for {name}: cannot find .model.layers")

    # Input embedding device is where tokenized inputs must go.
    try:
        emb = model.get_input_embeddings()
        input_device = next(emb.parameters()).device
    except Exception:
        input_device = next(model.parameters()).device

    logging.info(
        f"Loaded {name}: dtype={dtype}, gpu_ids={gpu_ids}, input_device={input_device}, "
        f"hidden_size={model.config.hidden_size}, n_layers={len(layers)}"
    )
    return LoadedModel(
        name=name, model=model, tokenizer=tok, is_instruct=is_instruct,
        layers=layers, hidden_size=model.config.hidden_size, input_device=input_device,
    )


def layer_device(loaded: LoadedModel, layer_idx: int) -> torch.device:
    """Device of a specific transformer layer (matters when the model is sharded)."""
    return next(loaded.layers[layer_idx].parameters()).device


def assert_tokenizer_match(a: LoadedModel, b: LoadedModel) -> None:
    assert a.tokenizer is not None and b.tokenizer is not None
    assert a.tokenizer.vocab_size == b.tokenizer.vocab_size, (
        f"Tokenizer vocab size mismatch: {a.name}={a.tokenizer.vocab_size} "
        f"vs {b.name}={b.tokenizer.vocab_size}"
    )
    sample = "The quick brown fox jumps over the lazy dog. 1234567890."
    ids_a = a.tokenizer(sample, add_special_tokens=False).input_ids
    ids_b = b.tokenizer(sample, add_special_tokens=False).input_ids
    assert ids_a == ids_b, (
        f"Tokenizers encode differently:\n  {a.name}: {ids_a}\n  {b.name}: {ids_b}"
    )
    logging.info(f"Tokenizer parity OK between {a.name} and {b.name}")


@contextmanager
def capture_residual(loaded: LoadedModel, layer_idx: int):
    captured: list[torch.Tensor | None] = [None]

    def hook(_module, _inputs, outputs):
        h = outputs[0] if isinstance(outputs, tuple) else outputs
        captured[0] = h.detach()

    handle = loaded.layers[layer_idx].register_forward_hook(hook)
    try:
        yield captured
    finally:
        handle.remove()


def aggregate_token_activations(
    h: torch.Tensor,
    attention_mask: torch.Tensor,
    method: str,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """(batch, seq, hidden) -> (batch, hidden), masking pad tokens. Output cast to out_dtype."""
    h = h.to(out_dtype)
    mask = attention_mask.to(dtype=out_dtype, device=h.device)
    if method == "mean":
        masked = h * mask.unsqueeze(-1)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return masked.sum(dim=1) / denom
    elif method == "last":
        lengths = mask.sum(dim=1).long() - 1
        idx = lengths.clamp_min(0)
        return h[torch.arange(h.size(0), device=h.device), idx]
    else:
        raise ValueError(f"Unknown aggregation: {method}")


@torch.no_grad()
def extract_activations(
    loaded: LoadedModel,
    texts: list[str],
    layer_idx: int,
    aggregation: str,
    max_length: int,
    batch_size: int = 8,
    out_dtype: torch.dtype = torch.float32,
) -> np.ndarray:
    """Run texts through the model and return aggregated residual-stream
    activations as a (N, hidden) float32 numpy array."""
    out_chunks: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        enc = loaded.tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True,
            max_length=max_length, add_special_tokens=True,
        )
        enc = {k: v.to(loaded.input_device) for k, v in enc.items()}
        with capture_residual(loaded, layer_idx) as cap:
            _ = loaded.model(**enc, use_cache=False)
        h = cap[0]
        agg = aggregate_token_activations(h, enc["attention_mask"], aggregation, out_dtype)
        out_chunks.append(agg.to("cpu").numpy().astype(np.float32))
    return np.concatenate(out_chunks, axis=0)
