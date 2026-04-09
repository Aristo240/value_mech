"""Determinism, device, config loading, and logging.

Importing this module does not have side effects beyond defining functions;
call set_seed() and load_config() explicitly.
"""
from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    """Best-effort full determinism. Some CUDA ops are still nondeterministic
    even with this; we use warn_only=True so the script doesn't crash, but the
    warnings should be reviewed in the logs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load config from `path`, or from $CONFIG env var, or from ./config.yaml."""
    if path is None:
        path = os.environ.get("CONFIG", "config.yaml")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    assert cfg["extraction"]["layer"] == cfg["steering"]["layer"], (
        "Pilot requires extraction and steering layers to match. "
        "Decoupling them is a separate experiment."
    )
    assert cfg["extraction"]["aggregation"] in ("mean", "last")
    return cfg


def get_device(cfg: dict[str, Any]) -> torch.device:
    requested = cfg["models"].get("device", "cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def setup_logging(cfg: dict[str, Any], run_name: str) -> Path:
    """Configure logging with an immediate-flush file handler so nohup-friendly
    log files are readable in real time."""
    out_dir = Path(cfg["logging"]["out_dir"]) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"

    class FlushingFileHandler(logging.FileHandler):
        def emit(self, record):
            super().emit(record)
            self.flush()

    class FlushingStreamHandler(logging.StreamHandler):
        def emit(self, record):
            super().emit(record)
            self.flush()

    logging.basicConfig(
        level=cfg["logging"].get("level", "INFO"),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[FlushingFileHandler(log_path), FlushingStreamHandler()],
        force=True,
    )
    with open(out_dir / "config_snapshot.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out_dir


def save_json(obj: Any, path: Path) -> None:
    """JSON dump that handles numpy types."""
    def default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Not JSON serializable: {type(o)}")
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=default)
