"""Append-only JSONL checkpointing so pilots resume cleanly after kill/crash/OOM.

Each record is `{"key": <stable-string>, "value": <json-dict>}`. Keys identify
work units (e.g., a (model, value, alpha) triple); values are whatever the work
unit produced. On startup the checkpoint reads all existing records and exposes
a set of done keys so the main loop can skip them.

All writes are flushed + fsynced after every record. That makes this safe
against SIGKILL / OOM-killer at the cost of some I/O per unit. For the pilots
(hundreds to low thousands of units) the I/O is negligible.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterator

import numpy as np


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Not JSON serializable: {type(o).__name__}")


class JSONLCheckpoint:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._done: set[str] = set()
        self._load_existing()

    @staticmethod
    def _serialize_key(key: Any) -> str:
        return json.dumps(key, sort_keys=True, default=str)

    def _load_existing(self) -> None:
        if not self.path.exists():
            return
        with open(self.path) as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    self._done.add(r["key"])
                except (json.JSONDecodeError, KeyError):
                    # A partial write (e.g., SIGKILL mid-flush). Stop there;
                    # rewrite everything up to the last good line.
                    self._truncate_after_last_good_line(lineno - 1)
                    return

    def _truncate_after_last_good_line(self, n_good_lines: int) -> None:
        """Rewrite the JSONL keeping only the first `n_good_lines` records."""
        if not self.path.exists():
            return
        with open(self.path) as f:
            good = []
            for i, line in enumerate(f):
                if i >= n_good_lines:
                    break
                good.append(line)
        with open(self.path, "w") as f:
            f.writelines(good)
            f.flush()
            os.fsync(f.fileno())

    def __contains__(self, key: Any) -> bool:
        return self._serialize_key(key) in self._done

    def __len__(self) -> int:
        return len(self._done)

    def append(self, key: Any, value: dict[str, Any]) -> None:
        k = self._serialize_key(key)
        rec = {"key": k, "value": value}
        line = json.dumps(rec, default=_json_default)
        with open(self.path, "a") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
        self._done.add(k)

    def items(self) -> Iterator[tuple[Any, dict[str, Any]]]:
        """Yield (key_deserialized, value) for every stored record."""
        if not self.path.exists():
            return
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                yield json.loads(r["key"]), r["value"]


def save_array_atomic(path: str | Path, arr: np.ndarray) -> None:
    """Write a numpy array atomically: write to tmp, fsync, rename.
    Safe against SIGKILL mid-write — either the old file is intact or the new
    file is complete."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    # np.save appends .npy if the name doesn't end in .npy, so we pass a file
    # object to force it to write exactly where we want.
    with open(tmp, "wb") as f:
        np.save(f, arr, allow_pickle=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_array_if_exists(path: str | Path) -> np.ndarray | None:
    path = Path(path)
    if not path.exists():
        return None
    return np.load(path)
