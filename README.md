# Schwartz Pilot Experiments

Pilot code for four pre-committed go/no-go tests that decide which of two
research directions to commit to:

- **Pilot B** (Option B / alignment inertia) — gated 3-step pipeline:
  - `pilot_b1_attractor.py` — is the instruct-vs-base "compliance attractor"
    actually inside the Schwartz value subspace, or mostly orthogonal to it?
  - `pilot_b2_steering.py` — for a 5-value subset, do steering slopes differ
    between matched base and instruct checkpoints at all?
  - `pilot_b3_gradient.py` — if B1 and B2 pass, fit the full 19-value
    inertia-vs-angle cosine law with a permutation null.
- **Pilot 5** (agent-modeling pivot) — independent:
  - `pilot_5_self_other.py` — does the residual stream linearly distinguish
    "value attributed to user" from "value performed by assistant"?

## Scientific rigor controls

All knobs live in `config.yaml`. No ad-hoc CLI flags — if you need a new knob,
add it to the config and document it there.

| Concern | Control |
| --- | --- |
| Reproducibility | Single global seed; `torch.use_deterministic_algorithms(True, warn_only=True)`; cuDNN deterministic. |
| Tokenizer parity | Asserted identical between base and instruct. |
| Layer choice | Pre-committed per model family in the config. Sensitivity sweep is a separate experiment, never post-hoc. |
| Token aggregation | Token-averaged residual stream over **non-pad** positions (attention mask applied), computed in float32. |
| Sample parity | DIM uses identical N for positive/negative; same ValueEval sentences for base and instruct (seeded). |
| Chat-template double-BOS | Apply chat template with `tokenize=False`, then re-tokenize with `add_special_tokens=False` so Llama-3's BOS is not doubled. |
| Preamble drift | PVQ prompt-seed rng is re-seeded to the same value-specific seed *per alpha*, so the only thing varying in a steering sweep is alpha. |
| Steering eval | `do_sample=False` (option-logit scoring, not generation); `use_cache=False`; vectors unit-normalized before scaling by alpha. |
| Zero-norm vectors | Skipped with a warning rather than dividing by zero. |
| Device handling | Steering hook queries each layer's device so sharded models work correctly. |
| Statistical claims | Bootstrap CIs; permutation null for cosine fit; calibration tested in the unit-test suite (`mean p ≈ 0.44` under H0). |

## Multi-GPU support

Every model has an explicit `gpu_ids` list in the config. GPU ids are **relative
to `CUDA_VISIBLE_DEVICES`** — if you launch with `CUDA_VISIBLE_DEVICES=4,5,6,7`
and pass `gpu_ids=[0,1]`, the model lands on physical GPUs 4 and 5.

Sizing guide (approximate, bf16):

| Model size | GPUs per model |
| --- | --- |
| ≤ 8B | 1 |
| 9 – 16B | 2 |
| 17 – 35B | 3 |
| 36 – 75B | 6 |

For pilots that load both base and instruct (B1/B2/B3), the two `gpu_ids` lists
must either be disjoint or (for small models) point to the same single GPU.

## Resumability — how killed runs pick up where they stopped

Every expensive computation is checkpointed incrementally:

- **Value vectors** — per-value JSONL checkpoint plus an atomic `.npy`
  fast-path. On resume, fully-cached arrays are loaded once and skipped.
- **Neutral-prompt activations (B1)** — cached as `.npy` files; the bootstrap
  runs over the cache, not re-extracting.
- **Per-alpha scores (B2, B3)** — appended to `sweep.jsonl` as soon as computed,
  fsynced. On resume, preloaded into an O(1) lookup dict.
- **Pilot 5 activations** — appended to `activations.jsonl`, fsynced, preloaded.

Atomic writes: every `.npy` goes to `path.tmp` first, is fsynced, then `os.replace`d.
JSONL appends are flushed and fsynced after every line. A SIGKILL mid-write
leaves either the previous complete file or nothing — never a corrupted file.
If the JSONL itself is mid-line-corrupt, the loader truncates to the last good
line on startup.

## Running — single experiment, single GPU (fastest iteration)

```bash
# 1. Unzip ValueEval24 into /tmp (or set data.valueeval_root in config.yaml)
unzip valueeval24.zip -d /tmp

# 2. Tests first — everything except the steering hook passes without torch
python tests/test_all.py

# 3. Then the pilots (Qwen2.5-1.5B defaults, single GPU)
CUDA_VISIBLE_DEVICES=0 bash scripts/run_gated.sh
```

Results land in `outputs/pilot_b1_attractor/results.json` etc. The decision
rule (pass/fail) is in each results JSON.

## Running — multi-GPU, nohup, one experiment

```bash
# Llama-3.1-8B pair, 2 GPUs, one model per GPU
bash scripts/launch_nohup.sh llama8
# -> writes outputs/llama8/pilot_*/results.json and logs/llama8.out
tail -f logs/llama8.out
```

## Running — multi-GPU, nohup, multiple experiments in parallel

```bash
# Launch Llama-8B pair on GPUs 0,1 AND Gemma-9B pair on GPUs 2,3,4,5
bash scripts/launch_nohup.sh parallel
```

Each experiment has its own config file, its own `logging.out_dir`, and its own
log file in `logs/`. There is no shared state between parallel runs, so nothing
collides. Kill one without affecting the other:

```bash
pkill -f 'CONFIG=configs/llama8.yaml'
```

Resume a killed run by re-launching the same scenario:

```bash
bash scripts/launch_nohup.sh llama8   # picks up where it left off
```

## Decision rules (pre-committed)

| Pilot | Pass iff |
| --- | --- |
| B1 | `projection_norm_ratio ≥ 0.30` AND bootstrap 95% CI lower bound `≥ 0.20` |
| B2 | at least 3/5 pilot values have both monotone steering (`|Spearman ρ| ≥ 0.7`) AND a bootstrap-95%-CI-excluding-zero slope difference between base and instruct |
| B3 | cosine fit `R² > 0` AND permutation `p < 0.05` |
| Pilot 5 | probe held-out accuracy `> 0.70` AND median `cos(v_user, v_asst)` across 19 values `< 0.7` |

**Do not silently weaken any rule.** If a pilot fails, document the failure and
pivot — don't re-tune the threshold after seeing the data.

## Layout

```
schwartz-pilot/
├── README.md
├── requirements.txt
├── config.yaml                     # default config (Qwen2.5-1.5B pair, 1 GPU)
├── configs/                        # generated on-demand by launch_nohup.sh
├── logs/                           # nohup logs
├── outputs/                        # results, checkpoints, cached activations
├── scripts/
│   ├── run_gated.sh                # gated runner (reads $CONFIG)
│   └── launch_nohup.sh             # multi-GPU parallel launch examples
├── src/
│   ├── __init__.py
│   ├── utils.py                    # seeding, config, flushing logging
│   ├── data.py                     # Schwartz angles, ValueEval loader, PVQ items (3p + 1p), neutral prompts
│   ├── checkpoint.py               # JSONL checkpoint + atomic .npy helpers
│   ├── models.py                   # multi-GPU loader, layer-device lookup, residual extraction
│   ├── vectors.py                  # DIM vectors (incremental), attractor, subspace projection, 2D circumplex
│   ├── steering.py                 # device-per-layer steering hook, PVQ option-logit scoring
│   ├── stats.py                    # bootstrap, monotonicity, cosine fit, permutation null
│   ├── pilot_b1_attractor.py
│   ├── pilot_b2_steering.py
│   ├── pilot_b3_gradient.py
│   └── pilot_5_self_other.py
└── tests/
    └── test_all.py                 # math + steering hook unit tests
```
