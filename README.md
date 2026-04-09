# value_mech

**Mechanistic interpretability of human values in large language models**

Do LLMs encode human values as geometric structure in their residual stream? This project investigates whether instruction-tuned models develop an interpretable *value subspace* aligned with [Schwartz's circumplex theory](https://doi.org/10.1016/j.jrp.2012.01.006) of 19 universal human values — and whether that structure can be causally steered.

## Research Questions

1. **Attractor geometry** — When an LLM is instruction-tuned, does the resulting behavioral shift (the "compliance attractor") live inside the value subspace, or is it orthogonal to it?
2. **Steering inertia** — Do base and instruct models respond differently to value-steering interventions? Does instruction tuning create measurable *resistance* to value manipulation?
3. **Circumplex law** — Does the inertia pattern follow a smooth cosine curve around the Schwartz circumplex, suggesting a mechanistic (not ad-hoc) alignment structure?
4. **Agent modeling** — Can the residual stream distinguish between values *attributed to the user* vs. *demonstrated by the assistant* in dialogue?

## Method

### Pipeline Architecture

```
ValueEval24 Dataset
        |
        v
  Difference-in-Means ──> 19 Value Vectors (per model)
        |
        ├──> B1: Attractor Projection ── PASS? ──> B2: Steering Slopes ── PASS? ──> B3: Cosine Law
        |                                                                              (full 19-value
        |                                                                               inertia fit)
        └──> Pilot 5: Self/Other Probe (independent)
```

The B-track is a **gated pipeline** — each stage must pass a pre-committed decision rule before the next stage runs. This prevents post-hoc threshold tuning: if a pilot fails, we document the failure and pivot.

### Pilot Experiments

| Pilot | Question | Method | Pass criterion |
|-------|----------|--------|----------------|
| **B1** | Is the attractor inside the value subspace? | Project attractor onto 19-D value basis; bootstrap CI on norm ratio | ratio >= 0.30, CI lower >= 0.20 |
| **B2** | Do steering slopes differ between base/instruct? | Sweep alpha in [-4, 4]; fit linear slopes; bootstrap slope difference | >= 3/5 values: monotone + CI excludes 0 |
| **B3** | Does inertia follow a cosine law? | Fit inertia(theta) = A*cos(theta - phi) + C; permutation null | R^2 > 0, permutation p < 0.05 |
| **5** | Can the model separate user vs. assistant values? | Logistic probe on residual activations; per-value cosine divergence | accuracy > 0.70, median cosine < 0.70 |

### Technical Approach

- **Value vectors**: Extracted via difference-in-means on [ValueEval24](https://touche.webis.de/clef24/touche24-web/value-identification.html) sentences, balanced positive/negative sampling with same-quadrant confound removal
- **Steering**: Activation injection at a pre-committed mid-layer — unit-normalized value vectors scaled by alpha, added to the residual stream via forward hooks
- **Scoring**: PVQ-style Likert items scored via next-token log-probabilities over option tokens, with preamble paraphrase variance reduction
- **Statistics**: Bootstrap CIs (n=1000), permutation nulls (n=1000), Spearman monotonicity, cosine-fit R^2

## Supported Models

| Model family | Parameters | GPUs needed | Status |
|-------------|-----------|-------------|--------|
| Qwen2.5 | 1.5B | 1 | Default pilot config |
| Llama 3.1 | 8B | 2 | Supported |
| Gemma 2 | 9B | 4 | Supported |
| Gemma 2 | 27B | 6 | Supported |
| Llama 3.1 | 70B | 6 | Pilot 5 only |

All models run in bfloat16 with multi-GPU sharding via `device_map="auto"` when needed.

## Quick Start

```bash
# Create environment
conda create -n value_mech python=3.11 -y
conda activate value_mech
pip install -r requirements.txt

# Prepare data
unzip valueeval24.zip -d /tmp

# Run unit tests (no GPU required)
python tests/test_all.py

# Run pilot on single GPU (Qwen2.5-1.5B, ~20 min)
CUDA_VISIBLE_DEVICES=0 bash scripts/run_gated.sh
```

Results are written to `outputs/<pilot_name>/results.json`.

### Multi-GPU / Multi-Experiment

```bash
# Llama-8B pair on 2 GPUs under nohup
bash scripts/launch_nohup.sh llama8
tail -f logs/llama8.out

# Two experiments in parallel across 6 GPUs
bash scripts/launch_nohup.sh parallel

# Resume a killed run (checkpoints pick up automatically)
bash scripts/launch_nohup.sh llama8
```

## Reproducibility Controls

| Concern | Control |
|---------|---------|
| Determinism | Single global seed, `torch.use_deterministic_algorithms`, cuDNN deterministic |
| Tokenizer parity | Asserted identical between base and instruct |
| Layer selection | Pre-committed per model family — never post-hoc |
| Sample balance | Identical N for positive/negative; same sentences for both models |
| Resumability | Atomic `.npy` writes + append-only JSONL with fsync — survives SIGKILL/OOM |
| Statistical rigor | Pre-committed thresholds; bootstrap CIs; permutation nulls; calibration tested in unit tests |

## Project Structure

```
value_mech/
├── config.yaml                  # Hyperparameters and GPU allocation
├── requirements.txt
├── src/
│   ├── data.py                  # Schwartz circumplex, ValueEval24 loader, PVQ items
│   ├── models.py                # Multi-GPU model loading, residual extraction
│   ├── vectors.py               # Value vectors, attractor, subspace projection
│   ├── steering.py              # Activation steering hooks, PVQ scoring
│   ├── stats.py                 # Bootstrap, permutation tests, cosine fitting
│   ├── checkpoint.py            # Crash-safe JSONL + atomic .npy checkpoints
│   ├── utils.py                 # Seeding, config loading, logging
│   ├── pilot_b1_attractor.py    # Attractor projection test
│   ├── pilot_b2_steering.py     # Steering slope comparison
│   ├── pilot_b3_gradient.py     # Full circumplex inertia fit
│   └── pilot_5_self_other.py    # User vs. assistant value encoding
├── scripts/
│   ├── run_gated.sh             # Gated pipeline runner
│   └── launch_nohup.sh          # Multi-GPU parallel launcher
└── tests/
    └── test_all.py              # Unit tests (math, statistics, steering)
```

## License

Research use only.
