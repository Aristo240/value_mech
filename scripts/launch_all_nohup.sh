#!/usr/bin/env bash
# Launch all model runs on 8x V100-32GB GPUs under nohup.
#
# Phase 1 (ALL PARALLEL on 8 GPUs):
#   GPU 0:       Qwen-1.5B re-run (both models, ~3GB, uses cache)
#   GPUs 0,1:    Llama-8B (1 GPU per model; GPU 0 reused after Qwen finishes fast)
#   GPUs 3,4:    Gemma-9B (1 GPU per model)
#   GPUs 2,5,6,7: Gemma-27B (2 GPUs per model)
#
# Phase 2 (after phase 1): Llama-70B Pilot 5 only (6 GPUs)
#
# Usage:
#   nohup bash scripts/launch_all_nohup.sh > logs/master.out 2>&1 &
#
# -------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs configs

export PYTHONUNBUFFERED=1

# HF_TOKEN must be set in the environment before running this script.
# e.g.: export HF_TOKEN=hf_xxxxx
if [ -z "${HF_TOKEN:-}" ]; then
    echo "WARNING: HF_TOKEN not set. Gated models (Llama, Gemma) will fail to download."
    echo "Run: export HF_TOKEN=<your_token>"
fi

launch() {
    local tag=$1 gpus=$2 config=$3
    echo "[$(date '+%H:%M:%S')] Launching $tag on GPUs $gpus"
    CUDA_VISIBLE_DEVICES=$gpus CONFIG=$config PYTHONUNBUFFERED=1 \
        nohup bash scripts/run_gated.sh \
        > "logs/$tag.out" 2> "logs/$tag.err" &
    echo "  pid=$! log=logs/$tag.out"
}

# ===== Phase 1: ALL FOUR models in parallel on all 8 GPUs =====
echo "============================================================"
echo "PHASE 1: All models in parallel"
echo "  GPU 0:       Qwen-1.5B"
echo "  GPU 1:       Llama-8B"
echo "  GPUs 3,4:    Gemma-9B"
echo "  GPUs 2,5,6,7: Gemma-27B"
echo "============================================================"

# --- Qwen-1.5B: both models on GPU 0 (uses cached results, fast re-run) ---
launch qwen15_v2 0 config.yaml

# --- Llama-8B: 1 GPU per model (GPU 0 = base, GPU 1 = instruct) ---
cat > configs/llama8.yaml <<'YAML'
seed: 0
models:
  base:     {name: "meta-llama/Llama-3.1-8B",          dtype: bfloat16, gpu_ids: [0]}
  instruct: {name: "meta-llama/Llama-3.1-8B-Instruct", dtype: bfloat16, gpu_ids: [1]}
  max_memory_per_gpu: "30GiB"
data: {valueeval_root: "/tmp", split: "training-english"}
extraction: {layer: 16, max_length: 128, aggregation: "mean", n_per_value: 200,
             activation_dtype: "float32", batch_size: 8}
steering: {alphas: [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0], layer: 16,
           pilot_value_indices: [0, 4, 8, 12, 16]}
pvq: {options: ["1","2","3","4","5","6"], n_prompt_seeds: 5, max_new_tokens: 1,
      use_chat_template_for_instruct: true}
neutral_prompts: {n: 50, max_new_tokens: 32}
stats: {n_bootstrap: 1000, n_permutations: 1000, ci: 0.95}
logging: {level: "INFO", out_dir: "outputs/llama8"}
YAML
launch llama8 0,1 configs/llama8.yaml

# --- Gemma-9B: 1 GPU per model ---
cat > configs/gemma9.yaml <<'YAML'
seed: 0
models:
  base:     {name: "google/gemma-2-9b",    dtype: bfloat16, gpu_ids: [0]}
  instruct: {name: "google/gemma-2-9b-it", dtype: bfloat16, gpu_ids: [1]}
  max_memory_per_gpu: "30GiB"
data: {valueeval_root: "/tmp", split: "training-english"}
extraction: {layer: 21, max_length: 128, aggregation: "mean", n_per_value: 200,
             activation_dtype: "float32", batch_size: 4}
steering: {alphas: [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0], layer: 21,
           pilot_value_indices: [0, 4, 8, 12, 16]}
pvq: {options: ["1","2","3","4","5","6"], n_prompt_seeds: 5, max_new_tokens: 1,
      use_chat_template_for_instruct: true}
neutral_prompts: {n: 50, max_new_tokens: 32}
stats: {n_bootstrap: 1000, n_permutations: 1000, ci: 0.95}
logging: {level: "INFO", out_dir: "outputs/gemma9"}
YAML
launch gemma9 3,4 configs/gemma9.yaml

# --- Gemma-27B: 2 GPUs per model ---
cat > configs/gemma27.yaml <<'YAML'
seed: 0
models:
  base:     {name: "google/gemma-2-27b",    dtype: bfloat16, gpu_ids: [0, 1]}
  instruct: {name: "google/gemma-2-27b-it", dtype: bfloat16, gpu_ids: [2, 3]}
  max_memory_per_gpu: "30GiB"
data: {valueeval_root: "/tmp", split: "training-english"}
extraction: {layer: 23, max_length: 128, aggregation: "mean", n_per_value: 200,
             activation_dtype: "float32", batch_size: 2}
steering: {alphas: [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0], layer: 23,
           pilot_value_indices: [0, 4, 8, 12, 16]}
pvq: {options: ["1","2","3","4","5","6"], n_prompt_seeds: 5, max_new_tokens: 1,
      use_chat_template_for_instruct: true}
neutral_prompts: {n: 50, max_new_tokens: 32}
stats: {n_bootstrap: 1000, n_permutations: 1000, ci: 0.95}
logging: {level: "INFO", out_dir: "outputs/gemma27"}
YAML
launch gemma27 2,5,6,7 configs/gemma27.yaml

echo "All Phase 1 jobs launched. Waiting..."
wait
echo "[$(date '+%H:%M:%S')] Phase 1 complete."

# ===== Phase 2: Llama-70B Pilot 5 only (6 GPUs) =====
echo "============================================================"
echo "PHASE 2: Llama-70B Pilot 5 only (GPUs 0-5)"
echo "============================================================"

cat > configs/llama70.yaml <<'YAML'
seed: 0
models:
  base:     {name: "meta-llama/Llama-3.1-70B",          dtype: bfloat16, gpu_ids: [0, 1, 2, 3, 4, 5]}
  instruct: {name: "meta-llama/Llama-3.1-70B-Instruct", dtype: bfloat16, gpu_ids: [0, 1, 2, 3, 4, 5]}
  max_memory_per_gpu: "30GiB"
data: {valueeval_root: "/tmp", split: "training-english"}
extraction: {layer: 40, max_length: 128, aggregation: "mean", n_per_value: 100,
             activation_dtype: "float32", batch_size: 1}
steering: {alphas: [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0], layer: 40,
           pilot_value_indices: [0, 4, 8, 12, 16]}
pvq: {options: ["1","2","3","4","5","6"], n_prompt_seeds: 5, max_new_tokens: 1,
      use_chat_template_for_instruct: true}
neutral_prompts: {n: 50, max_new_tokens: 32}
stats: {n_bootstrap: 500, n_permutations: 500, ci: 0.95}
logging: {level: "INFO", out_dir: "outputs/llama70"}
YAML

echo "Launching Llama-70B Pilot 5 only..."
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 CONFIG=configs/llama70.yaml PYTHONUNBUFFERED=1 \
    nohup python -m src.pilot_5_self_other \
    > "logs/llama70_pilot5.out" 2> "logs/llama70_pilot5.err" &
LLAMA70_PID=$!
echo "  pid=$LLAMA70_PID log=logs/llama70_pilot5.out"

echo "Waiting for Phase 2..."
wait $LLAMA70_PID

# Pilot 5 control uses cached activations (CPU-only, fast)
echo "Running Llama-70B Pilot 5 control..."
CONFIG=configs/llama70.yaml python -m src.pilot_5_control \
    >> "logs/llama70_pilot5.out" 2>> "logs/llama70_pilot5.err"

echo "============================================================"
echo "[$(date '+%H:%M:%S')] ALL PHASES COMPLETE"
echo "============================================================"
echo "Results:"
find outputs -name "results.json" -type f | sort
