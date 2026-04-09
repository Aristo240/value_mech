#!/usr/bin/env bash
# Wait for PID 1133479 (phase2_steering.py / llama-70B) to finish,
# then launch the Llama-3.1-8B pilot on GPUs 0,1 under nohup.
set -euo pipefail
cd "$(dirname "$0")/.."

TARGET_PID=1133479
echo "Waiting for PID $TARGET_PID (phase2_steering.py) to finish..."
echo "Started waiting at $(date)"

while kill -0 "$TARGET_PID" 2>/dev/null; do
    sleep 60
done

echo "PID $TARGET_PID finished at $(date). Launching Llama-8B pilot..."

# Ensure configs dir exists
mkdir -p configs logs

# Write Llama-8B config
cat > configs/llama8.yaml <<'YAML'
seed: 0
models:
  base:     {name: "meta-llama/Llama-3.1-8B",          dtype: bfloat16, gpu_ids: [0]}
  instruct: {name: "meta-llama/Llama-3.1-8B-Instruct", dtype: bfloat16, gpu_ids: [1]}
  max_memory_per_gpu: "75GiB"
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

# HF_TOKEN must be set in the environment before running this script.
# e.g.: export HF_TOKEN=hf_xxxxx
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN is not set. Export it before running this script."
    exit 1
fi
export PYTHONUNBUFFERED=1

# Launch on GPUs 0,1
CUDA_VISIBLE_DEVICES=0,1 CONFIG=configs/llama8.yaml \
    bash scripts/run_gated.sh \
    > logs/llama8.out 2> logs/llama8.err

echo "Llama-8B pilot finished at $(date)"
echo "Results in outputs/llama8/*/results.json"
