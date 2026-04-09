#!/usr/bin/env bash
# Example nohup launcher. Pick ONE of the scenarios below; they are examples,
# not something you run as-is.
#
# Every launch needs:
#   1. CUDA_VISIBLE_DEVICES — the physical GPU ids this process may use
#   2. CONFIG — path to a YAML config with matching gpu_ids (relative ids!) and
#              a UNIQUE logging.out_dir so parallel launches don't collide
#   3. A log file in logs/
#
# Pilots are RESUMABLE: if killed by OOM / SIGKILL / nohup hangup, re-running
# the same command picks up where it left off. No duplicate forward passes.
#
# Tail a running job:   tail -f logs/<tag>.out
# Kill a running job:   pkill -f 'CONFIG=configs/<tag>.yaml'
#
# -----------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs configs

usage() {
    echo "Usage: $0 <scenario>"
    echo "Scenarios:"
    echo "  qwen15       Qwen2.5-1.5B pair on 1 GPU       (cheap pilot)"
    echo "  llama8       Llama-3.1-8B pair on 2 GPUs"
    echo "  gemma9       Gemma-2-9B pair on 4 GPUs"
    echo "  gemma27      Gemma-2-27B pair on 6 GPUs"
    echo "  llama70      Llama-3.1-70B-Instruct ONLY (Pilot 5 only) on 6 GPUs"
    echo "  parallel     Llama-8B pair AND Gemma-9B pair simultaneously (6 GPUs)"
    exit 1
}
[[ $# -lt 1 ]] && usage
SCENARIO=$1

# The launch helper: backgrounds a run under nohup with a named log.
launch() {
    local tag=$1 gpus=$2 config=$3
    echo "Launching $tag on GPUs $gpus with $config"
    CUDA_VISIBLE_DEVICES=$gpus CONFIG=$config PYTHONUNBUFFERED=1 \
        nohup bash scripts/run_gated.sh \
        > "logs/$tag.out" 2> "logs/$tag.err" &
    echo "  pid=$! log=logs/$tag.out"
}

case "$SCENARIO" in
    qwen15)
        # Uses the default config.yaml (Qwen2.5-1.5B, gpu_ids=[0]).
        launch qwen15 0 config.yaml
        ;;

    llama8)
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
        launch llama8 0,1 configs/llama8.yaml
        ;;

    gemma9)
        cat > configs/gemma9.yaml <<'YAML'
seed: 0
models:
  base:     {name: "google/gemma-2-9b",    dtype: bfloat16, gpu_ids: [0, 1]}
  instruct: {name: "google/gemma-2-9b-it", dtype: bfloat16, gpu_ids: [2, 3]}
  max_memory_per_gpu: "75GiB"
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
        launch gemma9 0,1,2,3 configs/gemma9.yaml
        ;;

    gemma27)
        cat > configs/gemma27.yaml <<'YAML'
seed: 0
models:
  base:     {name: "google/gemma-2-27b",    dtype: bfloat16, gpu_ids: [0, 1, 2]}
  instruct: {name: "google/gemma-2-27b-it", dtype: bfloat16, gpu_ids: [3, 4, 5]}
  max_memory_per_gpu: "75GiB"
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
        launch gemma27 0,1,2,3,4,5 configs/gemma27.yaml
        ;;

    llama70)
        # 70B: instruct only, Pilot 5 only (B-track needs two 70Bs = 12 GPUs).
        cat > configs/llama70.yaml <<'YAML'
seed: 0
models:
  base:     {name: "meta-llama/Llama-3.1-70B",          dtype: bfloat16, gpu_ids: [0, 1, 2, 3, 4, 5]}
  instruct: {name: "meta-llama/Llama-3.1-70B-Instruct", dtype: bfloat16, gpu_ids: [0, 1, 2, 3, 4, 5]}
  max_memory_per_gpu: "75GiB"
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
        echo "WARNING: 70B B-track loads TWO 70Bs sequentially on the same 6 GPUs."
        echo "         Base is loaded first, then unloaded (GC), then instruct."
        echo "         That's not what the current loader does — it loads both at once."
        echo "         For 70B use Pilot 5 only (instruct only). Edit scripts accordingly."
        launch llama70 0,1,2,3,4,5 configs/llama70.yaml
        ;;

    parallel)
        # Two independent experiments in parallel on disjoint GPU sets.
        # Total: 6 GPUs (2 for Llama-8B pair, 4 for Gemma-9B pair).
        $0 llama8                              # uses GPUs 0,1 in its own CUDA_VISIBLE_DEVICES
        # Remap Gemma-9B launch to physical GPUs 2,3,4,5:
        cat > configs/gemma9_par.yaml <<'YAML'
seed: 0
models:
  base:     {name: "google/gemma-2-9b",    dtype: bfloat16, gpu_ids: [0, 1]}
  instruct: {name: "google/gemma-2-9b-it", dtype: bfloat16, gpu_ids: [2, 3]}
  max_memory_per_gpu: "75GiB"
data: {valueeval_root: "/tmp", split: "training-english"}
extraction: {layer: 21, max_length: 128, aggregation: "mean", n_per_value: 200,
             activation_dtype: "float32", batch_size: 4}
steering: {alphas: [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0], layer: 21,
           pilot_value_indices: [0, 4, 8, 12, 16]}
pvq: {options: ["1","2","3","4","5","6"], n_prompt_seeds: 5, max_new_tokens: 1,
      use_chat_template_for_instruct: true}
neutral_prompts: {n: 50, max_new_tokens: 32}
stats: {n_bootstrap: 1000, n_permutations: 1000, ci: 0.95}
logging: {level: "INFO", out_dir: "outputs/gemma9_par"}
YAML
        launch gemma9_par 2,3,4,5 configs/gemma9_par.yaml
        ;;

    *)
        usage
        ;;
esac

echo
echo "All jobs launched in background. Check with:"
echo "  ps -ef | grep run_gated"
echo "  tail -f logs/*.out"
