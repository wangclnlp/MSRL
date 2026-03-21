#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:-output/msrl/stage3_multimodal_grpo}
EVAL_DATASET=${EVAL_DATASET:-data/eval.jsonl}
INFER_BACKEND=${INFER_BACKEND:-vllm}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
MAX_PIXELS=${MAX_PIXELS:-1003520}
MAX_TOKENS=${MAX_TOKENS:-1024}
TEMPERATURE=${TEMPERATURE:-0.0}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.9}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-8192}
EVAL_NUM_PROC=${EVAL_NUM_PROC:-1}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
MAX_PIXELS=${MAX_PIXELS} \
swift eval \
    --model "${MODEL}" \
    --eval_dataset "${EVAL_DATASET}" \
    --eval_backend Native \
    --infer_backend "${INFER_BACKEND}" \
    --eval_generation_config "{\"max_tokens\": ${MAX_TOKENS}, \"temperature\": ${TEMPERATURE}, \"do_sample\": false}" \
    --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
    --vllm_max_model_len "${VLLM_MAX_MODEL_LEN}" \
    --eval_num_proc "${EVAL_NUM_PROC}"
