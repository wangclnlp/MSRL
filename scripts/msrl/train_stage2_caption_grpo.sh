#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:-output/msrl/stage1_grpo}
MODEL_TYPE=${MODEL_TYPE:-internvl3_5}
DATASET=${DATASET:-data/stage2_caption_rl.jsonl}
PLUGIN=${PLUGIN:-scripts/msrl/msrl_grpo_plugin.py}
OUTPUT_DIR=${OUTPUT_DIR:-output/msrl/stage2_caption_grpo}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
MAX_LENGTH=${MAX_LENGTH:-4096}
MAX_COMPLETION_LENGTH=${MAX_COMPLETION_LENGTH:-1024}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-1}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-4}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-128}
LEARNING_RATE=${LEARNING_RATE:-1e-6}
NUM_GENERATIONS=${NUM_GENERATIONS:-8}
SAVE_STEPS=${SAVE_STEPS:-50}
LOGGING_STEPS=${LOGGING_STEPS:-1}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-zero2}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.4}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-8192}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-$(( GLOBAL_BATCH_SIZE / (NPROC_PER_NODE * PER_DEVICE_TRAIN_BATCH_SIZE) ))}
SYSTEM_PROMPT=${SYSTEM_PROMPT:-"You are a multimodal reward model. Predict the task type in <type>...</type>, then reason in <think>...</think>, and finally output the preference in <answer>...</answer>."}

PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True} \
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
NPROC_PER_NODE=${NPROC_PER_NODE} \
swift rlhf \
    --rlhf_type grpo \
    --model "${MODEL}" \
    --model_type "${MODEL_TYPE}" \
    --dataset "${DATASET}" \
    --external_plugins "${PLUGIN}" \
    --reward_funcs msrl_accuracy_reward msrl_format_reward msrl_task_reward \
    --reward_weights 1.0 0.2 0.2 \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
    --vllm_tensor_parallel_size 1 \
    --vllm_max_model_len "${VLLM_MAX_MODEL_LEN}" \
    --sleep_level 1 \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --freeze_vit true \
    --freeze_aligner true \
    --max_length "${MAX_LENGTH}" \
    --max_completion_length "${MAX_COMPLETION_LENGTH}" \
    --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --learning_rate "${LEARNING_RATE}" \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit 5 \
    --logging_steps "${LOGGING_STEPS}" \
    --warmup_ratio 0.0 \
    --num_generations "${NUM_GENERATIONS}" \
    --temperature 1.0 \
    --system "${SYSTEM_PROMPT}" \
    --deepspeed "${DEEPSPEED_CONFIG}" \
    --log_completions true \
    --beta 0.0 \
    --epsilon 0.2 \
    --epsilon_high 0.28 \
    --scale_rewards none \
    --output_dir "${OUTPUT_DIR}" \
    --save_only_model true
