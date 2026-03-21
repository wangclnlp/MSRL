#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:-OpenGVLab/InternVL3_5-8B}
MODEL_TYPE=${MODEL_TYPE:-internvl3_5}
DATASET=${DATASET:-data/stage1_sft.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-output/msrl/stage1_sft}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
MAX_LENGTH=${MAX_LENGTH:-4096}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-3}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-2}
PER_DEVICE_EVAL_BATCH_SIZE=${PER_DEVICE_EVAL_BATCH_SIZE:-2}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-128}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
SAVE_STEPS=${SAVE_STEPS:-200}
EVAL_STEPS=${EVAL_STEPS:-200}
LOGGING_STEPS=${LOGGING_STEPS:-5}
DATASET_NUM_PROC=${DATASET_NUM_PROC:-4}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-4}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-zero2}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-$(( GLOBAL_BATCH_SIZE / (NPROC_PER_NODE * PER_DEVICE_TRAIN_BATCH_SIZE) ))}

PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True} \
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
NPROC_PER_NODE=${NPROC_PER_NODE} \
swift sft \
    --model "${MODEL}" \
    --model_type "${MODEL_TYPE}" \
    --dataset "${DATASET}" \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --freeze_vit true \
    --freeze_aligner true \
    --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
    --learning_rate "${LEARNING_RATE}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --save_steps "${SAVE_STEPS}" \
    --eval_steps "${EVAL_STEPS}" \
    --save_total_limit 3 \
    --logging_steps "${LOGGING_STEPS}" \
    --max_length "${MAX_LENGTH}" \
    --warmup_ratio 0.05 \
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
    --dataset_num_proc "${DATASET_NUM_PROC}" \
    --deepspeed "${DEEPSPEED_CONFIG}" \
    --output_dir "${OUTPUT_DIR}" \
    --save_only_model true
