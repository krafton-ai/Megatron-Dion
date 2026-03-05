#!/bin/bash
# =============================================================================
# nanoGPT 124M Dense + AdamW
# =============================================================================
#
# Model: 124M GPT (12 layers, 768 hidden, 6 heads)
# Data: Fineweb 10B tokens (GPT-2 tokenized)
# Optimizer: AdamW (beta1=0.9, beta2=0.95) with distributed optimizer
# LR Schedule: WSD (Warmup-Stable-Decay)
#
# Default: TP=2, PP=1, CP=1 (8 GPUs, DP=4)
#
# Usage:
#   bash examples/dion/run_nanogpt_adam.sh
#   TP_SIZE=4 bash examples/dion/run_nanogpt_adam.sh
#   CP_SIZE=2 bash examples/dion/run_nanogpt_adam.sh  # auto-enables TE
#
# =============================================================================

set -e

# Required for Tensor Parallelism
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Number of GPUs
NGPUS=${NGPUS:-8}
MASTER_PORT=${MASTER_PORT:-29500}

# =============================================================================
# Parallelism Configuration
# =============================================================================
TP_SIZE=${TP_SIZE:-2}
PP_SIZE=${PP_SIZE:-1}
CP_SIZE=${CP_SIZE:-1}

# TransformerEngine option (USE_TE=1 for TE)
USE_TE=${USE_TE:-1}

# CP requires TE
if [ "$CP_SIZE" -gt 1 ]; then
    USE_TE=1
fi

if [ "$USE_TE" = "1" ]; then
    TRANSFORMER_IMPL="transformer_engine"
    echo "Using TransformerEngine"
else
    TRANSFORMER_IMPL="local"
    echo "Using local transformer impl (no TE)"
fi

# =============================================================================
# Data Configuration
# =============================================================================
DATA_PATH="data/fineweb10B/fineweb_train"
CHECKPOINT_DIR="checkpoints/nanogpt_adam"
TENSORBOARD_DIR="tensorboard/nanogpt_adam"

# =============================================================================
# Model Configuration (124M GPT)
# =============================================================================
NUM_LAYERS=12
HIDDEN_SIZE=768
NUM_ATTENTION_HEADS=6
SEQ_LENGTH=1024

# =============================================================================
# Training Configuration
# =============================================================================
GLOBAL_BATCH_SIZE=512
MICRO_BATCH_SIZE=64
TRAIN_ITERS=19531         # 10B tokens / (512 * 1024)

# AdamW optimizer
LR=0.02
WEIGHT_DECAY=0.01
ADAM_BETA1=0.9
ADAM_BETA2=0.95

# LR Schedule: warmup (1%) -> stable -> decay (20%)
LR_WARMUP_ITERS=195       # 0.01 * 19531
LR_WSD_DECAY_ITERS=3906   # 0.2 * 19531

# =============================================================================
# Logging
# =============================================================================
LOG_INTERVAL=10
EVAL_INTERVAL=250
SAVE_INTERVAL=500

# =============================================================================
# Run Training
# =============================================================================
mkdir -p ${CHECKPOINT_DIR}
mkdir -p ${TENSORBOARD_DIR}

echo "============================================"
echo "nanoGPT 124M Dense + AdamW"
echo "============================================"
echo "GPUs: ${NGPUS}"
echo "Parallelism: TP=${TP_SIZE}, PP=${PP_SIZE}, CP=${CP_SIZE}, DP=$((NGPUS / (TP_SIZE * PP_SIZE * CP_SIZE)))"
echo "Model: ${NUM_LAYERS}L-${HIDDEN_SIZE}H-${NUM_ATTENTION_HEADS}A (124M)"
echo "Batch: ${GLOBAL_BATCH_SIZE} global, ${MICRO_BATCH_SIZE} micro"
echo "LR: ${LR}, WD: ${WEIGHT_DECAY}"
echo "Iterations: ${TRAIN_ITERS}"
echo "Transformer: ${TRANSFORMER_IMPL}"
echo "============================================"

torchrun --nproc_per_node=${NGPUS} --master_port=${MASTER_PORT} pretrain_gpt.py \
    --tensor-model-parallel-size ${TP_SIZE} \
    --pipeline-model-parallel-size ${PP_SIZE} \
    --context-parallel-size ${CP_SIZE} \
    --num-layers ${NUM_LAYERS} \
    --hidden-size ${HIDDEN_SIZE} \
    --num-attention-heads ${NUM_ATTENTION_HEADS} \
    --seq-length ${SEQ_LENGTH} \
    --max-position-embeddings ${SEQ_LENGTH} \
    --micro-batch-size ${MICRO_BATCH_SIZE} \
    --global-batch-size ${GLOBAL_BATCH_SIZE} \
    --train-iters ${TRAIN_ITERS} \
    --lr ${LR} \
    --min-lr 0.0 \
    --lr-warmup-iters ${LR_WARMUP_ITERS} \
    --lr-decay-style WSD \
    --lr-wsd-decay-style linear \
    --lr-wsd-decay-iters ${LR_WSD_DECAY_ITERS} \
    --weight-decay ${WEIGHT_DECAY} \
    --clip-grad 1.0 \
    --optimizer adam \
    --adam-beta1 ${ADAM_BETA1} \
    --adam-beta2 ${ADAM_BETA2} \
    --adam-eps 1e-8 \
    --use-distributed-optimizer \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --norm-epsilon 1e-5 \
    --squared-relu \
    --qk-layernorm \
    --disable-bias-linear \
    --hidden-dropout 0.0 \
    --attention-dropout 0.0 \
    --use-flash-attn \
    --transformer-impl ${TRANSFORMER_IMPL} \
    --no-data-sharding \
    --data-path ${DATA_PATH} \
    --split 100,0,0 \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model gpt2 \
    --bf16 \
    --log-interval ${LOG_INTERVAL} \
    --eval-interval ${EVAL_INTERVAL} \
    --eval-iters 0 \
    --save-interval ${SAVE_INTERVAL} \
    --save ${CHECKPOINT_DIR} \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --log-timers-to-tensorboard \
    --log-memory-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    --no-one-logger \
    "$@"
