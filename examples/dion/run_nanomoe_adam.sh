#!/bin/bash
# Megatron-LM training script matching nanoMoE configuration (Adam optimizer)
# Reference: nanoMoE config/train_nano_moe.py
#
# Parallelism: TP=2, EP=2, DP=2 (8 GPUs)
#
# nanoMoE config:
#   - Model: 6 layers, 6 heads, 384 hidden (small GPT)
#   - MoE: 8 experts, top-2, every other layer (stride=2)
#   - Training: lr=6e-4, weight_decay=0.1, warmup=2000, max_iters=50000
#   - Batch: ~491,520 tokens per step
#   - Dataset: Fineweb 10B tokens
#
# Usage:
#   bash examples/dion/run_nanomoe_adam.sh

set -e
export CUDA_DEVICE_MAX_CONNECTIONS=1

NGPUS=${NGPUS:-8}
MASTER_PORT=${MASTER_PORT:-29500}

# Parallelism: TP=2, EP=2, DP=2 (overridable via env vars)
TP_SIZE=${TP_SIZE:-2}
EP_SIZE=${EP_SIZE:-2}
# DP = 8 / (TP * EP) = 8 / 4 = 2

# ============================================
# nanoMoE Model Configuration
# ============================================
NUM_LAYERS=6
HIDDEN_SIZE=384
FFN_HIDDEN_SIZE=1536  # 4 * hidden_size
NUM_ATTENTION_HEADS=6
SEQ_LENGTH=1024

# MoE config
NUM_EXPERTS=8
MOE_ROUTER_TOPK=2
MOE_LAYER_FREQ=2  # stride=2 (every other layer is MoE)
MOE_CAPACITY_FACTOR=1.25  # nanoMoE: train_capacity=1.25

# ============================================
# Training Configuration
# ============================================
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-12}  # nanoMoE: batch_size=12
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-480}

# Learning rate (nanoMoE: 6e-4)
LR=${LR:-0.0006}
MIN_LR=${MIN_LR:-0.00006}

# Training iterations
TRAIN_ITERS=${TRAIN_ITERS:-50000}
LR_WARMUP_ITERS=${LR_WARMUP_ITERS:-2000}
LR_DECAY_ITERS=${LR_DECAY_ITERS:-50000}

# Weight decay (nanoMoE: 0.1)
WEIGHT_DECAY=0.1

# Auxiliary loss (nanoMoE: aux_loss_weight=0.01, router_z_loss_weight=0.001)
MOE_AUX_LOSS_COEFF=0.01
MOE_Z_LOSS_COEFF=0.001

# Data path
DATA_PATH=${DATA_PATH:-"data/fineweb10B/fineweb_train"}
VALID_PATH=${VALID_PATH:-"data/fineweb10B/fineweb_train"}

# Tensorboard
TENSORBOARD_DIR=${TENSORBOARD_DIR:-"tensorboard/nanomoe_adam_ep${EP_SIZE}"}

echo "============================================"
echo "nanoMoE Config - Adam Optimizer"
echo "============================================"
echo "GPUs: ${NGPUS}"
echo "Parallelism: TP=${TP_SIZE}, EP=${EP_SIZE}, DP=2"
echo "Model: ${NUM_LAYERS}L-${HIDDEN_SIZE}H-${NUM_ATTENTION_HEADS}A"
echo "MoE: ${NUM_EXPERTS} experts, top-${MOE_ROUTER_TOPK}, freq=${MOE_LAYER_FREQ}"
echo "Optimizer: Adam (lr=${LR})"
echo "Batch: micro=${MICRO_BATCH_SIZE}, global=${GLOBAL_BATCH_SIZE}"
echo "Training: ${TRAIN_ITERS} iters, warmup=${LR_WARMUP_ITERS}"
echo "============================================"

torchrun --nproc_per_node=${NGPUS} --master_port=${MASTER_PORT} pretrain_gpt.py \
    --tensor-model-parallel-size ${TP_SIZE} \
    --expert-model-parallel-size ${EP_SIZE} \
    --num-layers ${NUM_LAYERS} \
    --hidden-size ${HIDDEN_SIZE} \
    --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
    --num-attention-heads ${NUM_ATTENTION_HEADS} \
    --seq-length ${SEQ_LENGTH} \
    --max-position-embeddings ${SEQ_LENGTH} \
    --num-experts ${NUM_EXPERTS} \
    --moe-router-topk ${MOE_ROUTER_TOPK} \
    --moe-layer-freq ${MOE_LAYER_FREQ} \
    --moe-grouped-gemm \
    --moe-token-dispatcher-type alltoall \
    --moe-aux-loss-coeff ${MOE_AUX_LOSS_COEFF} \
    --moe-z-loss-coeff ${MOE_Z_LOSS_COEFF} \
    --moe-expert-capacity-factor ${MOE_CAPACITY_FACTOR} \
    --moe-pad-expert-input-to-capacity \
    --disable-bias-linear \
    --hidden-dropout 0.0 \
    --attention-dropout 0.0 \
    --micro-batch-size ${MICRO_BATCH_SIZE} \
    --global-batch-size ${GLOBAL_BATCH_SIZE} \
    --train-iters ${TRAIN_ITERS} \
    --lr ${LR} \
    --min-lr ${MIN_LR} \
    --lr-warmup-iters ${LR_WARMUP_ITERS} \
    --lr-decay-iters ${LR_DECAY_ITERS} \
    --lr-decay-style cosine \
    --weight-decay ${WEIGHT_DECAY} \
    --clip-grad 1.0 \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --use-distributed-optimizer \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --normalization LayerNorm \
    --use-flash-attn \
    --transformer-impl transformer_engine \
    --sequence-parallel \
    --bf16 \
    --train-data-path ${DATA_PATH} \
    --valid-data-path ${VALID_PATH} \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model gpt2 \
    --log-interval 10 \
    --eval-interval 500 \
    --eval-iters 200 \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --log-timers-to-tensorboard \
    --log-memory-to-tensorboard \
    --no-one-logger \
    "$@"
