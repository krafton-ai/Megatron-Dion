#!/bin/bash
# Megatron-LM training script for GPT2-Large based MoE (Dion optimizer)
# Model size: GPT2-2.4B-A1.1B (Qwen3 format)
#
# Parallelism: FS=2, TP=2, EP=2 (8 GPUs)
#
# gpt2-large config:
#   - Model: 36 layers, 20 heads, 1280 hidden
#   - MoE: 8 experts, top-2, every other layer (stride=2)
#   - Training: lr=3e-4, weight_decay=0.1, warmup=2000
#   - Dataset: OpenWebText (~9B tokens)
#
# Parameter calculation:
#   - Dense base: ~774M params
#   - MoE layers: 18 (stride=2)
#   - Expert addition: 18 * 7 * 13.1M ≈ 1.65B
#   - Total: ~2.4B params
#   - Active (top-2): ~1.1B params

set -e
export CUDA_DEVICE_MAX_CONNECTIONS=1

NGPUS=${NGPUS:-8}
MASTER_PORT=${MASTER_PORT:-29500}

# Parallelism: FS=2, TP=2, EP=2 (overridable via env vars)
FS_SIZE=${FS_SIZE:-2}
TP_SIZE=${TP_SIZE:-2}
EP_SIZE=${EP_SIZE:-2}

# ============================================
# GPT2-Large Model Configuration
# ============================================
NUM_LAYERS=36
HIDDEN_SIZE=1280
FFN_HIDDEN_SIZE=5120  # 4 * hidden_size
NUM_ATTENTION_HEADS=20
SEQ_LENGTH=1024

# MoE config
NUM_EXPERTS=8
MOE_ROUTER_TOPK=2
MOE_LAYER_FREQ=2  # stride=2 (every other layer is MoE)
MOE_CAPACITY_FACTOR=1.25

# ============================================
# Training Configuration
# ============================================
# Smaller micro batch due to larger model
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-4}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-256}

# Learning rate (same as Adam for fair comparison)
LR=${LR:-0.0003}
MIN_LR=${MIN_LR:-0.00003}

# Training iterations
TRAIN_ITERS=${TRAIN_ITERS:-50000}
LR_WARMUP_ITERS=${LR_WARMUP_ITERS:-500}      # 1% warmup
LR_WSD_DECAY_ITERS=${LR_WSD_DECAY_ITERS:-10000}  # 20% decay

# Weight decay
WEIGHT_DECAY=0.1

# Auxiliary loss
MOE_AUX_LOSS_COEFF=0.01
MOE_Z_LOSS_COEFF=0.001

# Data path
DATA_PATH=${DATA_PATH:-"data/fineweb10B/fineweb_train"}
VALID_PATH=${VALID_PATH:-"data/fineweb10B/fineweb_train"}

# Tensorboard
TENSORBOARD_DIR=${TENSORBOARD_DIR:-"tensorboard/nanomoe_large_dion_ep${EP_SIZE}"}

# Checkpoint
CHECKPOINT_DIR=${CHECKPOINT_DIR:-"checkpoints/nanomoe_large_dion"}
SAVE_INTERVAL=${SAVE_INTERVAL:-1000}

echo "============================================"
echo "GPT2-Large MoE - Dion Optimizer"
echo "Model: GPT2-2.4B-A1.1B"
echo "============================================"
echo "GPUs: ${NGPUS}"
echo "Parallelism: FS=${FS_SIZE}, TP=${TP_SIZE}, EP=${EP_SIZE}"
echo "Model: ${NUM_LAYERS}L-${HIDDEN_SIZE}H-${NUM_ATTENTION_HEADS}A"
echo "MoE: ${NUM_EXPERTS} experts, top-${MOE_ROUTER_TOPK}, freq=${MOE_LAYER_FREQ}"
echo "Optimizer: Dion + Lion (lr=${LR})"
echo "Batch: micro=${MICRO_BATCH_SIZE}, global=${GLOBAL_BATCH_SIZE}"
echo "Training: ${TRAIN_ITERS} iters, warmup=${LR_WARMUP_ITERS} (1%), decay=${LR_WSD_DECAY_ITERS} (20%)"
echo "LR Schedule: WSD (warmup -> stable -> linear decay)"
echo "============================================"

torchrun --nproc_per_node=${NGPUS} --master_port=${MASTER_PORT} pretrain_gpt.py \
    --fully-shard-model-parallel-size ${FS_SIZE} \
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
    --lr-decay-style WSD \
    --lr-wsd-decay-style linear \
    --lr-wsd-decay-iters ${LR_WSD_DECAY_ITERS} \
    --weight-decay ${WEIGHT_DECAY} \
    --clip-grad ${CLIP_GRAD:-1.0} \
    --optimizer dion \
    --dion-momentum 0.95 \
    --dion-rank-fraction 0.25 \
    --dion-scalar-optimizer adamw \
    --dion-beta1 0.9 \
    --dion-beta2 0.95 \
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
    --eval-iters 100 \
    --save-interval 999999 \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --log-timers-to-tensorboard \
    --log-memory-to-tensorboard \
    --no-one-logger \
    "$@"
