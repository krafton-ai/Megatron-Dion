#!/bin/bash
# =============================================================================
# nanoGPT 124M Dense + Dion
# =============================================================================
#
# Model: 124M GPT (12 layers, 768 hidden, 6 heads)
# Data: Fineweb 10B tokens (GPT-2 tokenized)
# Optimizer: DION (low-rank orthonormalized updates)
# LR Schedule: WSD (Warmup-Stable-Decay)
#
# Default: TP=2, FS=4, PP=1, CP=1 (8 GPUs)
#
# Usage:
#   bash examples/dion/run_nanogpt_dion.sh
#   TP_SIZE=4 FS_SIZE=2 bash examples/dion/run_nanogpt_dion.sh
#   CP_SIZE=2 bash examples/dion/run_nanogpt_dion.sh  # auto-enables TE
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
FS_SIZE=${FS_SIZE:-4}
RP_SIZE=${RP_SIZE:-1}
PP_SIZE=${PP_SIZE:-1}
CP_SIZE=${CP_SIZE:-1}

# TransformerEngine option (USE_TE=1 for TE, default is local)
USE_TE=${USE_TE:-0}
# Gradient accumulation fusion option (GRAD_FUSION=1 to enable, default is disabled for Dion)
GRAD_FUSION=${GRAD_FUSION:-0}
# Persist layer norm option (PERSIST_LN=1 to enable, default is disabled)
PERSIST_LN=${PERSIST_LN:-0}
# Sequence parallelism option (SEQ_PARALLEL=1 to enable, default is disabled)
SEQ_PARALLEL=${SEQ_PARALLEL:-0}
# TP communication overlap option (TP_COMM_OVERLAP=1 to enable, requires SEQ_PARALLEL=1)
TP_COMM_OVERLAP=${TP_COMM_OVERLAP:-0}

# CP requires TE
if [ "$CP_SIZE" -gt 1 ]; then
    USE_TE=1
fi

if [ "$USE_TE" = "1" ]; then
    TRANSFORMER_IMPL="transformer_engine"
    if [ "$GRAD_FUSION" = "1" ]; then
        GRAD_ACC_FUSION_FLAG=""  # Enable gradient_accumulation_fusion
        echo "Using TransformerEngine WITH gradient_accumulation_fusion"
    else
        GRAD_ACC_FUSION_FLAG="--no-gradient-accumulation-fusion"
        echo "Using TransformerEngine WITHOUT gradient_accumulation_fusion"
    fi
else
    TRANSFORMER_IMPL="local"
    GRAD_ACC_FUSION_FLAG="--no-gradient-accumulation-fusion"
    echo "Using local transformer impl (no TE)"
fi

if [ "$PERSIST_LN" = "1" ]; then
    PERSIST_LN_FLAG=""
    echo "Using persist-layer-norm (enabled)"
else
    PERSIST_LN_FLAG="--no-persist-layer-norm"
    echo "persist-layer-norm disabled"
fi

if [ "$SEQ_PARALLEL" = "1" ]; then
    SEQ_PARALLEL_FLAG="--sequence-parallel"
    echo "Sequence parallelism enabled"
else
    SEQ_PARALLEL_FLAG=""
    echo "Sequence parallelism disabled"
fi

if [ "$TP_COMM_OVERLAP" = "1" ]; then
    if [ "$SEQ_PARALLEL" != "1" ]; then
        echo "WARNING: --tp-comm-overlap requires --sequence-parallel, enabling it automatically"
        SEQ_PARALLEL_FLAG="--sequence-parallel"
    fi
    TP_COMM_OVERLAP_FLAG="--tp-comm-overlap"
    echo "TP communication overlap enabled"
else
    TP_COMM_OVERLAP_FLAG=""
    echo "TP communication overlap disabled"
fi

# =============================================================================
# DION-specific hyperparameters
# =============================================================================
DION_MOMENTUM=0.95        # Error feedback momentum (mu)
DION_RANK_FRACTION=${DION_RANK_FRACTION:-0.25}  # Low-rank fraction r/d
DION_SCALAR_OPT="adamw"   # Optimizer for 1D params
DION_BETA1=0.9            # Beta1 for scalar optimizer
DION_BETA2=0.95           # Beta2 for scalar optimizer

# =============================================================================
# Data Configuration
# =============================================================================
DATA_PATH="data/fineweb10B/fineweb_train"
CHECKPOINT_DIR="checkpoints/nanogpt_dion_rank${DION_RANK_FRACTION}"
TENSORBOARD_DIR="tensorboard/nanogpt_dion_rank${DION_RANK_FRACTION}"

# =============================================================================
# Model Configuration (124M GPT)
# =============================================================================
NUM_LAYERS=16
HIDDEN_SIZE=768
NUM_ATTENTION_HEADS=6
SEQ_LENGTH=1024

# =============================================================================
# Training Configuration
# =============================================================================
GLOBAL_BATCH_SIZE=512
MICRO_BATCH_SIZE=64
TRAIN_ITERS=19531         # 10B tokens / (512 * 1024)

# DION optimizer
LR=0.02
WEIGHT_DECAY=0.01

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
echo "nanoGPT 124M Dense + Dion"
echo "============================================"
echo "GPUs: ${NGPUS}"
echo "Parallelism: TP=${TP_SIZE}, FS=${FS_SIZE}, RP=${RP_SIZE}, PP=${PP_SIZE}, CP=${CP_SIZE}"
echo "Model: ${NUM_LAYERS}L-${HIDDEN_SIZE}H-${NUM_ATTENTION_HEADS}A (124M)"
echo "Batch: ${GLOBAL_BATCH_SIZE} global, ${MICRO_BATCH_SIZE} micro"
echo "LR: ${LR}, WD: ${WEIGHT_DECAY}"
echo "DION: mu=${DION_MOMENTUM}, rank_frac=${DION_RANK_FRACTION}"
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
    --optimizer dion \
    --dion-momentum ${DION_MOMENTUM} \
    --dion-rank-fraction ${DION_RANK_FRACTION} \
    --dion-scalar-optimizer ${DION_SCALAR_OPT} \
    --dion-beta1 ${DION_BETA1} \
    --dion-beta2 ${DION_BETA2} \
    --fully-shard-model-parallel-size ${FS_SIZE} \
    --replicate-model-parallel-size ${RP_SIZE} \
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
    ${PERSIST_LN_FLAG} \
    ${GRAD_ACC_FUSION_FLAG} \
    ${SEQ_PARALLEL_FLAG} \
    ${TP_COMM_OVERLAP_FLAG} \
    --no-data-sharding \
    --data-path ${DATA_PATH} \
    --split 100,0,0 \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model gpt2 \
    --bf16 \
    --log-interval ${LOG_INTERVAL} \
    --eval-interval ${EVAL_INTERVAL} \
    --eval-iters 0 \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --log-timers-to-tensorboard \
    --log-memory-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    --no-one-logger \
    "$@"
