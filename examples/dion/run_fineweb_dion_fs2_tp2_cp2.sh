#!/bin/bash
# =============================================================================
# DION Test: TP=2, CP=2, FS=2 (same settings as run_fineweb_dion_distributed.sh)
# =============================================================================
# 8 GPUs = TP(2) × CP(2) × FS(2)
# =============================================================================

set -e
export CUDA_DEVICE_MAX_CONNECTIONS=1

NGPUS=${NGPUS:-8}
MASTER_PORT=${MASTER_PORT:-29500}

# TransformerEngine options (enabled by default)
USE_TE=${USE_TE:-1}
GRAD_FUSION=${GRAD_FUSION:-1}
PERSIST_LN=${PERSIST_LN:-1}
SEQ_PARALLEL=${SEQ_PARALLEL:-0}
TP_COMM_OVERLAP=${TP_COMM_OVERLAP:-0}

if [ "$USE_TE" = "1" ]; then
    TRANSFORMER_IMPL="transformer_engine"
    if [ "$GRAD_FUSION" = "1" ]; then
        GRAD_ACC_FUSION_FLAG=""
    else
        GRAD_ACC_FUSION_FLAG="--no-gradient-accumulation-fusion"
    fi
else
    TRANSFORMER_IMPL="local"
    GRAD_ACC_FUSION_FLAG="--no-gradient-accumulation-fusion"
fi

if [ "$PERSIST_LN" = "1" ]; then
    PERSIST_LN_FLAG=""
else
    PERSIST_LN_FLAG="--no-persist-layer-norm"
fi

if [ "$SEQ_PARALLEL" = "1" ]; then
    SEQ_PARALLEL_FLAG="--sequence-parallel"
else
    SEQ_PARALLEL_FLAG=""
fi

if [ "$TP_COMM_OVERLAP" = "1" ]; then
    if [ "$SEQ_PARALLEL" != "1" ]; then
        SEQ_PARALLEL_FLAG="--sequence-parallel"
    fi
    TP_COMM_OVERLAP_FLAG="--tp-comm-overlap"
else
    TP_COMM_OVERLAP_FLAG=""
fi

# =============================================================================
# Parallelism: TP=2, CP=2, FS=2
# =============================================================================
TP_SIZE=2
CP_SIZE=2
PP_SIZE=1
FS_SIZE=2
RP_SIZE=1

# CP communication type
CP_COMM_TYPE=${CP_COMM_TYPE:-p2p}

# =============================================================================
# DION hyperparameters (same as baseline)
# =============================================================================
DION_MOMENTUM=0.95
DION_RANK_FRACTION=${DION_RANK_FRACTION:-0.25}
DION_SCALAR_OPT="adamw"
DION_BETA1=0.9
DION_BETA2=0.95

# =============================================================================
# Data Configuration
# =============================================================================
DATA_PATH="data/fineweb10B/fineweb_train"
CHECKPOINT_DIR="checkpoints/dion_fs${FS_SIZE}_tp${TP_SIZE}_cp${CP_SIZE}_rank${DION_RANK_FRACTION}"
TENSORBOARD_DIR="tensorboard/dion_fs${FS_SIZE}_tp${TP_SIZE}_cp${CP_SIZE}_rank${DION_RANK_FRACTION}"

# =============================================================================
# Model Configuration (124M GPT)
# =============================================================================
NUM_LAYERS=12
HIDDEN_SIZE=768
NUM_ATTENTION_HEADS=6
SEQ_LENGTH=1024

# =============================================================================
# Training Configuration (same as baseline)
# =============================================================================
GLOBAL_BATCH_SIZE=512
MICRO_BATCH_SIZE=64
TRAIN_ITERS=${TRAIN_ITERS:-19531}

LR=0.02
WEIGHT_DECAY=0.01
LR_WARMUP_ITERS=${LR_WARMUP_ITERS:-195}
LR_WSD_DECAY_ITERS=${LR_WSD_DECAY_ITERS:-3906}

# =============================================================================
# Logging
# =============================================================================
LOG_INTERVAL=10
EVAL_INTERVAL=250
SAVE_INTERVAL=500

mkdir -p ${CHECKPOINT_DIR}
mkdir -p ${TENSORBOARD_DIR}

echo "============================================"
echo "DION: TP=${TP_SIZE}, CP=${CP_SIZE}, FS=${FS_SIZE}"
echo "rank_fraction=${DION_RANK_FRACTION}"
echo "CP comm type=${CP_COMM_TYPE}"
echo "Transformer: ${TRANSFORMER_IMPL}"
echo "============================================"

torchrun --nproc_per_node=${NGPUS} --master_port=${MASTER_PORT} pretrain_gpt.py \
    --tensor-model-parallel-size ${TP_SIZE} \
    --pipeline-model-parallel-size ${PP_SIZE} \
    --context-parallel-size ${CP_SIZE} \
    --cp-comm-type ${CP_COMM_TYPE} \
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
    --ckpt-format torch \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --norm-epsilon 1e-5 \
    --squared-relu \
    --qk-layernorm \
    --disable-bias-linear \
    --use-flash-attn \
    --transformer-impl ${TRANSFORMER_IMPL} \
    ${PERSIST_LN_FLAG} \
    ${GRAD_ACC_FUSION_FLAG} \
    ${SEQ_PARALLEL_FLAG} \
    ${TP_COMM_OVERLAP_FLAG} \
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
    --no-one-logger \
    "$@"
