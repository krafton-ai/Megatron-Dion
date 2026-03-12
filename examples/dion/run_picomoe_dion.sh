#!/bin/bash
# Megatron-LM training script for a smaller picoMoE validation model (Dion optimizer)
#
# Parallelism defaults remain overridable via env vars. The topology legality is the
# same as nanoMoE on 8 GPUs, but the model is much smaller for faster validation.

set -e
export CUDA_DEVICE_MAX_CONNECTIONS=1

NGPUS=${NGPUS:-8}
MASTER_PORT=${MASTER_PORT:-29500}

# Parallelism (overridable via env vars)
FS_SIZE=${FS_SIZE:-2}
RP_SIZE=${RP_SIZE:-1}
TP_SIZE=${TP_SIZE:-2}
EP_SIZE=${EP_SIZE:-2}
PP_SIZE=${PP_SIZE:-1}
CP_SIZE=${CP_SIZE:-1}
ENABLE_SP=${ENABLE_SP:-1}

# ============================================
# picoMoE Model Configuration
# ============================================
NUM_LAYERS=${NUM_LAYERS:-8}
HIDDEN_SIZE=128
FFN_HIDDEN_SIZE=512
NUM_ATTENTION_HEADS=8
SEQ_LENGTH=1024

# MoE config
NUM_EXPERTS=8
MOE_ROUTER_TOPK=2
MOE_LAYER_FREQ=2
MOE_CAPACITY_FACTOR=1.25

# ============================================
# Training Configuration
# ============================================
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-12}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-480}

LR=${LR:-0.0006}
MIN_LR=${MIN_LR:-0.00006}

TRAIN_ITERS=${TRAIN_ITERS:-50000}
LR_WARMUP_ITERS=${LR_WARMUP_ITERS:-500}
LR_WSD_DECAY_ITERS=${LR_WSD_DECAY_ITERS:-10000}

WEIGHT_DECAY=0.1
MOE_AUX_LOSS_COEFF=0.01
MOE_Z_LOSS_COEFF=0.001

DATA_PATH=${DATA_PATH:-"data/openwebtext/openwebtext_train_text_document"}
VALID_PATH=${VALID_PATH:-"data/openwebtext/openwebtext_val_text_document"}
TENSORBOARD_DIR=${TENSORBOARD_DIR:-"tensorboard/picomoe_dion_ep${EP_SIZE}"}
LOG_INTERVAL=${LOG_INTERVAL:-10}
EVAL_INTERVAL=${EVAL_INTERVAL:-500}
EVAL_ITERS=${EVAL_ITERS:-200}

echo "============================================"
echo "picoMoE Config - Dion Optimizer"
echo "============================================"
echo "GPUs: ${NGPUS}"
echo "Parallelism: FS=${FS_SIZE}, RP=${RP_SIZE}, TP=${TP_SIZE}, EP=${EP_SIZE}, PP=${PP_SIZE}, CP=${CP_SIZE}"
echo "Sequence Parallel: ${ENABLE_SP}"
echo "Model: ${NUM_LAYERS}L-${HIDDEN_SIZE}H-${NUM_ATTENTION_HEADS}A"
echo "MoE: ${NUM_EXPERTS} experts, top-${MOE_ROUTER_TOPK}, freq=${MOE_LAYER_FREQ}"
echo "Optimizer: Dion (lr=${LR})"
echo "Batch: micro=${MICRO_BATCH_SIZE}, global=${GLOBAL_BATCH_SIZE}"
echo "Training: ${TRAIN_ITERS} iters, warmup=${LR_WARMUP_ITERS} (1%), decay=${LR_WSD_DECAY_ITERS} (20%)"
echo "LR Schedule: WSD (warmup -> stable -> linear decay)"
echo "============================================"

if [ "${PP_SIZE}" -gt "${NUM_LAYERS}" ]; then
    echo "ERROR: invalid PP=${PP_SIZE} for NUM_LAYERS=${NUM_LAYERS}; pipeline stages cannot exceed transformer layers"
    exit 1
fi
if [ $(( NUM_LAYERS % PP_SIZE )) -ne 0 ]; then
    echo "ERROR: invalid NUM_LAYERS=${NUM_LAYERS} for PP=${PP_SIZE}; NUM_LAYERS must be divisible by PP"
    exit 1
fi

SEQ_PARALLEL_ARG=""
if [ "${ENABLE_SP}" = "1" ]; then
    SEQ_PARALLEL_ARG="--sequence-parallel"
fi

torchrun --nproc_per_node=${NGPUS} --master_port=${MASTER_PORT} pretrain_gpt.py \
    --fully-shard-model-parallel-size ${FS_SIZE} \
    --replicate-model-parallel-size ${RP_SIZE} \
    --tensor-model-parallel-size ${TP_SIZE} \
    --pipeline-model-parallel-size ${PP_SIZE} \
    --context-parallel-size ${CP_SIZE} \
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
    --clip-grad 1.0 \
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
    ${SEQ_PARALLEL_ARG} \
    --bf16 \
    --no-data-sharding \
    --train-data-path ${DATA_PATH} \
    --valid-data-path ${VALID_PATH} \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model gpt2 \
    --log-interval ${LOG_INTERVAL} \
    --eval-interval ${EVAL_INTERVAL} \
    --eval-iters ${EVAL_ITERS} \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --log-timers-to-tensorboard \
    --log-memory-to-tensorboard \
    --no-one-logger \
    "$@"
