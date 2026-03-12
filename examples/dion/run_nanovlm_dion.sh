#!/bin/bash
# nanoVLM: SmolLM2-360M + SigLIP2-base + Dion Optimizer
#
# Reference: https://github.com/huggingface/nanoVLM
#   - models/config.py: VLMConfig, TrainConfig
#   - train.py: AdamW, cosine schedule, 3% warmup, min_lr=10% of max
#
# Architecture:
#   Vision:  SigLIP2-base (12L, 768H, 12heads, patch=16, img=512)
#   Language: SmolLM2-360M (32L, 960H, 15heads, 5kv, tie_weights=True)
#   Pixel shuffle factor=4 → 64 image tokens
#
# Note: nanoVLM uses per-group LR (MP=0.00512, ViT=5e-5, LM=5e-5).
#   Megatron uses single LR=5e-5 for all parameters.
#
# GBS=16 matches nanoVLM's batch_size=2 * grad_accum=8 on 1 GPU

export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

SOURCE=$(cd "$(dirname "$0")/../.." && pwd)
MODEL_NAME="nanovlm-siglip2base-smollm2"

OUTPUT="${SOURCE}/output/${MODEL_NAME}"
FINETUNE_DIR="${OUTPUT}/checkpoints"
TENSORBOARD_DIR="${OUTPUT}/tensorboard"

CHECKPOINT_DIR=${CHECKPOINT_DIR:-"${SOURCE}/checkpoints/${MODEL_NAME}"}
DATA_TRAIN=${DATA_TRAIN:-"${SOURCE}/examples/dion/finevision_pretrain.yaml"}

NUM_GPUS=${NUM_GPUS:-8}
GBS=${GBS:-16}
MBS=${MBS:-2}
TRAIN_ITERS=${TRAIN_ITERS:-40000}
LR=${LR:-5e-5}
MIN_LR=${MIN_LR:-5e-6}              # 10% of max LR (nanoVLM default)
WARMUP_ITERS=${WARMUP_ITERS:-1200}   # 3% of 40000 (nanoVLM default)
FREEZE_VIT=${FREEZE_VIT:-0}
FREEZE_LM=${FREEZE_LM:-0}

mkdir -p "${FINETUNE_DIR}" "${TENSORBOARD_DIR}"

FREEZE_ARGS=""
[ "$FREEZE_VIT" = "1" ] && FREEZE_ARGS="${FREEZE_ARGS} --freeze-ViT"
[ "$FREEZE_LM" = "1" ] && FREEZE_ARGS="${FREEZE_ARGS} --freeze-LM"

CKPT_ARGS=""
if [ -d "$CHECKPOINT_DIR" ]; then
    CKPT_ARGS="--pretrained-checkpoint ${CHECKPOINT_DIR} --allow-missing-vision-projection-checkpoint"
fi

export NVTE_APPLY_QK_LAYER_SCALING=0
# Allow callers (e.g., sweeps) to force determinism by setting this env var.
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=${NVTE_ALLOW_NONDETERMINISTIC_ALGO:-1}

torchrun --nproc_per_node ${NUM_GPUS} \
    ${SOURCE}/pretrain_vlm.py \
    --num-layers 12 --decoder-num-layers 32 \
    --hidden-size 960 --ffn-hidden-size 2560 \
    --num-attention-heads 15 --group-query-attention --num-query-groups 5 \
    --encoder-hidden-size 768 --encoder-ffn-hidden-size 3072 --encoder-num-attention-heads 12 \
    --seq-length 64 --dataloader-seq-length 4096 --max-position-embeddings 8192 \
    --img-h 512 --img-w 512 --patch-dim 16 \
    --vision-model-type siglip2_base \
    --pixel-shuffle --pixel-shuffle-factor 4 \
    --normalization RMSNorm --swiglu \
    --position-embedding-type rope --rotary-percent 1.0 --rotary-base 100000 \
    --disable-bias-linear \
    --use-flash-attn --transformer-impl transformer_engine --use-te --bf16 \
    --attention-softmax-in-fp32 --no-masked-softmax-fusion \
    --attention-dropout 0.0 --hidden-dropout 0.0 \
    --micro-batch-size ${MBS} --global-batch-size ${GBS} \
    --train-iters ${TRAIN_ITERS} --lr-decay-iters ${TRAIN_ITERS} \
    --lr-warmup-iters ${WARMUP_ITERS} \
    --lr ${LR} --min-lr ${MIN_LR} --lr-decay-style cosine \
    --clip-grad 1.0 --weight-decay 0.01 \
    --adam-beta1 0.9 --adam-beta2 0.95 --init-method-std 0.014 \
    --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 \
    --num-workers 2 --log-interval ${LOG_INTERVAL:-10} \
    --eval-iters 0 --eval-interval 99999 --save-interval 99999 \
    --split 100,0,0 --log-params-norm \
    --tokenizer-type MultimodalTokenizer \
    --tokenizer-model HuggingFaceTB/SmolLM2-360M-Instruct \
    --tokenizer-prompt-format chatml --language-model-type smollm2_360m \
    --distributed-timeout-minutes 60 --ckpt-format torch \
    --optimizer dion \
    --use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather \
    ${FREEZE_ARGS} ${CKPT_ARGS} \
    --dataloader-type external \
    --data-path ${DATA_TRAIN} \
    --prompt-path ${SOURCE}/examples/multimodal/manual_prompts.json \
    "$@"
