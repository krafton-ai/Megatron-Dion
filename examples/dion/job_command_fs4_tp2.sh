#!/usr/bin/env bash
set -euo pipefail

cd /home/jovyan/workspace/Megatron-Dion
source .venv/bin/activate

python examples/dion/speedrun_nanogpt_mcore.py \
  --fs-size 4 \
  --tp-size 2 \
  --rp-size 1 \
  --nproc-per-node 8 \
  --dion-rank-fraction 0.25 \
  --dion-scale-mode spectral
