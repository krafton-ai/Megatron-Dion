# MCore Dion Speedrun

This directory contains a Megatron-Core launcher for reproducing the `../dion`
NanoGPT speedrun shape through `pretrain_gpt.py`.

The default target is the `../dion/configs/dion_160m.yaml` model and optimizer
configuration, mapped onto the MCore backend:

- GPT: 12 layers, hidden size 768, 6 attention heads, sequence length 1024.
- Optimizer: Dion on transformer matrices; AdamW elementwise path for embedding
  and LM head surfaces. The launcher uses the current MCore Dion defaults
  `rank_fraction=0.25`, `scale_mode=spectral`, and `extra_scale_factor=0.2`.
  Dion momentum/Q/variance states are BF16 by default; QR/Cholesky math is
  promoted to FP32 inside the optimizer.
- Backend: `--use-distributed-optimizer`, `--transformer-impl transformer_engine`,
  `--attention-backend flash`, `--overlap-grad-reduce`, and
  `--overlap-param-gather`.
- Topology: MCore `TP` maps to tensor model parallelism, `FS` maps to
  `--fully-shard-model-parallel-size`, and `RP` maps to
  `--replicate-model-parallel-size`.

The built-in launcher defaults follow the current MCore Dion optimizer defaults.
To mirror `../dion/configs/dion_160m.yaml` exactly, pass the config explicitly:

```bash
python examples/dion/speedrun_nanogpt_mcore.py --config ../dion/configs/dion_160m.yaml
```

For the requested 8-GPU job with `FS=4, TP=2, RP=1`:

```bash
source .venv/bin/activate
python examples/dion/speedrun_nanogpt_mcore.py \
  --fs-size 4 \
  --tp-size 2 \
  --rp-size 1 \
  --dion-rank-fraction 0.25 \
  --dion-scale-mode spectral
```

## FineWeb10B Data

The launcher looks for prepared data under `../data/fineweb10B-gpt2` by default.
To prepare the dataset from the Dion GPT-2 FineWeb10B shards:

```bash
source .venv/bin/activate
python examples/dion/prepare_fineweb10b_mcore.py --num-train-shards 103
```

Pass `--overwrite` only when rebuilding existing Megatron `.bin/.idx` outputs.

This follows `../dion/train.py` split semantics: `fineweb_val_000000.bin` is the
validation set, and `fineweb_train_000001.bin` through
`fineweb_train_000103.bin` are the training set. The resulting prefixes are:

```text
../data/fineweb10B-gpt2/megatron-train/fineweb10B_gpt2_train_text_document
../data/fineweb10B-gpt2/megatron-val/fineweb10B_gpt2_val_text_document
```

When both prefixes exist, `speedrun_nanogpt_mcore.py` uses them automatically
with `--train-data-path` and `--valid-data-path`. If they are absent, the
launcher falls back to `--mock-data`.

For the job UI shown in `../samples.png`, use:

- Name: `dion-test`
- Resource: `cpu=80`, `memory=800Gi`, `gpu=8`
- Workspace Volume: the volume containing `/home/jovyan/workspace/Megatron-Dion`
- Run Command:

```bash
cd /home/jovyan/workspace/Megatron-Dion && \
source .venv/bin/activate && \
python examples/dion/speedrun_nanogpt_mcore.py \
  --fs-size 4 \
  --tp-size 2 \
  --rp-size 1 \
  --nproc-per-node 8 \
  --dion-rank-fraction 0.25 \
  --dion-scale-mode spectral
```

To print the exact `torchrun pretrain_gpt.py ...` command without launching:

```bash
python examples/dion/speedrun_nanogpt_mcore.py --dry-run
```

To override the prepared train/validation prefixes with a custom combined
Megatron indexed dataset:

```bash
python examples/dion/speedrun_nanogpt_mcore.py \
  --no-mock-data \
  --data-path 1.0 /path/to/megatron/indexed/prefix
```

For a local one-GPU smoke test:

```bash
python examples/dion/speedrun_nanogpt_mcore.py \
  --rp-size 1 --fs-size 1 --tp-size 1 --nproc-per-node 1 \
  --n-layer 1 --model-dim 64 --n-head 4 \
  --sequence-length 1024 --batch-size 1 --device-batch-size 1 \
  --num-iterations 1 --val-loss-every 1 --eval-iters 1 \
  --dion-rank-fraction 0.25 --dion-scale-mode spectral
```
