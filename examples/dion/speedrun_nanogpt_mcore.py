#!/usr/bin/env python3
"""Launch the ../dion NanoGPT speedrun on the Megatron-Core backend."""

from __future__ import annotations

import argparse
import math
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
FINEWEB10B_DATA_ROOT = REPO_ROOT.parent / "data" / "fineweb10B-gpt2"
FINEWEB10B_TRAIN_PREFIX = (
    FINEWEB10B_DATA_ROOT / "megatron-train" / "fineweb10B_gpt2_train_text_document"
)
FINEWEB10B_VAL_PREFIX = FINEWEB10B_DATA_ROOT / "megatron-val" / "fineweb10B_gpt2_val_text_document"
FINEWEB10B_GPT2_VOCAB = FINEWEB10B_DATA_ROOT / "tokenizer" / "gpt2-vocab.json"
FINEWEB10B_GPT2_MERGE = FINEWEB10B_DATA_ROOT / "tokenizer" / "gpt2-merges.txt"


def _indexed_prefix_exists(prefix: Path) -> bool:
    return prefix.with_suffix(".bin").exists() and prefix.with_suffix(".idx").exists()


def _dion_split_data_exists() -> bool:
    return _indexed_prefix_exists(FINEWEB10B_TRAIN_PREFIX) and _indexed_prefix_exists(
        FINEWEB10B_VAL_PREFIX
    )


DION_160M_DEFAULTS: dict[str, Any] = {
    "model_dim": 768,
    "n_layer": 12,
    "n_head": 6,
    "sequence_length": 1024,
    "batch_size": 1024,
    "device_batch_size": 32,
    "num_iterations": 3000,
    "warmup_ratio": 0.0,
    "warmdown_ratio": 0.2,
    "val_loss_every": 125,
    "val_tokens": 10485760,
    "optimizer": "dion",
    "scalar_opt": "adamw",
    "mu": 0.95,
    "weight_decay": 0.01,
    "ortho_fraction": 0.25,
    "lr": 0.02,
    "mixed_precision": True,
    "rcqr_oversample": 1.25,
    "dion_scale_mode": "spectral",
    "dion_extra_scale_factor": 0.2,
    "vocab_size": 50304,
    "rp_size": 1,
    "fs_size": 4,
    "tp_size": 2,
}


def _load_yaml_defaults(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("--config requires PyYAML to be installed") from exc

    with Path(path).open("r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream) or {}
    if not isinstance(loaded, dict):
        raise SystemExit(f"--config must contain a YAML mapping: {path}")

    result: dict[str, Any] = {}
    for key, value in loaded.items():
        normalized = str(key).replace("-", "_")
        if normalized == "dp_size":
            normalized = "rp_size"
        result[normalized] = value
    return result


def _add_defaulted(
    parser: argparse.ArgumentParser,
    defaults: dict[str, Any],
    name: str,
    *aliases: str,
    **kwargs: Any,
) -> None:
    flag = "--" + name.replace("_", "-")
    underscored = "--" + name
    option_strings = [flag]
    if underscored != flag:
        option_strings.append(underscored)
    option_strings.extend(aliases)
    parser.add_argument(*option_strings, dest=name, default=defaults.get(name), **kwargs)


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--config",
        default=None,
        help="Optional ../dion-style YAML config. CLI values override YAML values.",
    )
    pre_args, _ = pre_parser.parse_known_args()

    defaults = dict(DION_160M_DEFAULTS)
    defaults.update(_load_yaml_defaults(pre_args.config))
    if defaults.get("fs_size") is None:
        defaults["fs_size"] = DION_160M_DEFAULTS["fs_size"]
    if defaults.get("tp_size") is None:
        defaults["tp_size"] = DION_160M_DEFAULTS["tp_size"]
    if defaults.get("rp_size") is None:
        defaults["rp_size"] = DION_160M_DEFAULTS["rp_size"]

    parser = argparse.ArgumentParser(
        parents=[pre_parser],
        description="Run ../dion's NanoGPT speedrun shape via MCore pretrain_gpt.py.",
    )

    _add_defaulted(parser, defaults, "model_dim", type=int)
    _add_defaulted(parser, defaults, "n_layer", type=int)
    _add_defaulted(parser, defaults, "n_head", type=int)
    _add_defaulted(parser, defaults, "sequence_length", type=int)
    _add_defaulted(parser, defaults, "batch_size", type=int)
    _add_defaulted(parser, defaults, "device_batch_size", type=int)
    _add_defaulted(parser, defaults, "num_iterations", type=int)
    _add_defaulted(parser, defaults, "warmup_ratio", type=float)
    _add_defaulted(parser, defaults, "warmdown_ratio", type=float)
    _add_defaulted(parser, defaults, "val_loss_every", type=int)
    _add_defaulted(parser, defaults, "val_tokens", type=int)
    _add_defaulted(parser, defaults, "lr", type=float)
    _add_defaulted(parser, defaults, "mu", type=float)
    _add_defaulted(parser, defaults, "weight_decay", type=float)
    _add_defaulted(parser, defaults, "ortho_fraction", "--dion-rank-fraction", type=float)
    _add_defaulted(parser, defaults, "rcqr_oversample", type=float)
    _add_defaulted(
        parser,
        defaults,
        "dion_scale_mode",
        type=str,
        choices=("spectral", "unit_rms_norm", "shape_scaling"),
    )
    _add_defaulted(parser, defaults, "dion_extra_scale_factor", type=float)
    _add_defaulted(parser, defaults, "vocab_size", type=int)
    _add_defaulted(parser, defaults, "rp_size", type=int)
    _add_defaulted(parser, defaults, "fs_size", type=int)
    _add_defaulted(parser, defaults, "tp_size", type=int)

    parser.add_argument(
        "--optimizer",
        default=defaults.get("optimizer", "dion"),
        choices=("dion",),
        help="Optimizer to launch. This wrapper is scoped to Dion.",
    )
    parser.add_argument(
        "--scalar-opt",
        "--scalar_opt",
        default=defaults.get("scalar_opt", "adamw"),
        choices=("adamw", "lion"),
        help="Elementwise optimizer for embedding/output surfaces.",
    )
    parser.add_argument(
        "--mixed-precision",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults.get("mixed_precision", True)),
        help="Use bfloat16 Dion optimizer state dtypes.",
    )
    parser.add_argument(
        "--mock-data",
        action=argparse.BooleanOptionalAction,
        default=not _dion_split_data_exists(),
        help="Use Megatron mock data instead of the prepared FineWeb10B indexed dataset.",
    )
    parser.add_argument(
        "--data-path",
        nargs="*",
        default=None,
        help="Megatron indexed data blend. Prefer the default Dion-style train/valid split paths.",
    )
    parser.add_argument(
        "--train-data-path",
        nargs="*",
        default=None,
        help="Megatron indexed train data blend. Defaults to prepared ../data FineWeb10B train.",
    )
    parser.add_argument(
        "--valid-data-path",
        nargs="*",
        default=None,
        help="Megatron indexed validation data blend. Defaults to prepared ../data FineWeb10B val.",
    )
    parser.add_argument(
        "--test-data-path",
        nargs="*",
        default=None,
        help="Optional Megatron indexed test data blend.",
    )
    parser.add_argument("--split", default="969,30,1")
    parser.add_argument("--tokenizer-type", default="GPT2BPETokenizer")
    parser.add_argument("--vocab-file", default=str(FINEWEB10B_GPT2_VOCAB))
    parser.add_argument("--merge-file", default=str(FINEWEB10B_GPT2_MERGE))
    parser.add_argument("--eval-iters", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--attention-backend", default="flash")
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--nproc-per-node", type=int, default=None)
    parser.add_argument("--standalone", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--force-te-flash",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Set NVTE_* env vars so TE must use FlashAttention instead of fallback kernels.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--pretrain-script",
        default=str(REPO_ROOT / "pretrain_gpt.py"),
        help="Path to the MCore pretrain_gpt.py entrypoint.",
    )
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Additional pretrain_gpt.py args after '--'.",
    )
    args = parser.parse_args()
    if not args.mock_data and args.data_path is None and _dion_split_data_exists():
        if args.train_data_path is None:
            args.train_data_path = ["1.0", str(FINEWEB10B_TRAIN_PREFIX)]
        if args.valid_data_path is None:
            args.valid_data_path = ["1.0", str(FINEWEB10B_VAL_PREFIX)]
    return args


def _validate_topology(args: argparse.Namespace) -> int:
    for name in ("rp_size", "fs_size", "tp_size", "nnodes"):
        value = getattr(args, name)
        if value is None or value < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be >= 1")

    global_world_size = args.rp_size * args.fs_size * args.tp_size
    if args.nproc_per_node is None:
        if global_world_size % args.nnodes != 0:
            raise SystemExit(
                "RP * FS * TP must be divisible by --nnodes when --nproc-per-node is omitted"
            )
        args.nproc_per_node = global_world_size // args.nnodes

    launched_world_size = args.nproc_per_node * args.nnodes
    if launched_world_size != global_world_size:
        raise SystemExit(
            "Launcher world size must equal RP * FS * TP: "
            f"nproc_per_node={args.nproc_per_node} nnodes={args.nnodes} "
            f"RP={args.rp_size} FS={args.fs_size} TP={args.tp_size}"
        )
    return global_world_size


def _append_data_args(cmd: list[str], args: argparse.Namespace) -> None:
    cmd.extend(
        [
            "--tokenizer-type",
            args.tokenizer_type,
            "--vocab-size",
            str(args.vocab_size),
            "--make-vocab-size-divisible-by",
            "1",
        ]
    )
    if args.tokenizer_type == "GPT2BPETokenizer":
        cmd.extend(["--vocab-file", args.vocab_file, "--merge-file", args.merge_file])
    if args.mock_data:
        cmd.append("--mock-data")
        return
    per_split_paths = [
        ("--train-data-path", args.train_data_path),
        ("--valid-data-path", args.valid_data_path),
        ("--test-data-path", args.test_data_path),
    ]
    if any(path is not None for _, path in per_split_paths):
        for flag, path in per_split_paths:
            if path is not None:
                cmd.extend([flag, *path])
        return
    if not args.data_path:
        raise SystemExit("--no-mock-data requires --data-path with Megatron indexed data")
    cmd.extend(["--data-path", *args.data_path, "--split", args.split])


def build_command(args: argparse.Namespace) -> tuple[list[str], dict[str, str]]:
    _validate_topology(args)
    if args.optimizer != "dion":
        raise SystemExit("This launcher is intentionally scoped to --optimizer dion")

    eval_iters = args.eval_iters
    if eval_iters is None:
        if args.val_loss_every <= 0:
            eval_iters = 0
        else:
            tokens_per_global_batch = args.batch_size * args.sequence_length
            eval_iters = max(1, math.ceil(args.val_tokens / tokens_per_global_batch))

    warmdown_iters = max(1, round(args.warmdown_ratio * args.num_iterations))
    standalone = args.standalone
    if standalone is None:
        standalone = args.nnodes == 1

    torchrun = [
        "torchrun",
        "--nproc_per_node",
        str(args.nproc_per_node),
        "--nnodes",
        str(args.nnodes),
        "--node_rank",
        str(args.node_rank),
    ]
    if standalone:
        torchrun.append("--standalone")
    else:
        torchrun.extend(["--master_addr", args.master_addr, "--master_port", str(args.master_port)])

    cmd = [
        *torchrun,
        str(Path(args.pretrain_script).resolve()),
        "--num-layers",
        str(args.n_layer),
        "--hidden-size",
        str(args.model_dim),
        "--ffn-hidden-size",
        str(4 * args.model_dim),
        "--num-attention-heads",
        str(args.n_head),
        "--seq-length",
        str(args.sequence_length),
        "--max-position-embeddings",
        str(args.sequence_length),
        "--micro-batch-size",
        str(args.device_batch_size),
        "--global-batch-size",
        str(args.batch_size),
        "--train-iters",
        str(args.num_iterations),
        "--eval-interval",
        str(args.val_loss_every if args.val_loss_every > 0 else args.num_iterations + 1),
        "--eval-iters",
        str(eval_iters),
        "--lr",
        str(args.lr),
        "--min-lr",
        "0.0",
        "--lr-decay-style",
        "WSD",
        "--lr-decay-iters",
        str(args.num_iterations),
        "--lr-wsd-decay-style",
        "linear",
        "--lr-wsd-decay-iters",
        str(warmdown_iters),
        "--lr-warmup-fraction",
        str(args.warmup_ratio),
        "--weight-decay",
        str(args.weight_decay),
        "--bf16",
        "--transformer-impl",
        "transformer_engine",
        "--attention-backend",
        args.attention_backend,
        "--position-embedding-type",
        "rope",
        "--rotary-percent",
        "1.0",
        "--normalization",
        "RMSNorm",
        "--qk-layernorm",
        "--squared-relu",
        "--disable-bias-linear",
        "--untie-embeddings-and-output-weights",
        "--hidden-dropout",
        "0.0",
        "--attention-dropout",
        "0.0",
        "--optimizer",
        "dion",
        "--use-distributed-optimizer",
        "--overlap-grad-reduce",
        "--overlap-param-gather",
        "--tensor-model-parallel-size",
        str(args.tp_size),
        "--fully-shard-model-parallel-size",
        str(args.fs_size),
        "--replicate-model-parallel-size",
        str(args.rp_size),
        "--dion-momentum",
        str(args.mu),
        "--dion-rank-fraction",
        str(args.ortho_fraction),
        "--dion-oversample",
        str(args.rcqr_oversample),
        "--dion-elementwise-optimizer",
        args.scalar_opt,
        "--dion-beta1",
        "0.95",
        "--dion-beta2",
        "0.98",
        "--dion-scale-mode",
        args.dion_scale_mode,
        "--dion-extra-scale-factor",
        str(args.dion_extra_scale_factor),
        "--dion-split-qkv",
        "--log-interval",
        str(args.log_interval),
    ]

    if args.mixed_precision:
        cmd.extend(
            [
                "--dion-momentum-dtype",
                "bfloat16",
                "--dion-q-dtype",
                "bfloat16",
                "--dion-variance-dtype",
                "bfloat16",
            ]
        )

    _append_data_args(cmd, args)

    extra_args = list(args.extra_args)
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]
    cmd.extend(extra_args)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    if args.force_te_flash:
        env["NVTE_FLASH_ATTN"] = "1"
        env["NVTE_FUSED_ATTN"] = "0"
        env["NVTE_UNFUSED_ATTN"] = "0"
    return cmd, env


def main() -> int:
    args = parse_args()
    cmd, env = build_command(args)
    print(" ".join(shlex.quote(part) for part in cmd), flush=True)
    if args.dry_run:
        return 0
    return subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
