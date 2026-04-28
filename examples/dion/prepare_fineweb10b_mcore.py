#!/usr/bin/env python3
"""Prepare Dion FineWeb10B GPT-2 token shards as a Megatron indexed dataset."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from megatron.core.datasets import indexed_dataset


MAGIC = 20240520
VERSION = 1
HEADER_BYTES = 256 * 4
DEFAULT_REPO_ID = "kjj0/fineweb10B-gpt2"
DEFAULT_DATA_ROOT = REPO_ROOT.parent / "data" / "fineweb10B-gpt2"
DEFAULT_RAW_DIR = DEFAULT_DATA_ROOT / "raw"
DEFAULT_TRAIN_PREFIX = DEFAULT_DATA_ROOT / "megatron-train" / "fineweb10B_gpt2_train_text_document"
DEFAULT_VAL_PREFIX = DEFAULT_DATA_ROOT / "megatron-val" / "fineweb10B_gpt2_val_text_document"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help="Build one combined Megatron prefix. By default, build Dion-style train/val prefixes.",
    )
    parser.add_argument("--train-output-prefix", type=Path, default=DEFAULT_TRAIN_PREFIX)
    parser.add_argument("--val-output-prefix", type=Path, default=DEFAULT_VAL_PREFIX)
    parser.add_argument(
        "--num-train-shards",
        type=int,
        default=103,
        help="Number of fineweb_train_*.bin shards to include. Full FineWeb10B is 103.",
    )
    parser.add_argument("--train-start-index", type=int, default=1)
    parser.add_argument("--skip-val", action="store_true", help="Do not include fineweb_val_000000.bin")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def shard_names(args: argparse.Namespace) -> list[str]:
    names = []
    if not args.skip_val:
        names.append("fineweb_val_000000.bin")
    names.extend(
        f"fineweb_train_{index:06d}.bin"
        for index in range(args.train_start_index, args.train_start_index + args.num_train_shards)
    )
    return names


def download_shard(repo_id: str, raw_dir: Path, filename: str) -> Path:
    path = raw_dir / filename
    if path.exists():
        return path
    hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", local_dir=raw_dir)
    return path


def read_header(path: Path) -> int:
    with path.open("rb") as stream:
        header = np.frombuffer(stream.read(HEADER_BYTES), dtype=np.int32)
    if len(header) != 256 or int(header[0]) != MAGIC:
        raise RuntimeError(f"{path} is not a Dion FineWeb token shard")
    if int(header[1]) != VERSION:
        raise RuntimeError(f"{path} has unsupported shard version {int(header[1])}")
    token_count = int(header[2])
    expected_size = HEADER_BYTES + token_count * np.dtype(np.uint16).itemsize
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(f"{path} size mismatch: expected {expected_size}, got {actual_size}")
    return token_count


def build_indexed_dataset(paths: list[Path], output_prefix: Path, overwrite: bool) -> None:
    bin_path = output_prefix.with_suffix(".bin")
    idx_path = output_prefix.with_suffix(".idx")
    if (bin_path.exists() or idx_path.exists()) and not overwrite:
        raise SystemExit(f"{output_prefix} already exists; pass --overwrite to rebuild")

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for path in (bin_path, idx_path):
        if path.exists():
            path.unlink()

    builder = indexed_dataset.IndexedDatasetBuilder(str(bin_path), dtype=np.uint16)
    total_tokens = 0
    for path in tqdm(paths, desc="Converting shards", unit="shard"):
        token_count = read_header(path)
        tokens = np.memmap(path, mode="r", dtype=np.uint16, offset=HEADER_BYTES, shape=(token_count,))
        builder.add_document(tokens, [token_count])
        total_tokens += token_count
    builder.finalize(str(idx_path))

    print(f"wrote {total_tokens:,} tokens")
    print(f"data prefix: {output_prefix}")
    print(f"bin: {bin_path}")
    print(f"idx: {idx_path}")


def main() -> int:
    args = parse_args()
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    names = shard_names(args)

    paths = []
    if args.skip_download:
        paths = [args.raw_dir / name for name in names]
    else:
        for name in tqdm(names, desc="Downloading shards", unit="shard"):
            paths.append(download_shard(args.repo_id, args.raw_dir, name))

    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise SystemExit("missing raw shards:\n" + "\n".join(missing))

    if args.output_prefix is not None:
        build_indexed_dataset(paths, args.output_prefix, args.overwrite)
        return 0

    raw_by_name = {path.name: path for path in paths}
    train_paths = [
        raw_by_name[f"fineweb_train_{index:06d}.bin"]
        for index in range(args.train_start_index, args.train_start_index + args.num_train_shards)
    ]
    build_indexed_dataset(train_paths, args.train_output_prefix, args.overwrite)

    if not args.skip_val:
        build_indexed_dataset([raw_by_name["fineweb_val_000000.bin"]], args.val_output_prefix, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
