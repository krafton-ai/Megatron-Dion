#!/usr/bin/env python
# Copyright (c) 2024. All rights reserved.
"""
Convert dion fineweb .bin data format to Megatron indexed dataset format.

dion format:
- Header: 256 int32 values (magic=20240520, version=1, ntok, ...)
- Data: uint16 tokens

Megatron format:
- .idx file: Index file with sequence metadata
- .bin file: Token data

Usage:
    python tools/convert_dion_to_megatron.py \
        --input-pattern "/path/to/fineweb_train_*.bin" \
        --output-prefix "/path/to/output/fineweb_train" \
        --sequence-length 1024
"""

import argparse
import glob
import os
import struct
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


_INDEX_HEADER = b"MMIDIDX\x00\x00"

# Magic number for FineWeb binary format (from Microsoft Dion repository)
FINEWEB_BIN_MAGIC = 20240520


class DType:
    """NumPy dtype codes for Megatron indexed dataset."""
    uint8 = 1
    int8 = 2
    int16 = 3
    int32 = 4
    int64 = 5
    float64 = 6
    float32 = 7
    uint16 = 8

    @classmethod
    def code_from_dtype(cls, dtype):
        dtype_name = np.dtype(dtype).name
        return getattr(cls, dtype_name)


def read_dion_bin(filename):
    """Read dion format .bin file and return tokens as numpy array."""
    with open(filename, "rb") as f:
        # Read header: 256 int32 values
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)

        if header[0] != FINEWEB_BIN_MAGIC:
            raise ValueError(
                f"Invalid magic number in {filename}. "
                f"Expected {FINEWEB_BIN_MAGIC}, got {header[0]}"
            )

        version = header[1]
        if version != 1:
            raise ValueError(f"Unsupported version {version} in {filename}")

        ntok = header[2]

        # Read tokens as uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint16)

        if len(tokens) != ntok:
            raise ValueError(
                f"Token count mismatch in {filename}. "
                f"Header says {ntok}, but found {len(tokens)}"
            )

    return tokens


def write_megatron_indexed_dataset(output_prefix, all_tokens, sequence_length, dtype=np.uint16):
    """Write tokens in Megatron indexed dataset format.

    Args:
        output_prefix: Output path prefix (will create .bin and .idx files)
        all_tokens: numpy array of all tokens
        sequence_length: Length of each sequence (for document boundaries)
        dtype: Data type for tokens
    """
    bin_path = output_prefix + ".bin"
    idx_path = output_prefix + ".idx"

    # Calculate sequences
    # In GPT training, we need seq_length + 1 tokens per sample
    # (seq_length inputs + 1 for shifted targets)
    tokens_per_sample = sequence_length + 1
    num_samples = len(all_tokens) // tokens_per_sample
    total_tokens = num_samples * tokens_per_sample

    if num_samples == 0:
        raise ValueError(
            f"Not enough tokens to create any samples. "
            f"Have {len(all_tokens)} tokens, need at least {tokens_per_sample}"
        )

    # Trim tokens to exact multiple
    tokens = all_tokens[:total_tokens]

    print(f"Total tokens: {len(all_tokens):,}")
    print(f"Tokens per sample: {tokens_per_sample}")
    print(f"Number of samples: {num_samples:,}")
    print(f"Tokens used: {total_tokens:,}")

    # Write .bin file (raw tokens)
    print(f"Writing {bin_path}...")
    with open(bin_path, "wb") as f:
        tokens.tofile(f)

    # Write .idx file
    print(f"Writing {idx_path}...")
    with open(idx_path, "wb") as f:
        # Header
        f.write(_INDEX_HEADER)
        # Version (fixed at 1)
        f.write(struct.pack("<Q", 1))
        # Data type code
        f.write(struct.pack("<B", DType.code_from_dtype(dtype)))

        # Number of sequences
        f.write(struct.pack("<Q", num_samples))

        # Number of documents (treat each sequence as a document)
        f.write(struct.pack("<Q", num_samples))

        # Sequence lengths (each sequence has tokens_per_sample tokens)
        sequence_lengths = np.full(num_samples, tokens_per_sample, dtype=np.int32)
        f.write(sequence_lengths.tobytes(order="C"))

        # Sequence pointers (byte offsets)
        bytes_per_token = np.dtype(dtype).itemsize
        sequence_pointers = np.arange(num_samples, dtype=np.int64) * tokens_per_sample * bytes_per_token
        f.write(sequence_pointers.tobytes(order="C"))

        # Document indices (sequence indices marking end of each document)
        # Each document ends at the next sequence index (1-indexed end positions)
        document_indices = np.arange(1, num_samples + 1, dtype=np.int64)
        f.write(document_indices.tobytes(order="C"))

    print(f"Done! Created {bin_path} and {idx_path}")
    return num_samples


def main():
    parser = argparse.ArgumentParser(
        description="Convert dion fineweb data to Megatron indexed dataset format"
    )
    parser.add_argument(
        "--input-pattern",
        type=str,
        required=True,
        help="Glob pattern for input dion .bin files (e.g., 'data/fineweb_train_*.bin')",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        required=True,
        help="Output prefix for Megatron dataset (will create .bin and .idx files)",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=1024,
        help="Sequence length for GPT training (default: 1024)",
    )
    args = parser.parse_args()

    # Find input files
    input_files = sorted(glob.glob(args.input_pattern))
    if not input_files:
        raise FileNotFoundError(f"No files found matching pattern: {args.input_pattern}")

    print(f"Found {len(input_files)} input files")

    # Read all tokens
    all_tokens = []
    for i, input_file in enumerate(input_files):
        print(f"Reading {input_file} ({i+1}/{len(input_files)})...")
        tokens = read_dion_bin(input_file)
        all_tokens.append(tokens)
        print(f"  Read {len(tokens):,} tokens")

    # Concatenate all tokens
    all_tokens = np.concatenate(all_tokens)
    print(f"\nTotal tokens from all files: {len(all_tokens):,}")

    # Create output directory if needed
    output_dir = os.path.dirname(args.output_prefix)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Write Megatron format
    write_megatron_indexed_dataset(
        args.output_prefix,
        all_tokens,
        args.sequence_length,
    )


if __name__ == "__main__":
    main()
