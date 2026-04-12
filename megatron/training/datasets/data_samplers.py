# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Dataloaders."""


import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from megatron.core import mpu
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.datasets.utils import Split

from megatron.training import get_args
from megatron.training.dist_signal_handler import DistributedSignalHandler


def build_pretraining_data_loader(dataset, consumed_samples):
    """Build dataloader given an input dataset."""

    if dataset is None:
        return None
    args = get_args()
    proof_replay_dir = os.getenv("GPT_STEP_BATCH_REPLAY_DIR", "").strip()
    proof_record_dir = os.getenv("GPT_STEP_BATCH_RECORD_DIR", "").strip()
    if (proof_replay_dir or proof_record_dir) and args.dataloader_type != 'single':
        raise RuntimeError(
            "GPT step-batch record/replay harness currently requires dataloader_type=single"
        )

    if hasattr(dataset, 'split'):
        split = dataset.split
    elif hasattr(dataset, 'index_split'):
        split = dataset.index_split
    else:
        split = None

    if split == Split.valid and args.full_validation:
        batch_sampler = MegatronPretrainingSampler(
            total_samples=len(dataset),
            consumed_samples=0,
            micro_batch_size=args.micro_batch_size,
            data_parallel_rank=mpu.get_data_parallel_rank(),
            data_parallel_size=mpu.get_data_parallel_world_size(),
        )
    elif args.dataloader_type == 'single':
        if args.hybrid_context_parallel:
            batch_sampler = HybridCPMegatronPretrainingSampler(
                total_samples=len(dataset),
                consumed_samples=consumed_samples,
                micro_batch_size=args.micro_batch_size,
                global_batch_size=args.global_batch_size,
                data_parallel_rank=mpu.get_data_parallel_rank(),
                data_parallel_size=mpu.get_data_parallel_world_size())
        else:
            # Megatron sampler
            batch_sampler = MegatronPretrainingSampler(
                total_samples=len(dataset),
                consumed_samples=consumed_samples,
                micro_batch_size=args.micro_batch_size,
                data_parallel_rank=mpu.get_data_parallel_rank(),
                data_parallel_size=mpu.get_data_parallel_world_size())
    elif args.dataloader_type == 'cyclic':
        batch_sampler = MegatronPretrainingRandomSampler(
            dataset,
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=args.micro_batch_size,
            data_parallel_rank=mpu.get_data_parallel_rank(),
            data_parallel_size=mpu.get_data_parallel_world_size(),
            data_sharding=args.data_sharding,
        )
    elif args.dataloader_type == "external":
        # External dataloaders are passed through. User is expected to provide a
        # torch-compatible dataloader and define samplers, if needed.
        return dataset
    else:
        raise Exception('{} dataloader type is not supported.'.format(args.dataloader_type))

    def worker_init_fn(_):
        DistributedSignalHandler(args.exit_signal).__enter__()

    maybe_worker_init_fn = (
        worker_init_fn if args.exit_signal_handler and args.num_workers > 0 else None
    )
    # Torch dataloader.
    if args.hybrid_context_parallel:
        extra_kwargs = {"collate_fn": lambda x: x,}
    else:
        extra_kwargs = {}
    return torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        worker_init_fn=maybe_worker_init_fn,
        **extra_kwargs,
    )

class MegatronPretrainingSampler:
    """
    Sampler for Megatron pretraining dataloaders that divides data samples across
    data parallel workers. Each worker receives a contiguous chunk of data determined by
    its rank and the micro batch size. Supports dropping the last incomplete batch if
    specified, and keeps track of total and consumed samples. Designed to work with
    distributed training using Megatron's data parallelism.
    """

    def __init__(
        self,
        total_samples,
        consumed_samples,
        micro_batch_size,
        data_parallel_rank,
        data_parallel_size,
        drop_last=True,
    ):
        # Keep a copy of input params for later use.
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.micro_batch_times_data_parallel_size = self.micro_batch_size * data_parallel_size
        self.drop_last = drop_last

        # Sanity checks.
        assert self.total_samples > 0, 'no sample to consume: {}'.format(self.total_samples)
        assert (
            self.consumed_samples < self.total_samples
        ), 'no samples left to consume: {}, {}'.format(self.consumed_samples, self.total_samples)
        assert self.micro_batch_size > 0
        assert data_parallel_size > 0
        assert (
            self.data_parallel_rank < data_parallel_size
        ), 'data_parallel_rank should be smaller than data size: {}, ' '{}'.format(
            self.data_parallel_rank, data_parallel_size
        )

    def __len__(self):
        return self.total_samples

    def _proof_step_batch_record_dir(self) -> str:
        return os.getenv("GPT_STEP_BATCH_RECORD_DIR", "").strip()

    def _proof_step_batch_replay_dir(self) -> str:
        return os.getenv("GPT_STEP_BATCH_REPLAY_DIR", "").strip()

    def _proof_step_batch_max_steps(self) -> int:
        raw = os.getenv("GPT_STEP_BATCH_MAX_STEPS", "").strip()
        return int(raw) if raw else 0

    def _proof_step_batch_path(self, root: str, step_idx: int) -> Path:
        return Path(root) / f"shared_stream_step{step_idx:06d}.pt"

    def _proof_should_record(self) -> bool:
        if not torch.distributed.is_initialized():
            return True
        if mpu.get_data_parallel_rank(with_context_parallel=True) != 0:
            return False
        if mpu.get_tensor_model_parallel_rank() != 0:
            return False
        return mpu.is_pipeline_first_stage()

    def _proof_load_replay_step(self, replay_dir: str, step_idx: int, expected_count: int):
        path = self._proof_step_batch_path(replay_dir, step_idx)
        if not path.exists():
            raise RuntimeError(
                f"Missing GPT step-batch replay file for step {step_idx}: {path}"
            )
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, dict):
            sample_indices = payload.get("sample_indices")
            source_count = payload.get("global_batch_size")
        else:
            sample_indices = payload
            source_count = None
        if sample_indices is None:
            raise RuntimeError(f"Invalid GPT step-batch replay payload at {path}")
        if torch.is_tensor(sample_indices):
            sample_indices = sample_indices.tolist()
        else:
            sample_indices = list(sample_indices)
        if source_count is not None and int(source_count) != expected_count:
            raise RuntimeError(
                f"Replay batch-size mismatch at {path}: source={source_count} current={expected_count}"
            )
        if len(sample_indices) != expected_count:
            raise RuntimeError(
                f"Replay sample count mismatch at {path}: got={len(sample_indices)} expected={expected_count}"
            )
        return sample_indices

    def _proof_save_record_step(
        self, record_dir: str, step_idx: int, sample_indices, global_batch_size: int
    ):
        path = self._proof_step_batch_path(record_dir, step_idx)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "sample_indices": torch.tensor(sample_indices, dtype=torch.long),
            "global_batch_size": int(global_batch_size),
            "micro_batch_size": int(self.micro_batch_size),
            "data_parallel_size": int(
                self.micro_batch_times_data_parallel_size // self.micro_batch_size
            ),
            "num_microbatches": int(get_num_microbatches()),
        }
        torch.save(payload, path)

    def get_start_end_idx(self):
        """
        Calculate the start and end indices for the current data parallel worker's
        chunk within a batch.

        Returns:
            tuple: (start_idx, end_idx) indicating the slice of the batch for this worker.
        """
        start_idx = self.data_parallel_rank * self.micro_batch_size
        end_idx = start_idx + self.micro_batch_size
        return start_idx, end_idx

    def __iter__(self):
        batch = []
        record_dir = self._proof_step_batch_record_dir()
        replay_dir = self._proof_step_batch_replay_dir()
        if record_dir and replay_dir:
            raise RuntimeError("GPT step-batch harness cannot record and replay at the same time")
        num_microbatches = get_num_microbatches()
        step_global_batch_size = self.micro_batch_times_data_parallel_size * num_microbatches
        max_steps = self._proof_step_batch_max_steps()
        current_step = 1
        microbatch_in_step = 0
        record_step_samples = []
        replay_step_samples = None
        # Last batch will be dropped if drop_last is not set False
        for idx in range(self.consumed_samples, self.total_samples):
            batch.append(idx)
            if len(batch) == self.micro_batch_times_data_parallel_size:
                global_batch = batch
                if replay_dir:
                    if replay_step_samples is None and (not max_steps or current_step <= max_steps):
                        replay_step_samples = self._proof_load_replay_step(
                            replay_dir, current_step, step_global_batch_size
                        )
                    if replay_step_samples is not None:
                        mb_start = microbatch_in_step * self.micro_batch_times_data_parallel_size
                        mb_end = mb_start + self.micro_batch_times_data_parallel_size
                        global_batch = replay_step_samples[mb_start:mb_end]
                if record_dir and self._proof_should_record():
                    if not max_steps or current_step <= max_steps:
                        record_step_samples.extend(batch)
                start_idx, end_idx = self.get_start_end_idx()
                yield global_batch[start_idx:end_idx]
                microbatch_in_step += 1
                if microbatch_in_step == num_microbatches:
                    if record_dir and self._proof_should_record():
                        if not max_steps or current_step <= max_steps:
                            self._proof_save_record_step(
                                record_dir,
                                current_step,
                                record_step_samples,
                                step_global_batch_size,
                            )
                    current_step += 1
                    microbatch_in_step = 0
                    record_step_samples = []
                    replay_step_samples = None
                batch = []

        # Check the last partial batch and see drop_last is set
        if len(batch) > 0 and not self.drop_last:
            start_idx, end_idx = self.get_start_end_idx()
            yield batch[start_idx:end_idx]

class HybridCPMegatronPretrainingSampler(MegatronPretrainingSampler):
    """
    Data sampler for hybrid context parallel (Hybrid CP) format.
    This data sampler pulls in the entire global batch at once across all data parallel ranks.
    This helps provide the Hybrid CP Dataloader Wrapper to schedule and load balance sub-samples
    of the entire global batch.
    """

    def __init__(self, total_samples, consumed_samples, micro_batch_size, global_batch_size,
                 data_parallel_rank, data_parallel_size, drop_last=True):
        super().__init__(total_samples, consumed_samples, micro_batch_size, data_parallel_rank, data_parallel_size, drop_last)
        self.global_batch_size = global_batch_size
        self.data_parallel_size = data_parallel_size
        self.num_micro_batches = self.global_batch_size // self.micro_batch_times_data_parallel_size

    def __len__(self):
        return self.total_samples

    def get_start_end_idx_global_batch(self):
        start_idx = [self.data_parallel_rank * self.micro_batch_size + i * self.micro_batch_size * self.data_parallel_size for i in range(self.num_micro_batches)]
        end_idx = [start_idx[i] + self.micro_batch_size for i in range(self.num_micro_batches)]
        return start_idx, end_idx

    def __iter__(self):
        batch = []
        # Last batch will be dropped if drop_last is not set False
        for idx in range(self.consumed_samples, self.total_samples):
            batch.append(idx)
            if len(batch) == self.micro_batch_times_data_parallel_size * self.num_micro_batches:
                start_idx, end_idx = self.get_start_end_idx_global_batch()
                global_batch_idx = []
                for i in range(self.num_micro_batches):
                    global_batch_idx.extend(batch[start_idx[i]:end_idx[i]])
                yield global_batch_idx
                batch = []

        # Check the last partial batch and see drop_last is set
        if len(batch) > 0 and not self.drop_last:
            start_idx, end_idx = self.get_start_end_idx_global_batch()
            global_batch_idx = []
            for i in range(self.num_micro_batches):
                global_batch_idx.extend(batch[start_idx[i]:end_idx[i]])
            yield global_batch_idx

class RandomSeedDataset(Dataset):
    """
    A dataset wrapper that resets the random seed before each sample.

    This ensures deterministic behavior per sample by setting the RNG state
    for torch, numpy, and random before accessing each underlying data sample.
    The base seed is retrieved from training arguments, and can be varied per epoch
    using the set_epoch method to ensure different shuffling or augmentation each epoch.

    Args:
        dataset: The underlying dataset to wrap.

    Methods:
        set_epoch(epoch): Change the seed offset so each epoch produces different randomization.
        __getitem__(idx): Sets the seed based on the sample index and current epoch.
    """

    def __init__(self, dataset, seed):
        self.base_seed = seed
        self.curr_seed = seed
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def set_epoch(self, epoch):
        """
        Change the seed offset so each epoch produces different randomization.

        Args:
            epoch: The epoch number to use as the seed offset.
        """
        self.curr_seed = self.base_seed + epoch

    def __getitem__(self, idx):
        seed = idx + self.curr_seed
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        return self.dataset[idx]


class MegatronPretrainingRandomSampler:
    """
    Sampler for Megatron pretraining dataloaders that performs random sampling
    across data parallel workers. Supports data sharding to divide the dataset
    into buckets and shuffle within each bucket. Designed to work with distributed
    training using Megatron's data parallelism.
    """

    def __init__(
        self,
        dataset,
        total_samples,
        consumed_samples,
        micro_batch_size,
        data_parallel_rank,
        data_parallel_size,
        data_sharding,
    ):
        # Keep a copy of input params for later use.
        self.dataset = dataset
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_size = data_parallel_size
        self.data_sharding = data_sharding
        self.micro_batch_times_data_parallel_size = self.micro_batch_size * data_parallel_size
        self.last_batch_size = self.total_samples % self.micro_batch_times_data_parallel_size

        # Sanity checks.
        assert self.total_samples > 0, 'no sample to consume: {}'.format(self.total_samples)
        assert self.micro_batch_size > 0
        assert data_parallel_size > 0
        assert (
            self.data_parallel_rank < data_parallel_size
        ), 'data_parallel_rank should be smaller than data size: {}, ' '{}'.format(
            self.data_parallel_rank, data_parallel_size
        )

    def __len__(self):
        return self.total_samples

    def __iter__(self):
        active_total_samples = self.total_samples - self.last_batch_size
        self.epoch = self.consumed_samples // active_total_samples
        current_epoch_samples = self.consumed_samples % active_total_samples
        assert current_epoch_samples % self.micro_batch_times_data_parallel_size == 0

        if isinstance(self.dataset, RandomSeedDataset):
            self.dataset.set_epoch(self.epoch)

        # data sharding and random sampling
        if self.data_sharding:
            bucket_size = (
                self.total_samples // self.micro_batch_times_data_parallel_size
            ) * self.micro_batch_size
            bucket_offset = current_epoch_samples // self.data_parallel_size
            start_idx = self.data_parallel_rank * bucket_size

            g = torch.Generator()
            g.manual_seed(self.epoch)
            random_idx = torch.randperm(bucket_size, generator=g).tolist()
            idx_range = [start_idx + x for x in random_idx[bucket_offset:]]
        else:
            full_bucket_size = (self.total_samples // self.micro_batch_size) * self.micro_batch_size
            full_bucket_offset = current_epoch_samples
            g = torch.Generator()
            g.manual_seed(self.epoch)
            idx_range_total = torch.randperm(full_bucket_size, generator=g).tolist()
            idx_range_active = idx_range_total[full_bucket_offset:]
            idx_range = idx_range_active[self.data_parallel_rank :: self.data_parallel_size]

        batch = []
        # Last batch if not complete will be dropped.
        for idx in idx_range:
            batch.append(idx)
            if len(batch) == self.micro_batch_size:
                self.consumed_samples += self.micro_batch_times_data_parallel_size
                yield batch
                batch = []
