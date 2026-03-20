# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain vision language model."""
import os
import sys
import warnings
from copy import deepcopy
from functools import partial
from typing import List

import torch

from megatron.core import mpu, parallel_state, tensor_parallel
from megatron.core.datasets.blended_megatron_dataset_builder import (
    BlendedMegatronDatasetBuilder,
)
from megatron.core.datasets.multimodal_dataset import (
    MockMultimodalDataset,
    MultimodalDatasetConfig,
)
from megatron.core.enums import ModelType
from megatron.core.models.multimodal import context_parallel
from megatron.core.models.multimodal.llava_model import (
    DEFAULT_IMAGE_TOKEN_INDEX,
    IGNORE_INDEX,
    LLaVAModel,
)
from megatron.core.models.multimodal.llava_spec import (
    decoder_model_with_local_default_spec,
    decoder_model_with_transformer_engine_default_spec,
)
from megatron.core.models.vision.clip_vit_model import get_num_image_embeddings
from megatron.core.models.vision.vit_layer_specs import (
    get_vit_layer_with_local_spec,
    get_vit_layer_with_transformer_engine_spec,
)
from megatron.core.parallel_state import (
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_world_size,
    get_tensor_model_parallel_rank,
)
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.spec_utils import import_module
from megatron.training import (
    get_args,
    get_timers,
    get_tokenizer,
    pretrain,
    print_rank_0,
)
from megatron.training.arguments import core_transformer_config_from_args
from pretrain_gpt import loss_func as _gpt_loss_func

_BATCH_FP_DEBUG_COUNTER = 0
_POST_BCAST_FP_DEBUG_COUNTER = 0
_BATCH_KEY_DEBUG_COUNTER = 0
_COMBINED_FP_DEBUG_COUNTER = 0
_PACKED_FP_DEBUG_COUNTER = 0


def loss_func(loss_mask, output_tensor, model=None):
    """VLM loss function — delegates to standard GPT loss_func."""
    return _gpt_loss_func(loss_mask, output_tensor, model=model)


def _debug_log_init_params(model):
    patterns_raw = os.getenv("VLM_DEBUG_INIT_PARAM_PATTERNS", "").strip()
    if not patterns_raw:
        return
    patterns = [p.strip() for p in patterns_raw.split(",") if p.strip()]
    if not patterns:
        return

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
    pp_rank = get_pipeline_model_parallel_rank()
    tp_rank = get_tensor_model_parallel_rank()

    for name, param in model.named_parameters():
        if not any(p in name for p in patterns):
            continue
        data = param.detach().float()
        flat = data.flatten()
        limit = min(int(os.getenv("VLM_DEBUG_INIT_PARAM_PREVIEW", "8")), flat.numel())
        preview = ",".join(f"{float(x):.8f}" for x in flat[:limit].tolist())
        if flat.numel() > 0:
            weights = torch.arange(1, flat.numel() + 1, device=flat.device, dtype=flat.dtype)
            weighted = float((flat * weights).sum().item())
        else:
            weighted = 0.0
        print(
            "VLM_INIT_PARAM "
            f"rank={rank} pp={pp_rank} tp={tp_rank} "
            f"name={name} shape={tuple(param.shape)} "
            f"sum={float(data.sum().item()):.8f} "
            f"abs_sum={float(data.abs().sum().item()):.8f} "
            f"norm={float(torch.linalg.vector_norm(data).item()):.8f} "
            f"weighted_sum={weighted:.8f} "
            f"preview=[{preview}]",
            flush=True,
        )


def model_provider(
    pre_process=True,
    post_process=True,
    add_encoder=True,
    add_decoder=True,
    parallel_output=True,
    config=None,
    pg_collection=None,
) -> LLaVAModel:
    """Builds the model using examples/multimodal/model.py's model_provider with CP padding.

    Delegates to the existing multimodal model_provider (which supports CLIP, SigLIP2, etc.)
    and adds CP-specific padding for decoder_seq_length.
    """
    args = get_args()

    assert args.ckpt_format == 'torch', "Only ckpt-format torch is supported for VLM training currently."
    assert not (args.context_parallel_size > 1 and args.pipeline_model_parallel_size > 1), \
        "PP+CP is not yet supported by this script."

    # --- CP padding for decoder_seq_length ---
    # Compute num_image_embeddings to determine the correct decoder sequence length.
    vision_model_type = getattr(args, 'vision_model_type', None) or "clip"
    if vision_model_type.startswith("siglip"):
        class_token_len = 0
    else:
        class_token_len = 1
    ps_flag = getattr(args, 'pixel_shuffle', False)
    ps_factor = getattr(args, 'pixel_shuffle_factor', 2)
    pixel_shuffle = ps_factor if ps_flag else False

    num_image_embeddings = get_num_image_embeddings(
        args.img_h, args.img_w, args.patch_dim, vision_model_type, args.disable_vision_class_token,
        class_token_len=class_token_len, pixel_shuffle=pixel_shuffle, use_tile_tags=False
    )

    if args.dataloader_seq_length is None:
        args.dataloader_seq_length = args.seq_length
    decoder_seq_len = args.dataloader_seq_length + num_image_embeddings
    args.seq_length = args.encoder_seq_length = num_image_embeddings

    mp_padding_needed = context_parallel.get_padding(
        decoder_seq_len,
        args.context_parallel_size,
        args.tensor_model_parallel_size,
        args.sequence_parallel,
        getattr(args, 'decoder_tp_comm_overlap', False),
        args.decoder_seq_length
    )
    args.decoder_seq_length = decoder_seq_len + mp_padding_needed
    args.max_position_embeddings = max(args.max_position_embeddings, args.decoder_seq_length)

    # --- Delegate to examples/multimodal/model.py ---
    examples_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples", "multimodal")
    if examples_dir not in sys.path:
        sys.path.insert(0, examples_dir)
    from model import model_provider as _multimodal_model_provider

    # PP-aware module placement for LLaVA:
    # - Encoder runs only on the first PP stage.
    # - Decoder remains active on all PP stages (decoder starts at stage 0).
    # This avoids requiring image tensors on non-first PP stages.
    pp_world_size = get_pipeline_model_parallel_world_size()
    if pp_world_size > 1:
        add_encoder = bool(pre_process)
        add_decoder = True

    model = _multimodal_model_provider(
        pre_process=pre_process,
        post_process=post_process,
        add_encoder=add_encoder,
        add_decoder=add_decoder,
        parallel_output=parallel_output,
    )
    _debug_log_init_params(model)
    return model


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train, validation, and test sets.

    Returns:
        train_ds, val_ds, test_ds (megatron.core.datasets.multimodal_dataset.MockMultimodalDataset): Train, validation, and test datasets, respectively.
    """
    args = get_args()

    config = MultimodalDatasetConfig(
        random_seed=args.seed,
        split=args.split,
        sequence_length=args.dataloader_seq_length,
        tokenizer=get_tokenizer(),
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        image_h=args.img_h,
        image_w=args.img_w,
        preprocess_func=_preprocess_data_for_llava,
        mid_level_dataset_surplus=args.mid_level_dataset_surplus,
        allow_ambiguous_pad_tokens=args.allow_ambiguous_pad_tokens,
    )

    print_rank_0("> building train, validation, and test datasets for multimodal ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        MockMultimodalDataset,
        train_val_test_num_samples,
        lambda: parallel_state.get_tensor_model_parallel_rank() == 0,
        config,
    ).build()

    print_rank_0("> finished creating multimodal datasets ...")

    return train_ds, valid_ds, test_ds


def _preprocess_data_for_llava(data):
    """Preprocess data sample to the format expected by a LLaVA model.

    Note: This doesn't support all the different modes in the official LLaVA repo yet.

    Args:
        data (dict): Data sample with keys like 'image', 'tokens', etc.

    Returns:
        data (dict): Processed data sample suitable for the model.
    """
    # Prepend image token index to tokens.
    data["tokens"] = torch.cat(
        [
            DEFAULT_IMAGE_TOKEN_INDEX
            * torch.ones(1, dtype=data["tokens"].dtype, device=data["tokens"].device),
            data["tokens"],
        ]
    )
    # Prepend labels accordingly.
    data["labels"] = torch.cat([data["tokens"][1].unsqueeze(0), data["labels"]])
    # Zero loss mask for the image token index.
    data["loss_mask"] = torch.cat(
        [
            torch.zeros(1, dtype=data["loss_mask"].dtype, device=data["loss_mask"].device),
            data["loss_mask"],
        ]
    )
    # Add one more position id.
    data["position_ids"] = torch.cat(
        [data["position_ids"], data["position_ids"][-1].unsqueeze(0) + 1]
    )

    return data


def _is_first_or_last_stage(pp_size):
    """Check if the current pipeline parallel stage is the first or last stage."""
    if pp_size == 1:
        return True
    pp_rank = get_pipeline_model_parallel_rank()
    return pp_rank in (0, pp_size - 1)


def _is_dataloader_rank():
    """Check if we should have the dataloader on this TP and PP rank."""
    # Only run the dataloader on the leader of the TP×CP group for this DP rank:
    # - TP leader: tp_rank == 0
    # - CP leader: cp_rank == 0
    # This avoids redundant loading across CP ranks and enables explicit TP×CP
    # broadcast in get_batch_energon().
    is_first_rank = get_tensor_model_parallel_rank() == 0
    is_first_rank = is_first_rank and parallel_state.get_context_parallel_rank() == 0
    # IMPORTANT: When pipeline parallelism is enabled, the last stage must see the
    # *exact* same sample stream as the first stage (for labels/loss_mask). While
    # this is often true for standard GPT dataloaders, Energon shuffling/packing
    # can cause subtle divergences if both stages load independently.
    #
    # So, for PP>1 we only load on the *first* stage, and then broadcast the batch
    # to the last stage via the embedding group inside get_batch_energon().
    return is_first_rank and parallel_state.is_pipeline_first_stage()


def _get_energon_dataloader_provider():
    """Lazily import and return energon dataloader provider components.

    Adds examples/multimodal to sys.path so that dataset_helpers (TaskEncoder)
    and its dependencies can be resolved.
    """
    examples_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples", "multimodal")
    if examples_dir not in sys.path:
        sys.path.insert(0, examples_dir)

    from dataset_helpers import TaskEncoder, print_error_handler
    from megatron.energon import (
        LimitDataset,
        RepeatDataset,
        WorkerConfig,
        get_loader,
        get_savable_loader,
        get_train_dataset,
        get_val_datasets,
    )
    from megatron.core.num_microbatches_calculator import get_num_microbatches

    return TaskEncoder, print_error_handler, WorkerConfig, get_savable_loader, get_loader, get_train_dataset, get_val_datasets, LimitDataset, RepeatDataset, get_num_microbatches


class _EnergonDataloader:
    """A wrapper to use Megatron Energon dataloader with the Megatron-LM training loop."""
    def __init__(self, dataloader):
        self._dataloader = dataloader
        self._iter = iter(self._cyclic_iter(dataloader))

    @staticmethod
    def _cyclic_iter(it):
        while True:
            for x in it:
                yield x

    def __next__(self):
        return self._iter.__next__()

    def __iter__(self):
        return self._iter.__iter__()

    def save_state(self):
        return self._dataloader.save_state_rank()


class _InterleavedEnergonDataloader:
    """Round-robin over multiple Energon dataloaders.

    Energon shards by (rank, world_size). If we shard by the *current* DP world size, the
    global batch stream changes when DP changes (e.g., CP×FS sweeps). To make the global
    stream stable across DP sizes, treat `global_batch_size/micro_batch_size` as a fixed
    "virtual world size" and, when DP is smaller, create multiple loaders per DP rank and
    interleave them so that all DP configs see the same set of virtual ranks each iteration.
    """

    def __init__(self, dataloaders: List[_EnergonDataloader]):
        assert len(dataloaders) > 0
        self._dataloaders = dataloaders
        self._i = 0
        self._last_i = None

    def __iter__(self):
        return self

    def __next__(self):
        idx = self._i
        out = next(self._dataloaders[idx])
        self._i = (self._i + 1) % len(self._dataloaders)
        self._last_i = idx
        return out

    def save_state(self):
        return {
            "i": self._i,
            "states": [dl.save_state() for dl in self._dataloaders],
        }


def train_valid_test_dataloaders_provider(train_val_test_num_samples):
    """Build energon-based train, validation, and test dataloaders."""
    args = get_args()

    (TaskEncoder, print_error_handler, WorkerConfig, get_savable_loader,
     get_loader, get_train_dataset, get_val_datasets, LimitDataset,
     RepeatDataset, get_num_microbatches) = _get_energon_dataloader_provider()

    task_encoder = TaskEncoder()

    if not _is_dataloader_rank():
        return None, None, None

    rank = parallel_state.get_data_parallel_rank()
    world_size = parallel_state.get_data_parallel_world_size()
    data_parallel_group = parallel_state.get_data_parallel_group()

    # Virtual sharding: keep the effective Energon world size stable across DP sizes
    # so CP×FS sweeps see the same global batch stream.
    virtual_world_size = args.global_batch_size // args.micro_batch_size
    if virtual_world_size % world_size != 0:
        virtual_world_size = world_size
    num_virtual_ranks_per_dp = max(1, virtual_world_size // world_size)
    if os.getenv("VLM_DEBUG_BATCH_FINGERPRINT", "0") == "1":
        print_rank_0(
            "VLM_DATALOADER_CFG "
            f"dp_rank={rank} dp_world={world_size} "
            f"virtual_world={virtual_world_size} "
            f"num_virtual_ranks_per_dp={num_virtual_ranks_per_dp}"
        )

    dname = args.data_path[0] if isinstance(args.data_path, list) else args.data_path
    packing_buffer_size = getattr(args, 'packing_buffer_size', None)
    train_iterables: List[_EnergonDataloader] = []
    for i in range(num_virtual_ranks_per_dp):
        # IMPORTANT: Don't pass Megatron's DP process group here. With virtual_world_size,
        # worker_rank can exceed dp_world_size, and WorkerConfig.global_rank() would error
        # if a process group were provided.
        worker_config = WorkerConfig(
            rank=rank + i * world_size,
            world_size=virtual_world_size,
            num_workers=args.num_workers,
            data_parallel_group=None,
            worker_debug_path=None,
            worker_log_level=0,
        )
        train_dataset = get_train_dataset(
            dname,
            batch_size=args.micro_batch_size,
            task_encoder=task_encoder,
            virtual_epoch_length=1000,
            max_samples_per_sequence=100,
            shuffle_buffer_size=100,
            worker_config=worker_config,
            packing_buffer_size=packing_buffer_size,
            handler=print_error_handler,
            image_decode="pil",
        )
        train_dataloader = get_savable_loader(train_dataset, worker_config=worker_config)
        train_iterables.append(_EnergonDataloader(train_dataloader))
    val_datasets = get_val_datasets(
        dname,
        batch_size=args.micro_batch_size,
        task_encoder=task_encoder,
        worker_config=WorkerConfig(
            rank=rank,
            world_size=world_size,
            num_workers=args.num_workers,
            data_parallel_group=data_parallel_group,
            worker_debug_path=None,
            worker_log_level=0,
        ),
        packing_buffer_size=packing_buffer_size,
        handler=print_error_handler,
        image_decode="pil",
    )
    val_datasets_without_source = [
        LimitDataset(
            RepeatDataset(
                val_ds,
                worker_config=WorkerConfig(
                    rank=rank,
                    world_size=world_size,
                    num_workers=args.num_workers,
                    data_parallel_group=data_parallel_group,
                    worker_debug_path=None,
                    worker_log_level=0,
                ),
            ),
            length=args.eval_iters * get_num_microbatches(),
            worker_config=WorkerConfig(
                rank=rank,
                world_size=world_size,
                num_workers=args.num_workers,
                data_parallel_group=data_parallel_group,
                worker_debug_path=None,
                worker_log_level=0,
            ),
            reset_after_epoch=True,
        )
        for val_ds, _src_ds in val_datasets
    ]

    if len(train_iterables) == 1:
        train_iterable = train_iterables[0]
    else:
        train_iterable = _InterleavedEnergonDataloader(train_iterables)
    valid_dataloader = [
        _EnergonDataloader(
            get_loader(
                vds,
                worker_config=WorkerConfig(
                    rank=rank,
                    world_size=world_size,
                    num_workers=args.num_workers,
                    data_parallel_group=data_parallel_group,
                    worker_debug_path=None,
                    worker_log_level=0,
                ),
            )
        )
        for vds in val_datasets_without_source
    ]

    return train_iterable, valid_dataloader, _EnergonDataloader(None)


def _get_ltor_masks_and_position_ids(input_ids, target, pad_token):
    """Build masks and position ids for left to right model (energon path)."""
    seq_length = input_ids.shape[1]
    position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
    position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

    loss_mask = torch.ones(target.size(), dtype=torch.float, device=input_ids.device)
    loss_mask[target == pad_token] = 0.0
    loss_mask[target == IGNORE_INDEX] = 0.0

    return loss_mask, position_ids


def get_batch_energon(data_iterator):
    """Generate a batch from the energon external dataloader.

    Adapted from examples/multimodal/train.py get_batch().

    Returns:
        tokens, position_ids, labels, images, loss_mask, attention_mask,
        num_tiles, packed_seq_params
    """
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.parallel_state import is_pipeline_last_stage

    args = get_args()
    imgs = None
    tokens = None
    labels = None
    loss_mask = None
    attention_mask = None
    position_ids = None
    num_tiles = None
    packed_seq_params = None

    pp_size = get_pipeline_model_parallel_world_size()
    if not _is_first_or_last_stage(pp_size):
        return (
            tokens,
            position_ids,
            labels,
            imgs,
            loss_mask,
            attention_mask,
            num_tiles,
            packed_seq_params,
        )

    # We need the exact same sample stream on the first and last PP stages.
    is_first_stage = parallel_state.is_pipeline_first_stage()
    is_last_stage = is_pipeline_last_stage()

    # 1) Load on the first stage and broadcast within TP×CP for that stage.
    if is_first_stage:
        # Broadcast data across TP×CP (for this DP rank).
        # Only the TP×CP leader (rank 0 in the group) reads from the dataloader.
        tp_cp_group = parallel_state.get_tensor_and_context_parallel_group()
        if data_iterator is not None and tp_cp_group.rank() == 0:
            data = next(data_iterator)
            # Optional debug: print lightweight batch fingerprints to validate
            # that early-step sample streams are consistent across PP configs.
            if os.getenv("VLM_DEBUG_BATCH_FINGERPRINT", "0") == "1":
                global _BATCH_FP_DEBUG_COUNTER
                _BATCH_FP_DEBUG_COUNTER += 1
                virt_idx = getattr(data_iterator, "_last_i", -1)
                rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
                pp_rank = parallel_state.get_pipeline_model_parallel_rank()
                dp_rank = parallel_state.get_data_parallel_rank()
                tp_rank = parallel_state.get_tensor_model_parallel_rank()
                itype = type(data_iterator).__name__
                tok = data["tokens"].long()
                lab = data["labels"].long()
                tok_sum = int(tok.sum().item())
                tok_abs = int(tok.abs().sum().item())
                lab_sum = int(lab.sum().item())
                lab_abs = int(lab.abs().sum().item())
                print(
                    "VLM_BATCH_FP "
                    f"count={_BATCH_FP_DEBUG_COUNTER} "
                    f"rank={rank} pp={pp_rank} dp={dp_rank} tp={tp_rank} "
                    f"iter_type={itype} vloader={virt_idx} "
                    f"tok_sum={tok_sum} tok_abs={tok_abs} "
                    f"lab_sum={lab_sum} lab_abs={lab_abs}",
                    flush=True,
                )
            if os.getenv("VLM_DEBUG_BATCH_KEYS", "0") == "1":
                global _BATCH_KEY_DEBUG_COUNTER
                limit = int(os.getenv("VLM_DEBUG_BATCH_KEYS_LIMIT", "8"))
                if _BATCH_KEY_DEBUG_COUNTER < limit:
                    _BATCH_KEY_DEBUG_COUNTER += 1
                    virt_idx = getattr(data_iterator, "_last_i", -1)
                    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
                    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
                    dp_rank = parallel_state.get_data_parallel_rank()
                    tp_rank = parallel_state.get_tensor_model_parallel_rank()
                    keys = data.get("__key__", [])
                    if isinstance(keys, str):
                        keys = [keys]
                    joined = "|".join(str(k) for k in keys)
                    print(
                        "VLM_BATCH_KEYS "
                        f"count={_BATCH_KEY_DEBUG_COUNTER} "
                        f"rank={rank} pp={pp_rank} dp={dp_rank} tp={tp_rank} "
                        f"vloader={virt_idx} "
                        f"keys={joined}",
                        flush=True,
                    )
        else:
            data = None

        data_text = tensor_parallel.broadcast_data(
            ["tokens"], data, torch.int64, tp_group=tp_cp_group
        )["tokens"]
        labels = tensor_parallel.broadcast_data(
            ["labels"], data, torch.int64, tp_group=tp_cp_group
        )["labels"]
        imgs = tensor_parallel.broadcast_data(
            ["imgs"], data, torch.float32, tp_group=tp_cp_group
        )["imgs"]
        num_tiles = tensor_parallel.broadcast_data(
            ["num_tiles"], data, torch.int32, tp_group=tp_cp_group
        )["num_tiles"]
        cu_lengths = tensor_parallel.broadcast_data(
            ["cu_lengths"], data, torch.int32, tp_group=tp_cp_group
        )["cu_lengths"]
        max_lengths = tensor_parallel.broadcast_data(
            ["max_lengths"], data, torch.int32, tp_group=tp_cp_group
        )["max_lengths"]

    # 2) If pipelined, broadcast tokens/labels/packing metadata from the first stage to the
    #    last stage via the embedding group. (We intentionally do NOT broadcast images.)
    if pp_size > 1:
        embd_group = parallel_state.get_embedding_group(check_initialized=False)
        assert embd_group is not None and embd_group.size() == 2, (
            "Expected an embedding group of size 2 (first+last PP stages) when PP>1."
        )

        embd_data_i64 = None
        embd_data_i32 = None
        if embd_group.rank() == 0:
            # Sender: first stage rank for this TP partition.
            assert data_text is not None and labels is not None
            assert num_tiles is not None and cu_lengths is not None and max_lengths is not None
            embd_data_i64 = {"tokens": data_text, "labels": labels}
            embd_data_i32 = {
                "num_tiles": num_tiles,
                "cu_lengths": cu_lengths,
                "max_lengths": max_lengths,
            }

        out_i64 = tensor_parallel.broadcast_data(
            ["tokens", "labels"], embd_data_i64, torch.int64, tp_group=embd_group
        )
        out_i32 = tensor_parallel.broadcast_data(
            ["num_tiles", "cu_lengths", "max_lengths"],
            embd_data_i32,
            torch.int32,
            tp_group=embd_group,
        )

        data_text = out_i64["tokens"]
        labels = out_i64["labels"]
        num_tiles = out_i32["num_tiles"]
        cu_lengths = out_i32["cu_lengths"]
        max_lengths = out_i32["max_lengths"]

        # Last PP stage doesn't need images.
        if is_last_stage:
            imgs = None
        else:
            # Non-first ranks should never see images.
            if not is_first_stage:
                imgs = None

        # No image input (text-only sample). (On last stage, imgs is None, so use num_tiles.)
        if imgs is None:
            if num_tiles.numel() == 1 and int(num_tiles.view(-1)[0].item()) == 0:
                num_tiles = torch.tensor([], dtype=torch.int, device=data_text.device)
        else:
            if imgs.shape == torch.Size([1, 1]):
                imgs = torch.tensor([], dtype=torch.float32, device=data_text.device)
                num_tiles = torch.tensor([], dtype=torch.int, device=data_text.device)
    else:
        # PP==1: normalize Energon's dummy (text-only) image/tiles sentinel.
        if imgs is not None and imgs.shape == torch.Size([1, 1]):
            imgs = torch.tensor([], dtype=torch.float32, device=data_text.device)
            num_tiles = torch.tensor([], dtype=torch.int, device=data_text.device)

    # If cu_lengths and max_lengths are non-dummy, construct PackedSeqParams.
    if cu_lengths.shape != torch.Size([1, 1]):
        assert cu_lengths.shape[0] == max_lengths.shape[0] == 1, \
            "micro-batch-size must be 1 for packing"
        cu_lengths = cu_lengths[0]
        max_lengths = max_lengths[0]
        packed_seq_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_lengths,
            cu_seqlens_kv=cu_lengths,
            max_seqlen_q=max_lengths,
            max_seqlen_kv=max_lengths,
        )

    if os.getenv("VLM_DEBUG_PACKED_FP", "0") == "1":
        global _PACKED_FP_DEBUG_COUNTER
        limit = int(os.getenv("VLM_DEBUG_PACKED_FP_LIMIT", "16"))
        if _PACKED_FP_DEBUG_COUNTER < limit:
            _PACKED_FP_DEBUG_COUNTER += 1
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
            pp_rank = parallel_state.get_pipeline_model_parallel_rank()
            dp_rank = parallel_state.get_data_parallel_rank()
            tp_rank = parallel_state.get_tensor_model_parallel_rank()
            qkv_format = packed_seq_params.qkv_format if packed_seq_params is not None else "none"
            max_q = int(packed_seq_params.max_seqlen_q) if packed_seq_params is not None else -1
            max_kv = int(packed_seq_params.max_seqlen_kv) if packed_seq_params is not None else -1
            cu_last = (
                int(packed_seq_params.cu_seqlens_q[-1].item())
                if packed_seq_params is not None and packed_seq_params.cu_seqlens_q is not None
                else -1
            )
            print(
                "VLM_PACKED_FP "
                f"count={_PACKED_FP_DEBUG_COUNTER} "
                f"rank={rank} pp={pp_rank} dp={dp_rank} tp={tp_rank} "
                f"stage=batch "
                f"text_shape={tuple(data_text.shape)} "
                f"labels_shape={tuple(labels.shape)} "
                f"qkv_format={qkv_format} "
                f"max_q={max_q} max_kv={max_kv} cu_last={cu_last}",
                flush=True,
            )

    tokens_ = data_text.long()
    tokenizer = get_tokenizer()
    text_length = tokens_.shape[1]
    tokens = tokens_[:, :text_length].contiguous()
    labels = labels[:, 1:text_length + 1].contiguous()
    assert tokens.shape == labels.shape, f"tokens: {tokens.shape} != labels: {labels.shape}"

    loss_mask, position_ids = _get_ltor_masks_and_position_ids(tokens, labels, tokenizer.pad)

    if os.getenv("VLM_DEBUG_POST_BCAST_FP", "0") == "1":
        global _POST_BCAST_FP_DEBUG_COUNTER
        limit = int(os.getenv("VLM_DEBUG_POST_BCAST_FP_LIMIT", "16"))
        if _POST_BCAST_FP_DEBUG_COUNTER < limit and _is_first_or_last_stage(pp_size):
            _POST_BCAST_FP_DEBUG_COUNTER += 1
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
            pp_rank = parallel_state.get_pipeline_model_parallel_rank()
            dp_rank = parallel_state.get_data_parallel_rank()
            tp_rank = parallel_state.get_tensor_model_parallel_rank()
            tok_sum = int(tokens.sum().item())
            tok_abs = int(tokens.abs().sum().item())
            lab_sum = int(labels.sum().item())
            lab_abs = int(labels.abs().sum().item())
            mask_sum = float(loss_mask.sum().item())
            num_tiles_sum = int(num_tiles.sum().item()) if num_tiles is not None and num_tiles.numel() > 0 else 0
            print(
                "VLM_POST_BCAST_FP "
                f"count={_POST_BCAST_FP_DEBUG_COUNTER} "
                f"rank={rank} pp={pp_rank} dp={dp_rank} tp={tp_rank} "
                f"first={int(is_first_stage)} last={int(is_last_stage)} "
                f"text_len={int(text_length)} "
                f"tok_sum={tok_sum} tok_abs={tok_abs} "
                f"lab_sum={lab_sum} lab_abs={lab_abs} "
                f"mask_sum={mask_sum:.1f} tiles_sum={num_tiles_sum}",
                flush=True,
            )

    # CP/SP sharding.
    if args.context_parallel_size > 1 or args.sequence_parallel:
        # Current multimodal CP path only supports a single sample per microbatch.
        if args.context_parallel_size > 1:
            assert (
                tokens.shape[0] == 1
            ), f"micro-batch-size must be 1 with CP for VLM energon path, got {tokens.shape[0]}"

        image_token_index = DEFAULT_IMAGE_TOKEN_INDEX
        _vmt = getattr(args, 'vision_model_type', None) or "clip"
        _cls = 0 if _vmt.startswith("siglip") else 1
        _ps = getattr(args, 'pixel_shuffle_factor', 2) if getattr(args, 'pixel_shuffle', False) else False
        img_seq_len = get_num_image_embeddings(
            args.img_h, args.img_w, args.patch_dim, _vmt,
            args.disable_vision_class_token, _cls, _ps
        )

        # Compute per-sample net image token contribution (max across batch).
        # Each image token in the text is replaced by img_seq_len embeddings,
        # so net addition per image = img_seq_len - 1.
        image_token_mask = tokens == image_token_index
        num_images_per_sample = torch.sum(image_token_mask, dim=-1)  # [B]
        num_image_embeddings = (img_seq_len * num_images_per_sample - num_images_per_sample).max().item()

        # _preprocess_data always pads/truncates combined embeddings to decoder_seq_length,
        # so cu_seqlens_padded must reflect that length. Compute text padding accordingly.
        mp_padding_needed = max(0, args.decoder_seq_length - text_length - num_image_embeddings)
        tokens, position_ids, labels, loss_mask = [
            torch.nn.functional.pad(item, (0, mp_padding_needed))
            for item in (tokens, position_ids, labels, loss_mask)
        ]
        # Build packed_seq_params matching the mock get_batch path.
        # Use context_parallel.get_packed_seq_params which correctly computes
        # cu_seqlens and cu_seqlens_padded based on text + image token counts.
        packed_seq_params = context_parallel.get_packed_seq_params(
            tokens, num_image_embeddings, mp_padding_needed,
            args.context_parallel_size, True,
        )
        # Force SBHD format for energon path: llava_model CP sharding produces
        # [S/CP, B, H] (sbhd), not [T, H, D] (thd). THD would cause shape
        # mismatches in RoPE because tensor is 4D not 3D.
        # Set all cu_seqlens to None to avoid triggering THD RoPE path.
        packed_seq_params.qkv_format = 'sbhd'
        packed_seq_params.cu_seqlens_q = None
        packed_seq_params.cu_seqlens_kv = None
        packed_seq_params.cu_seqlens_q_padded = None
        packed_seq_params.cu_seqlens_kv_padded = None

    attention_mask = None
    return (tokens, position_ids, labels, imgs, loss_mask, attention_mask,
            num_tiles, packed_seq_params)


def get_batch(data_iterator):
    """Generate a batch (mock dataset path).

    Args:
        data_iterator: Iterable dataset.

    Returns:
        sample: A data sample with images, tokens, etc.
    """
    args = get_args()
    cp_size = args.context_parallel_size
    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None

    data_i = tensor_parallel.broadcast_data(["tokens", "position_ids", "labels"], data, torch.int64)
    data_f = tensor_parallel.broadcast_data(["image", "loss_mask"], data, torch.float32)

    batch = dict()
    packed_seq_params = None
    image_token_mask = None
    # Create batch with tokens and position_ids for CP sharding.
    tokens = data_i["tokens"].long()
    position_ids = data_i["position_ids"].long()
    labels = data_i["labels"].long()
    loss_mask = data_f["loss_mask"].float()
    images = data_f["image"].float()

    if cp_size > 1 or args.sequence_parallel:
        _vmt = getattr(args, 'vision_model_type', None) or "clip"
        _cls = 0 if _vmt.startswith("siglip") else 1
        _ps = getattr(args, 'pixel_shuffle_factor', 2) if getattr(args, 'pixel_shuffle', False) else False
        # Calculate the number of image embedding tokens will be added to text tokens
        num_image_embeddings_per_tile = get_num_image_embeddings(
            args.img_h, args.img_w, args.patch_dim, _vmt,
            args.disable_vision_class_token, _cls, _ps
        )
        # Pad to make sure the text sequence can be sharded equally by CP chunks.
        image_token_mask = tokens == DEFAULT_IMAGE_TOKEN_INDEX
        num_images_per_sample = torch.sum(image_token_mask, dim=-1)
        img_seq_len = (num_image_embeddings_per_tile * num_images_per_sample - num_images_per_sample).max()
        mp_padding_needed_for_text = context_parallel.get_padding(
            tokens.shape[1] + img_seq_len,
            args.context_parallel_size,
            args.tensor_model_parallel_size,
            args.sequence_parallel,
            args.decoder_tp_comm_overlap,
            args.decoder_seq_length
        )
        if mp_padding_needed_for_text > 0:
            tokens, position_ids, labels, loss_mask = [torch.nn.functional.pad(item, (0, mp_padding_needed_for_text)) for item in (tokens, position_ids, labels, loss_mask)]
        packed_seq_params = context_parallel.get_packed_seq_params(tokens, img_seq_len, mp_padding_needed_for_text, cp_size, args.use_packed_sequence)

        # SBHD format: cu_seqlens must be None to avoid triggering THD RoPE path
        # with a 4D tensor. cu_seqlens is only valid for THD format (3D tensors).
        if packed_seq_params.qkv_format != 'thd':
            packed_seq_params.cu_seqlens_q = None
            packed_seq_params.cu_seqlens_kv = None

        if packed_seq_params.qkv_format == 'thd':
            # Reshape from [B,S] to [T,1]
            tokens = (
                tokens.contiguous()
                .view(tokens.shape[0] * tokens.shape[1])
                .unsqueeze(0)
            )
            position_ids = (
                position_ids.contiguous()
                .view(position_ids.shape[0] * position_ids.shape[1])
                .unsqueeze(0)
            )
            labels = labels.view(labels.shape[0] * labels.shape[1]).unsqueeze(0)
            loss_mask = loss_mask.view(
                loss_mask.shape[0] * loss_mask.shape[1]
            ).unsqueeze(0)

    attention_mask = None  # Use the attention mask type defined in layer spec. Typically no mask for the vision model and causal mask for the vision model.

    # Return with num_tiles=None for mock path (model uses default single-tile logic).
    return tokens, position_ids, labels, images, loss_mask, attention_mask, None, packed_seq_params


def forward_step(data_iterator, model: LLaVAModel):
    """Forward training step.

    Args:
        data_iterator: Iterable dataset.
        model (megatron.core.models.multimodal.llava_model.LLaVAModel): Multimodal model

    Returns:
        output_tensor (torch.Tensor): Loss of shape [b, s] if labels are provided, otherwise logits of shape [b, s, vocab_size].
        loss_func (callable): Loss function with a loss mask specified.
    """
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    if getattr(args, 'dataloader_type', None) == 'external':
        tokens, position_ids, labels, images, loss_mask, attention_mask, num_tiles, packed_seq_params = get_batch_energon(data_iterator)
    else:
        tokens, position_ids, labels, images, loss_mask, attention_mask, num_tiles, packed_seq_params = get_batch(data_iterator)
    timers('batch-generator').stop()

    output_tensor, loss_mask = model(
        images, tokens, position_ids, attention_mask, labels, loss_mask,
        num_image_tiles=num_tiles, packed_seq_params=packed_seq_params
    )

    return output_tensor, partial(loss_func, loss_mask)


def add_vlm_extra_args(parser):
    """Extra arguments."""
    group = parser.add_argument_group(title='vision language model specific arguments')
    group.add_argument(
        '--freeze-LM', action='store_true', default=False, help="Freeze language model weights"
    )
    group.add_argument(
        '--freeze-ViT', action='store_true', default=False, help="Freeze vision model (ViT) weights"
    )
    group.add_argument(
        "--disable-vision-class-token",
        action="store_true",
        default=False,
        help="Drop vision model class token",
    )
    group.add_argument("--dataloader-seq-length", type=int, help="Make dataloader to produce sequences of specific length.")
    group.add_argument("--vision-model-type", type=str, default=None, help="Vision model type (clip, siglip, siglip2_base).")
    group.add_argument("--pixel-shuffle", action="store_true", default=False, help="Enable pixel shuffle for vision encoder.")
    group.add_argument("--pixel-shuffle-factor", type=int, default=2, help="Pixel shuffle downsampling factor.")
    group.add_argument("--encoder-hidden-size", type=int, default=None, help="Vision encoder hidden size (defaults to --hidden-size).")
    group.add_argument("--encoder-ffn-hidden-size", type=int, default=None, help="Vision encoder FFN hidden size.")
    group.add_argument("--encoder-num-attention-heads", type=int, default=None, help="Vision encoder attention heads.")
    group.add_argument("--decoder-tp-comm-overlap", action="store_true", default=False, help="Enables the overlap of "
                        "Tensor parallel communication and GEMM kernels in Decoder only. "
                        "Please provide decoder-seq-length when using this feature.")
    group.add_argument(
        "--use-packed-sequence",
        action="store_true",
        default=False,
        help="Use packed sequence",
    )
    # Energon external dataloader arguments.
    group.add_argument("--prompt-path", type=str, default=None, help="Path to manual prompts JSON file (energon).")
    group.add_argument("--language-model-type", type=str, default=None, help="Language model type (energon).")
    group.add_argument("--tokenizer-prompt-format", type=str, default=None,
                       choices=["mistral", "llama3", "chatml", "nvlm-yi-34b", "qwen2p0", "qwen2p5", "llama3p1",
                                "nemotron5", "nemotron5-aligned", None],
                       help="Prompt format to use with the tokenizer (energon).")
    group.add_argument("--packing-buffer-size", type=int, default=None,
                       help="Enable sample packing by setting the buffer size > 0 (energon).")
    group.add_argument("--packing-seq-length", type=int, default=0,
                       help="Packing sequence length. Must be > 0 if using packing (energon).")
    group.add_argument("--use-tiling", action="store_true", default=False, help="Use input image tiling (energon).")
    group.add_argument("--max-num-tiles", type=int, default=1, help="Maximum number of image tiles (energon).")
    group.add_argument("--use-thumbnail", action="store_true", default=False, help="Add image thumbnail as a tile (energon).")
    group.add_argument("--use-tile-tags", action="store_true", default=False, help="Use tile tags (energon).")
    group.add_argument("--allow-missing-vision-projection-checkpoint", action="store_true", default=False,
                       help="Allow loading checkpoint without vision projection weights.")
    group.add_argument("--use-loss-scaling", action="store_true", default=False,
                       help="Scale loss based on conversation turn length (energon).")
    group.add_argument("--use-area-weighted-aspect-ratio", action="store_true", default=False,
                       help="Use area-weighted aspect ratio for tiling (energon).")
    group.add_argument("--special-tokens", nargs="*", default=["<image>"],
                       help="Special tokens used in the multimodal model.")
    group.add_argument("--eos-id", type=int, default=None, help="Termination id for MultiModal Tokenizer (energon).")
    group.add_argument("--use-te", action="store_true", default=False, help="Use TransformerEngine for vision model.")
    group.add_argument("--image-tag-type", type=str, choices=["nvlm", "internvl", ""], default="",
                       help="Surround image tokens with tags.")
    group.add_argument("--num-frames", type=int, default=1, help="Number of video frames.")
    group.add_argument("--dataloader-save", type=str, default=None, help="Energon dataloader state save path.")
    group.add_argument("--force-system-message", action="store_true", default=False, help="Force system message.")
    group.add_argument("--recompute-vision", action="store_true", default=False, help="Activation checkpointing in ViT.")
    group.add_argument("--online-evaluation-config", type=str, default=None, help="Online evaluation config.")
    group.add_argument("--use-mcore-inference", action="store_true", default=False, help="Use MCore inference API.")
    return parser


def llava_embedding_ranks(pp_ranks):
    """LLaVA's embedding ranks consist of the first and last ranks of the pipeline.
    Args:
        pp_ranks: A list of global ranks that constitute a pipeline group.
    """
    first_rank = pp_ranks[0]
    last_rank = pp_ranks[-1]

    if len(pp_ranks) == 1:
        return [first_rank]
    else:
        return [first_rank, last_rank]


def llava_position_embedding_ranks(pp_ranks):
    """LLaVA's positional embeddings are on the first rank stage
    Args:
        pp_ranks: A list of global ranks that constitute a pipeline group.
    """
    return [pp_ranks[0]]


if __name__ == "__main__":
    # Determine dataloader type from command line to choose the right data provider.
    # We need to check before pretrain() parses all args because pretrain() needs
    # the provider at call time.
    _use_external = any(
        (a == '--dataloader-type' and i + 1 < len(sys.argv) and sys.argv[i + 1] == 'external')
        or a == '--dataloader-type=external'
        for i, a in enumerate(sys.argv)
    )
    if _use_external:
        # Energon external dataloader path.
        train_valid_test_dataloaders_provider.is_distributed = True
        pretrain(
            train_valid_test_dataloaders_provider,
            model_provider,
            ModelType.encoder_or_decoder,
            forward_step,
            args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
            extra_args_provider=add_vlm_extra_args,
            get_embedding_ranks=llava_embedding_ranks,
            get_position_embedding_ranks=llava_position_embedding_ranks,
        )
    else:
        # Mock dataset path (default).
        train_valid_test_datasets_provider.is_distributed = True
        pretrain(
            train_valid_test_datasets_provider,
            model_provider,
            ModelType.encoder_or_decoder,
            forward_step,
            args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
            extra_args_provider=add_vlm_extra_args,
            get_embedding_ranks=llava_embedding_ranks,
            get_position_embedding_ranks=llava_position_embedding_ranks,
        )
