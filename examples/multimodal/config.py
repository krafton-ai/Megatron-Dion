# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
from dataclasses import dataclass

import torch

from megatron.core.activations import fast_gelu, quick_gelu, squared_relu


def get_language_model_config(config):
    if config.language_model_type == "llama3_8b":
        config.activation_func = torch.nn.functional.silu
        config.add_bias_linear = False
        config.bias_activation_fusion = False
        config.gated_linear_unit = True
        config.apply_query_key_layer_scaling = False
        config.layernorm_zero_centered_gamma = (
            False  # Zero centered gamma not supported for RMSNorm
        )
        config.bias_dropout_fusion = False
        config.apply_rope_fusion = False
        config.attention_softmax_in_fp32 = True
        config.ffn_hidden_size = 14336
    elif config.language_model_type == "llama3.1_8b":
        config.activation_func = torch.nn.functional.silu
        config.add_bias_linear = False
        config.bias_activation_fusion = False
        config.gated_linear_unit = True
        config.apply_query_key_layer_scaling = False
        config.layernorm_zero_centered_gamma = (
            False  # Zero centered gamma not supported for RMSNorm
        )
        config.bias_dropout_fusion = False
        config.apply_rope_fusion = False
        config.attention_softmax_in_fp32 = True
        config.ffn_hidden_size = 14336
    elif config.language_model_type == "llama3.1_70B":
        config.activation_func = torch.nn.functional.silu
        config.add_bias_linear = False
        config.bias_activation_fusion = False
        config.gated_linear_unit = True
        config.apply_query_key_layer_scaling = False
        config.layernorm_zero_centered_gamma = (
            False  # Zero centered gamma not supported for RMSNorm
        )
        config.bias_dropout_fusion = False
        config.apply_rope_fusion = False
        config.attention_softmax_in_fp32 = True
        config.ffn_hidden_size = 28672
    elif config.language_model_type == "mistral_7b":
        config.activation_func = torch.nn.functional.silu
        config.add_bias_linear = False
        config.bias_activation_fusion = False
        config.gated_linear_unit = True
        config.apply_query_key_layer_scaling = False
        config.layernorm_zero_centered_gamma = (
            False  # Zero centered gamma not supported for RMSNorm
        )
        config.bias_dropout_fusion = False
        config.apply_rope_fusion = False
        config.attention_softmax_in_fp32 = True
        config.ffn_hidden_size = 14336
    elif config.language_model_type == "nemotron5-8b":
        config.add_bias_linear = False
        config.bias_activation_fusion = False
        config.gated_linear_unit = False
        config.bias_dropout_fusion = False
        config.apply_rope_fusion = False
        config.activation_func = squared_relu
        config.ffn_hidden_size = 21504
        config.masked_softmax_fusion = True
        config.attention_softmax_in_fp32 = True
    elif config.language_model_type == "yi-34b":
        config.activation_func = torch.nn.functional.silu
        config.add_bias_linear = False
        config.bias_activation_fusion = False
        config.gated_linear_unit = True
        config.apply_query_key_layer_scaling = False
        config.layernorm_zero_centered_gamma = (
            False  # Zero centered gamma not supported for RMSNorm
        )
        config.bias_dropout_fusion = False
        config.apply_rope_fusion = False
        config.attention_softmax_in_fp32 = True
        config.ffn_hidden_size = 20480
    elif config.language_model_type == "qwen2.0_72B":
        config.activation_func = torch.nn.functional.silu
        config.add_bias_linear = False
        config.add_qkv_bias = True
        config.bias_activation_fusion = False
        config.gated_linear_unit = True
        config.apply_query_key_layer_scaling = False
        config.layernorm_zero_centered_gamma = (
            False  # Zero centered gamma not supported for RMSNorm
        )
        config.bias_dropout_fusion = False
        config.apply_rope_fusion = False
        config.attention_softmax_in_fp32 = True
        config.ffn_hidden_size = 29568
    elif config.language_model_type == "qwen2.5_7B":
        config.activation_func = torch.nn.functional.silu
        config.add_bias_linear = False
        config.add_qkv_bias = True
        config.bias_activation_fusion = False
        config.gated_linear_unit = True
        config.apply_query_key_layer_scaling = False
        config.layernorm_zero_centered_gamma = (
            False  # Zero centered gamma not supported for RMSNorm
        )
        config.bias_dropout_fusion = False
        config.apply_rope_fusion = False
        config.attention_softmax_in_fp32 = True
        config.ffn_hidden_size = 18944
    elif config.language_model_type == "qwen2.5_72B":
        config.activation_func = torch.nn.functional.silu
        config.add_bias_linear = False
        config.add_qkv_bias = True
        config.bias_activation_fusion = False
        config.gated_linear_unit = True
        config.apply_query_key_layer_scaling = False
        config.layernorm_zero_centered_gamma = (
            False  # Zero centered gamma not supported for RMSNorm
        )
        config.bias_dropout_fusion = False
        config.apply_rope_fusion = False
        config.attention_softmax_in_fp32 = True
        config.ffn_hidden_size = 29568
    elif config.language_model_type == "nemotron5-hybrid-8b":
        config.activation_func = squared_relu
        config.squared_relu = True
        config.add_bias_linear = False
        config.bias_activation_fusion = False
        config.apply_query_key_layer_scaling = False
        config.gated_linear_unit = False
        config.layernorm_zero_centered_gamma = (
            False  # Zero centered gamma not supported for RMSNorm
        )
        config.bias_dropout_fusion = False
        config.attention_softmax_in_fp32 = True
        config.ffn_hidden_size = 21504
    elif config.language_model_type == "nemotron5-hybrid-56b":
        config.activation_func = squared_relu
        config.squared_relu = True
        config.add_bias_linear = False
        config.bias_activation_fusion = False
        config.apply_query_key_layer_scaling = False
        config.gated_linear_unit = False
        config.layernorm_zero_centered_gamma = (
            False  # Zero centered gamma not supported for RMSNorm
        )
        config.bias_dropout_fusion = False
        config.attention_softmax_in_fp32 = True
        config.ffn_hidden_size = 32768
        config.mamba_state_dim = 256
    elif config.language_model_type == "llama3.2_1b":
        config.activation_func = torch.nn.functional.silu
        config.add_bias_linear = False
        config.bias_activation_fusion = False
        config.gated_linear_unit = True
        config.apply_query_key_layer_scaling = False
        config.layernorm_zero_centered_gamma = (
            False  # Zero centered gamma not supported for RMSNorm
        )
        config.bias_dropout_fusion = False
        config.apply_rope_fusion = False
        config.attention_softmax_in_fp32 = True
        config.ffn_hidden_size = 8192
    elif config.language_model_type == "smollm2_360m":
        config.activation_func = torch.nn.functional.silu
        config.add_bias_linear = False
        config.bias_activation_fusion = False
        config.gated_linear_unit = True
        config.apply_query_key_layer_scaling = False
        config.layernorm_zero_centered_gamma = False
        config.bias_dropout_fusion = False
        config.apply_rope_fusion = False
        config.attention_softmax_in_fp32 = True
        config.ffn_hidden_size = 2560
    elif config.language_model_type.startswith("hf://"):
        # Loaded from HuggingFace config file.
        import transformers
        hf_config = transformers.AutoConfig.from_pretrained(config.language_model_type.split("hf://")[1])
        config.hf_config = hf_config
        config.hidden_size = hf_config.hidden_size
    else:
        raise ValueError(f"unknown language model type {config.language_model_type}")

    return config


def get_vision_model_config(config, apply_query_key_layer_scaling):
    if config.vision_model_type == "clip":
        config.num_layers = 24
        config.num_attention_heads = 16
        config.add_bias_linear = True
        config.add_qkv_bias = True
        config.hidden_size = 1024
        config.hidden_dropout = 0.0
        config.attention_dropout = 0.0
        config.ffn_hidden_size = 4096
        config.gated_linear_unit = False
        config.activation_func = quick_gelu
        config.kv_channels = 64
        config.num_query_groups = 16
        config.layernorm_zero_centered_gamma = False
        config.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        config.bias_activation_fusion = False
        config.bias_dropout_fusion = False
        config.attention_softmax_in_fp32 = True
        config.normalization = 'LayerNorm'
        config.apply_rope_fusion = False
    elif config.vision_model_type == "siglip":
        config.num_layers = 27
        config.num_attention_heads = 16
        config.add_bias_linear = True
        config.add_qkv_bias = True
        config.hidden_size = 1152
        config.hidden_dropout = 0.0
        config.attention_dropout = 0.0
        config.ffn_hidden_size = 4304
        config.gated_linear_unit = False
        config.activation_func = fast_gelu
        config.kv_channels = 72
        config.num_query_groups = 16
        config.layernorm_zero_centered_gamma = False
        config.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        config.bias_activation_fusion = False
        config.bias_dropout_fusion = False
        config.attention_softmax_in_fp32 = True
        config.normalization = 'LayerNorm'
        config.apply_rope_fusion = False
        config.qk_layernorm = False
        config.layernorm_epsilon = 1e-6
    elif config.vision_model_type == "siglip2_base":
        config.num_layers = 12
        config.add_bias_linear = True
        config.add_qkv_bias = True
        config.hidden_size = 768
        config.hidden_dropout = 0.0
        config.attention_dropout = 0.0
        config.ffn_hidden_size = 3072
        config.gated_linear_unit = False
        config.activation_func = fast_gelu
        # SigLIP2-base default is 12 heads, but TP=8 requires head count divisible by TP.
        # Keep defaults when valid; otherwise choose the smallest TP-compatible divisor of hidden size.
        base_num_attention_heads = 12
        override_num_attention_heads = getattr(config, "encoder_num_attention_heads", None)
        if override_num_attention_heads is not None:
            num_attention_heads = int(override_num_attention_heads)
        else:
            tp_size = max(1, int(getattr(config, "tensor_model_parallel_size", 1)))
            if base_num_attention_heads % tp_size == 0:
                num_attention_heads = base_num_attention_heads
            else:
                valid_heads = [
                    h
                    for h in range(tp_size, config.hidden_size + 1, tp_size)
                    if config.hidden_size % h == 0
                ]
                if not valid_heads:
                    raise ValueError(
                        f"No TP-compatible attention head count for hidden_size={config.hidden_size}, "
                        f"tp={tp_size}"
                    )
                num_attention_heads = next(
                    (h for h in valid_heads if h >= base_num_attention_heads), valid_heads[-1]
                )

        if config.hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({config.hidden_size}) must be divisible by num_attention_heads "
                f"({num_attention_heads}) for vision_model_type=siglip2_base"
            )

        config.num_attention_heads = num_attention_heads
        config.kv_channels = config.hidden_size // config.num_attention_heads
        config.num_query_groups = config.num_attention_heads
        config.layernorm_zero_centered_gamma = False
        config.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        config.bias_activation_fusion = False
        config.bias_dropout_fusion = False
        config.attention_softmax_in_fp32 = True
        config.normalization = 'LayerNorm'
        config.apply_rope_fusion = False
        config.qk_layernorm = False
        config.layernorm_epsilon = 1e-6
    elif config.vision_model_type == "internvit":
        config.num_layers = 45
        config.num_attention_heads = ((24 // config.tensor_model_parallel_size) + 1) * config.tensor_model_parallel_size
        config.num_query_groups = config.num_attention_heads
        config.add_bias_linear = True
        config.add_qkv_bias = False
        config.hidden_size = 3200
        config.hidden_dropout = 0.0
        config.attention_dropout = 0.0
        config.ffn_hidden_size = 12800
        config.gated_linear_unit = False
        config.activation_func = torch.nn.functional.gelu
        config.layernorm_zero_centered_gamma = False
        config.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        config.bias_activation_fusion = False
        config.bias_dropout_fusion = False
        config.attention_softmax_in_fp32 = True
        config.normalization = 'RMSNorm'
        config.layernorm_epsilon = 1e-6
        config.apply_rope_fusion = False
    elif config.vision_model_type == "internvit300M":
        config.num_layers = 24
        config.num_attention_heads = 16
        config.num_query_groups = config.num_attention_heads
        config.add_bias_linear = True
        config.add_qkv_bias = True
        config.hidden_size = 1024
        config.kv_channels = 64
        config.hidden_dropout = 0.0
        config.ffn_hidden_size = 4096
        config.gated_linear_unit = False
        config.activation_func = torch.nn.functional.gelu
        config.layernorm_zero_centered_gamma = False
        config.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        config.bias_activation_fusion = False
        config.bias_dropout_fusion = False
        config.attention_softmax_in_fp32 = True
        config.normalization = 'LayerNorm'
        config.layernorm_epsilon = 1e-6
        config.apply_rope_fusion = False
        config.qk_layernorm = False
    elif config.vision_model_type == "radio":
        config.num_layers = 32
        config.num_attention_heads = 16
        config.add_bias_linear = True
        config.add_qkv_bias = True
        config.hidden_size = 1280
        config.ffn_hidden_size = 5120
        config.gated_linear_unit = False
        config.activation_func = fast_gelu
        config.kv_channels = 80
        config.num_query_groups = 16
        config.layernorm_zero_centered_gamma = False
        config.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        config.bias_activation_fusion = False
        config.bias_dropout_fusion = False
        config.attention_softmax_in_fp32 = True
        config.normalization = 'LayerNorm'
        config.apply_rope_fusion = False
        config.qk_layernorm = False
        config.layernorm_epsilon = 1e-6
    elif config.vision_model_type == "radio-g":
        config.num_layers = 40
        config.num_attention_heads = 24
        config.add_bias_linear = True
        config.add_qkv_bias = True
        config.hidden_size = 1536
        config.ffn_hidden_size = 4096
        config.gated_linear_unit = True
        config.activation_func = torch.nn.functional.silu
        config.kv_channels = 64
        config.num_query_groups = 24
        config.layernorm_zero_centered_gamma = False
        config.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        config.bias_activation_fusion = False
        config.bias_dropout_fusion = False
        config.attention_softmax_in_fp32 = True
        config.normalization = 'LayerNorm'
        config.apply_rope_fusion = False
        config.qk_layernorm = False
        config.layernorm_epsilon = 1e-6
    elif config.vision_model_type == "cradio-g":
        config.num_layers = 40
        config.num_attention_heads = 24
        config.add_bias_linear = True
        config.add_qkv_bias = True
        config.hidden_size = 1536
        config.ffn_hidden_size = 6144
        config.gated_linear_unit = False
        config.activation_func = fast_gelu
        config.kv_channels = 64
        config.num_query_groups = 24
        config.layernorm_zero_centered_gamma = False
        config.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        config.bias_activation_fusion = False
        config.bias_dropout_fusion = False
        config.attention_softmax_in_fp32 = True
        config.normalization = 'LayerNorm'
        config.apply_rope_fusion = False
        config.qk_layernorm = False
        config.layernorm_epsilon = 1e-6
    elif config.vision_model_type.startswith("hf://"):
        import transformers
        hf_config = transformers.AutoConfig.from_pretrained(config.vision_model_type.split("hf://")[1])
        config.hf_config = hf_config
        config.hidden_size = hf_config.hidden_size
    else:
        raise ValueError(f"unknown vision model type {config.vision_model_type}")

    return config


def get_vision_projection_config(config, hidden_size):
    # If using FP8, then keep the whole vision projection in FP8.
    config.first_last_layers_bf16 = False
    config.num_layers_at_start_in_bf16 = 0
    config.num_layers_at_end_in_bf16 = 0

    config.gated_linear_unit = False
    config.bias_activation_fusion = False
    config.add_bias_linear = False
    config.hidden_size = hidden_size  # Used as the vision projection output size, i.e., the input to the language model.
    if config.language_model_type == "llama3_8b":
        config.ffn_hidden_size = 14336
        config.activation_func = torch.nn.functional.gelu
    elif config.language_model_type == "llama3.1_8b":
        config.ffn_hidden_size = 4096
        config.activation_func = torch.nn.functional.gelu
        config.layernorm_epsilon = 1e-5
        config.add_bias_linear = True
        config.normalization = "LayerNorm"
    elif config.language_model_type == "mistral_7b":
        config.ffn_hidden_size = 14336
        config.activation_func = torch.nn.functional.gelu
        config.normalization = None
    elif config.language_model_type == "yi-34b":
        config.ffn_hidden_size = 20480
        config.normalization = "LayerNorm"
        config.activation_func = torch.nn.functional.gelu
    elif config.language_model_type == "qwen2.0_72B":
        config.ffn_hidden_size = 29568
        config.normalization = "LayerNorm"
        config.activation_func = torch.nn.functional.gelu
    elif config.language_model_type == "qwen2.5_7B":
        config.ffn_hidden_size = 3584
        config.activation_func = torch.nn.functional.gelu
    elif config.language_model_type == "qwen2.5_72B":
        config.ffn_hidden_size = 29568
        config.normalization = "LayerNorm"
        config.activation_func = torch.nn.functional.gelu
    elif config.language_model_type == "nemotron5-hybrid-56b":
        config.ffn_hidden_size = 32768
        config.activation_func = squared_relu
    elif config.language_model_type in ("nemotron5-8b", "nemotron5-hybrid-8b"):
        config.ffn_hidden_size = 21504
        config.activation_func = squared_relu
    elif config.language_model_type == "llama3.2_1b":
        config.ffn_hidden_size = 2048
        config.activation_func = torch.nn.functional.gelu
        config.normalization = "LayerNorm"
    elif config.language_model_type == "smollm2_360m":
        config.ffn_hidden_size = 960
        config.activation_func = torch.nn.functional.gelu
        config.normalization = "LayerNorm"
    elif config.language_model_type.startswith("hf://"):
        config.activation_func = torch.nn.functional.gelu
        config.ffn_hidden_size = 4096
        config.normalization = "LayerNorm"
    else:
        raise ValueError(f"unknown language model type {config.language_model_type}")

    return config


@dataclass
class EvaluationConfig:
    """Evaluation related configuration."""
    task: str
    dataset: str = ""

    temperature: float = 1.0
    top_p: float = 0.0
    top_k: int = 0

    out_seq_length: int = 32

    output_path: str = ""

    input_image_path: str = ""
    gt_path: str = ""
    split: str = "validation"

    num_partitions: int = 0
    partition_id: int = 0
    num_samples_per_partition: int = 0
