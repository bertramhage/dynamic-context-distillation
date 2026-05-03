from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

# Hypernetwork output module short names -> Chronos-2 attention module names.
_ATTN_MODULE_MAP = {
    "q": "self_attention.q",
    "k": "self_attention.k",
    "v": "self_attention.v",
    "o": "self_attention.o",
}

NUM_ENCODER_LAYERS = 12


def save_adapter_to_disk(
    lora_dict: dict[str, dict[str, torch.Tensor]],
    sensor_idx: int,
    adapter_dir: Path,
    *,
    rank: int = 8,
    alpha: int = 16,
    d_model: int = 768,
    output_lora_d_in: int | None = None,
    output_lora_d_out: int | None = None,
    target_modules: list[str] | None = None,
    base_model_name: str = "amazon/chronos-2",
) -> None:
    """Save one sensor's hypernetwork-generated LoRA weights as a PEFT adapter.

    The hypernetwork output has shape:
        lora_dict[module_short]["A"]  ->  [batch, n_layers, r, d_in]
        lora_dict[module_short]["B"]  ->  [batch, n_layers, d_out, r]

    This function extracts ``sensor_idx`` from the batch dimension and writes
    a PEFT-compatible directory with ``adapter_model.safetensors`` and
    ``adapter_config.json``.

    Modules not covered by the hypernetwork (GroupSelfAttention layer.1 and
    output_patch_embedding) are filled with zeros so the adapter is loadable
    by PEFT with the standard target_modules config.
    """
    if target_modules is None:
        target_modules = [
            "self_attention.q",
            "self_attention.k",
            "self_attention.v",
            "self_attention.o",
            "output_patch_embedding.output_layer",
        ]

    adapter_dir.mkdir(parents=True, exist_ok=True)
    state_dict: dict[str, torch.Tensor] = {}

    for short_name, full_name in _ATTN_MODULE_MAP.items():
        if short_name not in lora_dict:
            continue

        A = lora_dict[short_name]["A"]  # [batch, n_layers, r, d_in]
        B = lora_dict[short_name]["B"]  # [batch, n_layers, d_out, r]

        for layer_idx in range(NUM_ENCODER_LAYERS):
            # TimeSelfAttention (layer.0)
            prefix = f"base_model.model.encoder.block.{layer_idx}.layer.0.{full_name}"
            state_dict[f"{prefix}.lora_A.weight"] = A[sensor_idx, layer_idx].cpu().contiguous()
            state_dict[f"{prefix}.lora_B.weight"] = B[sensor_idx, layer_idx].cpu().contiguous()

            # GroupSelfAttention (layer.1): zero weights (no effect on inference).
            prefix_group = f"base_model.model.encoder.block.{layer_idx}.layer.1.{full_name}"
            state_dict[f"{prefix_group}.lora_A.weight"] = torch.zeros(rank, d_model)
            state_dict[f"{prefix_group}.lora_B.weight"] = torch.zeros(d_model, rank)

    # output_patch_embedding — zero weights (hypernetwork doesn't produce these).
    # Use model-derived dimensions when provided, otherwise fall back to d_model.
    out_d_in = d_model if output_lora_d_in is None else int(output_lora_d_in)
    out_d_out = d_model if output_lora_d_out is None else int(output_lora_d_out)
    out_prefix = "base_model.model.output_patch_embedding.output_layer"
    if out_prefix not in {
        k.rsplit(".lora_", 1)[0] for k in state_dict
    }:
        state_dict[f"{out_prefix}.lora_A.weight"] = torch.zeros(rank, out_d_in)
        state_dict[f"{out_prefix}.lora_B.weight"] = torch.zeros(out_d_out, rank)

    save_file(state_dict, str(adapter_dir / "adapter_model.safetensors"))

    config = {
        "peft_type": "LORA",
        "base_model_name_or_path": base_model_name,
        "r": rank,
        "lora_alpha": alpha,
        "lora_dropout": 0.0,
        "bias": "none",
        "target_modules": target_modules,
        "inference_mode": True,
        "fan_in_fan_out": False,
        "init_lora_weights": True,
        "task_type": None,
        "modules_to_save": None,
        "rank_pattern": {},
        "alpha_pattern": {},
    }
    with open(adapter_dir / "adapter_config.json", "w") as f:
        json.dump(config, f, indent=2)
