"""Dynamic batched LoRA runtime for Chronos-2 evaluation.

This module applies per-sample LoRA weights directly in a single batched
forward pass by monkey-patching Chronos-2 TimeSelfAttention projection layers.
The implementation uses plain PyTorch batched matmul primitives and does not
rely on PEFT adapter switching.
"""

from __future__ import annotations

from functools import partial
from operator import attrgetter

import torch
import torch.nn as nn

# Match training-time injection: TimeSelfAttention only (layer.0).
_MODULE_PATHS = {
    "q": "layer.0.self_attention.q",
    "k": "layer.0.self_attention.k",
    "v": "layer.0.self_attention.v",
    "o": "layer.0.self_attention.o",
}


def has_non_base_adapters(
    adapter_ids: list[str],
    base_adapter_id: str = "__none__",
) -> bool:
    """Return True if any task in the batch uses a non-base adapter."""
    return any(str(adapter_id) != str(base_adapter_id) for adapter_id in adapter_ids)


def _dynamic_lora_forward(
    x: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    scaling: float,
    original_forward,
    *args,
    **kwargs,
) -> torch.Tensor:
    """Forward hook that adds batched per-sample LoRA deltas.

    Shapes:
        x: [batch, seq_len, d_in]
        A: [batch, r, d_in]
        B: [batch, d_out, r]
    """
    base_out = original_forward(x, *args, **kwargs)

    if x.shape[0] != A.shape[0] or x.shape[0] != B.shape[0]:
        raise ValueError(
            "Dynamic LoRA batch mismatch: "
            f"x batch={x.shape[0]}, A batch={A.shape[0]}, B batch={B.shape[0]}."
        )

    x_float = x.to(A.dtype)
    delta = torch.bmm(x_float, A.transpose(-2, -1))
    delta = torch.bmm(delta, B.transpose(-2, -1))
    delta = delta * scaling

    return (base_out + delta).to(base_out.dtype)


def apply_dynamic_lora_to_model(
    model: nn.Module,
    lora_batch: dict[str, dict[str, torch.Tensor]],
    scaling: float,
) -> list[tuple[nn.Module, callable]]:
    """Apply dynamic LoRA patches for one batched inference call.

    Args:
        model: Chronos-2 model (`pipeline.model`).
        lora_batch: Batched LoRA tensors:
            lora_batch[module_short]["A"] -> [batch, n_layers, r, d_in]
            lora_batch[module_short]["B"] -> [batch, n_layers, d_out, r]
        scaling: LoRA scaling factor (`alpha / rank`).

    Returns:
        List of `(module, original_forward)` to restore with
        `remove_dynamic_lora`.
    """
    encoder = model.encoder
    patches: list[tuple[nn.Module, callable]] = []

    for short_name, weights in lora_batch.items():
        if short_name not in _MODULE_PATHS:
            continue

        A_all = weights["A"]
        B_all = weights["B"]
        module_path = _MODULE_PATHS[short_name]

        for layer_idx, block in enumerate(encoder.block):
            module = attrgetter(module_path)(block)
            original_forward = module.forward

            A = A_all[:, layer_idx].to(device=module.weight.device)
            B = B_all[:, layer_idx].to(device=module.weight.device)

            module.forward = partial(
                _dynamic_lora_forward,
                A=A,
                B=B,
                scaling=scaling,
                original_forward=original_forward,
            )
            patches.append((module, original_forward))

    return patches


def remove_dynamic_lora(patches: list[tuple[nn.Module, callable]]) -> None:
    """Restore original forward methods after dynamic inference."""
    for module, original_forward in patches:
        module.forward = original_forward
