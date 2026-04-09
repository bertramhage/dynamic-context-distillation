"""Runtime LoRA injection for Chronos-2 during training.

Monkey-patches nn.Linear.forward in Chronos-2's encoder blocks to add
LoRA deltas produced by the hypernetwork. Unlike PEFT (used at evaluation),
this keeps gradients flowing through the LoRA weights back to the hypernetwork.

Adapted from D2L's lora_layer.py — simplified for Chronos-2 where each
sample in the batch gets the same LoRA (no multi-query packing).
"""

from __future__ import annotations

from functools import partial
from operator import attrgetter

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum


# Short name -> Chronos-2 module path within each encoder block's layer[0]
# (TimeSelfAttention only — layer[1] is GroupSelfAttention, left unpatched)
_MODULE_PATHS = {
    "q": "layer.0.self_attention.q",
    "k": "layer.0.self_attention.k",
    "v": "layer.0.self_attention.v",
    "o": "layer.0.self_attention.o",
}


def _lora_forward(
    x: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    scaling: float,
    original_forward,
    *args,
    **kwargs,
) -> torch.Tensor:
    """Modified forward that adds LoRA delta to the base linear output.

    Args:
        x: [batch, seq_len, d_in]
        A: [batch, r, d_in]
        B: [batch, d_out, r]
        scaling: LoRA scaling factor (alpha / r)
        original_forward: the original nn.Linear.forward bound method
    """
    base_out = original_forward(x, *args, **kwargs)

    # LoRA delta: x @ A^T @ B^T
    x_float = x.to(A.dtype)
    # [batch, seq_len, d_in] x [batch, d_in, r] -> [batch, seq_len, r]
    delta = torch.bmm(x_float, A.transpose(-2, -1))
    # [batch, seq_len, r] x [batch, r, d_out] -> [batch, seq_len, d_out]
    delta = torch.bmm(delta, B.transpose(-2, -1))
    delta = delta * scaling

    return (base_out + delta).to(base_out.dtype)


def apply_lora_to_model(
    model: nn.Module,
    lora_dict: dict[str, dict[str, torch.Tensor]],
    scaling: float = 2.0,
) -> list[tuple[nn.Module, callable]]:
    """Monkey-patch Chronos-2 encoder blocks with LoRA weights for one forward pass.

    Args:
        model: The Chronos-2 model (pipeline.model).
        lora_dict: {"q": {"A": [B, L, r, d_in], "B": [B, L, d_out, r]}, ...}
        scaling: LoRA scaling factor (typically alpha / rank).

    Returns:
        List of (module, original_forward) pairs for cleanup via remove_lora.
    """
    encoder = model.encoder
    patches = []

    for short_name, weights in lora_dict.items():
        if short_name not in _MODULE_PATHS:
            continue

        A_all = weights["A"]  # [batch, num_layers, r, d_in]
        B_all = weights["B"]  # [batch, num_layers, d_out, r]
        module_path = _MODULE_PATHS[short_name]

        for layer_idx, block in enumerate(encoder.block):
            module = attrgetter(module_path)(block)
            original_forward = module.forward

            A = A_all[:, layer_idx]  # [batch, r, d_in]
            B = B_all[:, layer_idx]  # [batch, d_out, r]

            module.forward = partial(
                _lora_forward,
                A=A, B=B, scaling=scaling,
                original_forward=original_forward,
            )
            patches.append((module, original_forward))

    return patches


def remove_lora(patches: list[tuple[nn.Module, callable]]) -> None:
    """Restore original forward methods after a training step."""
    for module, original_forward in patches:
        module.forward = original_forward
