"""HyperLoRA generator for Chronos-2.

Takes Perceiver output and produces LoRA A/B weight matrices for each
target module across all encoder layers.

Adapted from D2L's HyperLoRA: keeps ResMLPBlock pre-head processing and
EinMix projection heads, but simplified for Chronos-2's architecture.
"""

import math

import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import EinMix as Mix

from src.training.perceiver import PerceiverAggregator, RMSNorm

# Chronos-2 LoRA target module short names
TARGET_MODULES = ("q", "k", "v", "o")

# Chronos-2 Base architecture constants
NUM_LAYERS = 12
D_MODEL = 768


class ResMLPBlock(nn.Module):
    """Residual MLP block for pre-head processing (from D2L)."""

    def __init__(self, d: int, d_hidden: int, dropout: float = 0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(d),
            nn.Dropout(dropout),
            nn.Linear(d, d_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d),
            nn.LayerNorm(d),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(x)


class HyperLoRA(nn.Module):
    """Generates LoRA A/B matrices for Chronos-2 via a Perceiver + projection heads.

    Architecture:
        context [B, S, 768] -> Perceiver -> [B, L*M*r, d_latent]
        -> reshape to [B, L, M, r, d_latent]
        -> ResMLPBlock layers
        -> EinMix head -> [B, L, M, r, d_A + d_B]
        -> split into per-module A and B matrices

    Output format (matches orchestration layer expectations):
        {"q": {"A": [B, L, r, d_in], "B": [B, L, d_out, r]}, ...}
    """

    def __init__(
        self,
        d_input: int = D_MODEL,
        d_latent: int = 256,
        lora_rank: int = 8,
        num_layers: int = NUM_LAYERS,
        d_model: int = D_MODEL,
        target_modules: tuple[str, ...] = TARGET_MODULES,
        # Perceiver config
        n_latent_queries: int = 64,
        num_perceiver_blocks: int = 2,
        n_self_attn_per_block: int = 1,
        n_heads: int = 8,
        # Pre-head config
        num_pre_head_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.lora_rank = lora_rank
        self.target_modules = target_modules
        self.num_modules = len(target_modules)
        self.d_latent = d_latent

        n_output_queries = num_layers * self.num_modules * lora_rank

        # Perceiver aggregator
        self.perceiver = PerceiverAggregator(
            d_input=d_input,
            d_latent=d_latent,
            n_latent_queries=n_latent_queries,
            n_output_queries=n_output_queries,
            num_blocks=num_perceiver_blocks,
            n_self_attn_per_block=n_self_attn_per_block,
            n_heads=n_heads,
            dropout=dropout,
        )

        # Pre-head ResMLPBlocks
        self.pre_head = nn.Sequential(*[
            ResMLPBlock(d_latent, d_latent * 4, dropout)
            for _ in range(num_pre_head_layers)
        ])

        # All attention target modules have the same dims for Chronos-2:
        # A: [r, d_model], B: [d_model, r]  -> flattened: d_model + d_model
        d_lora = d_model + d_model

        # EinMix projection head: per-layer, per-module weights
        self.head = Mix(
            "bs n_layers n_modules r d_latent -> bs n_layers n_modules r d_lora",
            weight_shape="n_layers n_modules d_latent d_lora",
            bias_shape=None,
            n_layers=num_layers,
            n_modules=self.num_modules,
            d_latent=d_latent,
            r=lora_rank,
            d_lora=d_lora,
        )

        # Initialize EinMix head with small std to prevent wild initial LoRA outputs
        head_std = 0.5 / math.sqrt(d_latent + d_lora * lora_rank)
        nn.init.normal_(self.head.weight, mean=0.0, std=head_std)

        # Learnable bias and scaler for A and B (from D2L)
        self.bias_A = nn.ParameterDict({
            m: nn.Parameter(
                torch.normal(0, 0.2 / (d_model * lora_rank) ** 0.5,
                             (num_layers, lora_rank, d_model))
            ) for m in target_modules
        })
        self.bias_B = nn.ParameterDict({
            m: nn.Parameter(torch.zeros(num_layers, lora_rank, d_model))
            for m in target_modules
        })
        self.scaler_A = nn.ParameterDict({
            m: nn.Parameter(torch.ones(1, num_layers, lora_rank, 1))
            for m in target_modules
        })
        self.scaler_B = nn.ParameterDict({
            m: nn.Parameter(torch.zeros(1, num_layers, lora_rank, 1))
            for m in target_modules
        })

    def forward(
        self, context: torch.Tensor
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Generate LoRA weights from context encoder hidden states.

        Args:
            context: [batch, seq_len, d_input] from the context encoder.
                     Can also be [batch, num_layers, seq_len, d_input] for
                     per-layer context (will use last layer only).

        Returns:
            Dict mapping module short name -> {"A": [B, L, r, d_in], "B": [B, L, d_out, r]}.
        """
        # Handle per-layer context: use last layer's output
        if context.ndim == 4:
            context = context[:, -1]  # [B, S, D]

        # Perceiver: compress context into output queries
        # [B, n_output_queries, d_latent]
        emb = self.perceiver(context)

        # Reshape to [B, L, M, r, d_latent]
        emb = rearrange(
            emb, "b (l m r) d -> b l m r d",
            l=self.num_layers, m=self.num_modules, r=self.lora_rank,
        )

        # Pre-head processing
        emb = self.pre_head(emb)

        # Normalize before projection (from D2L)
        norm = torch.norm(emb, dim=-1, keepdim=True).clamp(min=1e-8)
        emb = emb / norm

        # Project to LoRA weight space: [B, L, M, r, d_A + d_B]
        flat = self.head(emb)

        # Split into per-module A and B
        lora_dict = {}
        for i, module in enumerate(self.target_modules):
            module_flat = flat[:, :, i]  # [B, L, r, d_A + d_B]
            A_raw, B_raw = module_flat.split(self.d_model, dim=-1)

            # Apply scaler and bias
            A = A_raw * self.scaler_A[module] + self.bias_A[module]
            B = B_raw * self.scaler_B[module] + self.bias_B[module]

            # A: [B, L, r, d_in], B: [B, L, d_out, r]
            lora_dict[module] = {
                "A": A,
                "B": rearrange(B, "b l r d -> b l d r"),
            }

        return lora_dict
