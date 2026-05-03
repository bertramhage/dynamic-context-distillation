import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * (x * rms)).to(dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int):
        super().__init__()
        self.gate = nn.Linear(d_in, d_hidden, bias=False)
        self.up = nn.Linear(d_in, d_hidden, bias=False)
        self.down = nn.Linear(d_hidden, d_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class CrossAttention(nn.Module):
    """Multi-head cross-attention: queries attend to context key-values."""

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, latents: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        """Args: latents [B, N, D], context [B, S, D]. Returns [B, N, D]."""
        B, N, _ = latents.shape
        S = context.shape[1]

        q = rearrange(self.q_proj(latents), "b n (h d) -> b h n d", h=self.n_heads)
        k = rearrange(self.k_proj(context), "b s (h d) -> b h s d", h=self.n_heads)
        v = rearrange(self.v_proj(context), "b s (h d) -> b h s d", h=self.n_heads)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.o_proj(out)


class SelfAttention(nn.Module):
    """Standard multi-head self-attention."""

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x [B, N, D]. Returns [B, N, D]."""
        B, N, _ = x.shape

        q = rearrange(self.q_proj(x), "b n (h d) -> b h n d", h=self.n_heads)
        k = rearrange(self.k_proj(x), "b n (h d) -> b h n d", h=self.n_heads)
        v = rearrange(self.v_proj(x), "b n (h d) -> b h n d", h=self.n_heads)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.o_proj(out)


class PerceiverBlock(nn.Module):
    """One block: cross-attention + optional self-attention layers + FFN."""

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        n_self_attn: int = 1,
        dropout: float = 0.0,
        ffn_mult: int = 4,
    ):
        super().__init__()
        # Cross-attention
        self.x_attn_norm_latents = RMSNorm(d_model)
        self.x_attn_norm_context = RMSNorm(d_model)
        self.x_attn = CrossAttention(d_model, n_heads, dropout)
        self.x_attn_post_norm = RMSNorm(d_model)

        # Self-attention layers
        self.self_attn_layers = nn.ModuleList()
        for _ in range(n_self_attn):
            self.self_attn_layers.append(nn.ModuleDict({
                "norm": RMSNorm(d_model),
                "attn": SelfAttention(d_model, n_heads, dropout),
                "post_norm": RMSNorm(d_model),
            }))

        # FFN
        self.ffn_pre_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_model * ffn_mult, d_model)
        self.ffn_post_norm = RMSNorm(d_model)

    def forward(
        self, latents: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        # Cross-attention
        residual = latents
        latents = self.x_attn_norm_latents(latents)
        ctx = self.x_attn_norm_context(context)
        latents = self.x_attn_post_norm(self.x_attn(latents, ctx))
        latents = residual + latents

        # Self-attention
        for sa in self.self_attn_layers:
            residual = latents
            latents = sa["norm"](latents)
            latents = sa["post_norm"](sa["attn"](latents))
            latents = residual + latents

        # FFN
        residual = latents
        latents = self.ffn_pre_norm(latents)
        latents = self.ffn_post_norm(self.ffn(latents))
        latents = residual + latents

        return latents


class PerceiverAggregator(nn.Module):
    """Perceiver that compresses context [B, S, D_in] -> [B, N_out, D_latent].

    Produces output queries shaped for the HyperLoRA heads:
        n_output_queries = num_layers * num_modules * r
    which get reshaped to [B, num_layers, num_modules, r, d_latent] downstream.
    """

    def __init__(
        self,
        d_input: int,
        d_latent: int,
        n_latent_queries: int,
        n_output_queries: int,
        num_blocks: int = 2,
        n_self_attn_per_block: int = 1,
        n_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_input = d_input
        self.d_latent = d_latent
        self.n_output_queries = n_output_queries

        # Project input features to latent dimension
        self.input_proj = SwiGLU(d_input, d_input * 4, d_latent)

        # Learned latent queries (bottleneck)
        self.latents_q = nn.Parameter(torch.randn(n_latent_queries, d_latent) * 0.02)

        # Encoder blocks: compress context into latent queries
        self.encoder_blocks = nn.ModuleList([
            PerceiverBlock(d_latent, n_heads, n_self_attn_per_block, dropout)
            for _ in range(num_blocks)
        ])

        # Decoder: output queries cross-attend to encoded latents
        self.output_queries = nn.Parameter(
            torch.randn(n_output_queries, d_latent) * 0.02
        )
        self.decoder_block = PerceiverBlock(
            d_latent, n_heads, n_self_attn=0, dropout=dropout
        )

        self.final_norm = RMSNorm(d_latent)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """Compress context into output queries.

        Args:
            context: [batch, seq_len, d_input] — encoder hidden states.

        Returns:
            [batch, n_output_queries, d_latent]
        """
        B = context.shape[0]

        # Project to latent dim
        context = self.input_proj(context)

        # Expand latent queries for batch
        latents = self.latents_q.unsqueeze(0).expand(B, -1, -1)

        # Encoder: cross-attend latents to context
        for block in self.encoder_blocks:
            latents = block(latents, context)

        # Decoder: output queries cross-attend to encoded latents
        out_q = self.output_queries.unsqueeze(0).expand(B, -1, -1)
        out = self.decoder_block(out_q, latents)
        out = self.final_norm(out)

        return out
