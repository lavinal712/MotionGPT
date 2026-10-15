import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# Cached keys/values: [batch, num_heads, cached_length, head_dim].
KVCache = Tuple[Tensor, Tensor]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


class RotaryEmbedding(nn.Module):
    """RoPE with explicit positions, supporting full and cached forward passes."""

    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        if dim <= 0 or dim % 2 != 0:
            raise ValueError("RoPE dimension must be positive and even.")
        if base <= 0:
            raise ValueError("RoPE base must be positive.")

        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        position_ids: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            q, k: [B, H, Q, D].
            position_ids: [Q] or [B, Q].
        """
        if position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0)

        if (
            position_ids.ndim != 2
            or position_ids.shape[-1] != q.shape[-2]
            or position_ids.shape[0] not in (1, q.shape[0])
        ):
            raise ValueError("position_ids must have shape [Q] or [B, Q].")

        # Keep trigonometric calculations in float32 under mixed precision.
        with torch.autocast(device_type=q.device.type, enabled=False):
            positions = position_ids.to(device=q.device, dtype=torch.float32)
            angles = positions.unsqueeze(-1) * self.inv_freq.float()
            angles = torch.cat((angles, angles), dim=-1).unsqueeze(1)
            cos = angles.cos().to(q.dtype)
            sin = angles.sin().to(q.dtype)

        return (
            q * cos + self._rotate_half(q) * sin,
            k * cos + self._rotate_half(k) * sin,
        )

    @staticmethod
    def _rotate_half(x: Tensor) -> Tensor:
        first, second = x.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)


class CausalSelfAttention(nn.Module):
    """
    Causal self-attention with optional QK normalization and KV caching.

    Masks use True for valid/visible positions.

    Cache is append-only. For sliding-window cache eviction, callers must
    preserve absolute RoPE positions by supplying position_ids explicitly.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        qk_norm: bool = True,
        rope_base: float = 10000.0,
        norm_eps: float = 1e-6,
        bias: bool = False,
    ):
        super().__init__()

        if hidden_dim <= 0 or num_heads <= 0 or hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout = dropout

        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=bias)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)

        self.q_norm = (
            RMSNorm(self.head_dim, norm_eps) if qk_norm else nn.Identity()
        )
        self.k_norm = (
            RMSNorm(self.head_dim, norm_eps) if qk_norm else nn.Identity()
        )
        self.rope = RotaryEmbedding(self.head_dim, rope_base)

    def forward(
        self,
        x: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_value: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[KVCache]]:
        """
        Args:
            x: [B, Q, hidden_dim], containing only new tokens when cached.
            key_padding_mask: bool [B, K]; True means valid key.
                K includes both cached and new tokens.
            attention_mask: optional bool [Q, K] or [B, Q, K].
                Additional restrictions, intersected with the causal mask.
            position_ids: [Q] or [B, Q], absolute positions of new tokens.
            past_key_value: cached rotated keys and values.
            use_cache: return the updated cache if True.

        Returns:
            output: [B, Q, hidden_dim].
            present_key_value: updated cache, or None.

        Padded query outputs are not generally zeroed here. The caller must
        exclude padded target positions from the training loss.
        """
        if x.ndim != 3:
            raise ValueError("x must have shape [B, Q, hidden_dim].")

        batch_size, query_length, hidden_dim = x.shape
        if query_length == 0:
            raise ValueError("At least one input token is required.")

        qkv = self.qkv(x).reshape(
            batch_size, query_length, 3, self.num_heads, self.head_dim
        )
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        past_length = 0
        if past_key_value is not None:
            past_k, past_v = past_key_value
            if (
                past_k.ndim != 4
                or past_k.shape != past_v.shape
                or past_k.shape[:2] != (batch_size, self.num_heads)
                or past_k.shape[-1] != self.head_dim
            ):
                raise ValueError("Invalid KV cache shape.")
            if past_k.device != k.device or past_v.device != v.device:
                raise ValueError("KV cache and input must use the same device.")
            if past_k.dtype != k.dtype or past_v.dtype != v.dtype:
                raise ValueError("KV cache and input must use the same dtype.")
            past_length = past_k.shape[-2]

        if position_ids is None:
            position_ids = torch.arange(
                past_length,
                past_length + query_length,
                device=x.device,
            )

        q, k = self.rope(self.q_norm(q), self.k_norm(k), position_ids)

        if past_key_value is not None:
            k = torch.cat((past_k, k), dim=-2)
            v = torch.cat((past_v, v), dim=-2)

        present_key_value = (k, v) if use_cache else None
        key_length = k.shape[-2]

        # Explicit offset is essential for cached decoding with Q != K.
        query_indices = past_length + torch.arange(
            query_length, device=x.device
        )
        key_indices = torch.arange(key_length, device=x.device)
        allowed = key_indices.unsqueeze(0) <= query_indices.unsqueeze(1)
        allowed = allowed.unsqueeze(0).unsqueeze(0)  # [1, 1, Q, K]

        if key_padding_mask is not None:
            if (
                key_padding_mask.dtype != torch.bool
                or key_padding_mask.shape != (batch_size, key_length)
            ):
                raise ValueError("key_padding_mask must be bool [B, K].")
            allowed = allowed & key_padding_mask.to(x.device)[:, None, None, :]

        if attention_mask is not None:
            if attention_mask.dtype != torch.bool:
                raise ValueError("attention_mask must be boolean.")
            if attention_mask.shape == (query_length, key_length):
                attention_mask = attention_mask[None, None, :, :]
            elif attention_mask.shape == (
                batch_size, query_length, key_length
            ):
                attention_mask = attention_mask[:, None, :, :]
            else:
                raise ValueError("attention_mask must be [Q, K] or [B, Q, K].")
            allowed = allowed & attention_mask.to(x.device)

        # Avoid fully masked softmax rows; discard their outputs afterwards.
        has_visible_key = allowed.any(dim=-1, keepdim=True)
        fallback = key_indices.eq(0)[None, None, None, :]
        safe_mask = allowed | (~has_visible_key & fallback)

        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=safe_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        output = output.transpose(1, 2).contiguous().reshape(
            batch_size, query_length, hidden_dim
        )
        output = self.out_proj(output)
        output = output.masked_fill(
            ~has_visible_key.squeeze(1), 0.0
        )

        return output, present_key_value


class TextCrossAttention(nn.Module):
    """Text attention with an optional learned-center Gaussian position bias.

    Local mode is soft, dense attention, not a sparse or monotonic window.
    Text K/V can be projected once and reused throughout a generation call.
    """

    def __init__(
        self,
        hidden_dim: int,
        text_dim: int,
        num_heads: int,
        attention_type: str = "local",
        sigma: float = 8.0,
        dropout: float = 0.0,
        qk_norm: bool = True,
        norm_eps: float = 1e-6,
        bias: bool = False,
    ):
        super().__init__()
        if min(hidden_dim, text_dim, num_heads) <= 0 or hidden_dim % num_heads:
            raise ValueError("Invalid cross-attention dimensions.")
        if attention_type not in ("global", "local"):
            raise ValueError("attention_type must be global or local.")
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("Cross-attention sigma must be finite and positive.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1).")

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.sigma = sigma
        self.dropout = dropout
        self.text_norm = RMSNorm(text_dim, norm_eps)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.kv_proj = nn.Linear(text_dim, 2 * hidden_dim, bias=bias)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.q_norm = RMSNorm(self.head_dim, norm_eps) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim, norm_eps) if qk_norm else nn.Identity()
        self.center_proj = nn.Linear(hidden_dim, 1) if attention_type == "local" else None

    def project_text(self, text: Tensor, text_mask: Tensor) -> KVCache:
        if text_mask.dtype != torch.bool or text_mask.shape != text.shape[:2]:
            raise ValueError("text_mask must be bool [B, L].")
        batch_size, length, _ = text.shape
        if length == 0:
            raise ValueError("Text memory needs at least one (possibly masked) token.")
        text = self.text_norm(text.masked_fill(~text_mask[..., None], 0.0))
        kv = self.kv_proj(text.to(self.kv_proj.weight.dtype)).reshape(
            batch_size, length, 2, self.num_heads, self.head_dim
        )
        k, v = kv.unbind(dim=2)
        return self.k_norm(k.transpose(1, 2)), v.transpose(1, 2)

    def forward(self, x: Tensor, text_key_value: KVCache, text_mask: Tensor) -> Tensor:
        batch_size, query_length, hidden_dim = x.shape
        k, v = text_key_value
        if (
            k.ndim != 4 or k.shape != v.shape
            or k.shape[:2] != (batch_size, self.num_heads)
            or k.shape[-1] != self.head_dim or k.shape[-2] == 0
            or text_mask.dtype != torch.bool
            or text_mask.shape != (batch_size, k.shape[-2])
        ):
            raise ValueError("Text cache and mask must match the cross-attention batch.")
        q = self.q_proj(x.to(self.q_proj.weight.dtype)).reshape(
            batch_size, query_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        q = self.q_norm(q)

        # CFG/text dropout may hide every key. Use a finite fallback and then
        # zero the complete projected residual, including any output bias.
        has_text = text_mask.any(dim=-1)
        fallback = torch.arange(k.shape[-2], device=x.device).eq(0)
        visible = text_mask | (~has_text[:, None] & fallback[None])
        attention_mask = visible[:, None, None, :]
        if self.center_proj is not None:
            center = self.center_proj(x.to(self.center_proj.weight.dtype)).float().sigmoid()
            center = center * (text_mask.sum(-1) - 1).clamp_min(0)[:, None, None]
            ranks = (text_mask.long().cumsum(-1) - 1).clamp_min(0).float()
            distance = (ranks[:, None, :] - center) / self.sigma
            attention_mask = (-0.5 * distance.square())[:, None].masked_fill(
                ~attention_mask, float("-inf")
            ).to(q.dtype)

        output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        output = output.transpose(1, 2).contiguous().reshape(
            batch_size, query_length, hidden_dim
        )
        output = self.out_proj(output)
        return output.masked_fill(~has_text[:, None, None], 0.0)


class FeedForward(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: Optional[int] = None,
        multiple_of: int = 256,
        bias: bool = False,
    ):
        super().__init__()

        if multiple_of <= 0:
            raise ValueError("multiple_of must be positive.")

        if intermediate_dim is None:
            intermediate_dim = int(8 * hidden_dim / 3)
            intermediate_dim = (
                (intermediate_dim + multiple_of - 1) // multiple_of
            ) * multiple_of

        if intermediate_dim <= 0:
            raise ValueError("intermediate_dim must be positive.")

        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=bias)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=bias)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class CausalTransformerBlock(nn.Module):
    """Pre-norm self-attention, optional motion-to-text attention, then FFN."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        intermediate_dim: Optional[int] = None,
        dropout: float = 0.0,
        qk_norm: bool = True,
        rope_base: float = 10000.0,
        norm_eps: float = 1e-6,
        bias: bool = False,
        text_dim: Optional[int] = None,
        cross_attention_type: Optional[str] = None,
        cross_attention_sigma: float = 8.0,
    ):
        super().__init__()

        self.attention_norm = RMSNorm(hidden_dim, norm_eps)
        self.attention = CausalSelfAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            qk_norm=qk_norm,
            rope_base=rope_base,
            norm_eps=norm_eps,
            bias=bias,
        )
        self.cross_attention = None
        if cross_attention_type is not None:
            if text_dim is None:
                raise ValueError("text_dim is required for cross-attention.")
            self.cross_attention_norm = RMSNorm(hidden_dim, norm_eps)
            self.cross_attention = TextCrossAttention(
                hidden_dim, text_dim, num_heads,
                attention_type=cross_attention_type,
                sigma=cross_attention_sigma, dropout=dropout,
                qk_norm=qk_norm, norm_eps=norm_eps, bias=bias,
            )
        self.ffn_norm = RMSNorm(hidden_dim, norm_eps)
        self.feed_forward = FeedForward(
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            bias=bias,
        )
        self.residual_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_value: Optional[KVCache] = None,
        use_cache: bool = False,
        text_key_value: Optional[KVCache] = None,
        text_mask: Optional[Tensor] = None,
        cross_query_start: int = 0,
    ) -> Tuple[Tensor, Optional[KVCache]]:
        attention_output, present_key_value = self.attention(
            self.attention_norm(x),
            key_padding_mask=key_padding_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        x = x + self.residual_dropout(attention_output)
        if self.cross_attention is not None:
            if text_key_value is None or text_mask is None:
                raise ValueError("Cross-attention requires text K/V and mask.")
            # In full forward, leave language/text prefix rows unchanged.
            motion = x[:, cross_query_start:]
            residual = self.cross_attention(
                self.cross_attention_norm(motion), text_key_value, text_mask
            )
            x = torch.cat((x[:, :cross_query_start],
                           motion + self.residual_dropout(residual)), dim=1)
        x = x + self.residual_dropout(
            self.feed_forward(self.ffn_norm(x))
        )
        return x, present_key_value
