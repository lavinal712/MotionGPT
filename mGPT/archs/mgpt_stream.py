from typing import Optional

import torch
from torch import Tensor, nn

from .tools.causal_transformer import CausalTransformerBlock, RMSNorm


class StreamGenerator(nn.Module):
    """
    Causal condition generator for streaming sign motion.

    Full forward:
        [language, text, BOS, motion_inputs]
        -> conditions at [BOS, motion_inputs].

    Cached forward:
        [new_motion_inputs] + past_key_values
        -> conditions at new motion positions.

    Masks use True for valid positions.
    KV cache is append-only.
    """

    def __init__(
        self,
        latent_dim: int = 48,
        text_dim: int = 768,
        hidden_dim: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        intermediate_dim: Optional[int] = None,
        num_languages: int = 3,
        dropout: float = 0.0,
        qk_norm: bool = True,
        rope_base: float = 10000.0,
        norm_eps: float = 1e-6,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.motion_proj = nn.Linear(latent_dim, hidden_dim)
        self.language_embed = (
            nn.Embedding(num_languages, hidden_dim)
            if num_languages > 0
            else None
        )

        self.bos = nn.Parameter(torch.empty(1, 1, hidden_dim))
        nn.init.normal_(self.bos, std=0.02)

        self.blocks = nn.ModuleList([
            CausalTransformerBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                intermediate_dim=intermediate_dim,
                dropout=dropout,
                qk_norm=qk_norm,
                rope_base=rope_base,
                norm_eps=norm_eps,
            )
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(hidden_dim, eps=norm_eps)
        self.end_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        motion_inputs: Tensor,
        text_features: Optional[Tensor] = None,
        text_mask: Optional[Tensor] = None,
        motion_mask: Optional[Tensor] = None,
        language_ids: Optional[Tensor] = None,
        past_key_values=None,
        past_mask: Optional[Tensor] = None,
        use_cache: bool = False,
    ):
        """
        Args:
            motion_inputs:
                [B, S, latent_dim].
                Full forward allows S=0 and inserts BOS internally.
                Cached forward receives only new motion latents.
            text_features:
                [B, L, text_dim], required for full forward only.
            text_mask:
                Bool [B, L], required for full forward only.
            motion_mask:
                Bool [B, S]. Defaults to all valid.
            language_ids:
                Long [B], required for full forward when language
                conditioning is enabled.
            past_key_values:
                One (key, value) pair per Transformer layer.
            past_mask:
                Bool [B, cached_length], returned by the previous call.
            use_cache:
                Whether to return updated key/value tensors.

        Returns:
            Dictionary containing:
                condition:
                    Full forward: [B, S+1, hidden_dim].
                    Cached forward: [B, S, hidden_dim].
                end_logits:
                    [B, S+1] or [B, S], predicting whether to stop next.
                past_key_values:
                    Updated cache, or None.
                attention_mask:
                    Validity mask for all cached/current positions.

        Training alignment:
            motion_inputs = target_latents[:, :-1]
            condition[:, i] predicts target_latents[:, i].

        Target preparation, END insertion, and loss masking are handled
        by the outer model.
        """
        batch_size, motion_length, _ = motion_inputs.shape
        device = motion_inputs.device

        if motion_mask is None:
            motion_mask = torch.ones(
                batch_size,
                motion_length,
                dtype=torch.bool,
                device=device,
            )
        elif (
            motion_mask.dtype != torch.bool
            or motion_mask.shape != motion_inputs.shape[:2]
        ):
            raise ValueError("motion_mask must be bool [B, S].")

        # Clear padding before projection.
        motion = motion_inputs.masked_fill(~motion_mask[..., None], 0.0)
        hidden = self.motion_proj(
            motion.to(self.motion_proj.weight.dtype)
        )

        prefix_length = 0

        if past_key_values is None:
            if past_mask is not None:
                raise ValueError("past_mask requires past_key_values.")
            if text_features is None or text_mask is None:
                raise ValueError("Full forward requires text features and mask.")
            if (
                text_mask.dtype != torch.bool
                or text_mask.shape != text_features.shape[:2]
                or text_features.shape[0] != batch_size
            ):
                raise ValueError("Invalid text features or text mask.")

            text = text_features.masked_fill(~text_mask[..., None], 0.0)
            prefix = self.text_proj(
                text.to(self.text_proj.weight.dtype)
            )
            prefix_mask = text_mask

            if self.language_embed is not None:
                if language_ids is None:
                    raise ValueError("language_ids is required.")

                language = self.language_embed(language_ids)
                prefix = torch.cat(
                    (language[:, None].to(prefix.dtype), prefix), dim=1
                )
                prefix_mask = torch.cat(
                    (
                        torch.ones(
                            batch_size, 1, dtype=torch.bool, device=device
                        ),
                        prefix_mask,
                    ),
                    dim=1,
                )

            prefix_length = prefix.shape[1]
            bos = self.bos.to(hidden.dtype).expand(batch_size, -1, -1)
            hidden = torch.cat((prefix, bos, hidden), dim=1)

            attention_mask = torch.cat(
                (
                    prefix_mask,
                    torch.ones(
                        batch_size, 1, dtype=torch.bool, device=device
                    ),
                    motion_mask,
                ),
                dim=1,
            )

            position_ids = (
                attention_mask.long().cumsum(dim=1) - 1
            ).clamp_min(0)

        else:
            if motion_length == 0:
                raise ValueError("Cached forward requires new motion inputs.")
            if len(past_key_values) != len(self.blocks):
                raise ValueError("Expected one cache per Transformer layer.")
            if past_mask is None:
                raise ValueError("Cached forward requires past_mask.")

            cached_length = past_key_values[0][0].shape[-2]
            if (
                past_mask.dtype != torch.bool
                or past_mask.shape != (batch_size, cached_length)
            ):
                raise ValueError("past_mask must match the cached sequence.")

            attention_mask = torch.cat((past_mask, motion_mask), dim=1)
            position_ids = (
                attention_mask.long().cumsum(dim=1) - 1
            ).clamp_min(0)[:, -motion_length:]

        new_cache = [] if use_cache else None

        for index, block in enumerate(self.blocks):
            hidden, present = block(
                hidden,
                key_padding_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=(
                    past_key_values[index]
                    if past_key_values is not None
                    else None
                ),
                use_cache=use_cache,
            )
            if use_cache:
                new_cache.append(present)

        condition = self.norm(hidden)[:, prefix_length:]

        return {
            "condition": condition,
            "end_logits": self.end_head(condition).squeeze(-1),
            "past_key_values": new_cache,
            "attention_mask": attention_mask,
        }
