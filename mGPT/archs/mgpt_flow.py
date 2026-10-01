import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class TimestepEmbedder(nn.Module):
    """Sinusoidal embedding followed by an MLP for continuous flow time."""

    def __init__(
        self,
        hidden_dim: int,
        frequency_dim: int = 256,
        time_scale: float = 1000.0,
    ):
        super().__init__()

        if frequency_dim < 2:
            raise ValueError("frequency_dim must be at least 2.")

        self.frequency_dim = frequency_dim
        self.time_scale = time_scale

        self.mlp = nn.Sequential(
            nn.Linear(frequency_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, timesteps: Tensor) -> Tensor:
        half = self.frequency_dim // 2

        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(
                half, device=timesteps.device, dtype=torch.float32
            )
            / half
        )
        angles = (
            timesteps.float()[:, None]
            * self.time_scale
            * frequencies[None, :]
        )
        embedding = torch.cat((angles.cos(), angles.sin()), dim=-1)

        if self.frequency_dim % 2:
            embedding = F.pad(embedding, (0, 1))

        return self.mlp(embedding.to(self.mlp[0].weight.dtype))


class AdaLNResBlock(nn.Module):
    """Residual MLP conditioned through shift, scale, and gating."""

    def __init__(
        self,
        hidden_dim: int,
        mlp_ratio: float = 1.0,
        norm_eps: float = 1e-6,
    ):
        super().__init__()

        intermediate_dim = int(hidden_dim * mlp_ratio)

        self.norm = nn.LayerNorm(
            hidden_dim,
            eps=norm_eps,
            elementwise_affine=False,
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.SiLU(),
            nn.Linear(intermediate_dim, hidden_dim),
        )
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 3 * hidden_dim),
        )

    def forward(self, hidden: Tensor, condition: Tensor) -> Tensor:
        shift, scale, gate = self.modulation(condition).chunk(3, dim=-1)
        residual = self.norm(hidden) * (1.0 + scale) + shift
        return hidden + gate * self.mlp(residual)


class FlowMatchingHead(nn.Module):
    """
    Predict the conditional velocity field v(x_t, t, condition).

    This module does not sample noise/times, construct flow targets,
    calculate losses, or integrate the ODE.
    """

    def __init__(
        self,
        latent_dim: int = 48,
        condition_dim: int = 512,
        hidden_dim: int = 512,
        num_layers: int = 6,
        mlp_ratio: float = 1.0,
        frequency_dim: int = 256,
        time_scale: float = 1000.0,
        norm_eps: float = 1e-6,
    ):
        super().__init__()

        if min(latent_dim, condition_dim, hidden_dim, num_layers) <= 0:
            raise ValueError("Dimensions and num_layers must be positive.")
        if int(hidden_dim * mlp_ratio) < 1:
            raise ValueError("MLP intermediate dimension must be positive.")

        self.latent_dim = latent_dim
        self.condition_dim = condition_dim

        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        self.condition_proj = nn.Linear(condition_dim, hidden_dim)
        self.time_embed = TimestepEmbedder(
            hidden_dim=hidden_dim,
            frequency_dim=frequency_dim,
            time_scale=time_scale,
        )

        self.blocks = nn.ModuleList([
            AdaLNResBlock(hidden_dim, mlp_ratio, norm_eps)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(
            hidden_dim,
            eps=norm_eps,
            elementwise_affine=False,
        )
        self.final_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
        )
        self.output_proj = nn.Linear(hidden_dim, latent_dim)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

        for index in (0, 2):
            nn.init.normal_(self.time_embed.mlp[index].weight, std=0.02)

        for block in self.blocks:
            nn.init.zeros_(block.modulation[-1].weight)
            nn.init.zeros_(block.modulation[-1].bias)

        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        noisy_latents: Tensor,
        timesteps: Tensor,
        condition: Tensor,
    ) -> Tensor:
        """
        Args:
            noisy_latents: [N, latent_dim].
            timesteps: [N], continuous flow times in [0, 1].
            condition: [N, condition_dim].

        Returns:
            velocity: [N, latent_dim].

        The outer model selects/reshapes valid sequence positions.
        """
        if (
            noisy_latents.ndim != 2
            or noisy_latents.shape[-1] != self.latent_dim
        ):
            raise ValueError("noisy_latents must be [N, latent_dim].")

        count = noisy_latents.shape[0]
        if timesteps.shape != (count,):
            raise ValueError("timesteps must be [N].")
        if condition.shape != (count, self.condition_dim):
            raise ValueError("condition must be [N, condition_dim].")

        dtype = self.input_proj.weight.dtype

        hidden = self.input_proj(noisy_latents.to(dtype))
        context = (
            self.time_embed(timesteps)
            + self.condition_proj(condition.to(dtype))
        )

        for block in self.blocks:
            hidden = block(hidden, context)

        shift, scale = self.final_modulation(context).chunk(2, dim=-1)
        hidden = self.final_norm(hidden) * (1.0 + scale) + shift

        return self.output_proj(hidden)
