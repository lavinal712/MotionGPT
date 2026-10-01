import math

import torch
import torch.nn as nn


class FSQ(nn.Module):
    def __init__(self, levels):
        super().__init__()
        if not levels or any(type(level) is not int or level < 2 for level in levels):
            raise ValueError("levels must be a nonempty sequence of integers >= 2")
        self.code_dim = len(levels)
        self.codebook_size = math.prod(levels)
        levels = torch.tensor(levels, dtype=torch.long)
        self.register_buffer("levels", levels)
        self.register_buffer("basis", torch.cat((torch.ones(1, dtype=torch.long),
                                                levels.cumprod(0)[:-1])))

    def forward(self, z):
        if z.shape[-1] != self.code_dim:
            raise ValueError(f"Expected {self.code_dim} latent channels, got {z.shape[-1]}")
        # Float32 keeps rounding at cell boundaries consistent under autocast.
        z = z.float()
        half_l = (self.levels - 1).float() * (1 + 1e-3) / 2
        offset = (self.levels % 2 == 0).float() * 0.5
        shift = (offset / half_l).atanh()
        bounded = (z + shift).tanh() * half_l - offset
        rounded = bounded + (bounded.round() - bounded).detach()
        codes = rounded / (self.levels // 2)
        return codes, self.codes_to_indices(codes)

    def codes_to_indices(self, codes):
        digits = (codes * (self.levels // 2)).round().long() + self.levels // 2
        return (digits * self.basis).sum(dim=-1)

    def indices_to_codes(self, indices):
        if indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("Token IDs must have integer dtype")
        if torch.any((indices < 0) | (indices >= self.codebook_size)):
            raise ValueError(f"Token IDs must be in [0, {self.codebook_size})")
        digits = (indices.long().unsqueeze(-1) // self.basis) % self.levels
        half_width = self.levels // 2
        return (digits - half_width).float() / half_width
