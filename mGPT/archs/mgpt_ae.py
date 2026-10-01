from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .mgpt_vq import Encoder, Decoder, CausalEncoder, CausalDecoder


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=-1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

    def sample(self):
        noise = torch.randn_like(self.mean)
        x = self.mean + self.std * noise
        return x

    def kl(self, latent_lengths=None):
        kl = 0.5 * (self.mean.pow(2) + self.var - 1.0 - self.logvar)

        if latent_lengths is not None:
            lengths = torch.as_tensor(latent_lengths, device=kl.device)
            valid = (
                torch.arange(kl.shape[1], device=kl.device)[None, :]
                < lengths[:, None]
            )
            kl = kl.masked_fill(~valid.unsqueeze(-1), 0.0)

        return kl.sum(dim=tuple(range(1, kl.ndim)))

    def mode(self):
        return self.mean


class AutoencoderKL(nn.Module):
    def __init__(
        self,
        nfeats: int,
        latent_dim: int = 16,
        down_t: int = 2,
        stride_t: int = 2,
        width: int = 512,
        depth: int = 3,
        dilation_growth_rate: int = 3,
        norm: Optional[str] = None,
        activation: str = "relu",
        is_causal: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.nfeats = nfeats
        self.latent_dim = latent_dim
        self.temporal_factor = down_t * stride_t

        encoder_cls = CausalEncoder if is_causal else Encoder
        decoder_cls = CausalDecoder if is_causal else Decoder

        encoder_output_dim = 2 * latent_dim

        self.encoder = encoder_cls(
            input_emb_width=nfeats,
            output_emb_width=encoder_output_dim,
            down_t=down_t,
            stride_t=stride_t,
            width=width,
            depth=depth,
            dilation_growth_rate=dilation_growth_rate,
            activation=activation,
            norm=norm,
        )
        self.quant_conv = nn.Conv1d(
            in_channels=encoder_output_dim,
            out_channels=encoder_output_dim,
            kernel_size=1,
        )
        self.post_quant_conv = nn.Conv1d(
            in_channels=latent_dim,
            out_channels=latent_dim,
            kernel_size=1,
        )
        self.decoder = decoder_cls(
            input_emb_width=nfeats,
            output_emb_width=latent_dim,
            down_t=down_t,
            stride_t=stride_t,
            width=width,
            depth=depth,
            dilation_growth_rate=dilation_growth_rate,
            activation=activation,
            norm=norm,
        )

    def encode(
        self,
        features: Tensor,
    ) -> DiagonalGaussianDistribution:
        x = features.transpose(1, 2).contiguous()
        padding = (-x.shape[-1]) % self.temporal_factor
        if padding:
            x = F.pad(x, (0, padding), mode="replicate")

        moments = self.encoder(x)
        moments = self.quant_conv(moments)
        moments = moments.transpose(1, 2).contiguous()
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(
        self,
        z: Tensor,
        length: Optional[int] = None,
    ) -> Tensor:
        z = z.transpose(1, 2).contiguous()
        z = self.post_quant_conv(z)
        rec = self.decoder(z)
        rec = rec.transpose(1, 2).contiguous()
        if length is not None:
            rec = rec[:, :length]
        return rec

    def forward(
        self,
        features: Tensor,
        sample_posterior: bool = True,
    ) -> Tuple[Tensor, DiagonalGaussianDistribution]:
        posterior = self.encode(features)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()

        rec = self.decode(z, features.shape[1])
        return rec, posterior
