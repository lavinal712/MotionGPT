# Partially from https://github.com/Mael-zys/T2M-GPT

from typing import List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions.distribution import Distribution
from .tools.resnet import Resnet1D, CausalResnet1D
from .tools.quantize_cnn import QuantizeEMAReset, Quantizer, QuantizeEMA, QuantizeReset
from .tools.quantize_fsq import FSQ
from collections import OrderedDict


class VQVae(nn.Module):

    def __init__(self,
                 nfeats: int,
                 quantizer: str = "ema_reset",
                 code_num=512,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 norm=None,
                 activation: str = "relu",
                 is_causal: bool = False,
                 **kwargs) -> None:

        super().__init__()

        self.code_dim = code_dim
        self.code_num = code_num
        self.nfeats = nfeats

        if not is_causal:
            self.encoder = Encoder(nfeats,
                                output_emb_width,
                                down_t,
                                stride_t,
                                width,
                                depth,
                                dilation_growth_rate,
                                activation=activation,
                                norm=norm)

            self.decoder = Decoder(nfeats,
                               output_emb_width,
                               down_t,
                               stride_t,
                               width,
                               depth,
                               dilation_growth_rate,
                               activation=activation,
                               norm=norm)
        else:
            self.encoder = CausalEncoder(
                nfeats,
                output_emb_width,
                down_t,
                stride_t,
                width,
                depth,
                dilation_growth_rate,
                activation=activation,
                norm=norm,
            )

            self.decoder = CausalDecoder(
                nfeats,
                output_emb_width,
                down_t,
                stride_t,
                width,
                depth,
                dilation_growth_rate,
                activation=activation,
                norm=norm,
            )

        if quantizer == "ema_reset":
            self.quantizer = QuantizeEMAReset(code_num, code_dim, mu=0.99)
        elif quantizer == "orig":
            self.quantizer = Quantizer(code_num, code_dim, beta=1.0)
        elif quantizer == "ema":
            self.quantizer = QuantizeEMA(code_num, code_dim, mu=0.99)
        elif quantizer == "reset":
            self.quantizer = QuantizeReset(code_num, code_dim)

    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0, 2, 1)
        return x

    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0, 2, 1)
        return x

    def forward(self, features: Tensor):
        # Preprocess
        x_in = self.preprocess(features)

        # Encode
        x_encoder = self.encoder(x_in)

        # quantization
        x_quantized, loss, perplexity = self.quantizer(x_encoder)

        # decoder
        x_decoder = self.decoder(x_quantized)
        x_out = self.postprocess(x_decoder)

        return x_out, loss, perplexity

    def encode(
        self,
        features: Tensor,
    ) -> Union[Tensor, Distribution]:

        N, T, _ = features.shape
        x_in = self.preprocess(features)
        x_encoder = self.encoder(x_in)
        x_encoder = self.postprocess(x_encoder)
        x_encoder = x_encoder.contiguous().view(-1,
                                                x_encoder.shape[-1])  # (NT, C)
        code_idx = self.quantizer.quantize(x_encoder)
        code_idx = code_idx.view(N, -1)

        # latent, dist
        return code_idx, None

    def decode(self, z: Tensor):

        x_d = self.quantizer.dequantize(z)
        x_d = x_d.view(1, -1, self.code_dim).permute(0, 2, 1).contiguous()

        # decoder
        x_decoder = self.decoder(x_d)
        x_out = self.postprocess(x_decoder)
        return x_out


class FSQVae(nn.Module):
    def __init__(
        self,
        nfeats: int,
        levels: List[int],
        width=512,
        depth=3,
        down_t=3,
        dilation_growth_rate=3,
        is_causal: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.nfeats = nfeats
        self.temporal_factor = 2 ** down_t
        self.quantizer = FSQ(levels)
        self.code_dim = self.quantizer.code_dim
        self.code_num = self.quantizer.codebook_size

        if not is_causal:
            self.encoder = Encoder(
                input_emb_width=nfeats,
                output_emb_width=self.code_dim,
                down_t=down_t,
                stride_t=2,
                width=width,
                depth=depth,
                dilation_growth_rate=dilation_growth_rate,
                # reverse_dilation=False,
            )
            self.decoder = Decoder(
                input_emb_width=nfeats,
                output_emb_width=self.code_dim,
                down_t=down_t,
                stride_t=2,
                width=width,
                depth=depth,
                dilation_growth_rate=dilation_growth_rate,
            )
        else:
            self.encoder = CausalEncoder(
                input_emb_width=nfeats,
                output_emb_width=self.code_dim,
                down_t=down_t,
                stride_t=2,
                width=width,
                depth=depth,
                dilation_growth_rate=dilation_growth_rate,
                # reverse_dilation=False,
            )
            self.decoder = CausalDecoder(
                input_emb_width=nfeats,
                output_emb_width=self.code_dim,
                down_t=down_t,
                stride_t=2,
                width=width,
                depth=depth,
                dilation_growth_rate=dilation_growth_rate,
            )

    def _encode(self, features):
        if features.ndim != 3 or features.shape[-1] != self.nfeats:
            raise ValueError(f"Expected features with shape (B, T, {self.nfeats})")
        if features.shape[0] == 0 or features.shape[1] == 0:
            raise ValueError("Batch and sequence must be nonempty")

        x = features.transpose(1, 2)
        padding = (-x.shape[-1]) % self.temporal_factor
        if padding:
            x = F.pad(x, (0, padding), mode="replicate")

        z = self.encoder(x).transpose(1, 2)
        return self.quantizer(z)

    def forward(self, features):
        codes, indices = self._encode(features)
        codes = codes.to(dtype=next(self.decoder.parameters()).dtype)
        reconstruction = self.decoder(codes.transpose(1, 2)).transpose(1, 2)
        reconstruction = reconstruction[:, :features.shape[1]]

        counts = torch.bincount(
            indices.reshape(-1), minlength=self.code_num
        ).float()
        probabilities = counts / counts.sum()
        perplexity = torch.exp(
            -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
        )

        return reconstruction, reconstruction.new_zeros(()), perplexity

    def encode(self, features):
        _, indices = self._encode(features)
        return indices, None

    def decode(self, indices, length=None):
        if indices.ndim == 1:
            indices = indices.unsqueeze(0)
        if indices.ndim != 2 or indices.shape[0] == 0 or indices.shape[1] == 0:
            raise ValueError("Expected nonempty token IDs with shape (N,) or (B, N)")

        codes = self.quantizer.indices_to_codes(indices)
        codes = codes.to(dtype=next(self.decoder.parameters()).dtype)
        reconstruction = self.decoder(codes.transpose(1, 2)).transpose(1, 2)

        if length is not None:
            if not 0 < length <= reconstruction.shape[1]:
                raise ValueError("length must be within the decoded sequence")
            reconstruction = reconstruction[:, :length]

        return reconstruction


class Encoder(nn.Module):

    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None,
                 reverse_dilation=True):
        super().__init__()

        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())

        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                Resnet1D(width,
                         depth,
                         dilation_growth_rate,
                         reverse_dilation=reverse_dilation,
                         activation=activation,
                         norm=norm),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):

    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        super().__init__()
        blocks = []

        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(width,
                         depth,
                         dilation_growth_rate,
                         reverse_dilation=True,
                         activation=activation,
                         norm=norm), nn.Upsample(scale_factor=2,
                                                 mode='nearest'),
                nn.Conv1d(width, out_dim, 3, 1, 1))
            blocks.append(block)
        blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class CausalConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
    ):
        super(CausalConv1d, self).__init__()
        self.pad = (kernel_size - 1) * dilation + (1 - stride)         
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size,                        
            stride=stride, 
            padding=0,  # no padding here
            dilation=dilation,
        )

    def forward(self, x):
        x = F.pad(x, (self.pad, 0))  # only pad on the left
        return self.conv(x)


class CausalEncoder(nn.Module):
    def __init__(
        self,
        input_emb_width: int = 3,
        output_emb_width: int = 512,
        down_t: int = 3,
        stride_t: int = 2,
        width: int = 512,
        depth: int = 3,
        dilation_growth_rate: int = 3,
        activation: str = 'relu',
        norm: Optional[str] = None,
    ):
        super().__init__()

        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(CausalConv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())

        for i in range(down_t):   
            input_dim = width
            block = nn.Sequential(
                CausalConv1d(input_dim, width, filter_t, stride_t, 1),
                CausalResnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm),
            )
            blocks.append(block)
        blocks.append(CausalConv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class CausalDecoder(nn.Module):
    def __init__(
        self,
        input_emb_width: int = 3,
        output_emb_width: int = 512,
        down_t: int = 3,
        stride_t: int = 2,
        width: int = 512,
        depth: int = 3,
        dilation_growth_rate: int = 3,
        activation: str = 'relu',
        norm: Optional[str] = None,
    ):
        super().__init__()
        blocks = []
        
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(CausalConv1d(output_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                CausalResnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
                nn.Upsample(scale_factor=2, mode='nearest'),
                CausalConv1d(width, out_dim, 3, 1, 1)
            )
            blocks.append(block)
        blocks.append(CausalConv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(CausalConv1d(width, input_emb_width, 3, 1, 1))

        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)
