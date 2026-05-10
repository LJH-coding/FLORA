from dataclasses import dataclass
from typing import List

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FrameCnnConfig:
    image_channels: int
    num_channels: int
    num_groups: int
    mult: List[int]
    down: List[int]

class FrameEncoder(nn.Module):
    def __init__(self, config: FrameCnnConfig) -> None:
        super().__init__()

        assert len(config.mult) == len(config.down)
        encoder_layers = [nn.Conv2d(config.image_channels, config.num_channels, kernel_size=3, stride=1, padding=1)]
        input_channels = config.num_channels

        for m, d in zip(config.mult, config.down):
            output_channels = m * config.num_channels
            encoder_layers.append(ResidualBlock(input_channels, output_channels, config.num_groups))
            input_channels = output_channels
            if d:
                encoder_layers.append(Downsample(output_channels))
        self.encoder = nn.Sequential(*encoder_layers)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        shape = x.shape

        # (B, T, C, H, W) -> (B*T, C, H, W) / (B, C, H, W) -> (B, C, H, W)
        x = x.reshape((-1,) + shape[-3:])

        # (N, C, H, W) -> (N, num_channels, H', W')
        x = self.encoder(x)

        # (N, num_channels, H', W') -> (B, T, num_channels, H', W') / (N, num_channels, H', W') -> (B, num_channels, H', W')
        x = x.reshape(shape[:-3] + x.shape[-3:])

        return x


class FrameDecoder(nn.Module):
    def __init__(self, config: FrameCnnConfig) -> None:
        super().__init__()

        assert len(config.mult) == len(config.down)
        decoder_layers = []
        output_channels = config.num_channels

        for m, d in zip(config.mult, config.down):
            input_channels = m * config.num_channels
            decoder_layers.append(ResidualBlock(input_channels, output_channels, config.num_groups))
            output_channels = input_channels
            if d:
                decoder_layers.append(Upsample(input_channels))
        decoder_layers.reverse()
        decoder_layers.extend([
            nn.GroupNorm(num_groups=config.num_groups, num_channels=config.num_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(config.num_channels, config.image_channels, kernel_size=3, stride=1, padding=1)
        ])
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        shape = x.shape

        # (B, T, num_channels, H', W') -> (B*T, num_channels, H', W') / (B, num_channels, H', W') -> (B, num_channels, H', W')
        x = x.reshape((-1,) + shape[-3:])

        # (N, num_channels, H', W') -> (N, image_channels, H', W')
        x = self.decoder(x)

        # (N, image_channels, H', W') -> (B, T, image_channels, H', W') / (N, image_channels, H', W') -> (N, image_channels, H', W')
        x = x.reshape(shape[:-3] + x.shape[-3:])
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_groups_norm: int = 32) -> None:
        super().__init__()

        self.f = nn.Sequential(
            nn.GroupNorm(num_groups_norm, in_channels), 
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups_norm, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )
        self.skip_projection = nn.Identity() if in_channels == out_channels else torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.skip_projection(x) + self.f(x) 


class Downsample(nn.Module):
    def __init__(self, num_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(num_channels, num_channels, kernel_size=2, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, num_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)

class RMSNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x * x, dim=1, keepdim=True) + self.eps)
        return x / rms * self.weight.view(1, -1, 1, 1)

class SimNorm(nn.Module):
    def __init__(self, simnorm_dim):
        super().__init__()
        self.dim = simnorm_dim
    def forward(self, x):
        shp = x.shape
        x = x.view(*shp[:-1], -1, self.dim)
        x = F.softmax(x, dim=-1)
        return x.view(*shp)
    def __repr__(self):
        return f"SimNorm(dim={self.dim})"

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class ResNeXtBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        cardinality: int = 32,        # 分組數 C
        bottleneck_width: int = 4,    # 每個 group 的寬度
        num_groups_norm: int = 32
    ) -> None:
        super().__init__()
        
        # bottleneck 中間層的 channel 數 = cardinality * bottleneck_width
        mid_channels = cardinality * bottleneck_width
        
        self.f = nn.Sequential(
            # 1x1 降維
            nn.GroupNorm(num_groups_norm, in_channels),
            nn.Mish(inplace=True),
            nn.Conv2d(in_channels, mid_channels, kernel_size=1),
            
            # 3x3 grouped conv（這裡是 ResNeXt 的核心）
            nn.GroupNorm(cardinality, mid_channels),  # num_groups = cardinality
            nn.Mish(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, groups=cardinality),
            
            # 1x1 升維
            nn.GroupNorm(num_groups_norm, mid_channels),
            nn.Mish(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1),
        )
        
        self.skip_projection = (
            nn.Identity() if in_channels == out_channels 
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.skip_projection(x) + self.f(x)

class ShiftAug(nn.Module):
	"""
	Random shift image augmentation.
	Adapted from https://github.com/facebookresearch/drqv2
	"""
	def __init__(self, pad=3):
		super().__init__()
		self.pad = pad
		self.padding = tuple([self.pad] * 4)

	def forward(self, x):
		x = x.float()
		n, _, h, w = x.size()
		assert h == w
		x = F.pad(x, self.padding, 'replicate')
		eps = 1.0 / (h + 2 * self.pad)
		arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
		arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
		base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
		base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
		shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
		shift *= 2.0 / (h + 2 * self.pad)
		grid = base_grid + shift
		return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)
