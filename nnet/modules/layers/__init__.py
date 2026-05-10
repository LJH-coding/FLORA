from .permute_channels import PermuteChannels
from .linear import Linear
from .dropout import Dropout
from .conv2d import Conv2d
from .conv_transpose_2d import ConvTranspose2d

# Layers Dictionary
layer_dict = {
    "PermuteChannels": PermuteChannels,
    "Linear": Linear,
    "Dropout": Dropout,
    "Conv2d": Conv2d,
    "ConvTranspose2d": ConvTranspose2d,
}