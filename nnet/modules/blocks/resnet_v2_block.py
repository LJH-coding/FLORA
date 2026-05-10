# Copyright (c) 2026 Computer Vision Lab, University of Wurzburg
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License"); you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Computer Vision Lab, University of Wurzburg.
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# PyTorch
import torch
import torch.nn as nn

# Neural Nets
from nnet import modules
from nnet.modules.layers import Conv2d
from nnet.utils import get_module_and_params

class ResNetV2Block(modules.Module):

    """ ResNetV2 Residual Block used by ResNet18V2 and ResNet34V2 networks.

    References: "Identity Mappings in Deep Residual Networks", He et al.
    https://arxiv.org/abs/1603.05027
    
    """

    def __init__(self, in_features, out_features, kernel_size, stride, norm="BatchNorm2d", act_fun="ReLU", dim=2, channels_last=False, weight_init="he_normal", bias_init="zeros", bias=False, joined_pre_norm=True, padding="same", kernel_size_down=None):
        super(ResNetV2Block, self).__init__()

        conv = {
            2: Conv2d
        }

        # Get act_fun and norm
        act_fun, act_fun_params = get_module_and_params(act_fun, modules.act_dict)
        norm, norm_params = get_module_and_params(norm, modules.norm_dict)

        # Kernel Size Down
        if kernel_size_down is None:
            kernel_size_down = kernel_size

        # Pre Norm
        self.joined_pre_norm = joined_pre_norm
        self.pre_norm = nn.Sequential(
            norm(in_features, **norm_params, channels_last=channels_last),
            act_fun(**act_fun_params)
        )

        # layers
        self.layers = nn.Sequential(
            conv[dim](in_channels=in_features, out_channels=out_features, kernel_size=kernel_size_down if torch.prod(torch.tensor(stride)) > 1 else kernel_size, stride=stride, channels_last=channels_last, bias=bias, weight_init=weight_init, bias_init=bias_init, padding=padding),

            norm(out_features, **norm_params, channels_last=channels_last),
            act_fun(**act_fun_params),
            conv[dim](in_channels=out_features, out_channels=out_features, kernel_size=kernel_size, channels_last=channels_last, weight_init=weight_init, bias_init=bias_init, padding=padding),
        )

        # Pooling Block
        if torch.prod(torch.tensor(stride)) > 1:
            self.pooling = nn.MaxPool2d(kernel_size=1, stride=stride)
            self.conv = None
        
        # Projection Block
        elif in_features != out_features:
            self.pooling = None
            self.conv = conv[dim](in_channels=in_features, out_channels=out_features, kernel_size=1, channels_last=channels_last, weight_init=weight_init, bias_init=bias_init)

        # Default Block
        else:
            self.pooling = None
            self.conv = None

    def forward(self, x):

        # Pooling Block
        if self.pooling != None: 
            x = self.layers(self.pre_norm(x)) + self.pooling(x)

        # Projection Block
        elif self.conv != None:
            if self.joined_pre_norm:
                x = self.pre_norm(x)
                x = self.layers(x) + self.conv(x)
            else:
                x = self.layers(self.pre_norm(x)) + self.conv(x)

        # Default Block
        else:
            x = self.layers(self.pre_norm(x)) + x

        return x