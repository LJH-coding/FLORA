# Copyright 2025, Maxime Burchi.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# PyTorch
import torch
import torch.nn as nn

# NeuralNets
from nnet import modules

class SimSiamNetwork(nn.Module):

    def __init__(
        self, 
        hidden_size=512, 
        out_size=512,
        feat_size=32*32+512, 
        embed_size=4*4*8*32,
        act_fun="ELU",
        num_layers=2
    ):
        super(SimSiamNetwork, self).__init__()

        self.mlp_feats = nn.Sequential( # projector 1
            nn.Linear(feat_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, out_size),
            nn.BatchNorm1d(out_size),
        )
        self.mlp_embed = nn.Sequential( # projector 2
            nn.Linear(embed_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, out_size),
            nn.BatchNorm1d(out_size),
        )
        self.predictor = nn.Sequential( # predictor
            nn.Linear(out_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, out_size),
        )

    def forward(self, feats, embed):
        feats_shape = feats.shape
        embed_shape = embed.shape
        feats = feats.reshape(-1, feats_shape[-1])
        embed = embed.reshape(-1, embed_shape[-1])

        # print(feats.shape, embed.shape)

        # Projector Layers
        z_1 = self.mlp_feats(feats)
        z_2 = self.mlp_embed(embed)

        p_1 = self.predictor(z_1)
        p_2 = self.predictor(z_2)

        z_1 = z_1.reshape(feats_shape[:-1] + (z_1.shape[-1],))
        z_2 = z_2.reshape(embed_shape[:-1] + (z_2.shape[-1],))
        p_1 = p_1.reshape(feats_shape[:-1] + (p_1.shape[-1],))
        p_2 = p_2.reshape(embed_shape[:-1] + (p_2.shape[-1],))

        return z_1, z_2, p_1, p_2
    