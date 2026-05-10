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

import torch
import torch.nn as nn
import torch.nn.functional as F


def create_foundation_model(model_type, model_id_map=None):
    import timm
    assert model_type in ["dinov2", "dinov3"], f"Unsupported foundation model type: {model_type}"
    default_id_map = {
        "dinov2": "hf-hub:timm/vit_large_patch14_dinov2.lvd142m",
        "dinov3": "hf-hub:timm/vit_large_patch16_dinov3.lvd1689m",
    }
    if model_id_map is not None:
        default_id_map.update(model_id_map)
    model_id = default_id_map[model_type]
    model = timm.create_model(model_id, pretrained=True, dynamic_img_size=True)
    model.requires_grad_(False)
    return model, 1024


class AuxFoundationModel(nn.Module):
    def __init__(self, model_type, model_id_map=None):
        super().__init__()
        self.model, self.feature_dim = create_foundation_model(model_type, model_id_map=model_id_map)
        self.model_type = model_type

    def _extract_patch_tokens(self, feat):
        # If prefix tokens are present (cls/register), strip them.
        tokens = feat
        prefix = self.model.num_prefix_tokens
        n = tokens.shape[1]
        if prefix > 0 and n > prefix:
            n_wo_prefix = n - prefix
            side = int(n_wo_prefix ** 0.5)
            if side * side == n_wo_prefix:
                tokens = tokens[:, prefix:, :]

        return tokens

    def _forward_tokens(self, x):
        # Foundation models are trained at 224 resolution.
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        feat = self.model.forward_features(x)
        patch_tokens = self._extract_patch_tokens(feat)
        n_tokens = patch_tokens.shape[1]
        side = int(n_tokens ** 0.5)
        return patch_tokens.reshape(patch_tokens.shape[0], side, side, patch_tokens.shape[-1]).permute(0, 3, 1, 2).contiguous()

    def forward(self, x):
        with torch.no_grad():
            return self._forward_tokens(x)