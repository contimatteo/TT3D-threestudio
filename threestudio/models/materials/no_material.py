import random
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.networks import get_encoding, get_mlp
from threestudio.utils.ops import dot, get_activation
from threestudio.utils.typing import *


@threestudio.register("no-material")
class NoMaterial(BaseMaterial):
    @dataclass
    class Config(BaseMaterial.Config):
        n_output_dims: int = 3
        color_activation: str = "sigmoid"
        input_feature_dims: Optional[int] = None
        mlp_network_config: Optional[dict] = None

    cfg: Config

    def configure(self) -> None:
        self.use_network = False
        if (
            self.cfg.input_feature_dims is not None
            and self.cfg.mlp_network_config is not None
        ):
            self.network = get_mlp(
                self.cfg.input_feature_dims,
                self.cfg.n_output_dims,
                self.cfg.mlp_network_config,
            )
            self.use_network = True

    def forward(
        self, features: Float[Tensor, "B ... Nf"], **kwargs
    ) -> Float[Tensor, "B ... Nc"]:
        if not self.use_network:
            assert (
                features.shape[-1] == self.cfg.n_output_dims
            ), f"Expected {self.cfg.n_output_dims} output dims, only got {features.shape[-1]} dims input."
            color = get_activation(self.cfg.color_activation)(features)
        else:
            color = self.network(features.view(-1, features.shape[-1])).view(
                *features.shape[:-1], self.cfg.n_output_dims
            )
            color = get_activation(self.cfg.color_activation)(color)
        return color

    def export(self, features: Float[Tensor, "*N Nf"], guidance=None, **kwargs) -> Dict[str, Any]:
        if guidance == None:
            color = self(features, **kwargs).clamp(0, 1)
        else:
            color = torch.zeros_like(features[..., :3], device=features.device)
            color_weights = torch.zeros((3*3, *features.shape[:2]), device=features.device)
            meshgrid = [ i.to(features.device) for i in torch.meshgrid(torch.arange(512), torch.arange(512)) ]
            for i in range(3):
                for j in range(3):
                    color_weights[i*3+j,i*256:i*256+512,j*256:j*256+512] += (512-1)/2 - torch.abs(meshgrid[0]-(512-1)/2) + 0.5
                    color_weights[i*3+j,i*256:i*256+512,j*256:j*256+512] += (512-1)/2 - torch.abs(meshgrid[1]-(512-1)/2) + 0.5
            color_weights = color_weights / color_weights.sum(0, keepdim=True)

            for i in range(3):
                for j in range(3):
                    latents = F.interpolate(
                        features[i*256:i*256+512, j*256:j*256+512].unsqueeze(0).permute(0, 3, 1, 2), (64, 64), mode="bilinear", align_corners=False
                    )
                    color[i*256:i*256+512, j*256:j*256+512] += guidance.decode_latents(
                        latents,
                        latent_height=64, latent_width=64,
                    ).permute(0, 2, 3, 1).squeeze(0).clamp(0, 1) * color_weights[i*3+j, i*256:i*256+512, j*256:j*256+512].unsqueeze(-1)

        assert color.shape[-1] >= 3, "Output color must have at least 3 channels"
        if color.shape[-1] > 3:
            threestudio.warn(
                "Output color has >3 channels, treating the first 3 as RGB"
            )
        return {"albedo": color[..., :3]}
