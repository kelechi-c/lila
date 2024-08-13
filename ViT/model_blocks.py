import torch
from torch import nn
from ViT.utils import config
from timm.layers import PatchEmbed
from einops import rearrange
from einops.layers.torch import Rearrange


class Patchify(nn.Module):
    def __init__(self, patches=16):
        super().__init__()
        self.patch_size = patches
        self.patch_module = nn.Unfold(kernel_size=patches, stride=patches)

    def forward(self, x: torch.Tensor):
        batch_s, ch, _, _ = x.shape

        x = self.patch_module(x)
        x = x.view(batch_s, ch, self.patch_size, self.patch_size, -1)
        x = x.permute(0, 4, 1, 2, 3)

        return x


class ViTBlock(nn.Module):
    def __init__(self, img_size, classes, embed_dim, in_channels, config=config):
        super().__init__()
        self.patchlayer = PatchEmbed()
        self.mlp_layer = nn.Sequential(
            nn.Linear(768, 1024), nn.GELU(), nn.Linear(1024, 768)
        )

        self.cls_token = nn.Parameter(torch.ones(1, 1, embed_dim))

    def forward(self, x):
        img_patches = self.patchlayer(x)
        x = self.mlp_layer(img_patches)

        return x
