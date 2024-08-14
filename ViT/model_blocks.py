import torch
import numpy as np
from torch import nn
from ViT.utils import config
from einops import repeat
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


def pos_embedding(seq_len, embed_dim=config.embed_dim):
    embeds = torch.ones(seq_len, embed_dim)
    for k in range(seq_len):
        for v in range(embed_dim):
            embeds[k][v] = (
                np.sin(k / pow(10000, v / embed_dim))
                if v % 2 == 0
                else np.cos(k / pow(10000, v - 1 / embed_dim))
            )

    return torch.tensor(embeds)


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim=config.embed_dim,
        hidden_dim=config.hidden_dim,
        in_channels=3,
        img_size=224,
        patch_size=config.patch_size,
    ):
        super().__init__()
        self.patch_projection = nn.Sequential(
            nn.Conv2d(
                in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
            ),
            Rearrange("b e (h) (w) -> b (h w) e"),
        )
        self.mlp_layer = nn.Sequential(
            nn.Linear(768, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 768)
        )

        self.cls_token = nn.Parameter(torch.rand(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            pos_embedding((img_size // patch_size) ** 2 + 1, embed_dim)
        )
        self.pos_embed.requires_grad = False

    def forward(self, x: torch.Tensor):
        bs, _, _, _ = x.shape
        img_patches = self.patchlayer(x)

        cls_token = repeat(self.cls_token, "() s e -> b s e", b=bs)

        x = torch.cat([img_patches, cls_token], dim=1)
        x = x + self.pos_embed

        return x


class ViTBlock(nn.Module):
    def __init__(self, img_size, embed_dim, hidden_dim, patch_size):
        super().__init__()
        self.patch_layer = PatchEmbedding()
        self.mlp_layer = nn.Sequential(
            nn.Linear(768, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 768)
        )

    def forward(self, x: torch.Tensor):
        x = self.patch_layer(x)

        return x
