import torch
import numpy as np
from torch import nn
from torch.nn import functional as func_nn
from .utils import config
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


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


class MultiSelfAttention(nn.Module):
    def __init__(self, embed_dim: int = 768, num_heads: int = 12, droprate=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.drop = nn.Dropout(droprate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = rearrange(self.query(x), "b n (h e) -> b h n e", h=self.num_heads)
        k = rearrange(self.key(x), "b n (h e) -> b h n e", h=self.num_heads)
        v = rearrange(self.value(x), "b n (h e) -> b h n e", h=self.num_heads)

        attn_weight = q @ k.transpose(3, 2) / self.num_heads**0.5

        attn_score = func_nn.softmax(attn_weight, dim=2)

        attn_x = self.drop(attn_score)

        x = attn_x @ v
        x = rearrange(x, "b h n e -> b n (h e)", h=self.num_heads)

        return x
