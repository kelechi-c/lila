import torch
from torch import nn
from .utils import config
from einops.layers.torch import Rearrange
from einops import repeat
import numpy as np


def random_shuffle(image: torch.Tensor, mask_ratio=0.75):
    B, L, D = image.shape

    keep_len = int((L * (1 - mask_ratio)))
    noise = torch.rand(B, L, device=image.device)  # random masking noise

    # get indexes for shuffled tokens
    shuffle_ids = torch.argsort(noise, dim=1)
    restore_ids = torch.argsort(noise, dim=1)
    retain_ids = shuffle_ids[:, :keep_len]  # remaining pixel ids

    x = torch.gather(
        image, dim=1, index=retain_ids.unsqueeze(-1).repeat(1, 1, D)
    )  # remaining image pixels/tokens

    # get binary mask
    mask = torch.ones([B, L], device=image.device)
    mask[:, :keep_len] = 0

    mask = torch.gather(mask, dim=1, index=restore_ids)  # masks

    return x, mask, restore_ids


class PatchEmbed(nn.Module):
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
            torch.zeros(1, num_patches, embed_dim), requires_grad=False
        )

    def _pos_embedding(self, seq_len, embed_dim=config.embed_dim):
        embeds = torch.ones(seq_len, embed_dim)
        for k in range(seq_len):
            for v in range(embed_dim):
                embeds[k][v] = (
                    np.sin(k / pow(10000, v / embed_dim))
                    if v % 2 == 0
                    else np.cos(k / pow(10000, v - 1 / embed_dim))
                )

        return embeds

    def forward(self, x: torch.Tensor):
        bs, _, _, _ = x.shape
        img_patches = self.patchlayer(x)

        cls_token = repeat(self.cls_token, "() s e -> b s e", b=bs)

        x = torch.cat([img_patches, cls_token], dim=1)
        x = x + self.pos_embed

        return x
