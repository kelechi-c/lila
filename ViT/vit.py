import torch
from torch import nn
from .model_blocks import MultiSelfAttention, PatchEmbedding


class ViTEncoderBlock(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.attn_block = MultiSelfAttention()

        self.feed_forward = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )

        self.dropout = nn.Dropout(0.1)
        self.layernorm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor):
        input = x
        x = self.layernorm(x)
        x = input + self.attn_block(x)
        x = x + self.drop(self.feed_forward(x))

        return x


class ViT(nn.Module):
    def __init__(self, embed_dim: int, num_layers=6, classes: int = 10):
        super().__init__()

        self.patch_embedding = PatchEmbedding()  # to patchify/tokenize images
        # multilayer transformer block
        self.transformer_encoder = nn.Sequential(
            *[ViTEncoderBlock(embed_dim) for _ in range(num_layers)]
        )

        self.ff_layer = nn.Linear(embed_dim, classes)

    def forward(self, x: torch.Tensor):
        x = self.patch_embedding(x)

        x = self.transformer_encoder(x)
        x = self.ff_layer(x[:, 0, :])

        return x
