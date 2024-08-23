import torch
from torch import nn
from ViT.utils import config
from .model_blocks import MultiSelfAttention, PatchEmbedding
from huggingface_hub import login, PyTorchModelHubMixin


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


class ViT(
    nn.Module,
    PyTorchModelHubMixin,
    pipeline_tag="image-classification",
    license="apache-2.0",
):
    def __init__(self, embed_dim: int = 768, num_layers=6, classes: int = 1000):
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


vit = ViT().to(config.dtype).to(config.device)

vit.save_pretrained("vit4HAR")
# push to the hub
vit.push_to_hub("tensorkelechi/vit4HAR")
