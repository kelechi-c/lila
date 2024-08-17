import torch
from torch import nn
from timm.models.vision_transformer import Block, PatchEmbed


class MaskedAutoencoder(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        img_size=224,
        in_channels=3,
        patch_size=16,
        decoder_dim=512,
        decoder_depth=8,
        num_heads=16,
        mlp_ratio=4,
        depth=12,
    ):
        self.patch_embed = PatchEmbed(
            img_size, patch_size, in_channels, embed_dim
        )  # initialize patchembed layer with custom configs

        # encoder section blocks

        self.layer_norm = nn.LayerNorm(embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, embed_dim)
        )  # encoder class token
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim),
            requires_grad=False,  # fixed sin-cos embedding
        )

        self.encoder_layers = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=self.layer_norm,
                )
                for _ in range(depth)
            ]
        )

        #### ______#####
        # decoder section/blocks
        self.mask_token = nn.Parameter(  # define mask tokens
            torch.zeros(1, 1, decoder_dim)
        )
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_dim), requires_grad=False
        )

        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.decoder_embedding = nn.Linear(
            embed_dim, decoder_dim, bias=True
        )  # map to decoder dimensionality

        self.decoder_layers = nn.ModuleList(  # blocks for shallow MAE decoder
            [
                Block(
                    decoder_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=self.decoder_norm,
                )
                for _ in range(decoder_depth)
            ]
        )

        self.output_layer = nn.Linear(
            decoder_dim, (patch_size**2) * in_channels, bias=True
        )  # output/prediction layer

    def _init_weights(self, ml):  # instantiate weights for seleccted layers/nn
        if isinstance(ml, nn.Linear):
            nn.init.xavier_uniform_(ml.weight)

        if isinstance(ml, nn.LayerNorm):
            nn.init.constant_(ml.weight, 0)
            nn.init.constant_(ml.bias, 0)

    def forward(self, x: torch.Tensor):
        return x
