import torch
from torch import isin, nn
from timm.models.vision_transformer import PatchEmbed


class MaskedAutoencoder(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        img_size=224,
        in_channels=3,
        patch_size=16,
        decoder_dim=512,
        decoder_depth=8,
    ):
        self.patch_embed = PatchEmbed(
            img_size, patch_size, in_channels, embed_dim
        )  # initialize patchembed layer with custom configs

        self.layer_norm = nn.LayerNorm(embed_dim)

        # encoder section blocks

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )
        self._pos_embed.requires_grad = False

        self.encoder_layers = nn.Sequential()

        # decoder section/blocks
        self.mask_token = nn.Parameter(  # # define mask tokens
            torch.zeros(1, 1, decoder_dim)
        )
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_dim), requires_grad=False
        )
        self.decoder_norm = nn.LayerNorm(decoder_dim)

    def _init_weights(self, ml):
        if isinstance(ml, nn.Linear):
            nn.init.xavier_uniform_(ml.weight)

        if isinstance(ml, nn.LayerNorm):
            nn.init.constant_(ml.weight, 0)
            nn.init.constant_(ml.bias, 0)
