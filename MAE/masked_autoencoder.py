from numpy import shape
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
            nn.init.constant_(ml.bias, 1.0)

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

    def encoder(self, x: torch.Tensor):  # encoder for featur extraction/learning
        x = self.patch_embed(x)

        cls_token = x[:, :1, :]
        x = x[:, 1:, :]

        x, mask, restore_ids = self.random_shuffle(x)

        x = torch.cat((cls_token, x), dim=1)
        x = self.encoder_layers(x)

        return x

    def decoder(self, x: torch.Tensor, restore_ids):  # decoder for reconstructon
        x = self.decoder_embedding(x)

        mask_tokens = self.mask_token.repeat(
            x.shape[0], self.mask_token.shape[0] + 1 - x.shape[1], 1
        )

        masked_x = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        masked_x = torch.gather(
            masked_x, dim=1, index=restore_ids.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )
        x = torch.cat([x[:, :1, :], masked_x], dim=1)

        x = x + self.decoder_pos_embed

        x = self.decoder_layers(x)
        x = self.output_layer(x)

        return x

    def forward(self, x: torch.Tensor):
        return x
