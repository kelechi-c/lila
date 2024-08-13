import torch
from torch import nn
from timm.layers import LayerNorm


class PatchEmbed(nn.Module):
    def __init__(self) -> None:
        super().__init__(*args, **kwargs)
