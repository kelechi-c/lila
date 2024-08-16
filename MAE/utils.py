import torch
import numpy
import random


def seed_everything(seed=22):
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class config:
    lr = 1e-4
    image_size = 224
    epoch_count = 50
    batch_size = 32
    decoder_depth = 8
    decoder_dim = 512
    embed_dim = 1024
    lr = 1.5e-4
    decay = 0.05
    hidden_dim = 3072
    att_heads = 12
    patch_size = 16
    mask_ratio = 0.75
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16
