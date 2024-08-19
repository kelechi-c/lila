import torch
import numpy
import random


def seed_everything(seed=22):
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def count_params(model: torch.nn.Module):
    p_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return p_count


class config:
    image_size = 224
    epoch_count = 50
    batch_size = 32
    grad_acc_step = 4
    decoder_depth = 8
    decoder_dim = 512
    embed_dim = 1024
    lr = 1.5e-4
    decay = 0.05
    hidden_dim = 3072
    model_file = "lil_mae.pth"
    safetensor_file = "lil_mae.safetensors"
    model_outpath = "lila"
    att_heads = 12
    patch_size = 16
    mask_ratio = 0.75
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16
    dataset_id = "timm/imagenet-w21-p"
    batch_size = 32
    split = 10000
