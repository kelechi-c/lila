import torch
import cv2
import numpy as np


class config:
    lr = 1e-4
    epoch_count = 50
    image_size = 224
    batch_size = 32
    embed_dim = 768
    hidden_dim = 3072
    att_heads = 12
    patch_size = 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = 5000
    dataset_id = "visual-layer/imagenet-1k-vl-enriched"
    dtype = torch.float32
    model_file = "vit_mini.pth"
    safetensor_file = "vit_mini.safetensors"
    model_outpath = "vit_mini"
    grad_acc_step = 4


def read_image(img):
    img = np.array(img)
    img = cv2.resize(img, (config.image_size, config.image_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0

    return img


def count_params(model: torch.nn.Module):
    p_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return p_count
