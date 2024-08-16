import numpy
import random
import torch


def seed_everything(seed=22):
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def random_shuffle(image: torch.Tensor, mask_ratio=0.75):
    B, L, D = image.shape

    keep_len = int((L / 4))
    noise = torch.rand(B, L, device=image.device)  # random masking noise

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
