import torch
import cv2
import numpy as np


class config:
    lr = 1e-4
    image_size = 224
    batch_size = 32
    embed_dim = 768
    hidden_dim = 3072
    patch_size = 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_id = ""
    split = 5000
    dtype = torch.float32


def read_image(img):
    img = np.array(img)
    img = cv2.resize(img, (config.image_size, config.image_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0

    return img
