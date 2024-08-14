import torch
from .utils import read_image, config
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset

dataset_id = ""

hfdata = load_dataset(dataset_id, plit="train", stream=True)


class ImageDataset(IterableDataset):
    def __init__(self, dataset=hfdata):
        super().__init__()
        self.dataset = dataset

    def __iter__(self):
        for item in self.dataset:
            image = read_image(item["image"])
            label = item["label"]

            image = torch.tensor(image, dtype=config.dtype)
            label = torch.tensor(label, dtype=config.dtype)

            yield image, label


img_dataset = ImageDataset()
train_loader = DataLoader(img_dataset, config.batch_size, shuffle=True)
