from os import cpu_count
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from functools import partial
from PIL import Image

from torchvision.datasets import StanfordCars

import torch

from torchvision import transforms as T


def get_dataloader(
    image_size: int = 128,
    augment_horizontal_flip=False,
    convert_image_to=None,
    data_dir: str = "./data",
    batch_size: int = 64,
):

    maybe_convert_fn = (
        partial(convert_image_to_fn, convert_image_to)
        if exists(convert_image_to)
        else torch.nn.Identity()
    )

    transform = T.Compose(
        [
            T.Lambda(maybe_convert_fn),
            T.Resize(image_size),
            T.RandomHorizontalFlip()
            if augment_horizontal_flip
            else torch.nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor(),
        ]
    )

    dataset = StanfordCars(data_dir, transform=transform, download=True)

    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=cpu_count(),
    )

    return dl


def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image


def exists(val):
    return val is not None
