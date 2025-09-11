import os
import re
import glob
import fnmatch
from os import PathLike
from typing import Union, List

import pandas as pd
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

import data
import transformations
from pathlib import Path


class CheckpointsStorage():
    dir : Path

    def __init__(self, *args, dir: Path, **kwargs):
        self.dir = dir
        super().__init__(*args, **kwargs)


    def touch(self):
        self.dir.mkdir(parents=True, exist_ok=True)

    def latest(self, reg: re.Pattern) -> (Path, int):
        if not isinstance(reg, re.Pattern):
            reg = re.compile(reg)

        def collect():
            for root, dirs, files in os.walk('.'):
                for file in files:
                    match = reg.match(file)
                    if match:
                        yield file, int(match.group(1))

        epochs = list(collect())

        if len(epochs) == 0:
            return None

        epochs.sort(key=lambda res: res[1], reverse=True)

        return self.dir / epochs[0][0], epochs[0][1]


class Metric():
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Epoch():
    epoch: int

    loss: List[float]

    def __init__(self, *args, model, epoch=0, **kwargs):
        self.epoch = epoch
        super().__init__(*args, **kwargs)


class Training():
    epoch: int

    loss: List[float]

    def __init__(self, *args, model, epoch=0, **kwargs):
        self.epoch = epoch
        super().__init__(*args, **kwargs)

#
#
# class ImageDataset(Dataset):
#     def __init__(self, *args, data_dir, data: pd.DataFrame, processor, aug=None, **kwargs):
#         self.data_dir = str(data_dir).rstrip('/') + '/'
#         self.data = data
#         self.processor = processor
#         self.aug = aug
#         super().__init__(*args, **kwargs)
#
#     def __len__(self):
#         return len(self.data)
#
#     def get_image(self, idx):
#         image_id = self.data.index[idx]
#
#         image = Image.open(self.data_dir + self.data.iloc[idx]["filepath"]).convert("RGB")
#
#         image = self.aug(image) if self.aug else image
#         image = self.processor(image)
#
#         return image, image_id
#
#
# class ImageDatasetWithLabel(DatasetWithLabel, ImageDataset):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#
#     def __getitem__(self, idx):
#         image, image_id = self.get_image(idx)
#
#         return image, self.get_labels(idx), image_id
#
#
# class DivideMixLabeledDataset(ImageDatasetWithLabel):
#     def __init__(self, *args, probs, **kwargs):
#         self.probs = probs
#         super().__init__(*args, **kwargs)
#
#     def __getitem__(self, idx):
#         image1, image_id = self.get_image(idx)
#         image2, _ = self.get_image(idx)
#
#         return image1, image2, self.get_labels(idx), self.probs[idx], image_id
#
# class DivideMixUnlabeledDataset(ImageDataset):
#     def __getitem__(self, idx):
#         image1, image_id = self.get_image(idx)
#         image2, _ = self.get_image(idx)
#
#         return image1, image2, image_id
#
