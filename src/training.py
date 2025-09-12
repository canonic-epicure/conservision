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
