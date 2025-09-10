# import pandas as pd
# import torch
# import numpy as np
# from PIL import Image
# from torch.utils.data import Dataset, DataLoader
#
# import data
# import transformations
#
#
# class Training():
#     def __init__(self, *args, model, **kwargs):
#         self.labels = labels
#         super().__init__(*args, **kwargs)
#
#     def get_labels(self, idx):
#         return torch.tensor(self.labels.iloc[idx].values).float()
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
