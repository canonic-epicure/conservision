import pandas as pd
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

import data
import transformations


class DatasetWithLabel(Dataset):
    def __init__(self, *args, labels: pd.DataFrame=None, **kwargs):
        self.labels = labels
        super().__init__(*args, **kwargs)

    def get_labels(self, idx):
        return torch.tensor(self.labels.iloc[idx].values).float()


class ImageDataset(Dataset):
    def __init__(self, *args, data_dir, data: pd.DataFrame, processor, aug=None, **kwargs):
        self.data_dir = str(data_dir).rstrip('/') + '/'
        self.data = data
        self.processor = processor
        self.aug = aug
        super().__init__(*args, **kwargs)

    def __len__(self):
        return len(self.data)

    def get_image(self, idx):
        image_id = self.data.index[idx]

        image = Image.open(self.data_dir + self.data.iloc[idx]["filepath"]).convert("RGB")

        image = self.aug(image) if self.aug else image
        image = self.processor(image)

        return image, image_id


class ImageDatasetWithoutLabel(ImageDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        image, image_id = self.get_image(idx)

        return image, image_id


class ImageDatasetWithLabel(DatasetWithLabel, ImageDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        image, image_id = self.get_image(idx)

        return image, self.get_labels(idx), image_id


class DivideMixLabeledDataset(ImageDatasetWithLabel):
    def __init__(self, *args, probs, **kwargs):
        self.probs = probs
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        image1, image_id = self.get_image(idx)
        image2, _ = self.get_image(idx)

        return image1, image2, self.get_labels(idx), self.probs[idx], image_id

class DivideMixUnlabeledDataset(ImageDataset):
    def __getitem__(self, idx):
        image1, image_id = self.get_image(idx)
        image2, _ = self.get_image(idx)

        return image1, image2, image_id


random_state = 42

class DividemixDataloaderFactory():
    def __init__(self, data_dir, batch_size, num_workers, x_train, y_train, x_eval, y_eval, preprocessor, aug_train, aug_inference):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = data_dir

        self.preprocessor = preprocessor
        self.aug_train = aug_train
        self.aug_inference = aug_inference

        self.x_train = x_train
        self.y_train = y_train
        self.x_eval = x_eval
        self.y_eval = y_eval

    def get_eval_train_dataloader(self) -> DataLoader:
        data_ = self.x_train #.sample(frac=0.01, random_state=random_state)
        labels_ = self.y_train.loc[ data_.index ]

        return DataLoader(
            dataset=ImageDatasetWithLabel(
                data_dir=self.data_dir,
                data=data_, labels=labels_,
                processor=self.preprocessor, aug=self.aug_inference
            ),
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=self.num_workers
        )

    def get_warmup_dataloader(self) -> DataLoader:
        data_ = self.x_train #.sample(frac=0.01, random_state=random_state)
        labels_ = self.y_train.loc[ data_.index ]

        return DataLoader(
            dataset=ImageDatasetWithLabel(
                data_dir=self.data_dir,
                data=data_, labels=labels_,
                processor=self.preprocessor, aug=self.aug_train
            ),
            batch_size=self.batch_size * 2,
            shuffle=True,
            num_workers=self.num_workers
        )

    def get_validation_dataloader(self) -> DataLoader:
        data_ = self.x_eval #.sample(frac=0.01, random_state=random_state)
        labels_ = self.y_eval.loc[ data_.index ]

        return DataLoader(
            dataset=ImageDatasetWithLabel(
                data_dir=self.data_dir,
                data=data_, labels=labels_,
                processor=self.preprocessor, aug=self.aug_inference
            ),
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=self.num_workers
        )

    # same as validation for now
    def get_test_dataloader(self) -> DataLoader:
        # data_ = self.x_eval #.sample(frac=0.01, random_state=random_state)
        # labels_ = self.y_eval.loc[ data_.index ]

        return DataLoader(
            dataset=ImageDatasetWithLabel(
                data_dir=self.data_dir,
                data=data.test_features, labels=data.test_labels,
                processor=self.preprocessor, aug=self.aug_inference
            ),
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=self.num_workers
        )

    def get_predict_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=ImageDatasetWithoutLabel(
                data_dir=self.data_dir,
                data=data.test_features,
                processor=self.preprocessor, aug=self.aug_inference
            ),
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=self.num_workers
        )

    def get_train_dataloader(self, ids, probs, mask) -> (DataLoader, DataLoader):
        labeled_ids = np.array(ids)[mask]

        labeled_dataset = DivideMixLabeledDataset(
            data_dir=self.data_dir,
            processor=self.preprocessor, aug=self.aug_train,
            data=self.x_train.loc[labeled_ids], labels=self.y_train.loc[labeled_ids], probs=probs[mask],
        )
        labeled_loader = DataLoader(
            dataset=labeled_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

        unlabeled_ids = np.array(ids)[~mask]
        unlabeled_dataset = DivideMixUnlabeledDataset(
            data_dir=self.data_dir,
            processor=self.preprocessor, aug=self.aug_train,
            data=self.x_train.loc[unlabeled_ids]
        )
        unlabeled_loader = DataLoader(
            dataset=unlabeled_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
        return labeled_loader, unlabeled_loader
