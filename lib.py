import glob
import re
from pathlib import Path
from typing import Tuple, List

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2

import random

import photo


def get_resolution(filename):
    with Image.open(filename) as img:
        return f'{img.size[0]}x{img.size[1]}'


def is_overexposed_torchvision(
    img_u8: torch.Tensor,
    roi_border_frac: float = 0.05,
    T_hi: int = 240,
    mu_thr: float = 240.0,
    sigma_thr: float = 8.0,
    p_hi_thr: float = 0.75,
    p_edge_thr: float = 0.05
):
    """
    img_u8: uint8 tensor [C,H,W] в диапазоне 0..255 (RGB)
    Возвращает (bool, dict метрик).
    """
    assert img_u8.dtype == torch.uint8 and img_u8.ndim == 3 and img_u8.shape[0] == 3
    C, H, W = img_u8.shape

    # 1) ROI: отрежем рамки/штампы по краям
    rb = int(round(roi_border_frac * H))
    cb = int(round(roi_border_frac * W))
    img = img_u8[:, rb:H-rb if H-2*rb>0 else H, cb:W-cb if W-2*cb>0 else W]

    # 2) в float
    x = img.float()  # [3,h,w]

    # 3) яркость (люминанс)
    Y = 0.2126 * x[0] + 0.7152 * x[1] + 0.0722 * x[2]  # [h,w]

    # 4) статистики яркости
    mu = Y.mean()
    sigma = Y.std(unbiased=False)

    # 5) доля пикселей, где ВСЕ каналы почти белые
    thr = torch.tensor(T_hi, dtype=torch.float32, device=x.device)
    p_hi = ((x[0] >= thr) & (x[1] >= thr) & (x[2] >= thr)).float().mean()

    # 6) «фактура»: плотность границ по Собелю
    #   задаём ядра Собеля (на batch=1, channel=1)
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32, device=x.device).view(1,1,3,3)
    ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32, device=x.device).view(1,1,3,3)
    Y1 = Y.unsqueeze(0).unsqueeze(0)  # [1,1,h,w]
    gx = F.conv2d(Y1, kx, padding=1)
    gy = F.conv2d(Y1, ky, padding=1)
    G = torch.sqrt(gx*gx + gy*gy).squeeze()  # [h,w]

    # порог для градиента возьмём «абсолютный» в u8-единицах
    tau = 12.0
    p_edge = (G > tau).float().mean()

    decision = (mu >= mu_thr) and (sigma <= sigma_thr) and (p_hi >= p_hi_thr) and (p_edge <= p_edge_thr)

    return bool(decision), {
        "mu": float(mu.item()),
        "sigma": float(sigma.item()),
        "p_hi": float(p_hi.item()),
        "p_edge": float(p_edge.item())
    }

def to_rgb(img_u8: torch.Tensor):
    if img_u8.shape[0] == 1:
        img_u8 = img_u8.repeat(3, 1, 1)
    return img_u8


def background_template(img_batch: torch.Tensor) -> torch.Tensor:
    assert img_batch.ndim == 4

    return img_batch.median(dim=0).values


def affine_params_to_background(x_i: torch.Tensor, background_i: torch.Tensor, eps:float =1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x_i.shape == background_i.shape, "x и background должны совпадать по форме."

    x = x_i
    background = background_i

    # Средние и СКО по всем пикселям каждого канала
    mux  = x.mean(dim=(1, 2))
    sdx  = x.std(dim=(1, 2), unbiased=False).clamp_min(eps)
    muB  = background.mean(dim=(1, 2))
    sdB  = background.std(dim=(1, 2), unbiased=False).clamp_min(eps)

    alpha = sdB / sdx            # (C,)
    beta  = muB - alpha * mux    # (C,)

    return alpha, beta

def similarity(img1_u8: torch.Tensor, img2_u8: torch.Tensor) -> float:
    img1_u8 = to_rgb(img1_u8)
    img2_u8 = to_rgb(img2_u8)

    assert img1_u8.dtype == torch.uint8 and img1_u8.ndim == 3 and img1_u8.shape[0] == 3
    assert img2_u8.dtype == torch.uint8 and img2_u8.ndim == 3 and img2_u8.shape[0] == 3

    C, H1, W1 = img1_u8.shape
    C, H2, W2 = img2_u8.shape

    assert H1 == H2 and W1 == W2


class LabCLAHE:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8),
                 apply_if_dark=True, dark_thr=0.35):
        """
        clip_limit: насколько ограничивать усиление контраста (>1 => сильнее).
        tile_grid_size: размер сетки CLAHE (в плитках).
        apply_if_dark: если True, применяем CLAHE только для "тёмных" кадров.
        dark_thr: порог по средней яркости L в [0,1], ниже которого включаем CLAHE.
        """
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.apply_if_dark = apply_if_dark
        self.dark_thr = dark_thr

        self.to_tensor = T.ToTensor()
        self.to_pil = T.ToPILImage()

    @torch.no_grad()
    def __call__(self, img):
        # Приводим к numpy uint8 RGB [0..255]
        if isinstance(img, torch.Tensor):
            img = to_rgb(img)
            # ожидаем CxHxW, [0,1]
            x = (img.clamp(0,1).mul(255).byte().permute(1,2,0).cpu().numpy())
        elif isinstance(img, Image.Image):
            img = img.convert("RGB")
            x = np.array(img)  # HxWxC, uint8, RGB
        else:
            raise TypeError("img must be PIL.Image or torch.Tensor")

        # RGB -> Lab (OpenCV ожидает RGB во флаге COLOR_RGB2LAB)
        lab = cv2.cvtColor(x, cv2.COLOR_RGB2LAB)  # uint8
        L = lab[:,:,0]        # L в [0..255]
        a = lab[:,:,1]        # a в [0..255] со сдвигом 128
        b = lab[:,:,2]        # b в [0..255] со сдвигом 128

        # Критерий "темноты" по средней L (нормируем к [0,1])
        mean_L = L.mean() / 255.0

        do_apply = True
        if self.apply_if_dark:
            do_apply = (mean_L < self.dark_thr)

        if do_apply:
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit,
                                    tileGridSize=self.tile_grid_size)
            L_eq = clahe.apply(L)
        else:
            L_eq = L  # оставляем как есть

        lab_eq = np.stack([L_eq, a, b], axis=2).astype(np.uint8)

        # Lab -> RGB
        rgb_eq = cv2.cvtColor(lab_eq, cv2.COLOR_Lab2RGB)  # uint8

        # обратно к тензору [0,1], CxHxW
        out = torch.from_numpy(rgb_eq).permute(2,0,1).float() / 255.0
        return out


class RandomInvertIfGrayscale:
    def __init__(self, p=0.5):
        self.p = p
        self.invert = v2.RandomInvert(p=1.0)

    @torch.no_grad()
    def __call__(self, img):
        if not isinstance(img, Image.Image):
            raise TypeError("img must be PIL.Image ")

        is_grayscale, _ = photo.looks_grayscale_ycbcr_cv(img)

        # no-op if image is not grayscale or random.random() > self.p
        if not is_grayscale or random.random() > self.p:
            return img

        return self.invert(img)


class ImagesDatasetResnet(Dataset):
    def __init__(self, x_df, y_df=None, learning=True):
        self.data = x_df
        self.label = y_df

        self.transform = v2.Compose(
            [
                LabCLAHE(),
                LabCLAHE(),
                v2.ToPILImage(),

                v2.ColorJitter() if learning else lambda x: x,
                v2.RandomAutocontrast() if learning else lambda x: x,
                v2.RandomEqualize() if learning else lambda x: x,
                v2.RandomAdjustSharpness(sharpness_factor=1.5) if learning else lambda x: x,
                v2.RandomHorizontalFlip() if learning else lambda x: x,
                v2.RandomRotation(degrees=15, interpolation=InterpolationMode.BICUBIC) if learning else lambda x: x,

                v2.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
                v2.ToDtype(torch.float32, scale=True)(),
                v2.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )

    def __getitem__(self, index):
        image = Image.open('data/' + self.data.iloc[index]["filepath"]).convert("RGB")
        image = self.transform(image)
        image_id = self.data.index[index]
        # if we don't have labels (e.g. for test set) just return the image and image id
        if self.label is None:
            sample = {"image_id": image_id, "image": image}
        else:
            label = torch.tensor(self.label.iloc[index].values, dtype=torch.float)
            sample = {"image_id": image_id, "image": image, "label": label}
        return sample

    def __len__(self):
        return len(self.data)


siglip2_training_transform = v2.Compose(
    [
        RandomInvertIfGrayscale(p=0.3),

        LabCLAHE(),
        LabCLAHE(),
        v2.ToPILImage(),

        v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.05),
        v2.RandomAutocontrast(p=0.1),
        v2.RandomEqualize(p=0.1),
        v2.RandomAdjustSharpness(p=0.3, sharpness_factor=1.5),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(degrees=15, interpolation=InterpolationMode.BICUBIC),
    ]
)

class ImageDatasetSigLip2(Dataset):
    def __init__(self, data: pd.DataFrame, labels: pd.DataFrame=None, processor=None, learning=True):
        self.data = data
        self.labels = labels
        self.processor = processor
        self.learning = learning

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        img = Image.open('data/' + self.data.iloc[idx]["filepath"]).convert("RGB")
        image_id = self.data.index[idx]

        if self.learning:
            img = siglip2_training_transform(img)

        # enc["pixel_values"]: (1, C, H, W) -> уберём размерность 0
        enc = self.processor(images=img, return_tensors="pt")
        if self.labels is None:
            return {
                "image_id": image_id,
                "pixel_values": enc["pixel_values"].squeeze(0)
            }
        else:
            return {
                "image_id": image_id,
                "pixel_values": enc["pixel_values"].squeeze(0),
                "labels": torch.tensor(self.labels.iloc[idx].values, dtype=torch.float)
            }

    # def collate_fn(batch):
    #     pixel_values = torch.stack([b["pixel_values"] for b in batch], dim=0)
    #     labels = torch.stack([b["labels"] for b in batch], dim=0)
    #     return {"pixel_values": pixel_values, "labels": labels}


def save_model(model, optimizer, file_name: str):
    Path(file_name).parent.mkdir(parents=True, exist_ok=True)

    torch.save({'model' : model, 'optimizer': optimizer}, file_name)

def model_checkpoints(glob_pattern: str) -> List[str]:
    files = glob.glob(glob_pattern)

    pattern = re.compile(r"checkpoint_(\d+)\.pth$")

    epochs = [ pattern.search(file)[1] for file in files if pattern.search(file) != None ]

    epochs.sort(reverse=True)

    return epochs


def predict_siglip(model, data_loader: DataLoader, accumulate_probs=True, accumulate_loss=False, T=1, desc='Predicting', criterion=None, columns=None):
    preds_collector = []

    # put the model in eval mode so we don't update any parameters
    model.eval()

    model.to(torch.device("cuda"))

    loss_acc = 0
    count = 0

    # we aren't updating our weights so no need to calculate gradients
    with torch.no_grad():
        for batch in tqdm.tqdm(data_loader, total=len(data_loader), desc=desc):
            # 1) run the forward step
            logits = model.forward(batch["pixel_values"].to(torch.device("cuda"))).logits

            if accumulate_loss:
                loss = criterion(logits, batch["labels"].to('cuda'))

                c = batch['pixel_values'].size(0)
                loss_acc += loss.item() * c
                count += c

            if accumulate_probs:
                # 2) apply softmax so that model outputs are in range [0,1]
                preds = F.softmax(logits / T, dim=1)
                # 3) store this batch's predictions in df
                # note that PyTorch Tensors need to first be detached from their computational graph before converting to numpy arrays
                preds_df = pd.DataFrame(
                    preds.detach().to('cpu').numpy(),
                    index=batch["image_id"],
                    columns=columns,
                )
                preds_collector.append(preds_df)

    return pd.concat(preds_collector) if accumulate_probs else None, loss_acc / count if accumulate_loss else None



def predict_siglip_ten_crop(model, data_loader: DataLoader, T=1, desc='Predicting', columns=None):
    preds_collector = []

    # put the model in eval mode so we don't update any parameters
    model.eval()

    model.to(torch.device("cuda"))

    # we aren't updating our weights so no need to calculate gradients
    with torch.no_grad():
        for batch in tqdm.tqdm(data_loader, total=len(data_loader), desc=desc):
            def process_image(image):
                w, h = image.shape[1:]

                resize = v2.Resize(size=(round(w * 1.5), round(h * 1.5)), interpolation=v2.InterpolationMode.BICUBIC)
                ten_crop = v2.TenCrop(size=(w, h))

                ten_crop_img = torch.stack(list(ten_crop(resize(image))))

                # 1) run the forward step
                logits = model.forward(ten_crop_img.to(torch.device('cuda'))).logits / T

                logp = F.log_softmax(logits, dim=1)  # (M, K)
                logp_mean = torch.logsumexp(logp, dim=0) - torch.log(torch.tensor(ten_crop_img.size(0)))  # (K,)

                return torch.exp(logp_mean).detach().to('cpu').numpy()

            process_image(batch["pixel_values"][0])

            preds_df = pd.DataFrame(
                [ process_image(image) for image in batch["pixel_values"] ],
                index=batch["image_id"],
                columns=columns,
            )
            preds_collector.append(preds_df)

    return pd.concat(preds_collector)


# import math, torch
# import torch.nn.functional as F
# from torchvision import transforms
# from PIL import Image
#
# v2.TenCrop(size=)
#
# # 1) TTA-трансформа: Resize→FiveCrop(384) + hflip для каждого кропа
# resize_448 = transforms.Resize(448, antialias=True)
# fivecrop   = transforms.FiveCrop(384)
#
# def make_tta_views(img: Image.Image):
#     crops = list(fivecrop(img))                  # 5 PIL-изображений
#     flips = [transforms.functional.hflip(c) for c in crops]
#     return crops + flips                         # всего M=10
#
# # 2) Прогоняем через processor→model и агрегируем лог-вероятности
# @torch.no_grad()
# def tta_logprob_mean(model, processor, img: Image.Image, device="cuda"):
#     views = make_tta_views(img)                                  # M PIL
#     enc = processor(images=views, return_tensors="pt")           # батч M
#     logits = model(enc["pixel_values"].to(device)).logits        # (M, K)
#     logp = F.log_softmax(logits, dim=1)                          # (M, K)
#     logp_mean = torch.logsumexp(logp, dim=0) - math.log(len(views))  # (K,)
#     return logp_mean  # агрегированные лог-вероятности по классам
