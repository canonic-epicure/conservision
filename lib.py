import glob
import re
import math
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
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
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
        LabCLAHE(),
        LabCLAHE(),
        v2.ToPILImage(),

        v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.05),
        v2.RandomAutocontrast(p=0.1),
        v2.RandomZoomOut(p=0.1),
        v2.RandomEqualize(p=0.1),
        v2.RandomAdjustSharpness(p=0.3, sharpness_factor=1.5),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(degrees=15, interpolation=InterpolationMode.BICUBIC),
    ]
)
siglip2_inference_transform = v2.Compose(
    [
        LabCLAHE(),
        LabCLAHE(),
        v2.ToPILImage(),
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
        else:
            img = siglip2_inference_transform(img)

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
            logits = model.forward(batch["pixel_values"].to(torch.device("cuda"))).logits / T

            if accumulate_loss:
                loss = criterion(logits, batch["labels"].to('cuda'))

                c = batch['pixel_values'].size(0)
                loss_acc += loss.item() * c
                count += c

            if accumulate_probs:
                # 2) apply softmax so that model outputs are in range [0,1]
                preds = F.softmax(logits, dim=1)
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


def sce_loss(logits, target, alpha=0.1, beta=1.0, num_classes=None, eps=1e-4):
    ce = F.cross_entropy(logits, target, reduction='none')  # [N]

    # one-hot с клампом (можно подать сюда и mixup-таргеты [N,C] без клампа)
    if num_classes is None:
        num_classes = logits.size(1)

    target = target.clamp_min(eps)  # [N,C]

    p = F.softmax(logits, dim=1)
    rce = -(p * target.log()).sum(dim=1)  # [N]

    loss = alpha * ce + beta * rce
    return loss.mean()


# # Утилита: делаем функцию-множитель для LambdaLR
# def make_cosine_multiplier(total_steps, warmup_steps, floor_ratio):
#     """
#     total_steps: на сколько шагов растянуть спад; после этого держим floor.
#     warmup_steps: сколько шагов линейно разогревать от 0 до 1.
#     floor_ratio: доля от базового LR, ниже которой не падаем (например, 0.1 = 10%).
#     Возвращает f(step_index) -> scale в [floor_ratio, 1.0].
#     """
#     total_steps = max(1, int(total_steps))
#     warmup_steps = max(0, int(warmup_steps))
#     floor_ratio = float(floor_ratio)
#
#     def f(step):
#         # В WARNING: в PyTorch первый scheduler.step() делает last_epoch=0.
#         # Поэтому step здесь – это last_epoch от LambdaLR.
#         if step < warmup_steps:
#             # линейный разогрев: 0 -> 1 (но не больше 1)
#             return (step + 1) / max(1, warmup_steps)
#         # прогресс внутри "косинусной" части [0..1]
#         t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
#         if t >= 1.0:
#             # вышли за горизонт: держим пол
#             return floor_ratio
#         # косинусный спад из [1..floor_ratio]
#         return floor_ratio + (1.0 - floor_ratio) * 0.5 * (1.0 + math.cos(math.pi * t))
#     return f


def choose_tau_from_val(logits, targets, T=1.0, target_prec=0.95, min_coverage=0.2):
    """
    logits_val: [N, K] логиты на валидации (без softmax), индекс i — объект, k — класс.
    y_val:      [N] целочисленные метки 0..K-1.
    T:          температура (T* после калибровки; если не калибровали — 1.0).
    """
    N = targets.numel()
    probs = F.softmax(logits / T, dim=1)        # p_{i,k}
    confidence, class_idx = probs.max(dim=1)                   # c_i, \hat y_i
    correct = (class_idx == targets).to(torch.int32)   # r_i

    # сортировка по уверенности по убыванию
    confidence_sorted, confidence_sorted_idx = torch.sort(confidence, descending=True)
    correct_sorted = correct[confidence_sorted_idx]

    # кумулятивная точность на префиксе длины k
    cum_correct = torch.cumsum(correct_sorted, dim=0)            # [N]
    k = torch.arange(1, N + 1, device=logits.device)
    prec_at_k = cum_correct.float() / k.float()
    coverage = k.float() / float(N)

    # ищем минимальный k с нужной точностью и покрытием
    mask = (prec_at_k >= target_prec) & (coverage >= min_coverage)
    if mask.any():
        print("Has something with both precision and coverage: ")
        k_star = torch.nonzero(mask, as_tuple=False)[0, 0].item()
    else:
        print("Fallback")
        # fallback: лучший k по точности при покрытии >= min_coverage,
        # иначе просто глобальный максимум точности
        mask2 = coverage >= min_coverage
        if mask2.any():
            k_star = torch.argmax(prec_at_k * mask2.float()).item()
        else:
            k_star = torch.argmax(prec_at_k).item()

    tau   = confidence_sorted[k_star].item()      # порог уверенности
    a_tau = prec_at_k[k_star].item()        # точность среди отобранных
    cov   = coverage[k_star].item()         # доля отобранных
    return tau, a_tau, cov


import torch
import torch.nn.functional as F

@torch.no_grad()
def choose_tau_per_class_from_val(
    logits_val: torch.Tensor,
    y_val: torch.Tensor,
    T: float = 1.0,
    target_prec: float = 0.95,
    min_coverage: float = 0.2,   # доля в пределах каждого предсказанного класса
    min_count: int = 30,         # минимум примеров предсказанного класса для устойчивой оценки
    measure: str = "conf",       # "conf" (=p_top1), "margin" (=p_top1 - p_top2), "entropy" (низкая лучше)
):
    """
    Вход:
      logits_val: [N, K] логиты на OOF/валидации (i — объект, k — класс)
      y_val:      [N] истинные метки классов (0..K-1)
      T:          температура (после калибровки)
      target_prec: целевая точность среди отобранных в каждом классе
      min_coverage: минимальная доля покрытия внутри каждого предсказанного класса
      min_count:   минимально допустимое число предсказаний класса для подбора порога
      measure:     как мерить «уверенность»:
                   - "conf":     s_i = p_top1
                   - "margin":   s_i = p_top1 - p_top2
                   - "entropy":  s_i = - H(p_i), т.е. чем больше, тем увереннее

    Выход:
      tau_per_class:   [K] тензор порогов по measure для предсказанного класса k (NaN если нельзя оценить)
      prec_per_class:  [K] достигнутая точность среди отобранных
      cov_per_class:   [K] достигнутое покрытие (доля в пределах предсказанного класса)
      counts_per_class:[K] число объектов, для которых \hat y_i = k
      extras: dict с p (N,K), pred (N,), score s (N,), conf (N,), correct (N,)
    """
    assert logits_val.ndim == 2, "logits_val должен быть [N, K]"
    N, K = logits_val.shape
    device = logits_val.device

    # 1) Вероятности и базовые величины
    p = F.softmax(logits_val / T, dim=1)               # p_{i,k}
    conf, pred = p.max(dim=1)                          # p_top1, \hat y_i
    correct = (pred == y_val).to(torch.int32)          # r_i

    # 2) Счёт уверенности s_i по выбранной мере
    if measure == "conf":
        s = conf                                        # p_top1
    elif measure == "margin":
        top2p, _ = p.topk(k=2, dim=1)                   # [N,2]
        s = top2p[:, 0] - top2p[:, 1]                   # p_top1 - p_top2
    elif measure == "entropy":
        # Низкая энтропия => выше уверенность. Возьмём s = -H, чтобы сортировать по убыванию s.
        eps = 1e-12
        H = -(p * (p.clamp_min(eps)).log()).sum(dim=1)  # [N]
        s = -H
    else:
        raise ValueError(f"Unknown measure: {measure}")

    tau_per_class   = torch.full((K,), float('nan'), device=device)
    prec_per_class  = torch.full((K,), float('nan'), device=device)
    cov_per_class   = torch.full((K,), float('nan'), device=device)
    counts_per_class= torch.zeros((K,), dtype=torch.long, device=device)

    # 3) Перебор классов по предсказанию \hat y_i = k
    for k in range(K):
        mask_k = (pred == k)
        n_k = int(mask_k.sum().item())
        counts_per_class[k] = n_k

        if n_k < max(1, min_count):
            # Недостаточно примеров — оставим NaN; можно позже подставить глобальный порог
            continue

        s_k       = s[mask_k]
        correct_k = correct[mask_k]

        # Сортируем по убыванию score внутри класса k
        s_sorted, idx = torch.sort(s_k, descending=True)
        correct_sorted = correct_k[idx]

        # Кумулятивная точность и покрытие относительно n_k
        cum_correct = torch.cumsum(correct_sorted, dim=0)                 # [n_k]
        kk = torch.arange(1, n_k + 1, device=device)
        prec_at_kk = cum_correct.float() / kk.float()                     # точность на префиксе
        cov_at_kk  = kk.float() / float(n_k)                              # покрытие внутри класса

        # Ищем минимальный префикс, удовлетворяющий целям
        mask_ok = (prec_at_kk >= target_prec) & (cov_at_kk >= min_coverage[k])
        if mask_ok.any():
            kk_star = torch.nonzero(mask_ok, as_tuple=False)[0, 0].item()
        else:
            # fallback 1: лучший по точности при покрытии >= min_coverage
            mask_cov = (cov_at_kk >= min_coverage[k])
            if mask_cov.any():
                # argmax точности на допустимом покрытии
                kk_star = torch.argmax(prec_at_kk * mask_cov.float()).item()
            else:
                # fallback 2: глобальный максимум точности на префиксе
                kk_star = torch.argmax(prec_at_kk).item()

        tau_k   = s_sorted[kk_star].item()
        a_k     = prec_at_kk[kk_star].item()
        cov_k   = cov_at_kk[kk_star].item()

        tau_per_class[k]  = tau_k
        prec_per_class[k] = a_k
        cov_per_class[k]  = cov_k

    extras = dict(p=p, pred=pred, score=s, conf=conf, correct=correct)
    return tau_per_class, prec_per_class, cov_per_class, counts_per_class, extras
