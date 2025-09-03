from typing import Tuple

import torch
import torch.nn.functional as F
from PIL import Image

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


import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import cv2

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
            # ожидаем CxHxW, [0,1]
            x = (img.clamp(0,1).mul(255).byte().permute(1,2,0).cpu().numpy())
        elif isinstance(img, Image.Image):
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