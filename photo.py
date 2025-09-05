from typing import List, Tuple, Union, Dict, Any
import math
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

import numpy as np
import cv2

def looks_grayscale_ycbcr_cv(img, std_thresh=12.0, far_thresh=10, far_frac=0.02):
    """
    Определяет, является ли изображение преимущественно серым.

    Параметры
    ---------
    img : PIL.Image | np.ndarray  HxWx{1,3}  uint8 | float
        Изображение. Если float, допускается диапазон [0,1] (будет приведён к 0..255).
    std_thresh : float
        Порог для E = sqrt(Var(Cb) + Var(Cr)) в 8-битной шкале (0..255).
    far_thresh : int
        Отступ от 128 в Cb/Cr, чтобы считать пиксель «заметно цветным».
    far_frac : float
        Допустимая доля заметно цветных пикселей.

    Возвращает
    ----------
    (is_gray: bool, stats: dict)
    """
    arr = np.asarray(img)

    # 1) Если изображение одноканальное — считаем серым сразу
    if arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] == 1):
        return True, {"reason": "single_channel"}

    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError("Ожидается HxWx3 или HxW")

    # 2) Привести к uint8 0..255 (OpenCV ожидает именно такой масштаб)
    a = arr[..., :3]
    if a.dtype != np.uint8:
        a = a.astype(np.float32)
        if a.max() <= 1.0:
            a = (a * 255.0).clip(0, 255)
        else:
            a = a.clip(0, 255)
        a = a.astype(np.uint8)

    # 3) RGB -> YCrCb с OpenCV (внимание: порядок каналов Y, Cr, Cb)
    # Если исходник в RGB (PIL/NumPy), используем конвертацию RGB2YCrCb
    a = np.ascontiguousarray(a)  # на всякий случай для cv2
    ycrcb = cv2.cvtColor(a, cv2.COLOR_RGB2YCrCb)
    Y  = ycrcb[..., 0]
    Cr = ycrcb[..., 1]
    Cb = ycrcb[..., 2]

    # 4) Энергия цветности и доля «далёких» от серого пикселей
    sig_cb = float(np.std(Cb))
    sig_cr = float(np.std(Cr))
    E = float(np.hypot(sig_cb, sig_cr))  # sqrt(sig_cb^2 + sig_cr^2)

    far = (np.abs(Cb.astype(np.int16) - 128) > far_thresh) | \
          (np.abs(Cr.astype(np.int16) - 128) > far_thresh)
    frac_far = float(np.mean(far))

    is_gray = (E < std_thresh) and (frac_far < far_frac)

    return bool(is_gray), {
        "E": E, "std_cb": sig_cb, "std_cr": sig_cr,
        "frac_far": frac_far,
        "params": {"std_thresh": std_thresh, "far_thresh": far_thresh, "far_frac": far_frac}
    }
