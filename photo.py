from typing import List, Tuple, Union, Dict, Any
import math
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

@torch.no_grad()
def detect_day_night(
    batch: Union[torch.Tensor, List[Image.Image], Image.Image],
    mu_thresh: float = 0.18,        # порог средней яркости (V) для ночи
    q95_thresh: float = 0.80,       # порог 95-перцентиля V для ночи
    sat_thresh: float = 0.10,       # доля пикселей V>0.98 => «вспышка»
    ratio_thresh: float = 1.50,     # mean(V центр)/mean(V бордюр) => «вспышка»
    return_flash: bool = True,      # вернуть ли флаг вспышки
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Принимает:
      - torch.Tensor  (B, C, H, W) или (C, H, W) в RGB; dtype float в [0,1] или uint8/float в [0,255]
      - список PIL.Image или один PIL.Image (любого размера; конвертируется в RGB)

    Возвращает словарь:
      - 'labels':  List[str] длины B: 'day' или 'night'
      - 'is_night': BoolTensor (B,)
      - 'is_flash': BoolTensor (B,)  # если return_flash=True
      - 'metrics': Dict[str, Tensor] с полями 'mu','q95','sat','ratio' (каждое (B,))
    """
    # ---- 1) Приводим вход к тензору (B, 3, H, W), float32 в [0,1] ----
    def _to_tensor_list(x) -> List[torch.Tensor]:
        if isinstance(x, Image.Image):
            x = [x]
        if isinstance(x, list):
            out = []
            for im in x:
                if not isinstance(im, Image.Image):
                    raise TypeError("Список должен содержать PIL.Image.Image")
                rgb = im.convert("RGB")
                t = torch.from_numpy(np.array(rgb))  # (H,W,3), uint8
                t = t.permute(2, 0, 1).contiguous().float() / 255.0  # -> (3,H,W) в [0,1]
                out.append(t)
            return out
        elif isinstance(x, torch.Tensor):
            t = x
            if t.dim() == 3:
                t = t.unsqueeze(0)  # (1,C,H,W) или (1,H,W,C)
            # если NHWC -> NCHW
            if t.shape[-1] == 3 and t.shape[1] not in (1, 3):
                t = t.permute(0, 3, 1, 2).contiguous()
            # если (B,1,H,W) → дублируем каналы
            if t.shape[1] == 1:
                t = t.repeat(1, 3, 1, 1)
            # в float [0,1]
            if t.dtype != torch.float32 and t.dtype != torch.float64:
                t = t.float()
            # нормализация масштаба
            # если значения выглядят как 0..255 → приведём к 0..1
            maxv = float(t.max().item()) if t.numel() > 0 else 1.0
            if maxv > 1.5:
                t = t / 255.0
            return [img for img in t]  # список из (3,H,W)
        else:
            raise TypeError("batch должен быть Tensor, PIL.Image или списком PIL.Image")

    imgs = _to_tensor_list(batch)
    B = len(imgs)

    # ---- 2) Считаем HSV и метрики на устройстве device ----
    mu   = torch.empty(B, device=device, dtype=torch.float32)
    q95  = torch.empty(B, device=device, dtype=torch.float32)
    sat  = torch.empty(B, device=device, dtype=torch.float32)
    ratio= torch.empty(B, device=device, dtype=torch.float32)

    for i, img in enumerate(imgs):
        x = img.to(device)  # (3,H,W) в [0,1]
        hsv = TF.rgb_to_hsv(x.unsqueeze(0))  # (1,3,H,W)
        V   = hsv[0, 2]  # (H,W), яркость
        H_, W_ = V.shape

        # средняя яркость
        mu[i] = V.mean()

        # 95-й перцентиль яркости
        q95[i] = torch.quantile(V.reshape(-1), 0.95)

        # доля пикселей, близких к клипу (засвет)
        sat[i] = (V > 0.98).float().mean()

        # «горячее пятно вспышки» центр vs бордюр
        h1, h2 = int(H_ * (1/3)), int(H_ * (2/3))
        w1, w2 = int(W_ * (1/3)), int(W_ * (2/3))
        center = V[h1:h2, w1:w2]
        # бордюр = всё, кроме прямоугольника центра
        border_mask = torch.ones_like(V, dtype=torch.bool)
        border_mask[h1:h2, w1:w2] = False
        border = V[border_mask]
        eps = 1e-6
        ratio[i] = center.mean() / (border.mean() + eps)

    # ---- 3) Решение: flash / night / day ----
    is_flash = (sat > sat_thresh) | (ratio > ratio_thresh)
    # Ночь: очень тёмно и нет признаков «вспышки»
    is_night = (~is_flash) & (mu < mu_thresh) & (q95 < q95_thresh)
    labels = ["night" if bool(is_night[j].item()) else "day" for j in range(B)]

    out = {
        "labels": labels,
        "is_night": is_night,
        "metrics": {"mu": mu, "q95": q95, "sat": sat, "ratio": ratio}
    }
    if return_flash:
        out["is_flash"] = is_flash
    return out
