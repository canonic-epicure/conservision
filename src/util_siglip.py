from typing import Literal

from transformers import Siglip2Model


def freeze(model: Siglip2Model, unfreezing: Literal['classifier_only', 'classifier_and_encoder'] = 'classifier_and_encoder', L: int = 4):
    # A) Сначала всё заморозим
    for p in model.parameters():
        p.requires_grad = False

    head_params = []
    enc_params  = []

    if unfreezing == 'classifier_only' or unfreezing == 'classifier_and_encoder':
        # unfreeze classifier if needed
        for name, p in model.named_parameters():
            if "classifier" in name:
                p.requires_grad = True  # голова остаётся обучаемой
                head_params.append(p)

    # unfreeze encoder if needed
    if unfreezing == 'classifier_and_encoder':
        # B) Разморозим последние L блоков визуального энкодера
        layers = model.vision_model.encoder.layers   # ModuleList
        for block in layers[-L:]:
            for p in block.parameters():
                p.requires_grad = True
                enc_params.append(p)

    return head_params, enc_params