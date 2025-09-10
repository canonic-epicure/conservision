from typing import Literal

from transformers import Siglip2Model


def set_freezing(
    model: Siglip2Model,
    optimizer,
    mode: Literal['classifier_only', 'classifier_and_encoder'] = 'classifier_and_encoder',
    L: int = 4
):
    # A) Сначала всё заморозим
    for p in model.parameters():
        p.requires_grad = False

    head_params = []
    enc_params  = []

    if mode == 'classifier_only' or mode == 'classifier_and_encoder':
        # unfreeze classifier if needed
        for name, p in model.named_parameters():
            if "classifier" in name:
                p.requires_grad = True  # голова остаётся обучаемой
                head_params.append(p)

    # unfreeze encoder if needed
    if mode == 'classifier_and_encoder':
        # B) Разморозим последние L блоков визуального энкодера
        layers = model.vision_model.encoder.layers   # ModuleList
        for block in layers[-L:]:
            for p in block.parameters():
                p.requires_grad = True
                enc_params.append(p)

    optimizer.param_groups = []

    optimizer.add_param_group({'name': "classifier", "params": head_params, "lr": 1e-3, "weight_decay": 0.01})
    optimizer.add_param_group({'name': "encoder", "params": enc_params, "lr": 1e-4, "weight_decay": 0.01})
