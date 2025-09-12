from typing import Literal, Union

import math
from adamp import AdamP
from torchvision.transforms.v2.functional import to_pil_image

import matplotlib.pyplot as plt
import pandas as pd
import torch
import asyncio
import tqdm
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForImageClassification

import data
import lib

from lib import predict_siglip

import ipywidgets as widgets
from IPython.display import display
import re
import fnmatch
import glob
import os
from pathlib import Path


# fnmatch.translate('siglip2_(*).py')
#%% md
# ## Model instantiation
#%%
restore_from_checkpoint: Union[int, bool] = True

model_id = "google/siglip2-base-patch16-256"  # FixRes вариант
model_preprocessor = AutoImageProcessor.from_pretrained(model_id)  # даст resize/normalize, mean/std/size

optimizer = None

# upcoming training epoch
epoch = 0

if restore_from_checkpoint == True or isinstance(restore_from_checkpoint, int) and not isinstance(restore_from_checkpoint, bool):
    if restore_from_checkpoint == True:
        epochs = lib.model_checkpoints(f'./models_siglip2_base_256/checkpoint_*.pth')

        if len(epochs) == 0:
            print('no models found')
            raise ValueError('No model found')

        checkpoint_num = epochs[ 0 ]
    else:
        checkpoint_num = restore_from_checkpoint

    print(f'Loading model from epoch { checkpoint_num }')

    checkpoint = torch.load(f'./models_siglip2_base_256/checkpoint_{ checkpoint_num }.pth', weights_only=False)

    model = checkpoint['model']
    optimizer = checkpoint['optimizer']

    epoch = model.epoch + 1
else:
    # Веса энкодера + НОВАЯ голова классификации (num_labels=2):
    model = AutoModelForImageClassification.from_pretrained(
        model_id,
        num_labels=len(data.species_labels),
        ignore_mismatched_sizes=True,  # создаст новую голову нужного размера
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    model.tracking_loss = []
    model.tracking_loss_val = []
    model.tracking_accuracy = []
    model.tracking_val_probs = []
    # the last epoch we finished training on
    model.epoch = None

tracking_loss = model.tracking_loss
tracking_loss_val = model.tracking_loss_val
tracking_accuracy = model.tracking_accuracy
tracking_val_probs = model.tracking_val_probs
#%% md
# ## Training
#%% md
# ### Data
#%%
train_ds = lib.ImageDatasetSigLip2(data.x_train, data.y_train, processor=model_preprocessor, learning=True)
val_ds   = lib.ImageDatasetSigLip2(data.x_eval, data.y_eval, processor=model_preprocessor, learning=False)

train_loader = DataLoader(train_ds, batch_size=192, shuffle=True, num_workers=6)
val_loader   = DataLoader(val_ds,   batch_size=128, shuffle=False, num_workers=6)
#%% md
# ### Optimizer
#%%
if optimizer is None:
    optimizer = AdamP([
        {'name': "encoder", "params": [],  "lr": 1e-4, "weight_decay": 0.05},
        {'name': "classifier", "params": [], "lr": 1e-3, "weight_decay": 0.01}
    ])
#%% md
# ### Freezing
#%%
unfreezing: Literal['classifier_only', 'classifier_and_encoder', 'all'] = 'classifier_and_encoder'
L = 6  # начните с 2–4; при достаточном VRAM можно 6–8

# C) Параметрические группы с «ступенчатым» LR: у головы LR выше, у энкодера ниже
head_params = []
enc_params  = []

if unfreezing == 'classifier_only':
    # 2) Заморозим всё, кроме головы (линейный пробинг)
    for name, p in model.named_parameters():
        p.requires_grad = "classifier" in name  # у HF-классификаторов голова обычно называется "classifier"

        if "classifier" in name:
            head_params.append(p)

elif unfreezing == 'classifier_and_encoder':
    # A) Сначала всё заморозим
    for p in model.parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        if "classifier" in name:
            p.requires_grad = True  # голова остаётся обучаемой
            head_params.append(p)

    # B) Разморозим последние L блоков визуального энкодера
    layers = model.vision_model.encoder.layers   # ModuleList
    for block in layers[-L:]:
        for p in block.parameters():
            p.requires_grad = True
            enc_params.append(p)
elif unfreezing == 'all':
    for p in model.parameters():
        p.requires_grad = True
else:
    raise ValueError(f"Unknown unfreezing mode: {unfreezing}")
#%%
optimizer.param_groups = []

optimizer.add_param_group({'name': "encoder", "params": enc_params,  "lr": 1e-5, "weight_decay": 0.01})
optimizer.add_param_group({'name': "classifier", "params": head_params, "lr": 2e-5, "weight_decay": 0.01})
#%% md
# ### Loss (possibly with weights)
#%%
# criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

criterion = lib.sce_loss
#%% md
# ### Cutmix + mixup
#%%
from torchvision.transforms import v2

use_cutmix_mixup = True

cutmix = v2.CutMix(alpha=0.3, num_classes=len(data.species_labels))
mixup = v2.MixUp(alpha=0.3, num_classes=len(data.species_labels))
cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
#%%
# steps_per_epoch = len(train_loader)
#
# def scheduler(step):
#

#%% md
# ### Loop
#%%
stop_button = widgets.Button(description="Stop")
stop_flag = { 'value' : False }

def on_click(b):
    stop_flag['value'] = True

stop_button.on_click(on_click)
display(stop_button)

num_epochs = 30

for cur_epoch in range(epoch, epoch + num_epochs):
    await asyncio.sleep(0)

    if stop_flag['value'] == True:
        break

    print(f"Starting epoch {cur_epoch}")

    model.train()

    loss_acc = 0
    count = 0

    for idx, batch in tqdm.tqdm(enumerate(train_loader), total=len(train_loader), desc='Training'):
        optimizer.zero_grad(set_to_none=True)

        images, labels = batch["pixel_values"].to(torch.device("cuda")), batch["labels"].to(torch.device("cuda"))

        if use_cutmix_mixup:
            images, labels = cutmix_or_mixup(images, labels)

        refined_labels_df = model(images)              # logits: (B, 2)
        loss = criterion(refined_labels_df.logits, labels)

        c = batch['pixel_values'].size(0)
        loss_acc += loss.item() * c
        count += c

        loss.backward()
        optimizer.step()

    tracking_loss.append(loss_acc / count)

    # валидация
    model.eval()

    probs, loss_acc = predict_siglip(
        model, val_loader, accumulate_probs=True, accumulate_loss=True, desc='Validation', columns=data.species_labels, criterion=criterion
    )
    tracking_val_probs.append(probs)
    tracking_loss_val.append(loss_acc)

    eval_predictions = probs.idxmax(axis=1)
    eval_true = data.y_eval.idxmax(axis=1)
    correct = (eval_predictions == eval_true).sum()
    accuracy = correct / len(eval_predictions)
    tracking_accuracy.append(accuracy.item())

    model.epoch = cur_epoch
    lib.save_model(model, optimizer, f"./models_siglip2_base_256/checkpoint_{str(cur_epoch).rjust(2, "0")}.pth")

    epoch = cur_epoch + 1

#%% md
# ## Training progress
#%%
pd.DataFrame({'tracking_loss' : tracking_loss, 'tracking_loss_val' : tracking_loss_val, 'tracking_accuracy' : tracking_accuracy }, index=range(len(tracking_accuracy)))
#%%
fig, ax = plt.subplots(figsize=(15, 5))

epochs_train = list(range(len(tracking_loss)))
epochs_val = list(range(len(tracking_loss_val)))

line1, = ax.plot(epochs_train, tracking_loss, label="Train loss")
line2, = ax.plot(epochs_val, tracking_loss_val, label="Validation loss")

ax.set_xlabel("Epoch (index)")
ax.set_ylabel("Loss")
ax.legend(loc="best", handles=[line1, line2])

ax.set_xticks(epochs_train)

ax.grid(True)
#%%
fig, ax = plt.subplots(figsize=(15, 5))

epochs_accuracy = list(range(len(tracking_accuracy)))

line1, = ax.plot(epochs_accuracy, tracking_accuracy, label="Accuracy", color="red")
ax.set_ylabel("Accuracy")

ax.legend(loc="best", handles=[line1])

ax.set_xticks(epochs_train)

ax.grid(True)
#%% md
# ## Validation
#%%
## search for optimal temperature
#%%
# temp_acc = {}
#%%
# for key in sorted(temp_acc.keys()):
#     print(f'T={key:.5f}: {temp_acc[key]:.5f}')
#%%
# import numpy as np
#
# for t in np.arange(0.785, 0.82, 0.0125):
#     _, loss = lib.predict_siglip(model, val_loader, accumulate_loss=True, accumulate_probs=False, criterion=criterion, T=t, desc='Searching', columns=data.species_labels)
#
#     print(f"T={t:.5f}: {loss:.4f}")
#
#     temp_acc[t] = loss
#%%
eval_preds_df = tracking_val_probs[-1]

# eval_preds_df_ten_crop = lib.predict_siglip_ten_crop(model, val_loader, T=1, desc='Predicting', columns=data.species_labels)
#%%
eval_preds_df.head()
#%%
# eval_preds_df_ten_crop.head()
#%%
print("True labels (training):")
data.y_train.idxmax(axis=1).value_counts(normalize=True)
#%%
print("Predicted labels (eval):")
eval_preds_df.idxmax(axis=1).value_counts(normalize=True)
#%%
print("True labels (eval):")
data.y_eval.idxmax(axis=1).value_counts(normalize=True)
#%%
eval_predictions = eval_preds_df.idxmax(axis=1)
# eval_predictions_ten_crop = eval_preds_df_ten_crop.idxmax(axis=1)
eval_true = data.y_eval.idxmax(axis=1)
#%%
# (eval_predictions_ten_crop != eval_predictions).sum()
#%%
print(f'Accuracy plain: { (eval_predictions == eval_true).mean() }')
# print(f'Accuracy ten crop: { (eval_predictions_ten_crop == eval_true).mean() }')
#%% md
# ### Predictions vs actual
#%%
eval_preds = eval_preds_df.copy()

eval_preds[ 'cls' ] = eval_preds_df.idxmax(axis=1)
eval_preds[ 'cls_true' ] = data.y_eval.idxmax(axis=1)

# eval_preds[(eval_preds[ 'cls' ] == 'blank') & (eval_preds[ 'cls_true' ] == 'leopard')]
#%%
data.species_labels
#%%
import math
from itertools import zip_longest
from PIL import Image
%matplotlib inline
# %matplotlib notebook
# %matplotlib widget
from torchvision.transforms.functional import to_pil_image

random_state = 41111

# rows = eval_preds[(eval_preds[ 'cls' ] == 'blank') & (eval_preds[ 'cls_true' ] == 'leopard')]
rows = eval_preds[(eval_preds[ 'cls' ] == 'blank') & (eval_preds[ 'cls_true' ] == 'leopard')]

rows = rows.sample(frac=0.2, random_state=random_state)

n_cols = 3
n_rows = math.ceil(len(rows) / n_cols)

fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(18, 35))

# fig.canvas.layout.width = '100%'   # ширину займёт вся ячейка
# Высоту ipywidgets не умеют «auto», укажи соотношение/высоту:
# fig.set_figheight(fig.get_figwidth() * 0.6)

invert = v2.RandomInvert(p=1)

# iterate through each species
print(f'Total rows: {len(rows)}')

clahe = lib.LabCLAHE()

for row, ax in zip_longest(list(rows.iterrows()), axes.flatten()):
    if row is None:
        if ax is not None:
            ax.remove()
        continue
    if ax is None:
        break
    img = Image.open('data/train_features/' + row[0] + '.jpg')
    ax.imshow(to_pil_image(clahe(clahe((img)))))
    ax.set_title(f"{row[1].name} ")

fig.tight_layout()
#%% md
# ### Confusion matrix
#%%
from sklearn.metrics import ConfusionMatrixDisplay

fig, ax = plt.subplots(figsize=(10, 10))
cm = ConfusionMatrixDisplay.from_predictions(
    data.y_eval.idxmax(axis=1),
    eval_preds_df.idxmax(axis=1),
    ax=ax,
    xticks_rotation=30,
    colorbar=True,
)
#%% md
# ## Create submission
#%%
test_dataset = lib.ImageDatasetSigLip2(data.test_features, processor=model_preprocessor, learning=False)
test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=6)
#%%
submission_df, _ = predict_siglip(model, test_dataloader, T=1, columns=data.species_labels)
#%%
submission_format = pd.read_csv("data/submission_format.csv", index_col="id")

assert all(submission_df.index == submission_format.index)
assert all(submission_df.columns == submission_format.columns)
#%%
submission_df.to_csv("submission.csv")