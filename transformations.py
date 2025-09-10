import torch
from torchvision.transforms import v2, InterpolationMode

from lib import LabCLAHE

resnet50_process = v2.Compose([
    v2.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
    v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


resnet50_transform_train = v2.Compose([
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
])

resnet50_transform_inference = v2.Compose([
    LabCLAHE(),
    LabCLAHE(),
    v2.ToPILImage(),
])

