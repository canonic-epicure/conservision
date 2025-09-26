#%%
# RTX 5090 Optimized Training Script
# Implements large-scale vision model training with cross-validation
print("RTX 5090 Training Mode Activated!")

# Automatic dependency management
import subprocess
import sys

def install_package(package):
    try:
        __import__(package)
        print(f"{package} installed")
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"{package} installation complete")

# Check and install required dependencies
required_packages = ['pandas', 'numpy', 'scikit-learn']
print("Checking and installing required packages...")
for package in required_packages:
    install_package(package)

# Install deep learning frameworks
try:
    import fastai
    print("fastai installed")
except ImportError:
    print("Installing fastai...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "fastai"])
    print("fastai installation complete")

try:
    import timm
    print("timm installed")
except ImportError:
    print("Installing timm...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "timm"])
    print("timm installation complete")

# Import all libraries
import pandas as pd
import numpy as np
from pathlib import Path
import gc
import torch
from fastai.vision.all import *
from sklearn.model_selection import StratifiedGroupKFold

print("All libraries imported successfully!")

# Configuration class
class CFG:
    # File paths
    BASE_PATH = Path('./data')
    TRAIN_FEATURES_PATH = BASE_PATH / 'train_features.csv'
    TRAIN_LABELS_PATH = BASE_PATH / 'train_labels.csv'
    TEST_FEATURES_PATH = BASE_PATH / 'test_features.csv'
    
    # RTX 5090 optimized settings
    # MODEL_ARCHITECTURE = 'convnext_large_in22k'  # Upgraded to larger model
    MODEL_ARCHITECTURE = 'timm/eva02_large_patch14_448.mim_m38m_ft_in22k_in1k'  # Upgraded to larger model

    IMAGE_SIZE = 448      # Higher resolution
    BATCH_SIZE = 16       # Optimized for RTX 5090
    N_FOLDS = 5
    EPOCHS = 15           # Moderate increase in training epochs
    
    # RTX 5090 optimization
    NUM_WORKERS = 12      # Optimized threading
    PIN_MEMORY = True
    PREFETCH_FACTOR = 4
    
    # Competition settings
    TARGET_COL = 'label'
    SEED = 42
    BASE_LR = 1e-3

print(f"RTX 5090 Configuration:")
print(f"   Model: {CFG.MODEL_ARCHITECTURE}")
print(f"   Resolution: {CFG.IMAGE_SIZE}x{CFG.IMAGE_SIZE}")
print(f"   Batch Size: {CFG.BATCH_SIZE}")
print(f"   Training Epochs: {CFG.EPOCHS}")

# RTX 5090 CUDA optimization settings
if torch.cuda.is_available():
    print(f"Detected: {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    print("RTX 5090 optimizations enabled")
    try:
        test_tensor = torch.randn(100, 100).cuda()
        result = torch.mm(test_tensor, test_tensor)
        print("CUDA test passed!")
        del test_tensor, result
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"CUDA test failed: {e}")
        print("Please check PyTorch CUDA installation!")
else:
    print("CUDA unavailable, using CPU mode")

# Set random seed for reproducibility
set_seed(CFG.SEED, reproducible=True)

# Data augmentation transforms
def get_transforms():
    return aug_transforms(
        size=CFG.IMAGE_SIZE,
        min_scale=0.7,
        max_rotate=20,
        max_lighting=0.4,
        max_warp=0.25,
        p_affine=0.9,
        p_lighting=0.9
    )

# Data preparation
print("\nPreparing data...")

# Check if data files exist
required_files = [CFG.TRAIN_FEATURES_PATH, CFG.TRAIN_LABELS_PATH, CFG.TEST_FEATURES_PATH]
for file_path in required_files:
    if not file_path.exists():
        print(f"File not found: {file_path}")
        print("Please ensure the following files are in the current directory:")
        print("  - train_features.csv")
        print("  - train_labels.csv")
        print("  - test_features.csv")
        raise FileNotFoundError(f"Missing required file: {file_path}")

train_features_df = pd.read_csv(CFG.TRAIN_FEATURES_PATH)
train_labels_df = pd.read_csv(CFG.TRAIN_LABELS_PATH)
test_features_df = pd.read_csv(CFG.TEST_FEATURES_PATH)

# Process labels - convert one-hot to categorical
train_labels_df['label'] = train_labels_df.iloc[:, 1:].idxmax(axis=1)
df = train_features_df.merge(train_labels_df[['id', 'label']], on='id')

# Create image paths
df['image_path'] = df['filepath'].apply(lambda x: CFG.BASE_PATH / x)
test_features_df['image_path'] = test_features_df['filepath'].apply(lambda x: CFG.BASE_PATH / x)

print(f"Data loaded successfully!")
print(f"   Training images: {len(df)}")
print(f"   Test images: {len(test_features_df)}")
print(f"   Number of classes: {df['label'].nunique()}")

# Check class distribution
print("\nClass distribution:")
print(df['label'].value_counts())

# Cross-validation setup
print("\nSetting up StratifiedGroupKFold...")
df['fold'] = -1
splitter = StratifiedGroupKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)

# Assign fold numbers
for fold, (train_idx, val_idx) in enumerate(splitter.split(df, df['label'], groups=df['site'])):
    df.loc[val_idx, 'fold'] = fold

print("Fold distribution:")
print(df.fold.value_counts())

# Training loop
print(f"\nStarting RTX 5090 Training - {CFG.N_FOLDS} Fold Cross Validation")

all_preds = []
all_oof_preds = []
fold_scores = []
vocab = None

for fold in range(CFG.N_FOLDS):
    print(f"\n{'='*50}")
    print(f"Fold {fold} - RTX 5090 Training")
    print(f"{'='*50}")

    # Create fold splitter function
    def get_splitter(fold_num):
        def _inner(o):
            val_mask = o['fold'] == fold_num
            train_mask = o['fold'] != fold_num
            return o.index[train_mask], o.index[val_mask]
        return _inner

    # DataBlock configuration
    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_x=ColReader('image_path'),
        get_y=ColReader(CFG.TARGET_COL),
        splitter=get_splitter(fold),
        item_tfms=Resize(CFG.IMAGE_SIZE, method=ResizeMethod.Pad, pad_mode=PadMode.Zeros),
        batch_tfms=[*get_transforms(), Normalize.from_stats(*imagenet_stats)]
    )
    
    print(f"Creating DataLoaders (batch size: {CFG.BATCH_SIZE})...")
    
    # Create DataLoaders with RTX 5090 optimizations
    if torch.cuda.is_available():
        dls = dblock.dataloaders(
            df,
            bs=CFG.BATCH_SIZE,
            num_workers=CFG.NUM_WORKERS,
            pin_memory=CFG.PIN_MEMORY,
            prefetch_factor=CFG.PREFETCH_FACTOR
        )
    else:
        dls = dblock.dataloaders(
            df,
            bs=16,
            num_workers=4
        )

    # Store vocabulary from first fold
    if vocab is None:
        vocab = dls.vocab
        print(f"Class vocabulary: {list(vocab)}")

    print(f"Creating {CFG.MODEL_ARCHITECTURE} model...")

    # Setup callbacks for training
    cbs = [
        EarlyStoppingCallback(monitor='valid_loss', patience=3),
        SaveModelCallback(monitor='valid_loss', fname=f'best_model_fold_{fold}')
    ]

    # Create learner with mixed precision
    learn = vision_learner(
        dls,
        CFG.MODEL_ARCHITECTURE,
        metrics=[error_rate, accuracy],
        cbs=cbs
    ).to_fp16()

    learn.load(f'best_model_fold_{fold}', device='cuda')

    # print(f"Starting training for {CFG.EPOCHS} epochs...")
    #
    # # Find optimal learning rate
    # try:
    #     lr_min, lr_steep = learn.lr_find()
    #     print(f"Suggested learning rate: {lr_steep:.2e}")
    #     final_lr = lr_steep
    # except Exception as e:
    #     print("Using default learning rate", e)
    #     final_lr = CFG.BASE_LR
    #
    # # Train the model
    # learn.fit_one_cycle(CFG.EPOCHS, lr_max=final_lr)

    # Record validation scores
    val_results = learn.validate()
    val_loss = float(val_results[0])
    val_acc = float(val_results[2])  # accuracy is the 2nd metric
    fold_scores.append({'fold': fold, 'val_loss': val_loss, 'val_acc': val_acc})
    print(f"Fold {fold} validation results: Loss={val_loss:.4f}, Acc={val_acc:.4f}")

    # Generate test predictions
    print("Generating predictions...")
    test_dl = dls.test_dl(test_features_df)
    preds, _ = learn.get_preds(dl=test_dl)
    all_preds.append(preds)

    # Get out-of-fold predictions
    val_dl = dls.valid
    oof_preds, _ = learn.get_preds(dl=val_dl)
    all_oof_preds.append(oof_preds)

    # Memory cleanup
    print("Memory cleanup...")
    del learn, dls, test_dl, val_dl
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print(f"\n{'='*50}")
print("RTX 5090 Training Complete!")
print(f"{'='*50}")

# Display fold results
print("\nCross-validation results:")
for score in fold_scores:
    print(f"Fold {score['fold']}: Loss={score['val_loss']:.4f}, Acc={score['val_acc']:.4f}")

avg_loss = np.mean([s['val_loss'] for s in fold_scores])
avg_acc = np.mean([s['val_acc'] for s in fold_scores])
print(f"\nAverage performance: Loss={avg_loss:.4f}, Acc={avg_acc:.4f}")

# Ensemble strategy - weighted by validation performance
print(f"\nExecuting ensemble strategy...")

val_accs = [s['val_acc'] for s in fold_scores]
val_loss = [s['val_loss'] for s in fold_scores]

weights_acc = torch.softmax(torch.tensor(val_accs) * 5, dim=0)
print(f"Fold weights by acc: {[f'{w:.3f}' for w in weights_acc.tolist()]}")

# Weighted ensemble predictions
ensemble_preds = sum(w * pred for w, pred in zip(weights_acc, all_preds))

weights_loss = torch.softmax((1 / torch.tensor(val_loss)) * 5, dim=0)
print(f"Fold weights by loss: {[f'{w:.3f}' for w in weights_loss.tolist()]}")

# Weighted ensemble predictions
ensemble_preds_loss = sum(w * pred for w, pred in zip(weights_loss, all_preds))

# Create submission file
print("\nCreating submission file by accuracy...")

submission_df = pd.DataFrame(columns=['id'] + list(vocab))
submission_df['id'] = test_features_df['id']
submission_df[list(vocab)] = ensemble_preds.numpy()

# Save submission
submission_df.to_csv('submission_fast_ai_k_fold_accuracy.csv', index=False)


print("\nCreating submission file by loss...")

submission_df = pd.DataFrame(columns=['id'] + list(vocab))
submission_df['id'] = test_features_df['id']
submission_df[list(vocab)] = ensemble_preds_loss.numpy()

# Save submission
submission_df.to_csv('submission_fast_ai_k_fold_loss.csv', index=False)


print(f"\n{'='*50}")
print("RTX 5090 Submission File Created!")
print(f"{'='*50}")
print("=== Configuration Summary ===")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"Model: {CFG.MODEL_ARCHITECTURE}")
print(f"Resolution: {CFG.IMAGE_SIZE}x{CFG.IMAGE_SIZE}")
print(f"Batch Size: {CFG.BATCH_SIZE}")
print(f"Total Training Epochs: {CFG.N_FOLDS * CFG.EPOCHS}")
print(f"Average Validation Accuracy: {avg_acc:.4f}")
print(f"Mixed Precision Training: {'Yes' if torch.cuda.is_available() else 'No'}")
print("")
print("Submission file: rtx5090_submission.csv")
print("Ready to dominate the competition!")