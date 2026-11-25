# ==============================================================================
# AgriBloom - EfficientNet-B4 Training on Google Colab
# Copy each section (marked by # === CELL X ===) into separate Colab cells
# ==============================================================================

# === CELL 1: Mount Google Drive & Setup =======================================
# (Upload PlantVillage_Cleaned.zip to your Google Drive root first)

from google.colab import drive
drive.mount('/content/drive')

# === CELL 2: Install Dependencies =============================================

!pip install efficientnet-pytorch torch torchvision matplotlib numpy scikit-learn tqdm -q

# === CELL 3: Unzip Dataset ====================================================

import os
import zipfile

# Change this path if your zip is in a subfolder
ZIP_PATH = "/content/drive/MyDrive/PlantVillage_Cleaned.zip"
EXTRACT_DIR = "/content/dataset"

if not os.path.exists(EXTRACT_DIR):
    print("Extracting dataset...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        z.extractall(EXTRACT_DIR)
    print("Done!")
else:
    print("Dataset already extracted.")

# Check what's inside
DATA_DIR = os.path.join(EXTRACT_DIR, "PlantVillage_Cleaned")
if not os.path.exists(DATA_DIR):
    # Maybe the zip extracts directly without a parent folder
    DATA_DIR = EXTRACT_DIR

classes = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
print(f"\nFound {len(classes)} classes:")
for c in classes:
    count = len(os.listdir(os.path.join(DATA_DIR, c)))
    print(f"  {c}: {count} images")

# === CELL 4: Check GPU =======================================================

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# === CELL 5: Training Configuration ==========================================

IMAGE_SIZE = 224
BATCH_SIZE = 32        # Reduce to 16 if you get OOM errors on Colab
EPOCHS = 30            # 30 is usually enough for PlantVillage
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
EARLY_STOPPING_PATIENCE = 7
NUM_WORKERS = 2        # Colab works best with 2
MIXED_PRECISION = True

OUTPUT_DIR = "/content/drive/MyDrive/AgriBloom_Model"  # Saves to Drive!
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Config ready!")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Output: {OUTPUT_DIR}")

# === CELL 6: Data Transforms & Loaders =======================================

import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

def get_transforms(mode="train"):
    if mode == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

# Load dataset
full_dataset = datasets.ImageFolder(DATA_DIR, transform=get_transforms("train"))
num_classes = len(full_dataset.classes)
class_names = full_dataset.classes

print(f"Total images: {len(full_dataset)}")
print(f"Number of classes: {num_classes}")

# Train/Val split (85/15)
val_size = int(len(full_dataset) * 0.15)
train_size = len(full_dataset) - val_size

train_dataset, val_dataset = torch.utils.data.random_split(
    full_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

# Val transforms
val_dataset.dataset = datasets.ImageFolder(DATA_DIR, transform=get_transforms("val"))

# Weighted sampler for class imbalance
targets = [full_dataset.targets[i] for i in train_dataset.indices]
class_counts = np.bincount(targets, minlength=num_classes)
class_weights = 1.0 / (class_counts + 1)
sample_weights = [class_weights[t] for t in targets]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True,
)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
    num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True,
)

print(f"Train: {train_size} images, Val: {val_size} images")
print(f"Class counts - min: {class_counts.min()}, max: {class_counts.max()}, mean: {class_counts.mean():.0f}")

# === CELL 7: Build Model =====================================================

import torch.nn as nn
from efficientnet_pytorch import EfficientNet

model = EfficientNet.from_pretrained("efficientnet-b4")
print("Loaded pretrained EfficientNet-B4")

# Replace classifier head
in_features = model._fc.in_features
model._fc = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(in_features, 512),
    nn.ReLU(inplace=True),
    nn.Dropout(0.2),
    nn.Linear(512, num_classes),
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Model on: {device}")
print(f"Configured for {num_classes} classes")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params: {total_params:,}")
print(f"Trainable params: {trainable_params:,}")

# === CELL 8: Train! ==========================================================

import time
import json
from torch.cuda.amp import GradScaler, autocast

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
scaler = GradScaler(enabled=MIXED_PRECISION)

best_val_acc = 0.0
patience_counter = 0
train_losses, val_losses, val_accs, train_accs = [], [], [], []

print(f"Starting training: {EPOCHS} epochs")
print("=" * 70)

for epoch in range(EPOCHS):
    epoch_start = time.time()

    # --- Train ---
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=MIXED_PRECISION):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / total
    train_acc = 100.0 * correct / total
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # --- Validate ---
    model.eval()
    v_loss, v_correct, v_total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast(enabled=MIXED_PRECISION):
                outputs = model(images)
                loss = criterion(outputs, labels)

            v_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            v_total += labels.size(0)
            v_correct += predicted.eq(labels).sum().item()

    val_loss = v_loss / max(v_total, 1)
    val_acc = 100.0 * v_correct / max(v_total, 1)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    scheduler.step()
    epoch_time = time.time() - epoch_start
    lr_now = scheduler.get_last_lr()[0]

    print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}% | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.1f}% | "
          f"LR: {lr_now:.6f} | Time: {epoch_time:.0f}s")

    # Save best
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": val_acc,
            "val_loss": val_loss,
            "class_names": class_names,
            "num_classes": num_classes,
        }, os.path.join(OUTPUT_DIR, "best_model.pth"))
        print(f"  ✅ New best! Val Acc: {val_acc:.1f}%")
    else:
        patience_counter += 1
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"  ⏹ Early stopping at epoch {epoch+1}")
            break

print("=" * 70)
print(f"Training complete! Best Val Acc: {best_val_acc:.1f}%")

# Save class labels
class_labels = {}
for idx, name in enumerate(class_names):
    parts = name.lower().split("___")
    crop = parts[0] if len(parts) >= 2 else name.split("_")[0]
    disease = parts[1] if len(parts) >= 2 else name
    class_labels[str(idx)] = {
        "class_name": name,
        "crop": crop,
        "disease": disease.replace("_", " ").title(),
    }

with open(os.path.join(OUTPUT_DIR, "class_labels.json"), "w") as f:
    json.dump(class_labels, f, indent=2)

print(f"Class labels saved to {OUTPUT_DIR}/class_labels.json")

# === CELL 9: Plot Training Curves ============================================

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(train_losses, label="Train Loss", color="#e74c3c")
axes[0].plot(val_losses, label="Val Loss", color="#3498db")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Training & Validation Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(train_accs, label="Train Acc", color="#e67e22")
axes[1].plot(val_accs, label="Val Acc", color="#2ecc71")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy (%)")
axes[1].set_title(f"Accuracy (Best Val: {best_val_acc:.1f}%)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot([scheduler.get_last_lr()[0]] * len(train_losses), color="#9b59b6")
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("Learning Rate")
axes[2].set_title("Learning Rate Schedule")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"), dpi=150, bbox_inches="tight")
plt.show()
print("Training curves saved!")

# === CELL 10: Export to ONNX (Optional) ======================================

# Uncomment to export:
# ckpt = torch.load(os.path.join(OUTPUT_DIR, "best_model.pth"), map_location="cpu")
# model_export = EfficientNet.from_name("efficientnet-b4")
# model_export._fc = nn.Sequential(
#     nn.Dropout(0.4),
#     nn.Linear(model_export._fc.in_features, 512),
#     nn.ReLU(inplace=True),
#     nn.Dropout(0.2),
#     nn.Linear(512, num_classes),
# )
# model_export.load_state_dict(ckpt["model_state_dict"])
# model_export.eval()
# dummy = torch.randn(1, 3, 224, 224)
# torch.onnx.export(
#     model_export, dummy, os.path.join(OUTPUT_DIR, "model.onnx"),
#     input_names=["image"], output_names=["prediction"],
#     dynamic_axes={"image": {0: "batch"}, "prediction": {0: "batch"}},
#     opset_version=14,
# )
# print("ONNX export done!")

# === CELL 11: Download Model from Drive ======================================
# After training, your model is saved in Google Drive at:
# /MyDrive/AgriBloom_Model/
#   ├── best_model.pth          (PyTorch checkpoint)
#   ├── class_labels.json       (label mapping)
#   └── training_curves.png     (loss/accuracy plots)
#
# You can download these from Drive or use:
# from google.colab import files
# files.download(os.path.join(OUTPUT_DIR, "best_model.pth"))
