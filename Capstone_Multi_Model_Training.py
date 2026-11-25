# ==============================================================================
# Capstone Project — Multi-Model Crop Disease Classification
# Train 3 models on PlantVillage (38 classes) and compare results
#
# Models: EfficientNet-B4, ResNet-50, ViT-Base-Patch16-224
# Dataset: PlantVillage Cleaned (38 classes, ~54,284 images)
#
# HOW TO USE:
#   1. Upload PlantVillage_Cleaned.zip to Google Drive root
#   2. Open Google Colab (colab.research.google.com)
#   3. Set Runtime > Change Runtime Type > GPU (T4)
#   4. Copy each CELL section into separate Colab cells
#   5. Run cells in order
# ==============================================================================


# === CELL 1: Mount Google Drive & Install Dependencies ========================

# from google.colab import drive
# drive.mount('/content/drive')

# !pip install efficientnet-pytorch torchvision matplotlib scikit-learn tqdm timm -q

# import torch
# print(f"PyTorch: {torch.__version__}")
# print(f"CUDA: {torch.cuda.is_available()}")
# if torch.cuda.is_available():
#     print(f"GPU: {torch.cuda.get_device_name(0)}")


# === CELL 2: Unzip Dataset ===================================================

# import os, zipfile
#
# ZIP_PATH = "/content/drive/MyDrive/PlantVillage_Cleaned.zip"
# EXTRACT_DIR = "/content/dataset"
#
# if not os.path.exists(EXTRACT_DIR):
#     print("Extracting dataset...")
#     with zipfile.ZipFile(ZIP_PATH, 'r') as z:
#         z.extractall(EXTRACT_DIR)
#     print("Done!")
#
# DATA_DIR = os.path.join(EXTRACT_DIR, "PlantVillage_Cleaned")
# if not os.path.exists(DATA_DIR):
#     DATA_DIR = EXTRACT_DIR
#
# classes = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
# print(f"\nFound {len(classes)} classes:")
# total = 0
# for c in classes:
#     count = len(os.listdir(os.path.join(DATA_DIR, c)))
#     total += count
#     print(f"  {c}: {count}")
# print(f"\nTotal images: {total}")


# === CELL 3: Configuration & Data Loading =====================================

import os
import json
import time
import copy
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from torchvision import datasets, transforms, models

# ── CONFIG ───────────────────────────────────────────────────────────────────
# Change DATA_DIR to your local path if not on Colab
DATA_DIR = "/content/dataset/PlantVillage_Cleaned"
# DATA_DIR = r"C:\Users\Admin\Downloads\capstone project\PlantVillage_Cleaned"  # Uncomment for local

OUTPUT_BASE = "/content/drive/MyDrive/Capstone_Results"
# OUTPUT_BASE = r"C:\Users\Admin\Downloads\capstone project\training_outputs"  # Uncomment for local

IMAGE_SIZE = 224
BATCH_SIZE = 32        # Reduce to 16 if OOM
EPOCHS = 25            # Enough for PlantVillage with pretrained models
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
EARLY_STOPPING_PATIENCE = 7
NUM_WORKERS = 2        # 2 for Colab, 4 for local
MIXED_PRECISION = True
VAL_SPLIT = 0.15
TEST_SPLIT = 0.10

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ── DATA TRANSFORMS ─────────────────────────────────────────────────────────
train_transforms = transforms.Compose([
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

val_test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── LOAD & SPLIT DATASET ────────────────────────────────────────────────────
print(f"\nLoading dataset from: {DATA_DIR}")
full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transforms)
num_classes = len(full_dataset.classes)
class_names = full_dataset.classes
total_images = len(full_dataset)

print(f"Total images: {total_images}")
print(f"Number of classes: {num_classes}")

# Split: 75% train, 15% val, 10% test
test_size = int(total_images * TEST_SPLIT)
val_size = int(total_images * VAL_SPLIT)
train_size = total_images - val_size - test_size

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

# Apply proper transforms to val and test sets
val_dataset_proper = datasets.ImageFolder(DATA_DIR, transform=val_test_transforms)
test_dataset_proper = datasets.ImageFolder(DATA_DIR, transform=val_test_transforms)

# Weighted sampler for class imbalance
train_targets = [full_dataset.targets[i] for i in train_dataset.indices]
class_counts = np.bincount(train_targets, minlength=num_classes)
class_weights = 1.0 / (class_counts + 1)
sample_weights = [class_weights[t] for t in train_targets]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)

val_loader = DataLoader(
    torch.utils.data.Subset(val_dataset_proper, val_dataset.indices),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
)

test_loader = DataLoader(
    torch.utils.data.Subset(test_dataset_proper, test_dataset.indices),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
)

print(f"\nSplit: Train={train_size}, Val={val_size}, Test={test_size}")
print(f"Class distribution: min={class_counts.min()}, max={class_counts.max()}, mean={class_counts.mean():.0f}")
print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")


# === CELL 4: Model Builders ===================================================

def build_efficientnet_b4(num_classes):
    """EfficientNet-B4 with custom classification head."""
    from efficientnet_pytorch import EfficientNet
    model = EfficientNet.from_pretrained("efficientnet-b4")
    in_features = model._fc.in_features
    model._fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes),
    )
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"EfficientNet-B4: {total_params:,} trainable params")
    return model


def build_resnet50(num_classes):
    """ResNet-50 with custom classification head."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes),
    )
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ResNet-50: {total_params:,} trainable params")
    return model


def build_vit_base(num_classes):
    """Vision Transformer (ViT-Base-Patch16-224) with custom head."""
    try:
        import timm
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"ViT-Base-Patch16-224: {total_params:,} trainable params")
        return model
    except ImportError:
        # Fallback: use torchvision ViT
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        model.heads.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.heads.head.in_features, num_classes),
        )
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"ViT-B-16 (torchvision): {total_params:,} trainable params")
        return model


print("Model builders ready!")
print(f"Will train for {num_classes} classes")


# === CELL 5: Training Engine ==================================================

def train_one_model(model_name, model, train_loader, val_loader, output_dir,
                    epochs=EPOCHS, lr=LEARNING_RATE):
    """
    Train a single model and save all results.
    Returns: history dict with train_loss, val_loss, val_acc, best_val_acc
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    scaler = torch.amp.GradScaler('cuda', enabled=(MIXED_PRECISION and DEVICE.type == 'cuda'))

    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
    }

    print(f"\n{'='*70}")
    print(f"  TRAINING: {model_name}")
    print(f"  Epochs: {epochs}, LR: {lr}, Batch: {BATCH_SIZE}")
    print(f"{'='*70}")

    total_start = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        # ── Train Phase ──────────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=(MIXED_PRECISION and DEVICE.type == 'cuda')):
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

            if (batch_idx + 1) % 100 == 0:
                print(f"  [{model_name}] Epoch {epoch+1}/{epochs} | "
                      f"Batch {batch_idx+1}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f} | Acc: {100.*correct/total:.1f}%")

        train_loss = running_loss / total
        train_acc = 100.0 * correct / total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        # ── Validation Phase ─────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                with torch.amp.autocast('cuda', enabled=(MIXED_PRECISION and DEVICE.type == 'cuda')):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss /= max(val_total, 1)
        val_acc = 100.0 * val_correct / max(val_total, 1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        scheduler.step()
        epoch_time = time.time() - epoch_start

        print(f"  [{model_name}] Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.1f}% | "
              f"Time: {epoch_time:.0f}s")

        # ── Save Best ────────────────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'class_names': class_names,
                'num_classes': num_classes,
                'model_name': model_name,
            }, output_path / "best_model.pth")
            print(f"  ✅ New best! Val Acc: {val_acc:.1f}%")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"  ⏹ Early stopping at epoch {epoch+1}")
                break

    total_time = time.time() - total_start
    print(f"\n  [{model_name}] Training complete in {total_time/60:.1f} min")
    print(f"  [{model_name}] Best Val Acc: {best_val_acc:.1f}%")

    # Load best weights
    model.load_state_dict(best_model_wts)

    # Save training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
    ax1.plot(history['val_loss'], label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{model_name} — Training & Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history['val_acc'], label='Val Accuracy', color='green', linewidth=2)
    ax2.plot(history['train_acc'], label='Train Accuracy', color='blue', linewidth=2, alpha=0.7)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title(f'{model_name} — Accuracy (Best: {best_val_acc:.1f}%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path / "training_curves.png", dpi=150)
    plt.close()

    # Save class labels
    labels_map = {}
    for idx, name in enumerate(class_names):
        parts = name.split("___")
        crop = parts[0] if len(parts) >= 2 else name.split("_")[0]
        disease = parts[1] if len(parts) >= 2 else name
        labels_map[str(idx)] = {
            "class_name": name,
            "crop": crop,
            "disease": disease.replace("_", " ").title(),
        }

    (output_path / "class_labels.json").write_text(json.dumps(labels_map, indent=2))

    # Save history
    (output_path / "training_history.json").write_text(json.dumps({
        'model_name': model_name,
        'best_val_acc': best_val_acc,
        'total_time_minutes': total_time / 60,
        'epochs_trained': len(history['train_loss']),
        'history': history,
    }, indent=2))

    return model, history, best_val_acc, total_time


print("Training engine ready!")


# === CELL 6: Evaluation Engine ================================================

from sklearn.metrics import (classification_report, confusion_matrix,
                              precision_recall_fscore_support, accuracy_score)


def evaluate_model(model, model_name, test_loader, output_dir):
    """
    Evaluate model on test set. Generate confusion matrix, per-class metrics.
    Returns: results dict
    """
    output_path = Path(output_dir)
    model.eval()

    all_preds = []
    all_labels = []
    inference_times = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE, non_blocking=True)

            start_t = time.time()
            with torch.amp.autocast('cuda', enabled=(MIXED_PRECISION and DEVICE.type == 'cuda')):
                outputs = model(images)
            inference_times.append(time.time() - start_t)

            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Overall metrics
    accuracy = accuracy_score(all_labels, all_preds) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    avg_inference_time = np.mean(inference_times) * 1000  # ms per batch

    print(f"\n{'='*70}")
    print(f"  TEST RESULTS: {model_name}")
    print(f"{'='*70}")
    print(f"  Accuracy:          {accuracy:.2f}%")
    print(f"  Weighted Precision: {precision*100:.2f}%")
    print(f"  Weighted Recall:    {recall*100:.2f}%")
    print(f"  Weighted F1:        {f1*100:.2f}%")
    print(f"  Macro F1:           {macro_f1*100:.2f}%")
    print(f"  Avg Inference:      {avg_inference_time:.1f} ms/batch")

    # Per-class report
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    report_text = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        zero_division=0
    )

    print(f"\nPer-class Report:\n{report_text}")

    # Save report
    (output_path / "classification_report.txt").write_text(
        f"Model: {model_name}\n"
        f"Test Accuracy: {accuracy:.2f}%\n"
        f"Weighted F1: {f1*100:.2f}%\n"
        f"Macro F1: {macro_f1*100:.2f}%\n\n"
        f"{report_text}"
    )

    (output_path / "classification_report.json").write_text(
        json.dumps(report, indent=2)
    )

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)

    fig, ax = plt.subplots(figsize=(20, 18))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title(f'{model_name} — Confusion Matrix\nAccuracy: {accuracy:.1f}%', fontsize=14)
    fig.colorbar(im, ax=ax, shrink=0.8)

    # Add labels
    tick_marks = np.arange(len(class_names))
    short_names = [n.replace("___", "\n") for n in class_names]
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(short_names, rotation=90, fontsize=6)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(short_names, fontsize=6)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.tight_layout()
    fig.savefig(output_path / "confusion_matrix.png", dpi=150)
    plt.close()

    results = {
        'model_name': model_name,
        'accuracy': accuracy,
        'weighted_precision': precision * 100,
        'weighted_recall': recall * 100,
        'weighted_f1': f1 * 100,
        'macro_precision': macro_p * 100,
        'macro_recall': macro_r * 100,
        'macro_f1': macro_f1 * 100,
        'avg_inference_ms': avg_inference_time,
        'total_params': sum(p.numel() for p in model.parameters()),
    }

    (output_path / "test_results.json").write_text(json.dumps(results, indent=2))

    return results


print("Evaluation engine ready!")


# === CELL 7: TRAIN MODEL 1 — EfficientNet-B4 =================================

print("\n" + "🔬"*35)
print("TRAINING MODEL 1: EfficientNet-B4")
print("🔬"*35)

efficientnet_model = build_efficientnet_b4(num_classes)
efficientnet_dir = os.path.join(OUTPUT_BASE, "efficientnet_b4")

efficientnet_model, eff_history, eff_best_acc, eff_time = train_one_model(
    model_name="EfficientNet-B4",
    model=efficientnet_model,
    train_loader=train_loader,
    val_loader=val_loader,
    output_dir=efficientnet_dir,
    epochs=EPOCHS,
    lr=LEARNING_RATE,
)

eff_results = evaluate_model(efficientnet_model, "EfficientNet-B4", test_loader, efficientnet_dir)

# Free GPU memory
del efficientnet_model
torch.cuda.empty_cache()
print("\n✅ EfficientNet-B4 DONE! GPU memory freed.\n")


# === CELL 8: TRAIN MODEL 2 — ResNet-50 =======================================

print("\n" + "🧠"*35)
print("TRAINING MODEL 2: ResNet-50")
print("🧠"*35)

resnet_model = build_resnet50(num_classes)
resnet_dir = os.path.join(OUTPUT_BASE, "resnet50")

resnet_model, res_history, res_best_acc, res_time = train_one_model(
    model_name="ResNet-50",
    model=resnet_model,
    train_loader=train_loader,
    val_loader=val_loader,
    output_dir=resnet_dir,
    epochs=EPOCHS,
    lr=LEARNING_RATE,
)

res_results = evaluate_model(resnet_model, "ResNet-50", test_loader, resnet_dir)

# Free GPU memory
del resnet_model
torch.cuda.empty_cache()
print("\n✅ ResNet-50 DONE! GPU memory freed.\n")


# === CELL 9: TRAIN MODEL 3 — ViT-Base ========================================

print("\n" + "⚡"*35)
print("TRAINING MODEL 3: ViT-Base-Patch16-224")
print("⚡"*35)

vit_model = build_vit_base(num_classes)
vit_dir = os.path.join(OUTPUT_BASE, "vit_base")

vit_model, vit_history, vit_best_acc, vit_time = train_one_model(
    model_name="ViT-Base-Patch16-224",
    model=vit_model,
    train_loader=train_loader,
    val_loader=val_loader,
    output_dir=vit_dir,
    epochs=EPOCHS,
    lr=LEARNING_RATE,
)

vit_results = evaluate_model(vit_model, "ViT-Base-Patch16-224", test_loader, vit_dir)

# Free GPU memory
del vit_model
torch.cuda.empty_cache()
print("\n✅ ViT-Base DONE! GPU memory freed.\n")


# === CELL 10: FINAL COMPARISON ================================================

print("\n" + "="*70)
print("  📊 FINAL MODEL COMPARISON")
print("="*70)

all_results = [eff_results, res_results, vit_results]
all_histories = [
    ("EfficientNet-B4", eff_history, eff_time),
    ("ResNet-50", res_history, res_time),
    ("ViT-Base-Patch16-224", vit_history, vit_time),
]

# Print comparison table
print(f"\n{'Model':<25} {'Accuracy':>10} {'W-Precision':>12} {'W-Recall':>10} "
      f"{'W-F1':>8} {'Macro-F1':>10} {'Params':>12} {'Time(min)':>10}")
print("-" * 100)
for r, (name, hist, t) in zip(all_results, all_histories):
    print(f"{r['model_name']:<25} {r['accuracy']:>9.2f}% {r['weighted_precision']:>11.2f}% "
          f"{r['weighted_recall']:>9.2f}% {r['weighted_f1']:>7.2f}% {r['macro_f1']:>9.2f}% "
          f"{r['total_params']:>11,} {t/60:>9.1f}")

# Find best model
best_idx = max(range(len(all_results)), key=lambda i: all_results[i]['accuracy'])
print(f"\n🏆 Best Model: {all_results[best_idx]['model_name']} "
      f"with {all_results[best_idx]['accuracy']:.2f}% accuracy!")

# ── Comparison Plot ──────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

# 1. Accuracy comparison bar chart
ax = axes[0, 0]
names = [r['model_name'] for r in all_results]
accs = [r['accuracy'] for r in all_results]
bars = ax.bar(names, accs, color=colors, edgecolor='white', linewidth=2)
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
ax.set_ylabel('Accuracy (%)')
ax.set_title('Test Accuracy Comparison', fontsize=13, fontweight='bold')
ax.set_ylim([min(accs) - 5, 100])
ax.grid(axis='y', alpha=0.3)

# 2. F1 Score comparison
ax = axes[0, 1]
f1s_w = [r['weighted_f1'] for r in all_results]
f1s_m = [r['macro_f1'] for r in all_results]
x = np.arange(len(names))
w = 0.35
ax.bar(x - w/2, f1s_w, w, label='Weighted F1', color=colors, alpha=0.8, edgecolor='white')
ax.bar(x + w/2, f1s_m, w, label='Macro F1', color=colors, alpha=0.5, edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=9)
ax.set_ylabel('F1 Score (%)')
ax.set_title('F1 Score Comparison', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# 3. Training curves overlay (validation accuracy)
ax = axes[1, 0]
for (name, hist, _), color in zip(all_histories, colors):
    ax.plot(hist['val_acc'], label=name, linewidth=2, color=color)
ax.set_xlabel('Epoch')
ax.set_ylabel('Validation Accuracy (%)')
ax.set_title('Validation Accuracy Curves', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# 4. Training curves overlay (validation loss)
ax = axes[1, 1]
for (name, hist, _), color in zip(all_histories, colors):
    ax.plot(hist['val_loss'], label=name, linewidth=2, color=color)
ax.set_xlabel('Epoch')
ax.set_ylabel('Validation Loss')
ax.set_title('Validation Loss Curves', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.suptitle('Crop Disease Classification — Multi-Model Comparison\n'
             f'Dataset: PlantVillage ({num_classes} classes, {total_images} images)',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()

comparison_path = os.path.join(OUTPUT_BASE, "model_comparison.png")
fig.savefig(comparison_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nComparison plot saved to: {comparison_path}")

# Save comparison JSON
comparison_data = {
    'dataset': {
        'name': 'PlantVillage Cleaned',
        'num_classes': num_classes,
        'total_images': total_images,
        'train_size': train_size,
        'val_size': val_size,
        'test_size': test_size,
        'class_names': class_names,
    },
    'models': all_results,
    'best_model': all_results[best_idx]['model_name'],
    'best_accuracy': all_results[best_idx]['accuracy'],
}
comparison_json_path = os.path.join(OUTPUT_BASE, "comparison_results.json")
with open(comparison_json_path, 'w') as f:
    json.dump(comparison_data, f, indent=2)

print(f"Comparison results saved to: {comparison_json_path}")
print(f"\n🎉 ALL TRAINING COMPLETE! Check {OUTPUT_BASE} for all results.")
