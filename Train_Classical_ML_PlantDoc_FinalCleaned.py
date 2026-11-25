"""
Train Classical Machine Learning Models (Random Forest & SVM)
on PlantDoc Final Noise-Cleaned Dataset (27 Classes)
====================================================================
Student: Bhavya Sodhi
Capstone Project: Plant Disease Classification Using Deep Learning

Dataset: PlantDoc_Final_noise_cleaned_Dataset
         (Noise-cleaned version with unwanted backgrounds & hand artifacts removed)

Features:
- Pretrained ResNet-50 feature embeddings (2048 dimensions) via PyTorch on GPU (RTX 3050)
- StandardScaler fitted on training splits only (no data leakage)
- Random Forest (150 trees, sqrt features)
- Linear SVM (SGDClassifier with hinge loss, robust convex optimization)
- Complete evaluation: Accuracy, Precision, Recall, F1, Confusion Matrices
- High-resolution visual outputs saved for presentation/report
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import os
import json
import time
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_fscore_support, accuracy_score
)

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"[INIT] PyTorch version: {torch.__version__}")
print(f"[INIT] Compute Device: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "PlantDoc_Final_Cleaned")
OUTPUT_DIR = os.path.join(BASE_DIR, "PlantDoc_FinalCleaned_Outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATASET_NAME = "PlantDoc Final Noise-Cleaned"
DATASET_TAG = "PlantDoc_FinalCleaned"

# Image Transform
IMAGE_SIZE = 224
BATCH_SIZE = 64
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Custom Dataset
class SimpleImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]
        with Image.open(path) as img:
            img = img.convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

# Load feature extractor model
print("\n[MODEL] Loading Pretrained ResNet-50 for Feature Extraction...")
feature_extractor = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
feature_extractor.fc = nn.Identity()  # 2048-dim feature output
feature_extractor = feature_extractor.to(DEVICE)
feature_extractor.eval()

def extract_features(loader, model, desc="Extracting"):
    features = []
    labels = []
    total_batches = len(loader)
    start_t = time.time()
    with torch.no_grad():
        for i, (imgs, lbls) in enumerate(loader):
            imgs = imgs.to(DEVICE, non_blocking=True)
            feats = model(imgs)
            features.append(feats.cpu().numpy())
            labels.extend(lbls.numpy())
            if (i + 1) % 15 == 0 or (i + 1) == total_batches:
                elapsed = time.time() - start_t
                print(f"  [{desc}] Batch {i+1}/{total_batches} ({elapsed:.1f}s)")
    return np.vstack(features), np.array(labels)


# ============================================================================
# 1. LOAD DATASET
# ============================================================================
print("\n" + "="*80)
print(f"  DATASET: {DATASET_NAME} (27 Classes, Noise-Cleaned In-the-Wild)")
print("="*80)

train_dir = os.path.join(DATASET_DIR, "train")
test_dir = os.path.join(DATASET_DIR, "test")

classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
class_to_idx = {c: i for i, c in enumerate(classes)}
print(f"Classes: {len(classes)}")

train_paths, train_labels = [], []
for c in classes:
    c_dir = os.path.join(train_dir, c)
    for fname in os.listdir(c_dir):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            train_paths.append(os.path.join(c_dir, fname))
            train_labels.append(class_to_idx[c])

test_paths, test_labels = [], []
for c in classes:
    c_dir = os.path.join(test_dir, c)
    if os.path.exists(c_dir):
        for fname in os.listdir(c_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                test_paths.append(os.path.join(c_dir, fname))
                test_labels.append(class_to_idx[c])

print(f"Train Samples: {len(train_paths)} | Test Samples: {len(test_paths)}")

# Print per-class distribution
print("\n[CLASS DISTRIBUTION]")
from collections import Counter
train_dist = Counter(train_labels)
test_dist = Counter(test_labels)
for c in classes:
    idx = class_to_idx[c]
    print(f"  {c:<35} Train: {train_dist.get(idx, 0):>4}  Test: {test_dist.get(idx, 0):>4}")

train_ds = SimpleImageDataset(train_paths, train_labels, transform=transform)
test_ds = SimpleImageDataset(test_paths, test_labels, transform=transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


# ============================================================================
# 2. FEATURE EXTRACTION
# ============================================================================
print("\nExtracting Features using ResNet-50...")
X_train, y_train = extract_features(train_loader, feature_extractor, "Train")
X_test, y_test = extract_features(test_loader, feature_extractor, "Test")

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)
print(f"Feature matrix: Train {X_train_sc.shape}, Test {X_test_sc.shape}")


# ============================================================================
# 3. TRAIN MODELS
# ============================================================================

# --- Random Forest ---
print("\n--- Training Random Forest ---")
t0 = time.time()
rf = RandomForestClassifier(n_estimators=150, max_features='sqrt', max_depth=25, n_jobs=-1, random_state=SEED)
rf.fit(X_train_sc, y_train)
rf_time = time.time() - t0
y_pred_rf = rf.predict(X_test_sc)
rf_acc = accuracy_score(y_test, y_pred_rf) * 100
rf_p_w, rf_r_w, rf_f1_w, _ = precision_recall_fscore_support(y_test, y_pred_rf, average='weighted', zero_division=0)
rf_p_m, rf_r_m, rf_f1_m, _ = precision_recall_fscore_support(y_test, y_pred_rf, average='macro', zero_division=0)
print(f"Random Forest -> Test Acc: {rf_acc:.2f}%, Weighted F1: {rf_f1_w*100:.2f}%, Macro F1: {rf_f1_m*100:.2f}%, Time: {rf_time:.2f}s")

# --- Linear SVM (SGDClassifier with hinge loss) ---
print("\n--- Training Linear SVM ---")
t0 = time.time()
svm = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-4, max_iter=1000, tol=1e-3, random_state=SEED)
svm.fit(X_train_sc, y_train)
svm_time = time.time() - t0
y_pred_svm = svm.predict(X_test_sc)
svm_acc = accuracy_score(y_test, y_pred_svm) * 100
svm_p_w, svm_r_w, svm_f1_w, _ = precision_recall_fscore_support(y_test, y_pred_svm, average='weighted', zero_division=0)
svm_p_m, svm_r_m, svm_f1_m, _ = precision_recall_fscore_support(y_test, y_pred_svm, average='macro', zero_division=0)
print(f"Linear SVM   -> Test Acc: {svm_acc:.2f}%, Weighted F1: {svm_f1_w*100:.2f}%, Macro F1: {svm_f1_m*100:.2f}%, Time: {svm_time:.2f}s")


# ============================================================================
# 4. GENERATE VISUALIZATIONS
# ============================================================================
print("\n" + "="*80)
print("  GENERATING HIGH-RESOLUTION PLOTS & SCREENSHOTS")
print("="*80)

def plot_confusion_matrix(cm, class_names, title, filename, cmap='Blues'):
    fig, ax = plt.subplots(figsize=(14, 12), dpi=150)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=11, fontweight='bold')
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=90, fontsize=6)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names, fontsize=6)

    # Add text annotations to each cell
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha='center', va='center', fontsize=5,
                    color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] {path}")
    return path

# 1. Confusion Matrix: Random Forest
cm_rf = confusion_matrix(y_test, y_pred_rf)
plot_confusion_matrix(cm_rf, classes,
                      f"{DATASET_NAME} — Random Forest Confusion Matrix\nTest Accuracy: {rf_acc:.2f}% | Weighted F1: {rf_f1_w*100:.2f}%",
                      f"{DATASET_TAG}_RF_Confusion_Matrix.png", cmap='Greens')

# 2. Confusion Matrix: SVM
cm_svm = confusion_matrix(y_test, y_pred_svm)
plot_confusion_matrix(cm_svm, classes,
                      f"{DATASET_NAME} — Linear SVM Confusion Matrix\nTest Accuracy: {svm_acc:.2f}% | Weighted F1: {svm_f1_w*100:.2f}%",
                      f"{DATASET_TAG}_SVM_Confusion_Matrix.png", cmap='Blues')

# 3. Per-Class Accuracy Bar Chart (side-by-side RF vs SVM)
fig, ax = plt.subplots(figsize=(18, 8), dpi=150)
per_class_rf = cm_rf.diagonal() / (cm_rf.sum(axis=1) + 1e-8) * 100
per_class_svm = cm_svm.diagonal() / (cm_svm.sum(axis=1) + 1e-8) * 100
x = np.arange(len(classes))
width = 0.38
bars_rf = ax.bar(x - width/2, per_class_rf, width, label=f'Random Forest ({rf_acc:.1f}%)', color='#2ca02c', edgecolor='black', alpha=0.85)
bars_svm = ax.bar(x + width/2, per_class_svm, width, label=f'Linear SVM ({svm_acc:.1f}%)', color='#1f77b4', edgecolor='black', alpha=0.85)
ax.set_ylabel('Per-Class Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title(f'{DATASET_NAME} — Per-Class Accuracy: Random Forest vs Linear SVM', fontsize=14, fontweight='bold', pad=12)
ax.set_xticks(x)
ax.set_xticklabels(classes, rotation=75, fontsize=7, ha='right')
ax.set_ylim(0, 115)
ax.legend(loc='upper right', fontsize=11, frameon=True)
ax.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
perclass_path = os.path.join(OUTPUT_DIR, f"{DATASET_TAG}_PerClass_Accuracy.png")
fig.savefig(perclass_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"  [SAVED] {perclass_path}")

# 4. Master Comparison Dashboard (Accuracy + F1)
fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

ax1 = axes[0]
model_names = ['Random Forest', 'Linear SVM']
accs = [rf_acc, svm_acc]
colors_acc = ['#2ca02c', '#1f77b4']
bars1 = ax1.bar(model_names, accs, color=colors_acc, edgecolor='black', linewidth=1.2, alpha=0.9, width=0.5)
for bar, acc in zip(bars1, accs):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
             f"{acc:.2f}%", ha='center', va='bottom', fontweight='bold', fontsize=14)
ax1.set_ylabel("Test Accuracy (%)", fontsize=12, fontweight='bold')
ax1.set_title(f"{DATASET_NAME}\nTest Accuracy Comparison", fontsize=13, fontweight='bold')
ax1.set_ylim(0, 110)
ax1.grid(axis='y', linestyle='--', alpha=0.4)

ax2 = axes[1]
x = np.arange(len(model_names))
width = 0.3
f1w = [rf_f1_w * 100, svm_f1_w * 100]
f1m = [rf_f1_m * 100, svm_f1_m * 100]
bars_w = ax2.bar(x - width/2, f1w, width, label='Weighted F1 (%)', color='#2b5c8f', edgecolor='black')
bars_m = ax2.bar(x + width/2, f1m, width, label='Macro F1 (%)', color='#e27d60', edgecolor='black')
for bar in bars_w:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.0,
             f"{bar.get_height():.1f}", ha='center', va='bottom', fontsize=11, fontweight='bold')
for bar in bars_m:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.0,
             f"{bar.get_height():.1f}", ha='center', va='bottom', fontsize=11, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(model_names, fontsize=11)
ax2.set_ylabel("Score (%)", fontsize=12, fontweight='bold')
ax2.set_title(f"{DATASET_NAME}\nWeighted F1 vs Macro F1", fontsize=13, fontweight='bold')
ax2.set_ylim(0, 115)
ax2.legend(loc='lower right', frameon=True)
ax2.grid(axis='y', linestyle='--', alpha=0.4)

plt.suptitle(f"Classical ML on {DATASET_NAME}\n(ResNet-50 Feature Embeddings, StandardScaler)",
             fontsize=14, fontweight='bold', y=1.03)
plt.tight_layout()
dashboard_path = os.path.join(OUTPUT_DIR, f"{DATASET_TAG}_Master_Comparison.png")
fig.savefig(dashboard_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"  [SAVED] {dashboard_path}")

# 5. Comparison with Original PlantDoc (if results exist)
old_json_path = os.path.join(BASE_DIR, "Classical_ML_Outputs", "classical_ml_summary.json")
if os.path.exists(old_json_path):
    print("\n--- Generating Old vs New Dataset Comparison ---")
    with open(old_json_path, 'r') as f:
        old_results = json.load(f)

    old_rf_acc = old_results['models']['PlantDoc_Random_Forest']['accuracy']
    old_svm_acc = old_results['models']['PlantDoc_Linear_SVM']['accuracy']
    old_rf_f1 = old_results['models']['PlantDoc_Random_Forest']['weighted_f1']
    old_svm_f1 = old_results['models']['PlantDoc_Linear_SVM']['weighted_f1']

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=150)

    # Accuracy comparison: Old vs New
    ax1 = axes[0]
    labels = ['RF (Old PlantDoc)', 'RF (Cleaned)', 'SVM (Old PlantDoc)', 'SVM (Cleaned)']
    vals = [old_rf_acc, rf_acc, old_svm_acc, svm_acc]
    colors_comp = ['#c0392b', '#27ae60', '#2980b9', '#3498db']
    bars = ax1.bar(labels, vals, color=colors_comp, edgecolor='black', linewidth=1.2, alpha=0.9)
    for bar, v in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.2,
                 f"{v:.2f}%", ha='center', va='bottom', fontweight='bold', fontsize=11)
    ax1.set_ylabel("Test Accuracy (%)", fontsize=12, fontweight='bold')
    ax1.set_title("Test Accuracy: Original vs Noise-Cleaned PlantDoc", fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 110)
    ax1.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
    ax1.grid(axis='y', linestyle='--', alpha=0.4)

    # F1 comparison: Old vs New
    ax2 = axes[1]
    f1_vals = [old_rf_f1, rf_f1_w * 100, old_svm_f1, svm_f1_w * 100]
    bars2 = ax2.bar(labels, f1_vals, color=colors_comp, edgecolor='black', linewidth=1.2, alpha=0.9)
    for bar, v in zip(bars2, f1_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.2,
                 f"{v:.2f}%", ha='center', va='bottom', fontweight='bold', fontsize=11)
    ax2.set_ylabel("Weighted F1 (%)", fontsize=12, fontweight='bold')
    ax2.set_title("Weighted F1: Original vs Noise-Cleaned PlantDoc", fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 110)
    ax2.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
    ax2.grid(axis='y', linestyle='--', alpha=0.4)

    plt.suptitle("Impact of Dataset Cleaning on Classical ML Performance\n(Same Pipeline: ResNet-50 Embeddings + StandardScaler)",
                 fontsize=14, fontweight='bold', y=1.03)
    plt.tight_layout()
    compare_path = os.path.join(OUTPUT_DIR, f"{DATASET_TAG}_vs_OldPlantDoc_Comparison.png")
    fig.savefig(compare_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] {compare_path}")

    # Improvement summary
    rf_delta = rf_acc - old_rf_acc
    svm_delta = svm_acc - old_svm_acc
    print(f"\n  [IMPROVEMENT] Random Forest: {old_rf_acc:.2f}% -> {rf_acc:.2f}% (delta: {rf_delta:+.2f}%)")
    print(f"  [IMPROVEMENT] Linear SVM:    {old_svm_acc:.2f}% -> {svm_acc:.2f}% (delta: {svm_delta:+.2f}%)")


# ============================================================================
# 5. SAVE TEXT REPORTS & JSON METRICS
# ============================================================================
results_summary = {
    "project": "Plant Disease Classification Using Deep Learning",
    "student": "Bhavya Sodhi",
    "dataset": DATASET_NAME,
    "dataset_description": "Noise-cleaned PlantDoc dataset with unwanted backgrounds & hand artifacts removed",
    "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "feature_extraction": "Pretrained ResNet-50 (2048-dimensional embeddings, PyTorch on RTX 3050 CUDA)",
    "scaling": "StandardScaler fitted on training split only (no data leakage)",
    "num_classes": len(classes),
    "total_train_samples": len(train_paths),
    "total_test_samples": len(test_paths),
    "classes": classes,
    "models": {
        "Random_Forest": {
            "algorithm": "RandomForestClassifier (n_estimators=150, max_features=sqrt, max_depth=25)",
            "accuracy": rf_acc,
            "weighted_precision": rf_p_w * 100,
            "weighted_recall": rf_r_w * 100,
            "weighted_f1": rf_f1_w * 100,
            "macro_precision": rf_p_m * 100,
            "macro_recall": rf_r_m * 100,
            "macro_f1": rf_f1_m * 100,
            "training_time_seconds": rf_time
        },
        "Linear_SVM": {
            "algorithm": "SGDClassifier (loss=hinge, penalty=l2, alpha=1e-4)",
            "accuracy": svm_acc,
            "weighted_precision": svm_p_w * 100,
            "weighted_recall": svm_r_w * 100,
            "weighted_f1": svm_f1_w * 100,
            "macro_precision": svm_p_m * 100,
            "macro_recall": svm_r_m * 100,
            "macro_f1": svm_f1_m * 100,
            "training_time_seconds": svm_time
        }
    }
}

json_path = os.path.join(OUTPUT_DIR, f"{DATASET_TAG}_classical_ml_summary.json")
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, indent=2)
print(f"\n[SAVED] Summary JSON: {json_path}")

# Write classification reports
with open(os.path.join(OUTPUT_DIR, f"{DATASET_TAG}_RF_Classification_Report.txt"), 'w', encoding='utf-8') as f:
    f.write(f"{'='*75}\n")
    f.write(f"  Random Forest Classification Report — {DATASET_NAME}\n")
    f.write(f"  Test Accuracy: {rf_acc:.2f}%  |  Weighted F1: {rf_f1_w*100:.2f}%  |  Macro F1: {rf_f1_m*100:.2f}%\n")
    f.write(f"{'='*75}\n\n")
    f.write(classification_report(y_test, y_pred_rf, target_names=classes, zero_division=0))

with open(os.path.join(OUTPUT_DIR, f"{DATASET_TAG}_SVM_Classification_Report.txt"), 'w', encoding='utf-8') as f:
    f.write(f"{'='*75}\n")
    f.write(f"  Linear SVM Classification Report — {DATASET_NAME}\n")
    f.write(f"  Test Accuracy: {svm_acc:.2f}%  |  Weighted F1: {svm_f1_w*100:.2f}%  |  Macro F1: {svm_f1_m*100:.2f}%\n")
    f.write(f"{'='*75}\n\n")
    f.write(classification_report(y_test, y_pred_svm, target_names=classes, zero_division=0))


# ============================================================================
# FINAL SCORECARD
# ============================================================================
print("\n" + "="*80)
print(f"  FINAL SCORECARD: {DATASET_NAME}")
print("="*80)
print(f"{'Model':<20} {'Test Acc':>10} {'Weighted F1':>14} {'Macro F1':>12} {'Train Time':>12}")
print("-" * 72)
print(f"{'Random Forest':<20} {rf_acc:>9.2f}% {rf_f1_w*100:>13.2f}% {rf_f1_m*100:>11.2f}% {rf_time:>10.2f}s")
print(f"{'Linear SVM':<20} {svm_acc:>9.2f}% {svm_f1_w*100:>13.2f}% {svm_f1_m*100:>11.2f}% {svm_time:>10.2f}s")
print("="*72)
print(f"\nAll outputs saved in: {OUTPUT_DIR}")
print(f"Files generated:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    fpath = os.path.join(OUTPUT_DIR, f)
    size_kb = os.path.getsize(fpath) / 1024
    print(f"  - {f} ({size_kb:.1f} KB)")
print("\nDone!")
