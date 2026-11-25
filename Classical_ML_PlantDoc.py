"""
Classical ML Models (Random Forest & SVM) on PlantDoc Dataset
=============================================================
This script trains Random Forest and SVM classifiers on the PlantDoc dataset.

Feature extraction: Pretrained ResNet-50 CNN embeddings (2048-dim)
Feature scaling: StandardScaler fitted on training data only
Evaluation: Using the provided train/test split from PlantDoc

Usage:
    python Classical_ML_PlantDoc.py
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import os
import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_fscore_support, accuracy_score
)

# ---- Try to use PyTorch for feature extraction (GPU) ----
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms, models
    USE_TORCH = True
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using PyTorch on device: {DEVICE}")
except ImportError:
    USE_TORCH = False
    from skimage.feature import hog
    from skimage.transform import resize
    from skimage import io, color
    print("PyTorch not available — falling back to HOG features")


# ============================================================================
# Configuration
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLANTDOC_DIR = os.path.join(SCRIPT_DIR, "PlantDoc_Cleaned")
TRAIN_DIR = os.path.join(PLANTDOC_DIR, "train")
TEST_DIR = os.path.join(PLANTDOC_DIR, "test")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "PlantDoc_Results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 64

print(f"\nPlantDoc Train Dir: {TRAIN_DIR}")
print(f"PlantDoc Test Dir:  {TEST_DIR}")
print(f"Output Dir:         {OUTPUT_DIR}")


# ============================================================================
# Dataset Exploration
# ============================================================================
print("\n" + "=" * 70)
print("  PLANTDOC DATASET SUMMARY")
print("=" * 70)

train_classes = sorted([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))])
test_classes = sorted([d for d in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, d))])

print(f"\nTrain classes: {len(train_classes)}")
print(f"Test classes:  {len(test_classes)}")

# Verify class alignment
if train_classes == test_classes:
    print("✅ Train and test classes match.")
else:
    print("⚠️ WARNING: Train and test classes do NOT match!")
    print(f"  In train but not test: {set(train_classes) - set(test_classes)}")
    print(f"  In test but not train: {set(test_classes) - set(train_classes)}")

class_names = train_classes
num_classes = len(class_names)

total_train = 0
total_test = 0
print(f"\n{'Class':<35} {'Train':>6} {'Test':>6}")
print("-" * 50)
for c in class_names:
    train_count = len(os.listdir(os.path.join(TRAIN_DIR, c)))
    test_count = len(os.listdir(os.path.join(TEST_DIR, c))) if c in test_classes else 0
    total_train += train_count
    total_test += test_count
    print(f"  {c:<33} {train_count:>6} {test_count:>6}")
print("-" * 50)
print(f"  {'TOTAL':<33} {total_train:>6} {total_test:>6}")


# ============================================================================
# Feature Extraction
# ============================================================================
print("\n" + "=" * 70)
print("  FEATURE EXTRACTION")
print("=" * 70)

if USE_TORCH:
    # ---- PyTorch: Pretrained ResNet-50 embeddings ----
    print("\nUsing pretrained ResNet-50 to extract 2048-dim feature vectors...")

    val_test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=val_test_transforms)
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=val_test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=0, pin_memory=True)

    # Verify class-to-index mapping matches
    print(f"\nTrain class_to_idx: {train_dataset.class_to_idx}")
    print(f"Test class_to_idx:  {test_dataset.class_to_idx}")

    # Build feature extractor
    feature_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    feature_model.fc = nn.Identity()  # Remove classification head
    feature_model = feature_model.to(DEVICE)
    feature_model.eval()

    def extract_features(loader, model, desc="Extracting"):
        all_features = []
        all_labels = []
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(loader):
                images = images.to(DEVICE, non_blocking=True)
                features = model(images)
                all_features.append(features.cpu().numpy())
                all_labels.extend(labels.numpy())
                if (batch_idx + 1) % 10 == 0:
                    print(f"  {desc}: batch {batch_idx+1}/{len(loader)}")
        return np.vstack(all_features), np.array(all_labels)

    print("\nExtracting training features...")
    X_train, y_train = extract_features(train_loader, feature_model, "Train")

    print("Extracting test features...")
    X_test, y_test = extract_features(test_loader, feature_model, "Test")

    del feature_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    feature_method = "Pretrained ResNet-50 embeddings (2048-dim)"
    feature_dim = 2048

else:
    # ---- Fallback: HOG features ----
    print("\nUsing HOG features (no GPU available)...")

    def extract_hog_features(data_dir, class_names):
        features = []
        labels = []
        for class_idx, class_name in enumerate(class_names):
            class_dir = os.path.join(data_dir, class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                try:
                    img = io.imread(img_path)
                    if len(img.shape) == 3:
                        img = color.rgb2gray(img)
                    img = resize(img, (128, 128))
                    feat = hog(img, orientations=9, pixels_per_cell=(8, 8),
                              cells_per_block=(2, 2), feature_vector=True)
                    features.append(feat)
                    labels.append(class_idx)
                except Exception as e:
                    print(f"  Skipping {img_path}: {e}")
        return np.array(features), np.array(labels)

    print("Extracting HOG features from training set...")
    X_train, y_train = extract_hog_features(TRAIN_DIR, class_names)

    print("Extracting HOG features from test set...")
    X_test, y_test = extract_hog_features(TEST_DIR, class_names)

    feature_method = "HOG features (9 orientations, 8x8 pixels/cell)"
    feature_dim = X_train.shape[1]

print(f"\nFeature extraction complete:")
print(f"  Method: {feature_method}")
print(f"  Feature dim: {feature_dim}")
print(f"  X_train: {X_train.shape}")
print(f"  X_test:  {X_test.shape}")


# ============================================================================
# Feature Scaling (fit on train only)
# ============================================================================
print("\n--- Scaling features (fitted on training data only) ---")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("  StandardScaler fitted on training data and applied to both splits.")


# ============================================================================
# Model Training & Evaluation Helper
# ============================================================================
def evaluate_and_save(model_name, y_true, y_pred, class_names, output_dir,
                      training_time, feature_method, extra_info=""):
    """Evaluate predictions and save all outputs."""
    os.makedirs(output_dir, exist_ok=True)

    accuracy = accuracy_score(y_true, y_pred) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0)

    print(f"\n{'='*70}")
    print(f"  TEST RESULTS: {model_name} on PlantDoc")
    print(f"{'='*70}")
    print(f"  Accuracy:          {accuracy:.2f}%")
    print(f"  Weighted Precision: {precision*100:.2f}%")
    print(f"  Weighted Recall:    {recall*100:.2f}%")
    print(f"  Weighted F1:        {f1*100:.2f}%")
    print(f"  Macro Precision:    {macro_p*100:.2f}%")
    print(f"  Macro Recall:       {macro_r*100:.2f}%")
    print(f"  Macro F1:           {macro_f1*100:.2f}%")
    print(f"  Training Time:      {training_time/60:.1f} minutes")

    # Classification report
    report_text = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0)
    print(f"\nPer-class Report:\n{report_text}")

    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Dataset: PlantDoc\n")
        f.write(f"Feature Extraction: {feature_method}\n")
        f.write(f"{extra_info}\n" if extra_info else "")
        f.write(f"Test Accuracy: {accuracy:.2f}%\n")
        f.write(f"Weighted F1: {f1*100:.2f}%\n")
        f.write(f"Macro F1: {macro_f1*100:.2f}%\n\n")
        f.write(report_text)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title(f'{model_name} on PlantDoc\nAccuracy: {accuracy:.1f}%', fontsize=14)
    fig.colorbar(im, ax=ax, shrink=0.8)
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=90, fontsize=6)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names, fontsize=6)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()

    # Save predictions
    np.save(os.path.join(output_dir, 'test_predictions.npy'), y_pred)
    np.save(os.path.join(output_dir, 'test_ground_truth.npy'), y_true)

    results = {
        'model_name': model_name,
        'dataset': 'PlantDoc',
        'accuracy': accuracy,
        'weighted_precision': precision * 100,
        'weighted_recall': recall * 100,
        'weighted_f1': f1 * 100,
        'macro_precision': macro_p * 100,
        'macro_recall': macro_r * 100,
        'macro_f1': macro_f1 * 100,
        'feature_extraction': feature_method,
        'training_time_minutes': training_time / 60,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'num_classes': len(class_names),
    }

    with open(os.path.join(output_dir, 'test_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    return results


# ============================================================================
# Train Random Forest on PlantDoc
# ============================================================================
print("\n" + "=" * 70)
print("  TRAINING: Random Forest on PlantDoc")
print("=" * 70)

rf_start = time.time()
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    n_jobs=-1,
    random_state=RANDOM_SEED,
    verbose=1,
)
rf_model.fit(X_train_scaled, y_train)
rf_time = time.time() - rf_start

rf_preds = rf_model.predict(X_test_scaled)
rf_dir = os.path.join(OUTPUT_DIR, "random_forest")
rf_results = evaluate_and_save(
    "Random Forest", y_test, rf_preds, class_names, rf_dir,
    rf_time, feature_method,
    "n_estimators=300, max_features=sqrt, min_samples_split=5"
)
print("\n✅ Random Forest on PlantDoc DONE.")


# ============================================================================
# Train SVM on PlantDoc
# ============================================================================
print("\n" + "=" * 70)
print("  TRAINING: SVM (Linear) on PlantDoc")
print("=" * 70)

svm_start = time.time()
svm_model = CalibratedClassifierCV(
    LinearSVC(
        C=1.0,
        max_iter=5000,
        random_state=RANDOM_SEED,
        verbose=1,
    ),
    cv=3,
)
svm_model.fit(X_train_scaled, y_train)
svm_time = time.time() - svm_start

svm_preds = svm_model.predict(X_test_scaled)
svm_dir = os.path.join(OUTPUT_DIR, "svm")
svm_results = evaluate_and_save(
    "SVM (Linear)", y_test, svm_preds, class_names, svm_dir,
    svm_time, feature_method,
    "Kernel=Linear, C=1.0, Calibrated with 3-fold CV"
)
print("\n✅ SVM on PlantDoc DONE.")


# ============================================================================
# Comparison Table for PlantDoc
# ============================================================================
print("\n" + "=" * 70)
print("  PLANTDOC COMPARISON: Random Forest vs SVM")
print("=" * 70)

all_plantdoc_results = [rf_results, svm_results]

print(f'\n{"Model":<25} {"Accuracy":>10} {"W-F1":>8} {"Macro-F1":>10} {"Train Time":>12}')
print("-" * 70)
for r in all_plantdoc_results:
    print(f'{r["model_name"]:<25} {r["accuracy"]:>9.2f}% {r["weighted_f1"]:>7.2f}% '
          f'{r["macro_f1"]:>9.2f}% {r["training_time_minutes"]:>10.1f} min')

# Save combined comparison
comparison = {
    'dataset': {
        'name': 'PlantDoc',
        'num_classes': num_classes,
        'train_samples': total_train,
        'test_samples': total_test,
        'split': 'Pre-defined train/test split provided by PlantDoc',
        'class_names': class_names,
    },
    'feature_extraction': feature_method,
    'feature_scaling': 'StandardScaler (fitted on train only)',
    'models': all_plantdoc_results,
}
with open(os.path.join(OUTPUT_DIR, 'plantdoc_comparison.json'), 'w') as f:
    json.dump(comparison, f, indent=2)

# Comparison bar chart
fig, ax = plt.subplots(figsize=(10, 6))
model_names = [r['model_name'] for r in all_plantdoc_results]
accuracies = [r['accuracy'] for r in all_plantdoc_results]
colors = ['#FFA07A', '#98D8C8']
bars = ax.bar(model_names, accuracies, color=colors, edgecolor='white', linewidth=2)
for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=14)
ax.set_ylabel('Test Accuracy (%)')
ax.set_title('PlantDoc Dataset — Classical ML Comparison', fontsize=14, fontweight='bold')
ax.set_ylim([min(accuracies) - 15, 100])
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'plantdoc_comparison.png'), dpi=150)
plt.close()

print(f"\n🎉 ALL PLANTDOC TRAINING COMPLETE!")
print(f"Results saved to: {OUTPUT_DIR}")
