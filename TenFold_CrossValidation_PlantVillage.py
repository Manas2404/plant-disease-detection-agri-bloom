"""
================================================================================
10-FOLD CROSS-VALIDATION — PlantVillage Dataset
================================================================================
Ma'am's Requirement: Seed 42 ke badle, 10-Fold Stratified Cross-Validation 
                     karke dikhao. Ye prove karega ki accuracy seed-independent 
                     hai aur model genuinely generalize karta hai.

This script:
  1. Splits PlantVillage into 10 stratified folds
  2. Trains EfficientNet-B4 on 9 folds, tests on 1 fold (rotates all 10)
  3. Reports per-fold accuracy + mean ± std deviation
  4. Compares with original fixed-seed result (99.52%)

Works on Google Colab (T4 GPU) — Copy this to Colab and run.
================================================================================
Student: Bhavya Sodhi
"""

import os
import json
import time
import copy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms, models
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ============================================================================
# CONFIG
# ============================================================================
# --- CHANGE THESE PATHS FOR YOUR SETUP ---
DATA_DIR = '/content/drive/MyDrive/PlantVillage_Cleaned'                    # Colab path
OUTPUT_BASE = '/content/drive/MyDrive/Capstone_Results/CrossValidation'     # Output path
# If running locally:
# DATA_DIR = r'c:\Users\Admin\Downloads\capstone project\PlantVillage_Cleaned'
# OUTPUT_BASE = r'c:\Users\Admin\Downloads\capstone project\CrossValidation_Outputs'

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10             # Slightly reduced per fold since we train 10 times
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
EARLY_STOPPING_PATIENCE = 4
NUM_WORKERS = 2
MIXED_PRECISION = True
K_FOLDS = 10
RANDOM_SEED = 42

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')
if DEVICE.type == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}')

# ============================================================================
# DATA TRANSFORMS
# ============================================================================
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


# ============================================================================
# LOAD DATASET
# ============================================================================
print('\n' + '='*70)
print('  10-FOLD STRATIFIED CROSS-VALIDATION')
print('='*70)

full_dataset = datasets.ImageFolder(DATA_DIR, transform=val_test_transforms)
num_classes = len(full_dataset.classes)
class_names = full_dataset.classes
total_images = len(full_dataset)
all_targets = np.array(full_dataset.targets)

print(f'Total images: {total_images}')
print(f'Number of classes: {num_classes}')
print(f'K-Folds: {K_FOLDS}')


# ============================================================================
# BUILD MODEL FUNCTION
# ============================================================================
def build_efficientnet_b4(num_classes):
    """Fresh EfficientNet-B4 with custom head (called once per fold)."""
    try:
        from efficientnet_pytorch import EfficientNet
        model = EfficientNet.from_pretrained('efficientnet-b4')
        in_features = model._fc.in_features
        model._fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )
    except ImportError:
        print('  [Fallback] Using ResNet-50 instead of EfficientNet-B4')
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )
    return model


def train_one_fold(model, train_indices, val_indices, fold_num, output_dir):
    """Train model on train_indices, evaluate on val_indices (one CV fold)."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create datasets with proper transforms
    train_ds_aug = datasets.ImageFolder(DATA_DIR, transform=train_transforms)
    val_ds_clean = datasets.ImageFolder(DATA_DIR, transform=val_test_transforms)
    
    # For early stopping, take 10% of training fold as internal validation
    n_train = len(train_indices)
    n_internal_val = int(n_train * 0.1)
    n_actual_train = n_train - n_internal_val
    
    # Shuffle training indices for internal split
    rng = np.random.RandomState(RANDOM_SEED + fold_num)
    shuffled_train = train_indices.copy()
    rng.shuffle(shuffled_train)
    actual_train_indices = shuffled_train[:n_actual_train]
    internal_val_indices = shuffled_train[n_actual_train:]
    
    train_loader = DataLoader(
        Subset(train_ds_aug, actual_train_indices),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True
    )
    internal_val_loader = DataLoader(
        Subset(val_ds_clean, internal_val_indices),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    test_fold_loader = DataLoader(
        Subset(val_ds_clean, val_indices),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.amp.GradScaler('cuda', enabled=(MIXED_PRECISION and DEVICE.type == 'cuda'))
    
    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0
    
    total_start = time.time()
    
    for epoch in range(EPOCHS):
        # Train phase
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
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
        
        train_acc = 100.0 * correct / total
        
        # Internal validation phase (for early stopping only)
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in internal_val_loader:
                images = images.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=(MIXED_PRECISION and DEVICE.type == 'cuda')):
                    outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100.0 * val_correct / max(val_total, 1)
        scheduler.step()
        
        print(f'    Fold {fold_num+1} | Epoch {epoch+1}/{EPOCHS} | '
              f'Train: {train_acc:.1f}% | Internal Val: {val_acc:.1f}%')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f'    >> Early stopping at epoch {epoch+1}')
                break
    
    total_time = time.time() - total_start
    model.load_state_dict(best_model_wts)
    
    # Evaluate on held-out FOLD (the actual test for this fold)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_fold_loader:
            images = images.to(DEVICE, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=(MIXED_PRECISION and DEVICE.type == 'cuda')):
                outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    fold_accuracy = accuracy_score(all_labels, all_preds) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    
    # Save fold results
    np.save(output_path / f'fold_{fold_num+1}_predictions.npy', all_preds)
    np.save(output_path / f'fold_{fold_num+1}_ground_truth.npy', all_labels)
    
    result = {
        'fold': fold_num + 1,
        'accuracy': fold_accuracy,
        'weighted_f1': f1 * 100,
        'macro_f1': macro_f1 * 100,
        'weighted_precision': precision * 100,
        'weighted_recall': recall * 100,
        'test_samples': len(val_indices),
        'train_samples': n_actual_train,
        'internal_val_samples': n_internal_val,
        'training_time_minutes': total_time / 60,
        'best_internal_val_acc': best_val_acc,
    }
    
    print(f'\n  Fold {fold_num+1} TEST Accuracy: {fold_accuracy:.2f}% '
          f'(Weighted F1: {f1*100:.2f}%, Macro F1: {macro_f1*100:.2f}%)')
    
    return result


# ============================================================================
# RUN 10-FOLD CROSS-VALIDATION
# ============================================================================
print('\n' + '='*70)
print('  STARTING 10-FOLD STRATIFIED CROSS-VALIDATION')
print('='*70)

skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)

all_indices = np.arange(total_images)
fold_results = []
overall_start = time.time()

for fold_num, (train_idx, test_idx) in enumerate(skf.split(all_indices, all_targets)):
    print(f'\n{"="*70}')
    print(f'  FOLD {fold_num+1}/{K_FOLDS}')
    print(f'  Train: {len(train_idx)} images | Test: {len(test_idx)} images')
    print(f'{"="*70}')
    
    # Build fresh model for each fold
    model = build_efficientnet_b4(num_classes)
    
    fold_output = os.path.join(OUTPUT_BASE, f'Fold_{fold_num+1}')
    result = train_one_fold(model, train_idx, test_idx, fold_num, fold_output)
    fold_results.append(result)
    
    # Free GPU memory between folds
    del model
    torch.cuda.empty_cache()

overall_time = (time.time() - overall_start) / 60

# ============================================================================
# SUMMARY AND COMPARISON
# ============================================================================
print('\n' + '='*70)
print('  10-FOLD CROSS-VALIDATION SUMMARY')
print('='*70)

accuracies = [r['accuracy'] for r in fold_results]
f1_scores = [r['weighted_f1'] for r in fold_results]
macro_f1_scores = [r['macro_f1'] for r in fold_results]

mean_acc = np.mean(accuracies)
std_acc = np.std(accuracies)
mean_f1 = np.mean(f1_scores)
std_f1 = np.std(f1_scores)
mean_macro_f1 = np.mean(macro_f1_scores)
std_macro_f1 = np.std(macro_f1_scores)

print(f'\n{"Fold":<8} {"Accuracy":>10} {"Weighted F1":>12} {"Macro F1":>10}')
print('-' * 45)
for r in fold_results:
    print(f'  {r["fold"]:<6} {r["accuracy"]:>9.2f}% {r["weighted_f1"]:>11.2f}% {r["macro_f1"]:>9.2f}%')
print('-' * 45)
print(f'  {"Mean":<6} {mean_acc:>9.2f}% {mean_f1:>11.2f}% {mean_macro_f1:>9.2f}%')
print(f'  {"Std":<6} {std_acc:>9.2f}% {std_f1:>11.2f}% {std_macro_f1:>9.2f}%')

print(f'\n  Mean ± Std: {mean_acc:.2f}% ± {std_acc:.2f}%')
print(f'  Total training time: {overall_time:.1f} minutes')

# Comparison with fixed-seed result
print(f'\n{"="*70}')
print(f'  COMPARISON WITH ORIGINAL FIXED-SEED RESULT')
print(f'{"="*70}')
print(f'  Original (seed=42, 75/15/10 split): 99.52%')
print(f'  10-Fold CV Mean:                    {mean_acc:.2f}% ± {std_acc:.2f}%')
print(f'  10-Fold CV Min:                     {min(accuracies):.2f}%')
print(f'  10-Fold CV Max:                     {max(accuracies):.2f}%')

if abs(mean_acc - 99.52) < 1.0:
    print(f'\n  ✅ CONCLUSION: 10-Fold CV result ({mean_acc:.2f}%) is within 1% of ')
    print(f'     the fixed-seed result (99.52%), confirming that:')
    print(f'     1. Seed=42 did NOT bias the result')
    print(f'     2. Model genuinely learns the patterns, not memorizing a lucky split')
    print(f'     3. Low std ({std_acc:.2f}%) means performance is stable across all splits')
else:
    print(f'\n  ⚠️ NOTE: 10-Fold CV result differs from fixed-seed by '
          f'{abs(mean_acc - 99.52):.2f}%. Investigate further.')

# ============================================================================
# GENERATE VISUALIZATION
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart of per-fold accuracy
ax = axes[0]
folds = [r['fold'] for r in fold_results]
bars = ax.bar(folds, accuracies, color='#4ECDC4', edgecolor='white', linewidth=2)
ax.axhline(y=mean_acc, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_acc:.2f}%')
ax.axhline(y=99.52, color='blue', linestyle=':', linewidth=2, label=f'Fixed-seed: 99.52%')
for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{acc:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_xlabel('Fold Number')
ax.set_ylabel('Accuracy (%)')
ax.set_title('10-Fold Cross-Validation: Per-Fold Accuracy')
ax.set_ylim([min(accuracies) - 1, 100.2])
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Box plot
ax = axes[1]
bp = ax.boxplot([accuracies, f1_scores, macro_f1_scores], 
                labels=['Accuracy', 'Weighted F1', 'Macro F1'],
                patch_artist=True)
colors = ['#4ECDC4', '#FF6B6B', '#45B7D1']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_ylabel('Score (%)')
ax.set_title(f'10-Fold CV Score Distribution\nMean Acc: {mean_acc:.2f}% ± {std_acc:.2f}%')
ax.grid(axis='y', alpha=0.3)

plt.suptitle('PlantVillage — 10-Fold Stratified Cross-Validation (EfficientNet-B4)',
             fontsize=13, fontweight='bold')
plt.tight_layout()

plot_path = os.path.join(OUTPUT_BASE, 'crossvalidation_results.png')
os.makedirs(OUTPUT_BASE, exist_ok=True)
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.show()
plt.close()

# Save full results JSON
cv_summary = {
    'method': '10-Fold Stratified Cross-Validation',
    'dataset': 'PlantVillage Cleaned',
    'num_classes': num_classes,
    'total_images': total_images,
    'k_folds': K_FOLDS,
    'model': 'EfficientNet-B4 (pretrained ImageNet)',
    'random_seed': RANDOM_SEED,
    'fold_results': fold_results,
    'summary': {
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc,
        'mean_weighted_f1': mean_f1,
        'std_weighted_f1': std_f1,
        'mean_macro_f1': mean_macro_f1,
        'std_macro_f1': std_macro_f1,
        'min_accuracy': min(accuracies),
        'max_accuracy': max(accuracies),
    },
    'comparison_with_fixed_seed': {
        'fixed_seed_accuracy': 99.52,
        'cv_mean_accuracy': mean_acc,
        'difference': abs(mean_acc - 99.52),
        'conclusion': 'Seed did not bias result' if abs(mean_acc - 99.52) < 1.0 
                      else 'Investigate further',
    },
    'total_training_time_minutes': overall_time,
}

with open(os.path.join(OUTPUT_BASE, 'crossvalidation_summary.json'), 'w') as f:
    json.dump(cv_summary, f, indent=2)

print(f'\nResults saved to: {OUTPUT_BASE}/')
print('DONE! 10-Fold Cross-Validation Complete!')

