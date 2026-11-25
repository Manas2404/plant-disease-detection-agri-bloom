"""
================================================================================
TWO-STAGE HIERARCHICAL PREDICTION — Plant Disease Classification
================================================================================
Ma'am's Requirement: Pehle predict karo ki leaf kaunse crop ki hai (Apple/Tomato/Corn etc.)
                     Fir us crop ke andar predict karo ki kaunsi disease/condition hai.

This script:
  Stage 1: Trains a 14-class CROP classifier (apple, tomato, corn, grape, etc.)
  Stage 2: For each crop with >1 condition, trains a CONDITION classifier
           (e.g., apple → scab, black_rot, cedar_rust, healthy)

Works on Google Colab (T4 GPU) — Copy this to Colab and run.
================================================================================
Student: Bhavya Sodhi
"""

import os
import json
import time
import copy
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset, WeightedRandomSampler
from torchvision import datasets, transforms, models
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, precision_recall_fscore_support)

# ============================================================================
# CONFIG
# ============================================================================
# --- CHANGE THESE PATHS FOR YOUR SETUP ---
DATA_DIR = '/content/drive/MyDrive/PlantVillage_Cleaned'           # Colab path
OUTPUT_BASE = '/content/drive/MyDrive/Capstone_Results/Hierarchical'  # Output path
# If running locally, change to:
# DATA_DIR = r'c:\Users\Admin\Downloads\capstone project\PlantVillage_Cleaned'
# OUTPUT_BASE = r'c:\Users\Admin\Downloads\capstone project\Hierarchical_Outputs'

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
EARLY_STOPPING_PATIENCE = 5
NUM_WORKERS = 2
MIXED_PRECISION = True
VAL_SPLIT = 0.15
TEST_SPLIT = 0.10
RANDOM_SEED = 42

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')
if DEVICE.type == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}')

# ============================================================================
# DATA TRANSFORMS (same as original pipeline)
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
# STEP 1: BUILD CROP HIERARCHY FROM FOLDER NAMES
# ============================================================================
print('\n' + '='*70)
print('  BUILDING CROP HIERARCHY FROM DATASET')
print('='*70)

full_dataset = datasets.ImageFolder(DATA_DIR, transform=val_test_transforms)
original_classes = full_dataset.classes  # 38 alphabetical classes
original_class_to_idx = full_dataset.class_to_idx

# Parse crop___condition from folder names
crop_to_conditions = {}
original_idx_to_crop = {}
original_idx_to_condition = {}

for cls_name, cls_idx in original_class_to_idx.items():
    parts = cls_name.split('___')
    crop = parts[0]
    condition = parts[1] if len(parts) > 1 else 'unknown'
    
    if crop not in crop_to_conditions:
        crop_to_conditions[crop] = []
    crop_to_conditions[crop].append((condition, cls_idx))
    original_idx_to_crop[cls_idx] = crop
    original_idx_to_condition[cls_idx] = condition

# Create crop-level mapping (Stage 1)
crop_names = sorted(crop_to_conditions.keys())
crop_to_crop_idx = {c: i for i, c in enumerate(crop_names)}
num_crops = len(crop_names)

print(f'\nStage 1 Classes (Crops): {num_crops}')
for crop in crop_names:
    conditions = crop_to_conditions[crop]
    print(f'  {crop_to_crop_idx[crop]:2d}. {crop} -> {len(conditions)} conditions: '
          f'{[c for c, _ in conditions]}')

# Create condition-level mapping per crop (Stage 2)
crop_condition_maps = {}
for crop in crop_names:
    conditions = sorted(crop_to_conditions[crop], key=lambda x: x[0])
    cond_names = [c for c, _ in conditions]
    cond_to_idx = {c: i for i, c in enumerate(cond_names)}
    original_idxs = [idx for _, idx in conditions]
    crop_condition_maps[crop] = {
        'condition_names': cond_names,
        'condition_to_idx': cond_to_idx,
        'original_class_idxs': original_idxs,
        'num_conditions': len(cond_names),
    }

# ============================================================================
# STEP 2: CREATE CROP-LEVEL LABELS FOR EACH IMAGE
# ============================================================================
print('\n' + '='*70)
print('  CREATING CROP-LEVEL LABELS')
print('='*70)

# For each sample in full_dataset, map original label -> crop label
all_crop_labels = []
for _, original_label in full_dataset.samples:
    crop = original_idx_to_crop[original_label]
    all_crop_labels.append(crop_to_crop_idx[crop])

total_images = len(full_dataset)
print(f'Total images: {total_images}')
print(f'Crop label distribution:')
crop_dist = Counter(all_crop_labels)
for crop_idx in sorted(crop_dist.keys()):
    print(f'  {crop_names[crop_idx]:15s} (ID {crop_idx:2d}): {crop_dist[crop_idx]:6d} images')

# ============================================================================
# STEP 3: SPLIT DATASET (same seed=42, same 75/15/10 ratio)
# ============================================================================
print('\n' + '='*70)
print('  SPLITTING DATASET (75/15/10, seed=42)')
print('='*70)

test_size = int(total_images * TEST_SPLIT)
val_size = int(total_images * VAL_SPLIT)
train_size = total_images - val_size - test_size

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(RANDOM_SEED)
)

val_dataset_proper = datasets.ImageFolder(DATA_DIR, transform=val_test_transforms)
test_dataset_proper = datasets.ImageFolder(DATA_DIR, transform=val_test_transforms)

print(f'Train: {train_size}, Val: {val_size}, Test: {test_size}')

# ============================================================================
# STEP 4: TRAIN STAGE-1 MODEL (14-Class Crop Classifier)
# ============================================================================
print('\n' + '='*70)
print('  STAGE 1: TRAINING 14-CLASS CROP CLASSIFIER (EfficientNet-B4)')
print('='*70)

# Custom dataset wrapper that returns crop labels instead of original labels
class CropLabelDataset(torch.utils.data.Dataset):
    """Wraps a dataset and returns crop-level labels instead of original labels."""
    def __init__(self, base_dataset, idx_to_crop_idx):
        self.base = base_dataset
        self.idx_to_crop_idx = idx_to_crop_idx
    
    def __len__(self):
        return len(self.base)
    
    def __getitem__(self, idx):
        image, original_label = self.base[idx]
        crop_label = self.idx_to_crop_idx[original_label]
        return image, crop_label

# Build crop label mapping: original_class_idx -> crop_idx
orig_to_crop_idx = {}
for cls_idx in range(len(original_classes)):
    crop = original_idx_to_crop[cls_idx]
    orig_to_crop_idx[cls_idx] = crop_to_crop_idx[crop]

# Create crop-label datasets
train_crop_ds = CropLabelDataset(
    Subset(datasets.ImageFolder(DATA_DIR, transform=train_transforms), train_dataset.indices),
    orig_to_crop_idx
)
val_crop_ds = CropLabelDataset(
    Subset(val_dataset_proper, val_dataset.indices),
    orig_to_crop_idx
)
test_crop_ds = CropLabelDataset(
    Subset(test_dataset_proper, test_dataset.indices),
    orig_to_crop_idx
)

# DataLoaders for Stage 1
train_crop_loader = DataLoader(train_crop_ds, batch_size=BATCH_SIZE, shuffle=True,
                                num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
val_crop_loader = DataLoader(val_crop_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
test_crop_loader = DataLoader(test_crop_ds, batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=NUM_WORKERS, pin_memory=True)

# Build Stage-1 model (EfficientNet-B4 for 14 crops)
def build_stage1_model():
    """EfficientNet-B4 for 14-class crop prediction."""
    try:
        from efficientnet_pytorch import EfficientNet
        model = EfficientNet.from_pretrained('efficientnet-b4')
        in_features = model._fc.in_features
        model._fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_crops),  # 14 crops
        )
    except ImportError:
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_crops),  # 14 crops
        )
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Stage-1 Model: {total_params:,} trainable params, {num_crops} output classes')
    return model


def train_model(model, model_name, train_loader, val_loader, output_dir, 
                num_output_classes, epochs=EPOCHS, lr=LEARNING_RATE):
    """Generic training function for any stage."""
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
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    total_start = time.time()
    
    for epoch in range(epochs):
        # Train
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
        
        train_loss = running_loss / total
        train_acc = 100.0 * correct / total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # Validate
        model.eval()
        val_loss_sum, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=(MIXED_PRECISION and DEVICE.type == 'cuda')):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                val_loss_sum += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss_sum / max(val_total, 1)
        val_acc = 100.0 * val_correct / max(val_total, 1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        scheduler.step()
        
        print(f'  [{model_name}] Epoch {epoch+1}/{epochs} | '
              f'Train: {train_acc:.1f}% | Val: {val_acc:.1f}% | '
              f'Gap: {train_acc-val_acc:.1f}%')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
            torch.save({'model_state_dict': model.state_dict(),
                        'val_acc': val_acc, 'model_name': model_name},
                       output_path / 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f'  >> Early stopping at epoch {epoch+1}')
                break
    
    total_time = time.time() - total_start
    model.load_state_dict(best_model_wts)
    print(f'  [{model_name}] Best Val Acc: {best_val_acc:.1f}%, Time: {total_time/60:.1f} min')
    
    return model, history, best_val_acc, total_time


# Train Stage 1
stage1_output = os.path.join(OUTPUT_BASE, 'Stage1_Crop')
stage1_model = build_stage1_model()
stage1_model, stage1_history, stage1_val_acc, stage1_time = train_model(
    stage1_model, 'Stage1-Crop-EfficientNet', 
    train_crop_loader, val_crop_loader, stage1_output,
    num_output_classes=num_crops
)

# Evaluate Stage 1 on test set
print('\n--- Stage 1 Test Evaluation ---')
stage1_model.eval()
stage1_preds, stage1_labels = [], []
with torch.no_grad():
    for images, labels in test_crop_loader:
        images = images.to(DEVICE, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=(MIXED_PRECISION and DEVICE.type == 'cuda')):
            outputs = stage1_model(images)
        _, predicted = outputs.max(1)
        stage1_preds.extend(predicted.cpu().numpy())
        stage1_labels.extend(labels.numpy())

stage1_preds = np.array(stage1_preds)
stage1_labels = np.array(stage1_labels)
stage1_acc = accuracy_score(stage1_labels, stage1_preds) * 100
print(f'Stage 1 Crop Accuracy: {stage1_acc:.2f}%')
print(f'\nStage 1 Classification Report:')
print(classification_report(stage1_labels, stage1_preds, target_names=crop_names))

np.save(os.path.join(stage1_output, 'test_crop_predictions.npy'), stage1_preds)
np.save(os.path.join(stage1_output, 'test_crop_ground_truth.npy'), stage1_labels)


# ============================================================================
# STEP 5: TRAIN STAGE-2 MODELS (Per-Crop Condition Classifiers)
# ============================================================================
print('\n' + '='*70)
print('  STAGE 2: TRAINING PER-CROP CONDITION CLASSIFIERS')
print('='*70)

# Only train Stage 2 for crops with >1 condition
multi_condition_crops = {crop: info for crop, info in crop_condition_maps.items() 
                         if info['num_conditions'] > 1}

print(f'\nCrops needing Stage 2 classifier: {len(multi_condition_crops)}')
for crop, info in multi_condition_crops.items():
    print(f'  {crop}: {info["num_conditions"]} conditions -> {info["condition_names"]}')

stage2_models = {}
stage2_results = {}

for crop, info in multi_condition_crops.items():
    print(f'\n{"="*70}')
    print(f'  Stage 2: Training {crop.upper()} classifier ({info["num_conditions"]} conditions)')
    print(f'{"="*70}')
    
    # Filter indices belonging to this crop
    crop_original_idxs = set(info['original_class_idxs'])
    
    # Build mapping: original_class_idx -> local_condition_idx
    orig_to_local = {}
    for cond_name, cond_idx in info['condition_to_idx'].items():
        for orig_cond, orig_idx in crop_to_conditions[crop]:
            if orig_cond == cond_name:
                orig_to_local[orig_idx] = cond_idx
    
    # Filter train/val/test indices for this crop
    train_crop_indices = [idx for idx in train_dataset.indices 
                          if full_dataset.targets[idx] in crop_original_idxs]
    val_crop_indices = [idx for idx in val_dataset.indices 
                        if full_dataset.targets[idx] in crop_original_idxs]
    test_crop_indices = [idx for idx in test_dataset.indices 
                         if full_dataset.targets[idx] in crop_original_idxs]
    
    print(f'  {crop}: Train={len(train_crop_indices)}, Val={len(val_crop_indices)}, '
          f'Test={len(test_crop_indices)}')
    
    if len(train_crop_indices) < 10 or len(test_crop_indices) < 5:
        print(f'  Skipping {crop}: too few samples')
        continue
    
    # Create datasets with local condition labels
    class ConditionLabelDataset(torch.utils.data.Dataset):
        def __init__(self, base_ds, indices, orig_to_local_map):
            self.base = base_ds
            self.indices = indices
            self.mapping = orig_to_local_map
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, idx):
            real_idx = self.indices[idx]
            image, orig_label = self.base[real_idx]
            local_label = self.mapping[orig_label]
            return image, local_label
    
    train_cond_ds = ConditionLabelDataset(
        datasets.ImageFolder(DATA_DIR, transform=train_transforms),
        train_crop_indices, orig_to_local
    )
    val_cond_ds = ConditionLabelDataset(
        val_dataset_proper, val_crop_indices, orig_to_local
    )
    test_cond_ds = ConditionLabelDataset(
        test_dataset_proper, test_crop_indices, orig_to_local
    )
    
    train_cond_loader = DataLoader(train_cond_ds, batch_size=BATCH_SIZE, shuffle=True,
                                    num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_cond_loader = DataLoader(val_cond_ds, batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=NUM_WORKERS, pin_memory=True)
    test_cond_loader = DataLoader(test_cond_ds, batch_size=BATCH_SIZE, shuffle=False,
                                   num_workers=NUM_WORKERS, pin_memory=True)
    
    # Build Stage-2 model (lighter ResNet-50 for per-crop)
    num_conds = info['num_conditions']
    s2_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    s2_model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(s2_model.fc.in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(256, num_conds),
    )
    
    s2_output = os.path.join(OUTPUT_BASE, f'Stage2_{crop}')
    s2_model, s2_hist, s2_val_acc, s2_time = train_model(
        s2_model, f'Stage2-{crop}', 
        train_cond_loader, val_cond_loader, s2_output,
        num_output_classes=num_conds, epochs=EPOCHS
    )
    
    # Evaluate Stage 2
    s2_model.eval()
    s2_preds, s2_labels = [], []
    with torch.no_grad():
        for images, labels in test_cond_loader:
            images = images.to(DEVICE, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=(MIXED_PRECISION and DEVICE.type == 'cuda')):
                outputs = s2_model(images)
            _, predicted = outputs.max(1)
            s2_preds.extend(predicted.cpu().numpy())
            s2_labels.extend(labels.numpy())
    
    s2_preds = np.array(s2_preds)
    s2_labels = np.array(s2_labels)
    s2_acc = accuracy_score(s2_labels, s2_preds) * 100
    
    print(f'\n  {crop} Stage-2 Accuracy: {s2_acc:.2f}%')
    print(classification_report(s2_labels, s2_preds, 
          target_names=info['condition_names'], zero_division=0))
    
    stage2_models[crop] = s2_model
    stage2_results[crop] = {
        'accuracy': s2_acc,
        'num_conditions': num_conds,
        'condition_names': info['condition_names'],
        'test_samples': len(test_crop_indices),
    }
    
    np.save(os.path.join(s2_output, 'test_predictions.npy'), s2_preds)
    np.save(os.path.join(s2_output, 'test_ground_truth.npy'), s2_labels)


# ============================================================================
# STEP 6: END-TO-END TWO-STAGE EVALUATION
# ============================================================================
print('\n' + '='*70)
print('  END-TO-END TWO-STAGE EVALUATION')
print('='*70)

# For each test image:
#   1. Stage 1 predicts crop
#   2. If crop has >1 condition, Stage 2 predicts condition
#   3. Map back to original 38-class label
#   4. Compare with original ground truth

test_loader_orig = DataLoader(
    Subset(test_dataset_proper, test_dataset.indices),
    batch_size=1, shuffle=False, num_workers=0, pin_memory=True
)

# Prepare for mapping: for single-condition crops, the answer is fixed
single_condition_crops = {crop: info['original_class_idxs'][0] 
                          for crop, info in crop_condition_maps.items() 
                          if info['num_conditions'] == 1}

hierarchical_preds = []
hierarchical_labels = []
stage1_correct_count = 0
stage2_correct_count = 0
end_to_end_correct = 0

stage1_model.eval()
for s2_model in stage2_models.values():
    s2_model.eval()

print('\nRunning end-to-end two-stage prediction on test set...')

with torch.no_grad():
    for i, (image, original_label) in enumerate(test_loader_orig):
        image = image.to(DEVICE, non_blocking=True)
        original_label = original_label.item()
        true_crop = original_idx_to_crop[original_label]
        true_crop_idx = crop_to_crop_idx[true_crop]
        
        # Stage 1: Predict crop
        with torch.amp.autocast('cuda', enabled=(MIXED_PRECISION and DEVICE.type == 'cuda')):
            crop_logits = stage1_model(image)
        _, pred_crop_idx = crop_logits.max(1)
        pred_crop_idx = pred_crop_idx.item()
        pred_crop = crop_names[pred_crop_idx]
        
        if pred_crop_idx == true_crop_idx:
            stage1_correct_count += 1
        
        # Stage 2: Predict condition within predicted crop
        if pred_crop in stage2_models:
            with torch.amp.autocast('cuda', enabled=(MIXED_PRECISION and DEVICE.type == 'cuda')):
                cond_logits = stage2_models[pred_crop](image)
            _, pred_cond_idx = cond_logits.max(1)
            pred_cond_idx = pred_cond_idx.item()
            pred_cond_name = crop_condition_maps[pred_crop]['condition_names'][pred_cond_idx]
            # Map back to original 38-class label
            pred_original = original_class_to_idx[f'{pred_crop}___{pred_cond_name}']
        elif pred_crop in single_condition_crops:
            pred_original = single_condition_crops[pred_crop]
        else:
            pred_original = -1  # fallback
        
        hierarchical_preds.append(pred_original)
        hierarchical_labels.append(original_label)
        
        if pred_original == original_label:
            end_to_end_correct += 1
        
        if (i + 1) % 1000 == 0:
            print(f'  Processed {i+1}/{test_size} images...')

hierarchical_preds = np.array(hierarchical_preds)
hierarchical_labels = np.array(hierarchical_labels)

e2e_accuracy = end_to_end_correct / test_size * 100
stage1_accuracy = stage1_correct_count / test_size * 100

print(f'\n{"="*70}')
print(f'  TWO-STAGE HIERARCHICAL RESULTS')
print(f'{"="*70}')
print(f'  Stage 1 (Crop Prediction) Accuracy: {stage1_accuracy:.2f}%')
print(f'  End-to-End (Crop + Condition) Accuracy: {e2e_accuracy:.2f}%')
print(f'  Test samples: {test_size}')

# Compare with flat model
print(f'\n  COMPARISON:')
print(f'  Flat 38-class model (EfficientNet-B4):  99.52%')
print(f'  Two-Stage Hierarchical:                 {e2e_accuracy:.2f}%')

if e2e_accuracy < 99.52:
    print(f'\n  NOTE: Two-stage is {99.52 - e2e_accuracy:.2f}% lower because:')
    print(f'  - Stage 1 crop error propagates to Stage 2 (cascading error)')
    print(f'  - If Stage 1 predicts wrong crop, Stage 2 looks at wrong classes')
else:
    print(f'\n  Two-stage approach improved accuracy by {e2e_accuracy - 99.52:.2f}%')

# Save comprehensive results
results = {
    'approach': 'Two-Stage Hierarchical Classification',
    'stage1_crop_classes': num_crops,
    'stage1_crop_accuracy': stage1_accuracy,
    'stage2_results': stage2_results,
    'end_to_end_accuracy': e2e_accuracy,
    'flat_model_accuracy': 99.52,
    'comparison': 'Flat is better' if e2e_accuracy < 99.52 else 'Hierarchical is better',
    'test_samples': test_size,
    'dataset': 'PlantVillage',
    'total_images': total_images,
}

os.makedirs(OUTPUT_BASE, exist_ok=True)
with open(os.path.join(OUTPUT_BASE, 'hierarchical_results.json'), 'w') as f:
    json.dump(results, f, indent=2)

np.save(os.path.join(OUTPUT_BASE, 'hierarchical_test_predictions.npy'), hierarchical_preds)
np.save(os.path.join(OUTPUT_BASE, 'hierarchical_test_ground_truth.npy'), hierarchical_labels)

print(f'\nAll results saved to: {OUTPUT_BASE}/')
print('DONE!')

