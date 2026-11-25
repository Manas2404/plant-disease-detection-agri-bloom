"""
Generate Visual Comparison of Datasets (PlantVillage vs PlantDoc)
and Consolidate All Result Screenshots into a Single Clean Folder
================================================================
Student: Bhavya Sodhi
"""

import os
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLANTDOC_DIR = os.path.join(BASE_DIR, "PlantDoc_Cleaned")
PLANTVILLAGE_DIR = os.path.join(BASE_DIR, "PlantVillage_Cleaned")
CLASSICAL_DIR = os.path.join(BASE_DIR, "Classical_ML_Outputs")
DEST_DIR = os.path.join(BASE_DIR, "ALL_RESULTS_SCREENSHOTS")
ARTIFACT_DIR = r"C:\Users\Admin\.gemini\antigravity-ide\brain\8bfb6b40-0fc0-483a-a46a-b32499278095"

os.makedirs(DEST_DIR, exist_ok=True)

# ----------------------------------------------------------------------------
# 1. Visual Comparison of Actual Images (PlantVillage vs PlantDoc)
# ----------------------------------------------------------------------------
pairs = [
    ("Apple Scab",
     os.path.join(PLANTVILLAGE_DIR, "apple___apple_scab"),
     os.path.join(PLANTDOC_DIR, "train", "Apple Scab Leaf")),
    ("Corn Common Rust",
     os.path.join(PLANTVILLAGE_DIR, "corn___common_rust"),
     os.path.join(PLANTDOC_DIR, "train", "Corn rust leaf")),
    ("Tomato Early Blight",
     os.path.join(PLANTVILLAGE_DIR, "tomato___early_blight"),
     os.path.join(PLANTDOC_DIR, "train", "Tomato Early blight leaf")),
    ("Potato Early Blight",
     os.path.join(PLANTVILLAGE_DIR, "potato___early_blight"),
     os.path.join(PLANTDOC_DIR, "train", "Potato leaf early blight")),
]

fig, axes = plt.subplots(4, 2, figsize=(11, 14), dpi=150)
fig.patch.set_facecolor('#F8F9FA')

for i, (title, pv_dir, pd_dir) in enumerate(pairs):
    # PlantVillage Image
    pv_img_name = sorted(os.listdir(pv_dir))[0]
    pv_img_path = os.path.join(pv_dir, pv_img_name)
    with Image.open(pv_img_path) as img:
        axes[i, 0].imshow(img)
    axes[i, 0].set_title(f"PlantVillage: {title}\n(Clean Lab Background | Homogeneous Lighting)",
                         fontsize=10, fontweight='bold', color='#1B5E20')
    axes[i, 0].axis('off')
    
    # PlantDoc Image
    pd_img_name = sorted(os.listdir(pd_dir))[0]
    pd_img_path = os.path.join(pd_dir, pd_img_name)
    with Image.open(pd_img_path) as img:
        axes[i, 1].imshow(img)
    axes[i, 1].set_title(f"PlantDoc: {title}\n(In-The-Wild Field Conditions | Complex Background)",
                         fontsize=10, fontweight='bold', color='#B71C1C')
    axes[i, 1].axis('off')

plt.suptitle("VISUAL COMPARISON: PlantVillage (Laboratory) vs. PlantDoc (In-The-Wild Field)\n"
             "Why Lab Models Reach 99.5% While Field Classification Drops to 35%–45%",
             fontsize=13, fontweight='bold', y=0.99)
plt.tight_layout()

visual_comp_path = os.path.join(DEST_DIR, "01_Dataset_Visual_Comparison_Lab_vs_Field.png")
fig.savefig(visual_comp_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"[SAVED] {visual_comp_path}")


# ----------------------------------------------------------------------------
# 2. Accuracy Drop Analysis (Lab vs Field Comparison)
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
fig.patch.set_facecolor('#FFFFFF')

models = ['Linear SVM', 'Random Forest', 'Deep Learning CNN\n(Literature ResNet / MobileNet)']
lab_scores = [90.53, 83.82, 99.45]
field_scores = [44.49, 35.59, 38.00]  # Field scores from our experiments and Singh et al. (2019)

x = np.arange(len(models))
width = 0.35

rects1 = ax.bar(x - width/2, lab_scores, width, label='PlantVillage (Controlled Lab)',
                color='#2E7D32', edgecolor='black', linewidth=1.2)
rects2 = ax.bar(x + width/2, field_scores, width, label='PlantDoc (Real Field)',
                color='#D32F2F', edgecolor='black', linewidth=1.2)

for bar in rects1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.2,
            f"{bar.get_height():.1f}%", ha='center', va='bottom', fontweight='bold', fontsize=11)
for bar in rects2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.2,
            f"{bar.get_height():.1f}%", ha='center', va='bottom', fontweight='bold', fontsize=11)

# Annotate drops
for i in range(len(models)):
    drop = lab_scores[i] - field_scores[i]
    ax.annotate(f"Drop:\n-{drop:.1f}%",
                xy=(x[i] + width/2, field_scores[i] + 4),
                xytext=(x[i], field_scores[i] + 25),
                ha='center', fontsize=10, fontweight='bold', color='#880E4F',
                arrowprops=dict(arrowstyle="->", color='#880E4F', lw=1.5))

ax.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Generalization Gap: Laboratory vs. Real Agricultural Field\n(Same Crop Diseases Evaluated on Both Datasets)',
             fontsize=13, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.set_ylim(0, 115)
ax.legend(loc='upper right', frameon=True, fontsize=11)
ax.grid(axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()
drop_path = os.path.join(DEST_DIR, "08_Accuracy_Drop_Analysis_Lab_vs_Field.png")
fig.savefig(drop_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"[SAVED] {drop_path}")


# ----------------------------------------------------------------------------
# 3. Consolidate & Number All Result Screenshots
# ----------------------------------------------------------------------------
copy_map = {
    os.path.join(CLASSICAL_DIR, "Grand_Model_Leaderboard.png"):
        os.path.join(DEST_DIR, "02_Grand_Model_Leaderboard.png"),
    os.path.join(CLASSICAL_DIR, "Master_Classical_ML_Comparison.png"):
        os.path.join(DEST_DIR, "03_Master_Classical_ML_Comparison.png"),
    os.path.join(CLASSICAL_DIR, "PlantVillage_SVM_Confusion_Matrix.png"):
        os.path.join(DEST_DIR, "04_PlantVillage_SVM_Confusion_Matrix.png"),
    os.path.join(CLASSICAL_DIR, "PlantVillage_RF_Confusion_Matrix.png"):
        os.path.join(DEST_DIR, "05_PlantVillage_RF_Confusion_Matrix.png"),
    os.path.join(CLASSICAL_DIR, "PlantDoc_SVM_Confusion_Matrix.png"):
        os.path.join(DEST_DIR, "06_PlantDoc_SVM_Confusion_Matrix.png"),
    os.path.join(CLASSICAL_DIR, "PlantDoc_RF_Confusion_Matrix.png"):
        os.path.join(DEST_DIR, "07_PlantDoc_RF_Confusion_Matrix.png"),
}

for src, dst in copy_map.items():
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"[COPIED] {os.path.basename(dst)}")

# Also copy the summary json and txt reports
for fname in os.listdir(CLASSICAL_DIR):
    if fname.endswith(('.txt', '.json')):
        shutil.copy2(os.path.join(CLASSICAL_DIR, fname), os.path.join(DEST_DIR, fname))

# Copy everything to Artifact Directory for Markdown rendering
if os.path.exists(ARTIFACT_DIR):
    for fname in os.listdir(DEST_DIR):
        if fname.endswith('.png'):
            shutil.copy2(os.path.join(DEST_DIR, fname), os.path.join(ARTIFACT_DIR, fname))

print(f"\nAll images and reports consolidated in: {DEST_DIR}")
