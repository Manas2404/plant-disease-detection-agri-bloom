# Capstone Project CSET400C — Research Paper Sections
**Author:** Bhavya Sodhi  
**Project:** Plant Disease Classification Using Deep Learning & Classical Machine Learning  
**Formatting Standard:** Times New Roman, 12pt, Single Spaced  
**Target:** IEEE / Springer Conference Proceedings  

---

## 3. Research Methodology

This section describes the research methodology used in this project. It explains the datasets used for training and testing, the preprocessing steps applied to prepare the images, the deep learning and classical machine learning models selected for classification, and the evaluation metrics used to measure model accuracy and stability.

### 3.1 Datasets

To test how well computer vision models work in different agricultural conditions, we used two separate datasets. The first is a clean laboratory dataset, and the second is an unconstrained real-world field dataset.

* **PlantVillage Dataset (Laboratory Controlled):**  
  The PlantVillage dataset is a standard benchmark in agricultural artificial intelligence. It contains 54,284 high-resolution leaf images categorized into 38 distinct classes. These classes cover 14 different crop species, including Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, and Tomato. The dataset includes 26 different fungal, bacterial, and viral diseases, along with healthy leaf samples. All images were taken inside a controlled laboratory environment with uniform artificial gray or black backgrounds and fixed lighting conditions.  
  *Footnote 1: PlantVillage Dataset URL: https://github.com/spMohanty/PlantVillage-Dataset*

* **PlantDoc Dataset (In-the-Wild Real Field):**  
  The PlantDoc dataset is a challenging benchmark designed to test disease detection under authentic farming conditions. Unlike PlantVillage, PlantDoc images were photographed outdoors in real fields. The images contain complex backgrounds with soil, weeds, other crops, strong sunlight, shadows, and human hands holding leaves. The original raw dataset contained 2,540 images across 27 classes and 13 crop species. After applying our automated noise cleaning and duplicate removal pipeline, the final cleaned benchmark consists of 2,174 images across all 27 categories.  
  *Footnote 2: PlantDoc Dataset URL: https://github.com/pratikkayal/PlantDoc-Dataset*

#### Table 1: Structural Comparison of Evaluated Agricultural Datasets
| Dataset Attribute | PlantVillage (Laboratory) | PlantDoc (Final Cleaned) |
| :--- | :--- | :--- |
| **Environmental Domain** | Controlled Laboratory (Studio) | In-the-Wild Real Field |
| **Background Type** | Uniform Artificial (Black/Gray) | Complex Natural Soil & Weeds |
| **Total Number of Images** | **54,284** | **2,174** |
| **Total Number of Classes** | **38** | **27** |
| **Number of Crops** | 14 Major Species | 13 Crop Species |
| **Evaluation Protocol** | 10-Fold Stratified CV & 75/15/10 Split | 10-Fold Stratified Cross-Validation |
| **Public Reference** | Hughes & Salathé (2015) | Singh et al. (2019) |

---

### 3.2 Preprocessing

Raw agricultural images cannot be used directly for model training because they contain noise, duplicates, and size differences. We applied a step-by-step preprocessing pipeline to make the data clean, consistent, and ready for modeling:

1. **Duplicate Image Removal:**  
   Online image datasets often contain duplicate files. We computed cryptographic MD5 hashes for all images. Identical duplicate files (360 duplicates in PlantDoc) were removed to prevent unfair memory bias during evaluation.
2. **Hand Artifact and Blur Filtering:**  
   In field images, human hands frequently hold leaves, which distracts the model from actual disease features. Images where hands or non-foliar objects occupied more than 50% of the area, or where heavy motion blur was present, were removed (366 images pruned), leaving a clean set of 2,174 images.
3. **Resizing and Center Cropping:**  
   All images were resized to 256 × 256 pixels and center-cropped to 224 × 224 pixels. This matches the standard input size needed by convolutional neural networks and Vision Transformers.
4. **Channel Normalization:**  
   Pixel brightness was normalized using standard ImageNet mean values (0.485, 0.456, 0.406) and standard deviations (0.229, 0.224, 0.225) across the Red, Green, and Blue channels.
5. **Data Augmentation (Training Only):**  
   To help models generalize and avoid memorizing training images, augmentations were applied only to training folds: random cropping, horizontal flips, vertical flips, small rotations up to 20 degrees, and slight color brightness changes.
6. **Leakage-Free Partitioning:**  
   For all classical machine learning models, 10-Fold Stratified Cross-Validation was implemented. The feature scaler was fitted strictly on training folds, ensuring that zero test fold information leaked into training.

---

### 3.3 Methods

We investigated two distinct classification paradigms: modern end-to-end Deep Learning models and Classical Machine Learning models powered by deep feature representations.

* **ResNet-50 (Residual Network Baseline):**  
  ResNet-50 is a 50-layer convolutional neural network containing 25.5 million parameters. It uses skip connections that let information jump across layers, which prevents gradients from vanishing during backpropagation. We replaced the final layer with a custom classifier head containing Dropout (0.4), a 512-unit hidden layer with ReLU activation, Dropout (0.2), and a final 38-class linear output layer.
* **EfficientNet-B4 (Compound-Scaled CNN):**  
  EfficientNet-B4 uses compound scaling, which balances network depth, width, and image resolution simultaneously using a fixed formula. It has 19.3 million parameters and uses Mobile Inverted Bottleneck Convolution (MBConv) blocks with Squeeze-and-Excitation attention, allowing it to capture subtle disease spots efficiently with fewer parameters.
* **Vision Transformer (ViT-Base-Patch16-224):**  
  ViT replaces standard convolutions with self-attention mechanisms. The 224 × 224 input image is split into 196 non-overlapping patches of 16 × 16 pixels. Each patch is projected into a 768-dimensional token vector. Using 12 Multi-Head Self-Attention layers (86.6 million parameters), ViT learns relationships between all parts of the leaf simultaneously.
* **ResNet-50 Feature Extraction for Classical ML:**  
  Classical models cannot process raw pixels directly. We passed every image through a pretrained ResNet-50 backbone (with the classification head removed) to extract a dense 2048-dimensional feature vector describing leaf color and texture. Features were standardized using StandardScaler fitted strictly on training folds.
* **Random Forest (RF):**  
  An ensemble classifier of 150 decision trees. Each tree was trained on bootstrap samples of the data and constrained to a maximum depth of 25, using the square root of total features (45 features) at each split to prevent overfitting.
* **Linear Support Vector Machine (Linear SVM):**  
  Trained using Stochastic Gradient Descent (SGD) with hinge loss and L2 regularization penalty (alpha = 0.0001). Linear SVM finds the best maximum-margin hyperplane in the continuous 2048-dimensional feature space.

---

### 3.4 Evaluation Metrics

To measure model performance thoroughly and fairly, we used five standard evaluation metrics:

1. **Accuracy:**  
   The percentage of total leaf images correctly identified across all classes. It gives an overall measure of system performance.
2. **Precision:**  
   Measures how many leaves predicted as a specific disease actually had that disease. High precision prevents false alarms and avoids unnecessary pesticide application.
3. **Recall (Sensitivity):**  
   Measures how many actual diseased leaves the model successfully detected. High recall ensures that infections are caught early and not missed.
4. **F1-Score:**  
   The harmonic mean of precision and recall. We report Weighted F1 (which weights each class by its sample count) and Macro F1 (which weights all classes equally).
5. **10-Fold Stratified Cross-Validation:**  
   The dataset is divided into 10 balanced folds with identical class proportions. The model trains on 9 folds and tests on 1 fold, repeating 10 times so every image is tested. The average and standard deviation are reported to ensure results do not depend on a single random data split.

---

## 4. Evaluation Results and Discussion

#### Table 2: Deep Learning Benchmark Results on PlantVillage Dataset (38 Classes)
| Model Architecture | Parameters | Test Accuracy (%) | Weighted F1 (%) | Training Time |
| :--- | :--- | :--- | :--- | :--- |
| **EfficientNet-B4** | 19.3 M | **99.52%** | **99.51%** | ~110 min |
| **ResNet-50** | 25.5 M | **99.45%** | **99.44%** | ~85 min |
| **ViT-Base-Patch16** | 86.6 M | **99.38%** | **99.37%** | ~140 min |

#### Table 3: 10-Fold Stratified Cross-Validation Results on PlantVillage (54,284 Images)
| Model | Mean Accuracy (%) | Std Dev (±%) | Min Accuracy (%) | Max Accuracy (%) | Mean Weighted F1 (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Linear SVM** | **95.30%** | **±0.21%** | 95.04% | 95.71% | **95.25% ± 0.21%** |
| **Random Forest** | **89.60%** | **±0.31%** | 89.15% | 90.13% | **88.68% ± 0.35%** |

#### Table 4: 10-Fold Stratified Cross-Validation Results on PlantDoc Final Cleaned (2,174 Images)
| Model | Mean Accuracy (%) | Std Dev (±%) | Min Accuracy (%) | Max Accuracy (%) | Mean Weighted F1 (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Linear SVM** | **44.89%** | **±0.84%** | 43.58% | 46.33% | **44.27% ± 1.03%** |
| **Random Forest** | **40.11%** | **±1.66%** | 36.87% | 42.86% | **35.93% ± 1.93%** |

---

### Paragraph 1 (Controlled Benchmark Results & Model Comparison — 205 words)
The experimental results on the controlled PlantVillage dataset show that deep learning models perform very well when leaf images are clean. EfficientNet-B4 achieved the highest test accuracy of 99.52% and a Weighted F1-Score of 99.51%. ResNet-50 achieved 99.45% accuracy and 99.44% F1-Score, while the Vision Transformer achieved 99.38% accuracy. These high scores show that convolutional neural networks are very good at learning leaf disease symptoms from uniform backgrounds. EfficientNet-B4 performed best because its compound scaling balances network depth, width, and image resolution efficiently. For classical machine learning models trained on ResNet-50 feature vectors, 10-Fold Stratified Cross-Validation produced reliable results across all 54,284 images. Linear SVM achieved a mean accuracy of 95.30% with a very low standard deviation of only 0.21%. Random Forest achieved a mean accuracy of 89.60% with a standard deviation of 0.31%. Linear SVM performed 5.7 percentage points better than Random Forest. This happens because the 2048-dimensional feature vectors from ResNet-50 are easily separated by linear hyperplanes. Random Forest works on one feature at a time with decision trees, which makes it less effective for continuous deep features. In summary, both deep learning and classical models give strong results on clean laboratory data, confirming that transfer learning works well for plant disease recognition.

### Paragraph 2 (In-the-Wild Field Results & Domain Gap Analysis — 204 words)
When we tested the models on the real-world PlantDoc dataset, classification accuracy dropped significantly. In our 10-Fold Stratified Cross-Validation across all 2,174 cleaned field images, Linear SVM achieved a mean accuracy of 44.89% with a standard deviation of 0.84%. Random Forest achieved a mean accuracy of 40.11% with a standard deviation of 1.66%. The weighted F1-Score was 44.27% for Linear SVM and 35.93% for Random Forest. Compared to the 95.30% accuracy on PlantVillage, Linear SVM dropped by more than 50 percentage points on PlantDoc. This large performance decrease happens because field images have complex natural backgrounds with soil, weeds, and varying sunlight. Studio-trained models learn to rely on plain backgrounds, so they get confused by outdoor visual noise and overlapping leaves. Furthermore, field images often show multiple leaves at different distances, which makes symptom identification much harder. In addition, real agricultural images suffer from motion blur and shadow occlusions during outdoor photography. However, our data cleaning pipeline helped improve field performance. Removing duplicate images and human hand occlusions increased Random Forest accuracy from 35.59% on the raw dataset to 40.11% on the cleaned dataset. This improvement of 4.52 percentage points proves that data quality cleaning is necessary before deploying computer vision models in agriculture.

### Paragraph 3 (Cross-Validation Stability & Methodology Discussion — 204 words)
A central goal of our investigation was to evaluate whether earlier results depended on an accidental lucky data split with seed 42. By completing 10-Fold Stratified Cross-Validation on both datasets, every single image served as an unseen test sample exactly once. The cross-validation findings confirm that model performance is very stable and not biased by data partitioning. On PlantVillage, Linear SVM accuracy stayed between 95.04% and 95.71% across all 10 folds, giving a tiny spread of only 0.67 percentage points. Random Forest varied between 89.15% and 90.13%, with a spread under one percentage point. On the noisy PlantDoc dataset, Linear SVM accuracy ranged from 43.58% to 46.33% with a standard deviation of 0.84%, while Random Forest ranged from 36.87% to 42.86%. These narrow standard deviations across ten folds prove that the 2048-dimensional ResNet-50 features are stable across different data subsets. Moreover, the single-split test accuracy of 44.39% for SVM falls right inside the 10-fold range. This proves that seed 42 did not introduce unfair bias. Therefore, our reported accuracy numbers are statistically dependable and show the true capabilities of the models. These results provide strong evidence that our experimental workflow is rigorous and that the observed models can generalize consistently within each data distribution.

---

## 5. Conclusion and Future Work (310 words)

In this capstone project, we conducted a rigorous comparative evaluation of deep learning architectures and classical machine learning algorithms for plant leaf disease classification. We analyzed EfficientNet-B4, ResNet-50, and Vision Transformer alongside Random Forest and Linear SVM across two complementary datasets. The laboratory-controlled PlantVillage dataset provided 54,284 clean leaf images across 38 classes, while the in-the-wild PlantDoc dataset provided 2,174 real field images across 27 classes. On the PlantVillage dataset, EfficientNet-B4 achieved the best overall performance with 99.52% test accuracy, followed closely by ResNet-50 at 99.45% and ViT-Base at 99.38%. Applying 10-Fold Stratified Cross-Validation on ResNet-50 feature embeddings, Linear SVM achieved a stable mean accuracy of 95.30% (±0.21%) and Random Forest reached 89.60% (±0.31%). However, cross-domain evaluation on real-world PlantDoc field images revealed a steep performance decline to 44.89% (±0.84%) for Linear SVM and 40.11% (±1.66%) for Random Forest. This performance gap of approximately 50 percentage points demonstrates that models trained in studio conditions struggle severely with soil backgrounds, shifting sunlight, and leaf occlusions. Our data cleaning pipeline improved field accuracy by 4.52 percentage points, proving that removing duplicate files and hand occlusions is critical. 

For future work, six promising research directions should be explored. First, domain adaptation methods like adversarial training and style transfer should be implemented to bridge the gap between laboratory and field images. Second, end-to-end deep learning models should be fine-tuned on larger open-field datasets. Third, the trained models should be exported to ONNX format and quantized for mobile devices, enabling farmers to perform real-time offline disease diagnosis using ordinary smartphones. Fourth, drone-assisted photography can collect aerial foliar images over entire agricultural fields. Fifth, hybrid ensemble architectures combining convolutional networks with transformer attention should be studied for leaves displaying multiple simultaneous diseases. Finally, connecting the automated classifier to an advisory system providing fungicide and treatment guidance will make this technology directly beneficial for sustainable crop management.
