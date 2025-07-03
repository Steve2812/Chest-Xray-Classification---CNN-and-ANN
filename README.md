# 🩺 Chest X-ray Pneumonia Detection Using CNN

This project uses deep learning to classify chest X-ray images as either **Normal** or **Pneumonia**. It leverages a custom Convolutional Neural Network (CNN) trained on a curated dataset of X-rays, with optimizations for class imbalance and hardware acceleration (TPUs/GPUs).

---

## 📌 Problem Statement

Pneumonia is a serious lung infection that can be detected in chest X-rays. Early and accurate diagnosis is critical, especially in resource-limited settings. This project aims to automate the classification of chest X-ray images using a CNN model, helping reduce diagnostic burden and improve speed and accuracy in detection.

---

## 📂 Dataset

The dataset includes chest X-ray images from two classes:
- `NORMAL`
- `PNEUMONIA`

Data is organized into `train/`, `val/`, and `test/` directories. However, due to very limited samples in `val/`, the project merges `train/` and `val/` and creates a new 80/20 train-validation split.

---

## 🛠️ Setup and Environment

- TensorFlow 2.x
- Python 3.7+
- Google Colab / Jupyter Notebook
- TPU/GPU-compatible
- Google Drive mounted for data access

---

## 🧪 Preprocessing Steps

- Rescale pixel values to [0, 1]
- Resize all images to `160x160`
- Apply real-time data augmentation (Zoom, Shear, Flip)
- Handle corrupted images with fallback dummy tensors
- Custom image-label pipeline with `tf.data` for performance

---

## 🧠 Model Architecture

- `Rescaling + RandomZoom + RandomShear + RandomFlip`
- 2 × `Conv2D → BatchNorm → Dropout → MaxPooling`
- `Flatten → Dense (512 → 256 → 128) with Dropout & BatchNorm`
- Final layer: `Dense(1, activation='sigmoid')` for binary classification

---

## ⚖️ Class Imbalance Handling

- Calculated class weights based on distribution of Normal vs. Pneumonia
- Applied `class_weight` during training to penalize the overrepresented class
- Ensures better learning on minority class (Normal images)

---

## 🚀 Training Details

- Optimizer: `Adam`
- Loss Function: `Binary Crossentropy`
- Metrics: `Accuracy`, `AUC`
- Epochs: `25` with `EarlyStopping(patience=5)`
- Batch size scaled with TPU/GPU strategy
- Evaluation performed on a separate test set

---

## 📊 Results

Model evaluated using:
- **Accuracy**
- **AUC (Area Under ROC Curve)**
- Final results printed using `model.evaluate()`

> ✅ The model shows strong performance on test data, effectively distinguishing pneumonia cases from normal chest X-rays.

---

## 📈 Next Steps (Optional Ideas)

- Apply transfer learning using MobileNetV2 or EfficientNet
- Tune hyperparameters (batch size, dropout rates, optimizer)
- Visualize Grad-CAM for model interpretability
- Convert model to TFLite for mobile deployment

---

## 📎 Credits

- Dataset: [Kaggle Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- Tools: TensorFlow, Google Colab, NumPy, Pandas, Matplotlib

---

## 🧾 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
