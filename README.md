# 🍃 Optimized Vision Transformer (ViT) for Plant Disease Detection

This project is a **PyTorch** implementation of an optimized Vision Transformer model, based on the research paper *"Optimized Vision Transformers for Superior Plant Disease Detection"*.

The goal is to build a plant disease recognition model that achieves **up to 99.77% accuracy** on the PlantVillage dataset, while remaining lightweight (~13M parameters) — making it suitable for deployment on resource-constrained **Edge/IoT devices**.

---

## ✨ Architecture Highlights

Unlike Google's original ViT-Base, this model has been carefully tuned in its hyperparameters and core components to maximize performance on agricultural data:

| Component | Detail |
|---|---|
| **Image Size** | 224 × 224 |
| **Patch Size** | 16 |
| **Embedding Dimension** | 512 |
| **Depth** | 6 |
| **Attention Heads** | 8 |
| **MLP Dimension** | 1024 |
| **Positional Encoding** | Fixed Sine/Cosine (instead of Learnable Parameters) |
| **Activation Function** | ReLU (instead of GELU) |
| **Total Parameters** | ~13.02M (~7× lighter than ViT-Base, ~10× lighter than VGG19) |

Key design choices:
- **Fixed Sine/Cosine Positional Encoding** — better preserves spatial information and reduces model size compared to learnable positional embeddings.
- **ReLU Activation** in the Feed-Forward Network — a simpler, faster alternative to GELU that maintains strong performance.
- **Compact Design** — at just ~13M parameters, the model is practical for real-world agricultural deployment.

---

## 📁 Project Structure

```text
vit_pytorch_project/
├── data/                          # (Create manually) Image dataset directory
│   ├── Train/                     # Training set (organized by class folders)
│   └── Test/                      # Test set (organized by class folders)
├── dataloaders/                   # Data input pipeline
│   ├── __init__.py
│   ├── dataset.py                 # Custom DataLoader using ImageFolder
│   └── transforms.py              # Data Augmentation (Flip, Rotation, Affine, Crop)
├── models/                        # Model architecture
│   ├── __init__.py
│   ├── patch_embed.py             # Image-to-Patch splitting & Linear Projection
│   ├── transformer.py             # Multi-Head Attention & MLP Blocks
│   └── vit.py                     # Full ViT model assembly
├── utils/                         # Helper utilities
│   ├── __init__.py
│   ├── engine.py                  # Train/Eval loop
│   └── helpers.py                 # Random seed setup
├── weights/                       # (Auto-created) Saved model checkpoints (.pth)
├── train.py                       # Main training script
├── eval.py                        # Evaluation script & Confusion Matrix plotting
└── README.md
```

---

## 🚀 Installation & Usage

### 1. System Requirements

- **Python:** 3.11 or 3.12
- **PyTorch:** CUDA-enabled build (CUDA 12.1 or 12.4 recommended for faster training)
- **Other libraries:** `torchvision`, `scikit-learn`, `matplotlib`, `seaborn`, `tqdm`

Install all dependencies:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install scikit-learn matplotlib seaborn tqdm
```

### 2. Prepare the Dataset

Organize your image files inside the `data/` directory as follows. The subfolder names will automatically be used as **class labels**:

```text
data/
  Train/
    Bacterial_spot/
    Healthy/
    Powdery_mildew/
    ...
  Test/
    Bacterial_spot/
    Healthy/
    Powdery_mildew/
    ...
```

### 3. Train the Model

The model uses the **AdamW optimizer** with a learning rate of `0.0001`. Start training with:

```bash
python train.py
```

The best-performing model checkpoint will be saved automatically to `weights/optimized_vit_best.pth`.

### 4. Evaluate the Model

To assess model performance on the test set and generate a detailed report (Accuracy, Precision, Recall, F1-Score) along with a Confusion Matrix heatmap, run:

```bash
python eval.py
```

The confusion matrix will be exported as a high-resolution `confusion_matrix.png` file.

---

## 📚 References

Ouamane, A., et al. (2025). *"Optimized Vision Transformers for Superior Plant Disease Detection"*. **IEEE Access**.