# ConvNeXt-Exploration
ConvNeXt Architecture Exploration for Rice Leaf Disease Classification

> Analyzing the effect of model size, activation function, and regularization strategy on ConvNeXt performance for rice leaf disease identification under data-constrained conditions.

---

## Overview

This repository contains the implementation and experimental notebooks for an architectural analysis of ConvNeXt variants applied to rice leaf disease classification. The study systematically examines how model size, activation function (GELU vs SiLU), and regularization strategy (Stochastic Depth vs Standard Dropout) affect classification performance on a small-scale dataset of 3,353 images across 7 disease categories.

---

## Dataset

**Rice Leaf Disease Identification Dataset** — publicly released by Wang et al. (2023)  
Source: [Kaggle](https://www.kaggle.com/datasets/wangxiaoqii/rice-leaf-disease-identification-dataset)

| Class | Images |
|---|---|
| Bacterial Blight | 261 |
| Bacterial Leaf Streak | 168 |
| Brown Spot | 834 |
| Healthy | 864 |
| Hispa | 729 |
| Leaf Blast | 699 |
| Sheath Blight | 126 |
| **Total** | **3,353** |

Split: **60% train / 20% val / 20% test** (fixed seed = 42)

---

## Experiments

### Ablation Study (ConvNeXt-Tiny)

All experiments use: ImageNet-1K initialization · AdamW · Cosine LR decay · 30 epochs · Batch size 64

| # | Activation | Dropout | Norm | Accuracy | F1-Score |
|---|---|---|---|---|---|
| 01 | GELU | Stochastic Depth | LayerNorm | — | — |
| 02 | GELU | Stochastic Depth | BatchNorm | — | — |
| 03 | GELU | Standard Dropout | LayerNorm | — | — |
| 04 | GELU | Standard Dropout | BatchNorm | — | — |
| 05 | SiLU | Stochastic Depth | LayerNorm | — | — |
| 06 | SiLU | Stochastic Depth | BatchNorm | — | — |
| 07 | SiLU | Standard Dropout | LayerNorm | — | — |
| 08 | SiLU | Standard Dropout | BatchNorm | — | — |

*Results will be updated as experiments complete.*

---

## Repository Structure

```
ConvNeXt-Exploration/
│
├── RawSourceCode/
│   ├── 01_ConvNeXt-Tiny | GELU | StochDepth | LayerNorm.ipynb
│   ├── 02_ConvNeXt-Tiny | GELU | StochDepth | BatchNorm.ipynb
│   ├── 03_ConvNeXt-Tiny | GELU | Dropout | LayerNorm.ipynb
│   ├── 04_ConvNeXt-Tiny | GELU | Dropout | BatchNorm.ipynb
│   ├── 05_ConvNeXt-Tiny | SiLU | StochDepth | LayerNorm.ipynb
│   ├── 06_ConvNeXt-Tiny | SiLU | StochDepth | BatchNorm.ipynb
│   ├── 07_ConvNeXt-Tiny | SiLU | Dropout | LayerNorm.ipynb
│   └── 08_ConvNeXt-Tiny | SiLU | Dropout | BatchNorm.ipynb
│
├── Output(html)/
│   ├── 00_ConvNeXt-Base | GELU | StochDepth | Layernorm.html
│   ├── 01_ConvNeXt-Tiny | GELU | StochDepth | LayerNorm.html
│   ├── 02_ConvNeXt-Tiny | GELU | StochDepth | BatchNorm.html
│   └── 03_ConvNeXt-Tiny | GELU | Dropout | LayerNorm.html
│
└── README.md
```

---

## Setup & Usage

### Requirements

```bash
torch>=2.0.0
torchvision>=0.15.0
scikit-learn
matplotlib
seaborn
pandas
numpy
kaggle
```

### Running on Google Colab

1. Open any notebook in Colab
2. Set runtime to **GPU**
3. Run **Cell 0a** — mount Google Drive
4. Run **Cell 0b** — upload `kaggle.json` ([get it here](https://www.kaggle.com/settings))
5. Run **Cell 0c** — download & copy dataset to local storage
6. Run all remaining cells sequentially

### Reproducing Results

All experiments use `SEED = 42` and identical train/val/test splits. To reproduce:

```python
SEED = 42
generator = torch.Generator().manual_seed(SEED)
train_set, val_set, test_set = random_split(
    full_dataset, [n_train, n_val, n_test], generator=generator
)
```

---

## Training Configuration

| Parameter | Value |
|---|---|
| Image size | 224 × 224 |
| Batch size | 64 |
| Epochs | 30 |
| Optimizer | AdamW |
| Initial LR | 1e-3 |
| Final LR | 1e-6 |
| LR schedule | Cosine annealing + 1 epoch warmup |
| Weight decay | 0.05 |
| Pretrained weights | ImageNet-1K |
| Normalization | ImageNet stats [0.485, 0.456, 0.406] |

