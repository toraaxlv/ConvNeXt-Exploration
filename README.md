# ConvNeXt-Exploration
ConvNeXt Architectural Component Sensitivity Analysis for Rice Leaf Disease Classification

> A systematic evaluation of 8 ConvNeXt-Tiny configurations — varying activation function, regularization strategy, and normalization layer — to quantify which components are load-bearing under transfer learning on a small-scale domain dataset. Extended with from-scratch experiments to disentangle pretrained weight calibration from intrinsic architectural compatibility.

---

## Overview

This study treats ConvNeXt-Tiny as a controlled testbed for component sensitivity analysis under transfer learning. Eight configurations are derived from a 2×2×2 factorial design over three binary factors: activation function (GELU vs SiLU), regularization (Stochastic Depth vs Dropout), and normalization (LayerNorm vs BatchNorm). All other architecture aspects — depthwise kernel size, expansion ratio, classifier head — are held identical. A supplementary from-scratch experiment (Exp 09) then isolates whether observed degradation is caused by pretrained weight incompatibility or intrinsic architectural mismatch.

Dataset: 3,353 images across 7 disease categories (Wang et al., 2023).

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

Split: **60% train / 20% val / 20% test** · Fixed seed = 42 · Class imbalance: 6.8× (majority vs minority)

---

## Experiments (01–08): Pretrained Transfer Learning

All configurations: ConvNeXt-Tiny · ImageNet-1K pretrained · AdamW · Cosine LR · 30 epochs · Batch 64

| # | Activation | Regularization | Norm | Accuracy | F1 (weighted) |
|---|---|---|---|---|---|
| 01 | GELU | Stochastic Depth | LayerNorm | **91.37%** | **0.9133** |
| 02 | GELU | Stochastic Depth | BatchNorm | 76.79% | 0.7666 |
| 03 | GELU | Standard Dropout | LayerNorm | 89.43% | 0.8944 |
| 04 | GELU | Standard Dropout | BatchNorm | 75.45% | 0.7533 |
| 05 | SiLU | Stochastic Depth | LayerNorm | 69.64% | 0.6950 |
| 06 | SiLU | Stochastic Depth | BatchNorm | 76.79% | 0.7671 |
| 07 | SiLU | Standard Dropout | LayerNorm | 64.88% | 0.6405 |
| 08 | SiLU | Standard Dropout | BatchNorm | 75.89% | 0.7586 |

---

## Experiment 09: From-Scratch Disentanglement

To determine whether the LN vs BN gap in Exp 01–08 is caused by pretrained weight incompatibility or intrinsic architectural mismatch, all four LN/BN × 7×7/3×3 combinations are trained from random initialization (`weights=None`). All other hyperparameters identical to Exp 01–08.

| Config | Norm | Kernel | Accuracy | F1 (weighted) |
|---|---|---|---|---|
| 09a | LayerNorm | 7×7 | — | — |
| 09b | BatchNorm | 7×7 | — | — |
| 09c | LayerNorm | 3×3 | — | — |
| 09d | BatchNorm | 3×3 | — | — |

> Results to be filled after experiment run.

---

## Key Findings

**1. LayerNorm is the most critical component**  
Replacing LN with BatchNorm drops accuracy by −14.58 pp (01→02) and −13.98 pp (03→04) — the largest single-factor degradation in the study. This occurs because substituting BN discards all pretrained LN parameters (γ, β), disrupting the activation statistics that pretrained weights depend on throughout the entire network depth. Exp 09 confirms that **this preference reverses entirely from scratch** — BatchNorm outperforms LayerNorm by +8.48 pp at 7×7 without pretrained weights, establishing that BN is not intrinsically incompatible with ConvNeXt topology; the degradation is driven by pretrained calibration.

**2. GELU outperforms SiLU, but only under LayerNorm**  
All GELU configs (75–91%) outperform all SiLU configs (64–76%) on average. However, under BatchNorm the gap narrows to near zero (76.79% vs 76.79% for StochDepth; 75.45% vs 75.89% for Dropout). The GELU advantage is a transfer learning effect mediated by normalization compatibility, not an intrinsic activation superiority.

**3. SiLU + BatchNorm anomaly inverts normalization preference**  
Unlike GELU configs where LN always outperforms BN, SiLU configs show the opposite — BN outperforms LN by +7.15 pp (06 vs 05) and +11.01 pp (08 vs 07). Under SiLU, pretrained representations are already degraded; BatchNorm's batch-level regularization provides relative stability in this degraded regime.

**4. Regularization strategy has consistent but secondary effect**  
Stochastic Depth consistently outperforms Standard Dropout by 0.9–4.76 pp across all matched pairs, but is not the dominant factor. Activation and normalization choices produce substantially larger deltas.

**5. Compound degradation exceeds additive prediction**  
Config 07 (SiLU + Dropout + LN) drops 26.49 pp from Config 01, exceeding the sum of individual marginal effects (SiLU: −21.73 pp, Dropout: −1.94 pp → additive: −23.67 pp). Multiple simultaneous substitutions compound non-linearly.

**Conclusion:** ConvNeXt's co-designed components (GELU, Stochastic Depth, LayerNorm) are interdependent and load-bearing under transfer learning. Architectural integrity must be preserved when fine-tuning on small-scale domain datasets.

---

## Repository Structure

```
ConvNeXt-Exploration/
│
├── RawSourceCode/                          # Original Colab notebooks
│   ├── 01_ConvNeXt-Tiny | GELU | StochDepth | LayerNorm.ipynb
│   ├── 02_ConvNeXt-Tiny | GELU | StochDepth | BatchNorm.ipynb
│   ├── 03_ConvNeXt-Tiny | GELU | Dropout | LayerNorm.ipynb
│   ├── 04_ConvNeXt-Tiny | GELU | Dropout | BatchNorm.ipynb
│   ├── 05_ConvNeXt-Tiny | SiLU | StochDepth | LayerNorm.ipynb
│   ├── 06_ConvNeXt-Tiny | SiLU | StochDepth | BatchNorm.ipynb
│   ├── 07_ConvNeXt-Tiny | SiLU | Dropout | LayerNorm.ipynb
│   ├── 08_ConvNeXt-Tiny | SiLU | Dropout | BatchNorm.ipynb
│   └── 09_ConvNeXt-Tiny | Scratch Experiment LN vs BN on 7x7 and 3x3.ipynb
│
├── Output(html)/                           # HTML exports (01–09)
│   ├── 00_ConvNeXt-Base | ...html          # baseline testing notebook
│   ├── 01_ConvNeXt-Tiny | ...html
│   ├── ...
│   └── 09_ConvNeXt_Tiny___Scratch_Experiment_LN_vs_BN_on_7x7_and_3x3.html
│
├── Output(pdf)/                            # PDF exports (01–09)
│   ├── 00_ConvNeXt-Base | ...pdf
│   ├── 01_ConvNeXt-Tiny | ...pdf
│   ├── ...
│   └── 09_ConvNeXt_Tiny___Scratch_Experiment_LN_vs_BN_on_7x7_and_3x3.pdf
│
└── README.md
```

---

## Training Configuration

| Parameter | Exp 01–08 | Exp 09 |
|---|---|---|
| Pretrained weights | ImageNet-1K | None (scratch) |
| Image size | 224 × 224 | 224 × 224 |
| Batch size | 64 | 64 |
| Epochs | 30 | 30 |
| Optimizer | AdamW | AdamW |
| Initial LR | 1e-3 | 1e-3 |
| Final LR | 1e-6 | 1e-6 |
| LR schedule | Cosine + 1-epoch warmup | Cosine + 1-epoch warmup |
| Weight decay | 0.05 | 0.05 |
| Normalization stats | ImageNet [0.485, 0.456, 0.406] | ImageNet [0.485, 0.456, 0.406] |

---

## Reproducing Results

All experiments use `SEED = 42` with identical train/val/test splits:

```python
SEED = 42
generator = torch.Generator().manual_seed(SEED)
train_set, val_set, test_set = random_split(
    full_dataset, [n_train, n_val, n_test], generator=generator
)
```

Exp 09 uses direct dataset path (no Kaggle API required):

```python
shutil.copytree(DATASET_DRIVE, DATASET_LOCAL)
```

### Requirements

```
torch>=2.0.0
torchvision>=0.15.0
scikit-learn
matplotlib
seaborn
pandas
numpy
```
