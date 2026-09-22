# Methodology

## Objective

This project investigates JEPA-style self-supervised representation
learning for breast histopathology image classification.

## Dataset

BreaKHis v1.

Task:

- Benign
- Malignant

Total images:

7909

## Pipeline

Image
↓
Patch Embedding
↓
Vision Transformer Encoder
↓
Context Representation
↓
JEPA Predictor
↓
Target Representation
↓
Representation Prediction Loss
↓
Fine-tuning
↓
Binary Classification

## Model Configuration

- Image size: 224 × 224
- Patch size: 16 × 16
- Number of patches: 196
- Embedding dimension: 384
- Encoder depth: 6
- Attention heads: 6
- Predictor depth: 4
- EMA coefficient: 0.996

## Self-Supervised Learning

The context encoder receives visible patches while the target encoder
produces representations of the target region.

The predictor attempts to predict target representations from context
representations.

The target encoder is updated using exponential moving average.

## Downstream Classification

The pretrained encoder is fine-tuned using a binary classifier.

The classifier uses:

- LayerNorm
- Linear layer
- GELU
- Dropout
- Linear output layer

## Evaluation

The experiments use:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

## Evaluation Splits

Two types of evaluation are considered:

1. Image-level split
2. Specimen/group-level split

The group-level experiment separates images belonging to the same
specimen/group across partitions to reduce possible information leakage.