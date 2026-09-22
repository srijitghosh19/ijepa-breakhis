# Experiments

## Experiment 1 — Image-Level Split

Dataset:

BreaKHis v1

Task:

Benign vs malignant classification.

Preliminary results:

| Metric | Result |
|---|---:|
| Accuracy | 89.06% |
| Precision | 94.19% |
| Recall | 89.59% |
| F1-score | 91.84% |
| ROC-AUC | 0.9579 |

This experiment is considered preliminary because correlated images from
the same specimen may occur in different partitions.

## Experiment 2 — Specimen/Group-Level Split

A group-level split is used to separate images originating from the same
specimen between training, validation and testing.

Status:

Experimental evaluation.

Purpose:

To obtain a more rigorous estimate of generalization to previously unseen
specimens.