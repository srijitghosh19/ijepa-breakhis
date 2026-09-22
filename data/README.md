# Dataset

This project uses the BreakHis breast histopathology dataset.

The dataset is not included in this repository.

## Dataset

BreaKHis v1

## Task

Binary breast histopathology classification.

## Classes

- Benign
- Malignant

## Dataset Size

7909 images used in the current experiment.

## Expected Structure

BreaKHis_v1/
└── histology_slides/
    └── breast/
        ├── benign/
        └── malignant/

The dataset must be downloaded separately and its local path should be
provided when running the notebooks or training scripts.

## Results

### I-JEPA Pretraining

![Pretraining Loss](results/figures/pretraining_loss.png)

### Fine-tuning

![Fine-tuning Loss](results/figures/finetuning_loss.png)

![Fine-tuning Accuracy](results/figures/finetuning_accuracy.png)

### Classification Performance

![Confusion Matrix](results/figures/confusion_matrix.png)

![ROC Curve](results/figures/roc_curve.png)