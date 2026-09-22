from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


def calculate_metrics(y_true, y_pred, y_prob=None):

    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }

    if y_prob is not None:
        results["roc_auc"] = roc_auc_score(
            y_true,
            y_prob
        )

    results["confusion_matrix"] = confusion_matrix(
        y_true,
        y_pred
    )

    return results