from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score
)


def evaluate_model(y_true, y_pred, y_probs):
    """
    Evaluate the model's performance and print metrics.

    Args:
        y_true (numpy array): True labels.
        y_pred (numpy array): Predicted labels.
        y_probs (numpy array): Predicted probabilities for the positive class.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    accuracy = accuracy_score(y_true, y_pred)
    print("Classification Report:")
    print(classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary'
    )

    roc_auc = roc_auc_score(y_true, y_probs)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

    metrics = {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc
    }

    return metrics
