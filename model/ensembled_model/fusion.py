import numpy as np
from sklearn.metrics import roc_auc_score
from itertools import product


def weighted_sum_fusion(svm_probs, lr_probs, weight_svm=0.6, weight_lr=0.4, threshold=0.5):
    """
    Combine probabilities from SVM and Logistic Regression models using weighted sum.

    Args:
        svm_probs (numpy array): Predicted probabilities from the SVM model.
        lr_probs (numpy array): Predicted probabilities from the Logistic Regression model.
        weight_svm (float): Weight for the SVM model.
        weight_lr (float): Weight for the Logistic Regression model.
        threshold (float): Threshold for final classification.

    Returns:
        numpy array: Final binary predictions.
        numpy array: Combined probabilities.
    """
    # Normalize weights
    total_weight = weight_svm + weight_lr
    weight_svm_norm = weight_svm / total_weight
    weight_lr_norm = weight_lr / total_weight

    # Compute weighted probabilities
    combined_probs = (weight_svm_norm * svm_probs) + (weight_lr_norm * lr_probs)

    # Final predictions based on threshold
    final_predictions = (combined_probs >= threshold).astype(int)

    return final_predictions, combined_probs


def grid_search_weights(svm_probs, lr_probs, y_true, weight_range=np.linspace(0, 1, 11)):
    """
    Perform grid search to find the best weights for fusion based on ROC-AUC.

    Args:
        svm_probs (numpy array): Predicted probabilities from the SVM model.
        lr_probs (numpy array): Predicted probabilities from the Logistic Regression model.
        y_true (numpy array): True labels.
        weight_range (numpy array): Range of weights to search.

    Returns:
        tuple: Best weights and the corresponding ROC-AUC score.
    """
    best_auc = 0
    best_weights = (0.5, 0.5)

    for w_svm, w_lr in product(weight_range, repeat=2):
        if w_svm + w_lr == 0:
            continue  # Skip if both weights are zero
        # Normalize weights
        w_total = w_svm + w_lr
        w_svm_norm = w_svm / w_total
        w_lr_norm = w_lr / w_total
        # Compute combined probabilities
        combined = (w_svm_norm * svm_probs) + (w_lr_norm * lr_probs)
        # Calculate ROC-AUC
        auc = roc_auc_score(y_true, combined)
        if auc > best_auc:
            best_auc = auc
            best_weights = (w_svm_norm, w_lr_norm)

    print(f"Best Weights - SVM: {best_weights[0]:.2f}, LR: {best_weights[1]:.2f} with ROC-AUC: {best_auc:.4f}")
    return best_weights, best_auc
