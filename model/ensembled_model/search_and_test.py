# main.py

import os
import joblib
from sklearn.model_selection import train_test_split
from utils import load_all_data
from model.utils.preprocess import preprocess_features
from model.single_mode.base import svm_train_and_save, logistic_regression_train_and_save
from model.ensembled_model.evaluation import evaluate_model
from model.ensembled_model.fusion import weighted_sum_fusion, grid_search_weights


def search_and_test(text_feature_path, text_label, audio_feature_path, audio_label, text_model_path, audio_model_path):
    # Load all data
    print("Loading data...")
    X_text, X_audio, y = load_all_data(text_feature_path, text_label, audio_feature_path, audio_label)
    print(f"Text features shape: {X_text.shape}")
    print(f"Audio features shape: {X_audio.shape}")
    print(f"Labels shape: {y.shape}")

    # Split the data
    print("Splitting data into training and testing sets...")
    X_text_train, X_text_test, X_audio_train, X_audio_test, y_train, y_test = train_test_split(
        X_text, X_audio, y, test_size=0.3, stratify=y, random_state=42
    )

    # Preprocess text features
    print("Preprocessing text features...")
    X_text_train_scaled, X_text_test_scaled, pca_text, selector_text, scaler_text = preprocess_features(
        X_text_train, X_text_test, y_train
    )

    # Preprocess audio features
    print("Preprocessing audio features...")
    X_audio_train_scaled, X_audio_test_scaled, pca_audio, selector_audio, scaler_audio = preprocess_features(
        X_audio_train, X_audio_test, y_train
    )

    # Train SVM model on text features
    print("Training SVM model on text features...")
    svm_model = svm_train_and_save(X_text_train_scaled, y_train, text_model_path, retrain=True, return_model=True)

    # Train Logistic Regression model on audio features
    print("Training Logistic Regression model on audio features...")
    lr_model = logistic_regression_train_and_save(X_audio_train_scaled, y_train, audio_model_path, retrain=True,
                                                  return_model=True)

    # Get prediction probabilities
    print("Getting prediction probabilities...")
    svm_probs = svm_model.predict_proba(X_text_test_scaled)[:, 1]
    lr_probs = lr_model.predict_proba(X_audio_test_scaled)[:, 1]

    # Perform weighted sum fusion with default weights
    print("Performing weighted sum fusion with default weights...")
    final_predictions, combined_probs = weighted_sum_fusion(
        svm_probs, lr_probs, weight_svm=0.4, weight_lr=0.6, threshold=0.5
    )

    # Evaluate the fused model
    print("=== Fused Model Performance ===")
    evaluate_model(y_test, final_predictions, combined_probs)

    # Optional: Grid search for optimal weights
    print("Performing grid search to find optimal weights...")
    best_weights, best_auc = grid_search_weights(svm_probs, lr_probs, y_test)

    # Apply the best weights found
    print("Applying the best weights found from grid search...")
    best_final_predictions, best_combined_probs = weighted_sum_fusion(
        svm_probs, lr_probs, weight_svm=best_weights[0], weight_lr=best_weights[1], threshold=0.5
    )

    # Evaluate the fused model with best weights
    print("=== Fused Model Performance with Best Weights ===")
    metrics = evaluate_model(y_test, best_final_predictions, best_combined_probs)

    # Save the best weights and models if necessary
    # Example: Save weights to a file
    weights = {
        'weight_svm': best_weights[0],
        'weight_lr': best_weights[1]
    }
    weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'fusion_weights.joblib')
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    joblib.dump(weights, weights_path)
    print(f"Selected {weights}")
    print(f"Best fusion weights saved to {weights_path}")
    return metrics
