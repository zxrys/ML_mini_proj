from utils import load_features
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import joblib
import os


def data_split(feature_path, label_path, test_size=0.3, random_state=42):
    feature_matrix, labels = load_features(feature_path, label_path)

    print(f"Feature matrix shape: {feature_matrix.shape}")
    print(f"Labels shape: {labels.shape}")
    X = feature_matrix
    y = labels

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Apply PCA
    pca = PCA(n_components=0.95, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    print(f"Training set shape after PCA: {X_train_pca.shape}")
    print(f"Test set shape after PCA: {X_test_pca.shape}")

    # Feature selection
    selector = SelectKBest(score_func=f_classif, k=3)
    X_train_selected = selector.fit_transform(X_train_pca, y_train)
    X_test_selected = selector.transform(X_test_pca)

    print(f"Training set shape after feature selection: {X_train_selected.shape}")
    print(f"Test set shape after feature selection: {X_test_selected.shape}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)

    return X_train_scaled, y_train, X_test_scaled, y_test


def random_forest_train_and_save(X_train_scaled, y_train, model_path, random_state=42, retrain=True, return_model=False):
    print("Preprocessing objects saved.")

    # Check if retraining is needed
    if os.path.exists(model_path) and not retrain:
        model = joblib.load(model_path)
    else:
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        model.fit(X_train_scaled, y_train)

    # Save the trained model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    if return_model:
        return model
    return model_path


def logistic_regression_train_and_save(X_train_scaled, y_train, model_path, random_state=42, retrain=True, return_model=False):
    # Check if retraining is needed
    if os.path.exists(model_path) and not retrain:
        model = joblib.load(model_path)
    else:
        # Initialize and train Logistic Regression model
        model = LogisticRegression(
            random_state=random_state,
            max_iter=1000,  # Increase max iterations to ensure convergence
            solver='lbfgs'  # Recommended solver
        )
        model.fit(X_train_scaled, y_train)

    # Ensure model save directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Save model
    joblib.dump(model, model_path)
    print(f"Logistic Regression model saved to {model_path}")
    if return_model:
        return model
    return model_path


def svm_train_and_save(X_train_scaled, y_train, model_path, random_state=42, retrain=True, return_model=False):
    # Check if retraining is needed
    if os.path.exists(model_path) and not retrain:
        model = joblib.load(model_path)
    else:
        # Initialize and train SVM model
        model = SVC(
            kernel='rbf',  # Radial Basis Function kernel
            random_state=random_state,
            probability=True  # Enable probability estimation
        )
        model.fit(X_train_scaled, y_train)

    # Ensure model save directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Save model
    joblib.dump(model, model_path)
    print(f"Support Vector Machine model saved to {model_path}")
    if return_model:
        return model
    return model_path


def test_model(model_path, X_test_scaled, y_test):
    model_path = model_path
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")

    # predict on test dataset
    y_pred = model.predict(X_test_scaled)

    # evaluate
    accuracy = accuracy_score(y_test, y_pred)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    return accuracy, cm
