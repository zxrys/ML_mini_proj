from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler


def preprocess_features(X_train, X_test, y_train, random_state=42):
    """
    Apply PCA, feature selection, and scaling to the data.

    Args:
        X_train (numpy array): Training feature matrix.
        X_test (numpy array): Testing feature matrix.
        y_train (numpy array): Training labels.
        random_state (int): random state seed

    Returns:
        tuple: Preprocessed training and testing feature matrices,
               and the fitted PCA, selector, and scaler objects.
    """
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

    # Standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)

    return X_train_scaled, X_test_scaled, pca, selector, scaler
