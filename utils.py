import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def save_data(data, path):
    """Save data to a specified path using joblib."""
    joblib.dump(data, path)
    print(f"Data saved to {path}")


def load_data(path):
    """Load data from a specified path using joblib."""
    data = joblib.load(path)
    print(f"Data loaded from {path}")
    return data


def reduce_dimensionality(features, n_components=100):
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(features)


def normalize_features(features):
    scaler = StandardScaler()
    return scaler.fit_transform(features)


def save_numpy_array(array, filepath):
    np.save(filepath, array)


def load_numpy_array(filepath):
    return np.load(filepath)


def save_features(feature_matrix, labels, feature_path, label_path):
    """Save feature matrix and labels to disk."""
    joblib.dump(feature_matrix, feature_path)
    joblib.dump(labels, label_path)
    print(f"Features saved to {feature_path}")
    print(f"Labels saved to {label_path}")


def load_features(feature_path, label_path):
    """Load feature matrix and labels from disk."""
    feature_matrix = joblib.load(feature_path)
    labels = joblib.load(label_path)
    print(f"Features loaded from {feature_path}")
    print(f"Labels loaded from {label_path}")
    return feature_matrix, labels
