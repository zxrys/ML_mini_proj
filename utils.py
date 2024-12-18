import joblib
import numpy as np
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


def load_all_data(text_feature_path, text_label, audio_feature_path, audio_label):
    """
    Load text features, audio features, and labels.

    Returns:
        tuple: Text features, audio features, and labels.
    """
    X_text, y_text = load_features(text_feature_path, text_label)
    X_audio, y_audio = load_features(audio_feature_path, audio_label)

    # Ensure that labels are consistent
    if not np.array_equal(y_text, y_audio):
        raise ValueError("Labels for text and audio features do not match.")

    return X_text, X_audio, y_text
