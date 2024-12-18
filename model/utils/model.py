import joblib
import os



def load_model(model_path):
    """
    Load a trained model from a file.

    Args:
        model_path (str): Path to the model file.

    Returns:
        sklearn estimator: Loaded model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    return model
