import joblib

def save_data(data, path):
    """Save data to a specified path using joblib."""
    joblib.dump(data, path)
    print(f"Data saved to {path}")

def load_data(path):
    """Load data from a specified path using joblib."""
    data = joblib.load(path)
    print(f"Data loaded from {path}")
    return data
