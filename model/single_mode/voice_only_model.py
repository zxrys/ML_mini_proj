from model.single_mode.base import *

def train_and_test(feature_path,
                   label_path,
                   model_type="random forest",
                   test_size=0.3):
    x_train, y_train, x_test, y_test = data_split(feature_path, label_path, test_size=test_size)
    if model_type == "random forest":
        model_path = random_forest_train_and_save(x_train, y_train, "models/audio_only/random_forest.joblib")
    else:
        raise ValueError("Model not implemented")

    acc, cm = test_model(model_path, x_test, y_test)

    return acc, cm

