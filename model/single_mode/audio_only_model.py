"""
    This file is part of ML_mini_proj project
    Copyright (C) 2024 Yao Shu  <springrainyszxr@gmail.com>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from model.single_mode.base import *

def train_and_test(feature_path,
                   label_path,
                   model_type="random forest",
                   test_size=0.3):
    x_train, y_train, x_test, y_test = data_split(feature_path, label_path, test_size=test_size)
    if model_type == "random forest":
        model_path = random_forest_train_and_save(
            x_train,
            y_train,
            "models/audio_only/random_forest.joblib")
    elif model_type == "logistic regression":
        model_path = logistic_regression_train_and_save(
            x_train,
            y_train,
            "models/audio_only/logistic_regressio.joblib")
    elif model_type == "svm":
        model_path = svm_train_and_save(
            x_train,
            y_train,
            "models/audio_only/svm.joblib")
    else:
        raise ValueError("Model not implemented")

    acc, cm = test_model(model_path, x_test, y_test)

    return acc, cm

