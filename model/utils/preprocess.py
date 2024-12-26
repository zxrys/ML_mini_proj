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
