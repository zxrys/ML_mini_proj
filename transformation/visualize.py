def analyze_features(audio_features_path, text_features_path, labels_path):
    """
    Analyze and visualize audio and text features along with their labels.

    Parameters:
    - audio_features_path (str): Path to the audio_features.joblib file.
    - text_features_path (str): Path to the text_features.joblib file.
    - labels_path (str): Path to the labels.joblib file.

    Returns:
    - None
    """

    # ---------------------------
    # 1. Import Necessary Libraries
    # ---------------------------
    import joblib
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.model_selection import train_test_split
    import plotly.express as px

    # ---------------------------
    # 2. Configure Plotting Aesthetics
    # ---------------------------
    sns.set(style='whitegrid', palette='muted')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['axes.grid'] = False

    # ---------------------------
    # 3. Data Loading
    # ---------------------------
    print("Loading data...")

    try:
        # Load audio features
        audio_features = joblib.load(audio_features_path)
        print(f"Loaded Audio Features from {audio_features_path} with shape {audio_features.shape}")

        # Load text features
        text_features = joblib.load(text_features_path)
        print(f"Loaded Text Features from {text_features_path} with shape {text_features.shape}")

        # Load labels
        labels = joblib.load(labels_path)
        print(f"Loaded Labels from {labels_path} with shape {labels.shape}\n")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # ---------------------------
    # 4. Preliminary Data Inspection
    # ---------------------------
    print("Preliminary Data Inspection:")
    print(f"Audio Features Shape: {audio_features.shape}")
    print(f"Text Features Shape: {text_features.shape}")
    print(f"Labels Shape: {labels.shape}")
    print(f"Unique Labels: {np.unique(labels)}\n")

    # Convert labels to categorical if they are not already
    if labels.dtype != 'object' and labels.dtype != 'str':
        labels = labels.astype(str)
        print("Converted labels to string type.\n")

    # ---------------------------
    # 5. Feature Concatenation and Scaling
    # ---------------------------
    print("Concatenating and scaling features...")

    # Concatenate audio and text features
    combined_features = np.concatenate((audio_features, text_features), axis=1)  # Shape: (100, 1601)
    print(f"Combined Features Shape (Before Scaling): {combined_features.shape}")

    # Standardize the combined features
    scaler = StandardScaler()
    combined_features_scaled = scaler.fit_transform(combined_features)

    print(f"Combined Features Shape (Scaled): {combined_features_scaled.shape}\n")

    # ---------------------------
    # 6. Dimensionality Reduction and Visualization
    # ---------------------------
    def plot_pca(features, labels, title):
        pca = PCA(n_components=2, random_state=42)
        principal_components = pca.fit_transform(features)
        pc_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
        pc_df['Label'] = labels

        plt.figure(figsize=(10, 8))
        sns.scatterplot(x='PC1', y='PC2', hue='Label', data=pc_df, palette='viridis', s=100, alpha=0.7)
        plt.title(title, fontsize=16)
        plt.xlabel('Principal Component 1', fontsize=14)
        plt.ylabel('Principal Component 2', fontsize=14)
        plt.legend(title='Label', fontsize=12, title_fontsize=14)
        plt.tight_layout()
        plt.show()

    def plot_tsne(features, labels, title):
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
        tsne_results = tsne.fit_transform(features)
        tsne_df = pd.DataFrame(data=tsne_results, columns=['Dim1', 'Dim2'])
        tsne_df['Label'] = labels

        plt.figure(figsize=(10, 8))
        sns.scatterplot(x='Dim1', y='Dim2', hue='Label', data=tsne_df, palette='deep', s=100, alpha=0.7)
        plt.title(title, fontsize=16)
        plt.xlabel('Dimension 1', fontsize=14)
        plt.ylabel('Dimension 2', fontsize=14)
        plt.legend(title='Label', fontsize=12, title_fontsize=14)
        plt.tight_layout()
        plt.show()

    print("Performing PCA and t-SNE on different feature sets...\n")

    # Apply PCA and Plot
    plot_pca(audio_features, labels, 'PCA of Audio Features')
    plot_pca(text_features, labels, 'PCA of Text Features')
    plot_pca(combined_features_scaled, labels, 'PCA of Combined Features')

    # Apply t-SNE and Plot
    plot_tsne(audio_features, labels, 't-SNE of Audio Features')
    plot_tsne(text_features, labels, 't-SNE of Text Features')
    plot_tsne(combined_features_scaled, labels, 't-SNE of Combined Features')

    # ---------------------------
    # 7. Class Distribution Visualization
    # ---------------------------
    def plot_class_distribution(labels, title):
        label_counts = pd.Series(labels).value_counts()
        plt.figure(figsize=(6, 6))
        sns.barplot(x=label_counts.index, y=label_counts.values, palette='pastel')
        plt.title(title, fontsize=16)
        plt.xlabel('Class', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        for index, value in enumerate(label_counts.values):
            plt.text(index, value + 1, str(value), ha='center', fontsize=12)
        plt.tight_layout()
        plt.show()

    print("Visualizing class distributions...\n")

    # Plot Class Distribution for Entire Dataset
    plot_class_distribution(labels, 'Class Distribution in Entire Dataset')

    # Split the data into training and validation sets
    print("Splitting data into training and validation sets...\n")
    X_train, X_val, y_train, y_val = train_test_split(
        combined_features_scaled, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Plot Class Distribution for Training Set
    plot_class_distribution(y_train, 'Class Distribution in Training Set')

    # Plot Class Distribution for Validation Set
    plot_class_distribution(y_val, 'Class Distribution in Validation Set')

    # ---------------------------
    # 8. Feature Distribution Visualization
    # ---------------------------
    print("Visualizing feature distributions...\n")

    # Select Top High-Variance Features
    selector = VarianceThreshold(threshold=0.8 * (1 - 0.8))  # Example threshold
    selector.fit(combined_features_scaled)
    high_variance_features = selector.transform(combined_features_scaled)

    print(f"Original Number of Features: {combined_features_scaled.shape[1]}")
    print(f"Number of High-Variance Features: {high_variance_features.shape[1]}\n")

    # Convert to DataFrame for Easier Handling
    combined_df = pd.DataFrame(combined_features_scaled, columns=[f'Feature_{i}' for i in range(combined_features_scaled.shape[1])])
    combined_df['Label'] = labels

    # Select a Few High-Variance Features to Plot
    selected_features = combined_df.columns[:10]  # Adjust as needed

    # Plot Histograms
    combined_df[selected_features].hist(bins=15, figsize=(20, 15), layout=(5, 2))
    plt.suptitle('Histograms of Selected High-Variance Features', fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # Box Plots for Feature Comparison Across Classes
    features_to_plot = ['Feature_0', 'Feature_1', 'Feature_2', 'Feature_3', 'Feature_4']

    # Check if the features exist
    available_features = [f for f in features_to_plot if f in combined_df.columns]
    if not available_features:
        print("No features available for box plotting.")
    else:
        # Melt the DataFrame for Seaborn Boxplot
        melted_df = combined_df[available_features + ['Label']].melt(id_vars='Label', var_name='Feature', value_name='Value')

        plt.figure(figsize=(15, 10))
        sns.boxplot(x='Feature', y='Value', hue='Label', data=melted_df, palette='Set2')
        plt.title('Box Plots of Selected Features by Label', fontsize=16)
        plt.xlabel('Feature', fontsize=14)
        plt.ylabel('Value', fontsize=14)
        plt.legend(title='Label', fontsize=12, title_fontsize=14)
        plt.tight_layout()
        plt.show()

    # ---------------------------
    # 9. Correlation Analysis
    # ---------------------------
    print("Performing correlation analysis...\n")

    # Compute Correlation Matrix for a Subset of Features
    subset_features = combined_df.iloc[:, :50]  # First 50 features
    corr_matrix = subset_features.corr()

    # Plot Heatmap
    plt.figure(figsize=(20, 18))
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0, linewidths=.5, annot=False)
    plt.title('Correlation Heatmap of First 50 Features', fontsize=20)
    plt.tight_layout()
    plt.show()

    # ---------------------------
    # 10. Optional: Interactive PCA Plot using Plotly
    # ---------------------------
    def plot_pca_interactive(features, labels, title):
        pca = PCA(n_components=2, random_state=42)
        principal_components = pca.fit_transform(features)
        pc_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
        pc_df['Label'] = labels

        fig = px.scatter(pc_df, x='PC1', y='PC2', color='Label',
                         title=title,
                         labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'},
                         width=800, height=600)
        fig.update_traces(marker=dict(size=12, opacity=0.8),
                          selector=dict(mode='markers'))
        fig.show()

    print("Creating interactive PCA plot...\n")
    plot_pca_interactive(combined_features_scaled, labels, 'Interactive PCA of Combined Features')

    print("Analysis Complete.")

