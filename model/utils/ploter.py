from matplotlib import pyplot as plt
import seaborn as sns


def plot_res(accuracy, cm, header=""):
    print(f"Test Accuracy: {accuracy:.4f}")

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'],
                yticklabels=['Class 0', 'Class 1'])
    plt.title(f'Confusion Matrix {header}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
