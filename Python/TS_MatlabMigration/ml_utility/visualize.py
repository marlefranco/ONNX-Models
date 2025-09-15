import matplotlib.pyplot as plt
import numpy as np

def visualize_data(X_train, y_train, X_test, y_test):
    """
    Plots sample spectra from train and test sets.
    
    Parameters:
    - X_train, y_train: Training features and labels.
    - X_test, y_test: Testing features and labels.
    """
    plt.figure(figsize=(12, 5))
    
    for i in range(5):  # Plot 5 random samples
        plt.subplot(1, 2, 1)
        plt.plot(X_train[i], label=f"Train Target: {y_train[i]}")
        plt.title("Training Set Spectra")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(X_test[i], label=f"Test Target: {y_test[i]}")
        plt.title("Test Set Spectra")
        plt.legend()

    plt.show()