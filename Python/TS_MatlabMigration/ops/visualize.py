import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    
def plot_all_spectra(xtrain, ytrain, wavelength_df):
    """
    Plots all spectral samples for tissue and non-tissue, overlaying them for visualization.

    Parameters:
        xtrain (pd.DataFrame or np.array): Spectral intensity data (samples x wavelengths).
        ytrain (pd.Series or np.array): Labels ('tissue' or 'non-tissue') corresponding to samples.
        wavelength_df (pd.DataFrame or np.array): Wavelength values corresponding to spectral features.

    Returns:
        None (Displays a plot).
    """

    # Convert inputs to NumPy arrays if they are Pandas objects
    xtrain = xtrain.values if isinstance(xtrain, pd.DataFrame) else xtrain
    ytrain = ytrain.values if isinstance(ytrain, pd.Series) else ytrain
    wavelengths = wavelength_df.values.flatten() if isinstance(wavelength_df, pd.DataFrame) else wavelength_df.flatten()

    # Separate spectra by class
    tissue_spectra = xtrain[ytrain == 'Tissue']
    non_tissue_spectra = xtrain[ytrain == 'Non-Tissue']

    # # Create plot
    # plt.figure(figsize=(12, 6))

    # # Plot all tissue spectra in blue
    # for spectrum in tissue_spectra:
    #     plt.plot(wavelengths, spectrum, color='blue', alpha=0.3, linewidth=0.7)

    # # Plot all non-tissue spectra in red
    # for spectrum in non_tissue_spectra:
    #     plt.plot(wavelengths, spectrum, color='red', alpha=0.3, linewidth=0.7)

    # # Labels and legend
    # plt.xlabel("Wavelength (nm)")
    # plt.ylabel("Intensity")
    # plt.title("All Spectral Samples for Tissue and Non-Tissue")
    # plt.legend(["Tissue", "Non-Tissue"])
    # plt.show()
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Plot tissue spectra
    for spectrum in tissue_spectra:
        axes[0].plot(wavelengths, spectrum, color='blue', alpha=0.3, linewidth=0.7)
    axes[0].set_title("Tissue Spectra")
    axes[0].set_xlabel("Wavelength (nm)")
    axes[0].set_ylabel("Intensity")

    # Plot non-tissue spectra
    for spectrum in non_tissue_spectra:
        axes[1].plot(wavelengths, spectrum, color='red', alpha=0.3, linewidth=0.7)
    axes[1].set_title("Non-Tissue Spectra")
    axes[1].set_xlabel("Wavelength (nm)")

    plt.suptitle("Spectral Samples for Tissue and Non-Tissue")
    plt.tight_layout()
    plt.show()