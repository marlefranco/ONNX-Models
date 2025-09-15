import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import trapz

# Function to extract features from intensity data
def extract_features(x_train, wavelength_df, labels):
    """
    Extracts features like power ratios, AUC, standard deviation, and peak-to-trough values from spectral data.
    
    Parameters:
    - x_train: Intensity data (spectra).
    - wavelength_df: Corresponding wavelength values.
    - labels: Tissue/Non-Tissue labels for the spectra.

    Returns:
    - features_df: DataFrame containing the extracted features.
    """
    # Define the ranges for power ratios and features (using the index of the wavelengths)
    range1_start = np.where(wavelength_df >= 460)[0][0]
    range1_end = np.where(wavelength_df <= 480)[0][-1]
    range2_start = np.where(wavelength_df >= 560)[0][0]
    range2_end = np.where(wavelength_df <= 580)[0][-1]
    range3_start = np.where(wavelength_df >= 660)[0][0]
    range3_end = np.where(wavelength_df <= 680)[0][-1]
    
    features = []
    
    # Loop through each spectrum (row in x_train)
    for i, spectrum in enumerate(x_train.values):  # Use .values to get the underlying array of intensities
        # Power Ratios
        range1_intensity = np.mean(spectrum[range1_start:range1_end+1])
        range2_intensity = np.mean(spectrum[range2_start:range2_end+1])
        range3_intensity = np.mean(spectrum[range3_start:range3_end+1])
        
        power_ratio1 = range1_intensity / range2_intensity
        power_ratio2 = range2_intensity / range3_intensity
        
        # Additional Features (AUC, Standard Deviation, Peak-to-Trough, etc.)
        auc_range1 = trapz(spectrum[range1_start:range1_end+1], wavelength_df[range1_start:range1_end+1])
        auc_range2 = trapz(spectrum[range2_start:range2_end+1], wavelength_df[range2_start:range2_end+1])
        auc_range3 = trapz(spectrum[range3_start:range3_end+1], wavelength_df[range3_start:range3_end+1])
        
        std_range1 = np.std(spectrum[range1_start:range1_end+1])
        std_range2 = np.std(spectrum[range2_start:range2_end+1])
        std_range3 = np.std(spectrum[range3_start:range3_end+1])
        
        peak_to_trough_range1 = np.max(spectrum[range1_start:range1_end+1]) / np.min(spectrum[range1_start:range1_end+1])
        peak_to_trough_range2 = np.max(spectrum[range2_start:range2_end+1]) / np.min(spectrum[range2_start:range2_end+1])
        
        smoothed_spectrum = np.convolve(spectrum, np.ones(5)/5, mode='same')
        smoothed_std = np.std(smoothed_spectrum[range1_start:range1_end+1])
        
        # Combine features for this spectrum into a list
        features.append([power_ratio1, power_ratio2, auc_range1, auc_range2, auc_range3,
                         std_range1, std_range2, std_range3, peak_to_trough_range1, peak_to_trough_range2, smoothed_std, labels[i]])
    
    # Convert the list of features into a DataFrame
    columns = ['Power Ratio 1', 'Power Ratio 2', 'AUC Range 1', 'AUC Range 2', 'AUC Range 3',
               'STD Range 1', 'STD Range 2', 'STD Range 3', 'Peak-to-Trough Range 1', 
               'Peak-to-Trough Range 2', 'Smoothed STD', 'Label']
    features_df = pd.DataFrame(features, columns=columns)
    
    return features_df


# Function to plot features
def plot_features(features_df):
    """
    Plot pairwise relationships (scatter plots) between different features.
    
    Parameters:
    - features_df: DataFrame containing the extracted features.
    """
    # Plot pairwise relationships (e.g., scatter plots) between different features
    sns.pairplot(features_df, hue="Label", vars=['Power Ratio 1', 'Power Ratio 2', 'AUC Range 1', 'AUC Range 2', 'AUC Range 3'])
    plt.title('Feature Relationships')
    plt.show()

    # Example of plotting the distribution of specific features
    plt.figure(figsize=(12, 6))

    # Power Ratio 1 Distribution
    plt.subplot(2, 2, 1)
    sns.histplot(features_df['Power Ratio 1'], kde=True, hue=features_df['Label'])
    plt.title('Power Ratio 1 Distribution')

    # Power Ratio 2 Distribution
    plt.subplot(2, 2, 2)
    sns.histplot(features_df['Power Ratio 2'], kde=True, hue=features_df['Label'])
    plt.title('Power Ratio 2 Distribution')

    # AUC Range 1 Distribution
    plt.subplot(2, 2, 3)
    sns.histplot(features_df['AUC Range 1'], kde=True, hue=features_df['Label'])
    plt.title('AUC Range 1 Distribution')

    # AUC Range 2 Distribution
    plt.subplot(2, 2, 4)
    sns.histplot(features_df['AUC Range 2'], kde=True, hue=features_df['Label'])
    plt.title('AUC Range 2 Distribution')

    plt.tight_layout()
    plt.show()

# Function to plot individual spectra with regions of interest
def plot_spectrum_with_regions(spectrum, wavelength_df):
    """
    Plot an individual spectrum with highlighted regions of interest (e.g., AUC ranges).
    
    Parameters:
    - spectrum: Intensity values of the spectrum.
    - wavelength_df: Corresponding wavelength values.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(wavelength_df, spectrum, label='Spectrum')

    # Highlight the regions of interest (e.g., ranges for AUC calculations)
    plt.axvspan(460, 480, color='orange', alpha=0.3, label='Range 1 (460-480 nm)')
    plt.axvspan(560, 580, color='green', alpha=0.3, label='Range 2 (560-580 nm)')
    plt.axvspan(660, 680, color='blue', alpha=0.3, label='Range 3 (660-680 nm)')

    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity')
    plt.title('Spectrum with Highlighted Regions of Interest')
    plt.legend()
    plt.show()
