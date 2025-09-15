import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from joblib import Parallel, delayed  # Import joblib for parallelization

# Detect only top peaks based on prominence and distance
def detect_top_peaks(spectrum, wavelengths, num_peaks=3, prominence=0.02, distance=10):
    # Detect initial peaks
    peaks, properties = find_peaks(spectrum, prominence=prominence, distance=distance)
    
    if len(peaks) == 0:
        return [], []  # No peaks found
    
    # Get prominences of detected peaks
    prominences = properties["prominences"]
    
    # Sort peaks by prominence in descending order
    sorted_indices = prominences.argsort()[::-1]
    
    # Select top 'num_peaks' based on prominence
    top_peaks_indices = peaks[sorted_indices][:num_peaks]
    top_peaks_wavelengths = wavelengths[top_peaks_indices]
    top_peaks_intensities = spectrum[top_peaks_indices]
    
    return top_peaks_wavelengths, top_peaks_intensities

# Detect peaks across all samples with given prominence and distance
def detect_peaks_across_samples(spectra, wavelengths, num_peaks=3, prominence=0.02, distance=10):
    peak_positions = []
    for index, spectrum in spectra.iterrows():  # iterrows() iterates over DataFrame rows
        intensities = spectrum.values  # Extract intensities (numeric values) for this spectrum
        peaks_wavelengths, _ = detect_top_peaks(intensities, wavelengths, num_peaks, prominence, distance)
        peak_positions.extend(peaks_wavelengths)  # Append corresponding wavelengths of detected peaks
    return np.array(peak_positions)

# Plot histogram of detected peak wavelengths
def plot_peak_histogram(peak_positions, title="Peak Histogram", bin_width=1):
    plt.figure(figsize=(10,6))
    bins = np.arange(peak_positions.min(), peak_positions.max() + bin_width, bin_width)
    plt.hist(peak_positions, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Peak Count')
    plt.title(f'{title} - Histogram of Detected Peak Wavelengths')
    plt.grid(True)
    plt.show()

# Get the peak ranges based on a tolerance around the peaks
def get_peak_ranges(peak_positions, tolerance):
    peak_ranges = []
    for peak in peak_positions:
        peak_ranges.append((peak - tolerance, peak + tolerance))
    return peak_ranges

# Extract features (average intensity, area under curve) from defined peak ranges
def extract_features_from_ranges(spectra, wavelengths, peak_ranges, n_jobs=-1):
    # Precompute wavelength indices for each range to avoid redundant calculations
    peak_range_indices = []
    for start, end in peak_ranges:
        peak_range_indices.append(np.where((wavelengths >= start) & (wavelengths <= end))[0])
    
    # Define the feature extraction function for a single spectrum
    def extract_features_from_single_spectrum(spectrum_values):
        features = []
        for indices in peak_range_indices:
            intensities_in_range = spectrum_values[indices]
            mean_intensity = np.mean(intensities_in_range)
            features.append(mean_intensity)
        return features
    
    # Use joblib's Parallel to apply this function to each spectrum in parallel
    extracted_features = Parallel(n_jobs=n_jobs)(
        delayed(extract_features_from_single_spectrum)(spectrum.values) for _, spectrum in spectra.iterrows()
    )
    
    return np.array(extracted_features)


# Function to get tissue and non-tissue samples based on y_train labels
def get_tissue_non_tissue_samples(X_train, y_train):
    tissue_spectra = X_train[y_train == 'Tissue']
    non_tissue_spectra = X_train[y_train == 'Non-Tissue']
    return tissue_spectra, non_tissue_spectra

# Main function to extract features and plot results
def extract_and_plot_features(X_train, y_train, wavelength_df, tolerance=5, num_peaks=3, prominence=0.02, distance=10):
    tissue_spectra, non_tissue_spectra = get_tissue_non_tissue_samples(X_train, y_train)
    wavelengths = wavelength_df
    
    # Detect top peaks for tissue and non-tissue
    tissue_peaks = detect_peaks_across_samples(tissue_spectra, wavelengths, num_peaks, prominence, distance)
    non_tissue_peaks = detect_peaks_across_samples(non_tissue_spectra, wavelengths, num_peaks, prominence, distance)
    
    # Plot histograms of detected peaks
    # plot_peak_histogram(tissue_peaks, title='Tissue Spectra')
    # plot_peak_histogram(non_tissue_peaks, title='Non-Tissue Spectra')
    
    # Define peak ranges based on detected peaks with a tolerance (e.g., +/- 5 nm)
    tissue_peak_ranges = get_peak_ranges(tissue_peaks, tolerance)
    non_tissue_peak_ranges = get_peak_ranges(non_tissue_peaks, tolerance)
    
    # Extract features from the defined peak ranges for both tissue and non-tissue
    tissue_features = extract_features_from_ranges(tissue_spectra, wavelengths, tissue_peak_ranges)
    non_tissue_features = extract_features_from_ranges(non_tissue_spectra, wavelengths, non_tissue_peak_ranges)
    
    # Combine the features into a single feature matrix (tissue + non-tissue)
    feature_matrix = np.vstack([tissue_features, non_tissue_features])
    
    # Store the extracted features (e.g., in a DataFrame for future use)
    feature_df = pd.DataFrame(feature_matrix, columns=[f"Feature_{i+1}" for i in range(feature_matrix.shape[1])])
    
    # Plot the separation of features (e.g., using PCA or the first two features for simplicity)
    plt.figure(figsize=(10, 6))
    plt.scatter(tissue_features[:, 0], tissue_features[:, 1], color='blue', label='Tissue')
    plt.scatter(non_tissue_features[:, 0], non_tissue_features[:, 1], color='red', label='Non-Tissue')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Separation of Tissue and Non-Tissue Based on Extracted Features')
    plt.legend()
    plt.show()
    
    # Return the extracted features for further processing (e.g., model training)
    return feature_df
