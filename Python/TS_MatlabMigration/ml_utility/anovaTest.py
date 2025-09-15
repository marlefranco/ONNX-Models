import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f_oneway


import numpy as np
from scipy.stats import f_oneway

def anova_significant_peaks(xtrain, ytrain, wavelength_df, alpha=0.05):


    if hasattr(xtrain, 'values'):
        xtrain = xtrain.values
    if hasattr(ytrain, 'values'):
        ytrain = ytrain.values
    if hasattr(wavelength_df, 'values'):
        wavelength_df = wavelength_df.values.flatten() 
    p_values = []
    for wavelength in range(xtrain.shape[1]):
        tissue_intensities = xtrain[ytrain == 'Tissue', wavelength]
        non_tissue_intensities = xtrain[ytrain == 'Non-Tissue', wavelength]
        f_stat, p_value = f_oneway(tissue_intensities, non_tissue_intensities)
        p_values.append(p_value)


   # significant_peaks = np.where(np.array(p_values) < alpha)[0]
   # significant_wavelengths = wavelength_df[significant_peaks] 
    #return significant_wavelengths, np.array(p_values)
    p_values = np.array(p_values)
    significant_indices = np.where(p_values < alpha)[0]
    return wavelength_df[significant_indices], significant_indices, p_values

# Visualize the results
# def plot_significant_peaks(xtrain, ytrain, significant_peaks):
#     """
#     Plot average spectra and highlight significant peaks.
#     """
#     # Compute average spectra
#     average_spectrum_tissue = np.mean(xtrain[ytrain == 'tissue', :], axis=0)
#     average_spectrum_non_tissue = np.mean(xtrain[ytrain == 'non-tissue', :], axis=0)
#     wavelengths = np.arange(xtrain.shape[1])  # Wavelength indices

#     # Plot average spectra
#     plt.figure(figsize=(10, 6))
#     plt.plot(wavelengths, average_spectrum_tissue, label='Tissue', color='blue')
#     plt.plot(wavelengths, average_spectrum_non_tissue, label='Non-Tissue', color='red')

#     # Highlight significant peaks
#     for peak in significant_peaks:
#         plt.axvline(x=peak, color='green', linestyle='--', alpha=0.5)

#     # Add labels and legend
#     plt.xlabel('Wavelength Index')
#     plt.ylabel('Intensity')
#     plt.title('Average Spectra with Significant Wavelength Regions (ANOVA)')
#     plt.legend()
#     plt.show()

def plot_significant_peaks(xtrain, ytrain, wavelength_df, significant_wavelengths):
    """
    Plot average spectra and highlight significant peaks.
    """
    # Convert to NumPy arrays if needed
    if hasattr(xtrain, 'values'):
        xtrain = xtrain.values
    if hasattr(ytrain, 'values'):
        ytrain = ytrain.values
    if hasattr(wavelength_df, 'values'):
        wavelength_df = wavelength_df.values.flatten()

    # Compute average spectra
    average_spectrum_tissue = np.mean(xtrain[ytrain == 'Tissue', :], axis=0)
    average_spectrum_non_tissue = np.mean(xtrain[ytrain == 'Non-Tissue', :], axis=0)

    # Plot average spectra
    plt.figure(figsize=(8, 5), dpi=80)
    plt.plot(wavelength_df, average_spectrum_tissue, label='Tissue', color='blue')
    plt.plot(wavelength_df, average_spectrum_non_tissue, label='Non-Tissue', color='red')

    # Highlight significant peaks
    for wavelength in significant_wavelengths:
        plt.axvline(x=wavelength, color='green', linestyle='--', alpha=0.5)

    # Add labels and legend
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity')
    plt.title('Average Spectra with Significant Wavelength Regions (ANOVA)')
    plt.legend()
    plt.show()

def find_significant_wavelengths(xtrain, ytrain, wavelengths, alpha=0.05):
    """
    Identifies significant wavelengths using ANOVA for tissue vs. non-tissue classification.

    Parameters:
        xtrain (np.array): Training intensity data (samples × wavelengths).
        ytrain (np.array): Labels ('tissue' or 'non-tissue').
        wavelengths (np.array): Wavelength values corresponding to columns in xtrain.
        alpha (float): Significance level for selecting important features.

    Returns:
        significant_wavelengths (np.array): Wavelengths where spectral differences are significant.
        p_values (np.array): P-values for all wavelengths.
    """
    if hasattr(xtrain, 'values'):
        xtrain = xtrain.values
    if hasattr(ytrain, 'values'):
        ytrain = ytrain.values
    if hasattr(wavelengths, 'values'):
        wavelengths = wavelengths.values.flatten() 
    p_values = []
    unique_labels = np.unique(ytrain)
    
    for i in range(xtrain.shape[1]):  # Loop over all wavelengths
        groups = [xtrain[ytrain == label, i] for label in unique_labels]
        f_stat, p_value = f_oneway(*groups)
        p_values.append(p_value)

    p_values = np.array(p_values)
    significant_wavelengths = wavelengths[p_values < alpha]  # Select wavelengths with p < alpha
    return significant_wavelengths, p_values

def find_roi_anova(xtrain, ytrain, wavelengths, alpha=0.05, min_gap=5):
    """
    Identifies regions of interest (ROI) using ANOVA for tissue vs. non-tissue classification.

    Parameters:
        xtrain (np.array): Spectral intensity data (samples x wavelengths).
        ytrain (np.array): Labels (samples,).
        wavelengths (np.array): Wavelength values corresponding to columns in xtrain.
        alpha (float): Significance level for ANOVA test (default: 0.05).
        min_gap (int): Minimum gap (in nm) allowed between significant wavelengths to merge into a region.

    Returns:
        roi_list (list of tuples): List of ROI as (start_wavelength, end_wavelength).
        significant_wavelengths (np.array): Array of all significant wavelengths.
    """
    if hasattr(xtrain, 'values'):
        xtrain = xtrain.values
    if hasattr(ytrain, 'values'):
        ytrain = ytrain.values
    if hasattr(wavelengths, 'values'):
        wavelengths = wavelengths.values.flatten() 
    # Convert ytrain to NumPy array if it's a pandas Series
    ytrain = np.array(ytrain)
    unique_labels = np.unique(ytrain)

    # Perform ANOVA for each wavelength
    p_values = np.array([
        f_oneway(*[xtrain[ytrain == label, i] for label in unique_labels])[1]
        for i in range(xtrain.shape[1])
    ])
    
    # Identify significant wavelengths
    significant_wavelengths = wavelengths[p_values < alpha]

    # Group consecutive wavelengths into ROI
    if len(significant_wavelengths) == 0:
        return [], significant_wavelengths  # No significant regions found

    roi_list = []
    start_wl = significant_wavelengths[0]

    for i in range(1, len(significant_wavelengths)):
        if significant_wavelengths[i] - significant_wavelengths[i - 1] > min_gap:
            # If gap is too large, finalize the previous ROI
            roi_list.append((start_wl, significant_wavelengths[i - 1]))
            start_wl = significant_wavelengths[i]
    
    # Add the last region
    roi_list.append((start_wl, significant_wavelengths[-1]))

    return roi_list, significant_wavelengths

def plot_spectral_data_with_roi(xtrain, ytrain, wavelengths, roi_list):
    """
    Plots the spectral data with highlighted ROIs.
    
    """
    if hasattr(xtrain, 'values'):
        xtrain = xtrain.values
    if hasattr(ytrain, 'values'):
        ytrain = ytrain.values
    if hasattr(wavelengths, 'values'):
        wavelengths = wavelengths.values.flatten() 
    plt.figure(figsize=(10, 5))

    # Plot all spectra
    for i in range(xtrain.shape[0]):
        plt.plot(wavelengths, xtrain[i, :], alpha=0.3, color='gray')

    # Highlight ROIs
    for start, end in roi_list:
        plt.axvspan(start, end, color='red', alpha=0.3, label="ROI" if 'ROI' not in plt.gca().get_legend_handles_labels()[1] else "")

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity")
    plt.title("Spectral Data with Highlighted ROI")
    plt.legend()
    plt.show()