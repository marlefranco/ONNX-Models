

# import numpy as np
# import pandas as pd

# def calculate_power_ratios(x_train, y_train, wavelength_df):
#     # Convert wavelength column to numeric
#     #wavelength_df = wavelength_df.apply(pd.to_numeric, errors='coerce')
#     x_train = np.nan_to_num(x_train * 1000, nan=0, posinf=0, neginf=0).astype(int)
    
#     # Extract wavelength values
#     wavelengths = wavelength_df
    
#     # Define wavelength ranges
#     ranges = {
#         "460-490": (460, 490),
#         "515-540": (515, 540),
#         "550-680": (550, 680)
#     }
    
#     # ranges = {
#     #     "460-480": (460, 480),
#     #     "560-580": (560, 580),
#     #     "660-680": (660, 680)
#     # }
    
#     # Identify indices for each range
#     range_indices = {
#         key: np.where((wavelengths >= val[0]) & (wavelengths <= val[1]))[0]
#         for key, val in ranges.items()
#     }
    
#     # Initialize lists to store power ratio features
#     features = {
#         "460-490 / 515-540": [],
#         "550-680 / 515-540": []
#     }
#     labels = []
    
#     # Iterate through each sample in x_train
#     for i, sample in enumerate(x_train):  # Transpose to iterate over columns
        
#         # # Get intensity values for each range
#         # power_460_480 = np.abs(sample[range_indices["460-480"]]).sum()
#         # power_560_580 = np.abs(sample[range_indices["560-580"]]).sum()
#         # power_660_680 = np.abs(sample[range_indices["660-680"]]).sum()
        
#         # # Compute power ratios (avoid division by zero)
#         # ratio_460_480_560_580 = np.round(power_460_480 / power_560_580, 2) if power_560_580 != 0 else None
#         # ratio_560_580_660_680 = np.round(power_560_580 / power_660_680, 2) if power_660_680 != 0 else None
        
#         # # Append to feature lists
#         # features["460-480 / 560-580"].append(ratio_460_480_560_580)
#         # features["560-580 / 660-680"].append(ratio_560_580_660_680)
#         # Get intensity values for each range
#         power_460_490 = np.abs(sample[range_indices["460-490"]]).sum()
#         power_515_540 = np.abs(sample[range_indices["515-540"]]).sum()
#         power_550_680 = np.abs(sample[range_indices["550-680"]]).sum()
        
#         # Compute power ratios (avoid division by zero)
#         ratio_460_490_515_540 = np.round(power_460_490 / power_515_540, 2) if power_515_540 != 0 else None
#         ratio_550_680_515_540 = np.round(power_550_680 / power_515_540, 2) if power_515_540 != 0 else None
        
#         # Append to feature lists
#         features["460-490 / 515-540"].append(ratio_460_490_515_540)
#         features["550-680 / 515-540"].append(ratio_460_490_515_540)
#         labels.append(y_train.iloc[i])
    
#     # Create a DataFrame with extracted features and labels
#     feature_df = pd.DataFrame(features)
#     feature_df["Label"] = labels
    
#     return feature_df
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# def calculate_power_ratios(x_train, y_train, wavelength_df, ratio_ranges):
#     """
#     Computes power ratios dynamically based on given wavelength ranges.

#     Parameters:
#     - x_train: Spectral intensity data (numpy array or DataFrame).
#     - y_train: Labels corresponding to the data.
#     - wavelength_df: Wavelength values corresponding to spectral data.
#     - ratio_ranges: Dictionary with two power ratio definitions.
#       Example: 
#       ratio_ranges = {
#           "Ratio 1": ("460-490", "515-540"), 
#           "Ratio 2": ("550-680", "515-540")
#       }

#     Returns:
#     - feature_df: DataFrame with calculated power ratios and labels.
#     """
    
#     # Convert spectral data to integers (handling NaN/Inf)
#     x_train = np.nan_to_num(x_train * 1000, nan=0, posinf=0, neginf=0).astype(int)
    
#     # Extract wavelength values
#     wavelengths = wavelength_df

#     # Identify indices for each range
#     range_indices = {}
#     for key in set(val for pair in ratio_ranges.values() for val in pair):  # Get all unique ranges
#         start, end = map(int, key.split('-'))
#         range_indices[key] = np.where((wavelengths >= start) & (wavelengths <= end))[0]

#     # Initialize feature storage
#     features = {key: [] for key in ratio_ranges.keys()}
#     labels = []

#     # Iterate through each sample in x_train
#     for i, sample in enumerate(x_train):
#         power_values = {}
        
#         # Compute power for each unique range
#         for key, indices in range_indices.items():
#             power_values[key] = np.abs(sample[indices]).sum()

#         # Compute power ratios
#         for ratio_name, (num_range, denom_range) in ratio_ranges.items():
#             numerator = power_values[num_range]
#             denominator = power_values[denom_range]
#             ratio = np.round(numerator / denominator, 2) if denominator != 0 else None
#             features[ratio_name].append(ratio)

#         labels.append(y_train.iloc[i])

#     # Create a DataFrame with extracted features and labels
#     feature_df = pd.DataFrame(features)
#     feature_df["Label"] = labels
    
#     return feature_df


import numpy as np
import pandas as pd
from scipy.integrate import trapezoid as trapz # For AUC calculation
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_spectral_features(x_train, y_train, wavelength_df, ratio_ranges):
    """
    Computes power ratios along with additional spectral features: AUC, Peak-to-Trough, STD, Mean Intensity.

    Parameters:
    - x_train: Spectral intensity data (numpy array or DataFrame).
    - y_train: Labels corresponding to the data.
    - wavelength_df: Wavelength values corresponding to spectral data.
    - ratio_ranges: Dictionary with power ratio definitions.
      Example: 
      ratio_ranges = {
          "Ratio 1": ("460-490", "515-540"), 
          "Ratio 2": ("550-680", "515-540")
      }

    Returns:
    - feature_df: DataFrame with calculated spectral features and labels.
    """
    
    # Convert spectral data to handle NaN/Inf and scale to integers
    x_train = np.nan_to_num(x_train * 1000, nan=0, posinf=0, neginf=0).astype(int)
    
    # Extract wavelength values
    wavelengths = wavelength_df

    # Identify indices for each range
    range_indices = {}
    for key in set(val for pair in ratio_ranges.values() for val in pair):  # Get all unique ranges
        start, end = map(int, key.split('-'))
        range_indices[key] = np.where((wavelengths >= start) & (wavelengths <= end))[0]
    print(range_indices)

    # Initialize feature storage
    features = {key: [] for key in ratio_ranges.keys()}  # Power ratios
    additional_features = {
        "AUC_1": [], "AUC_2": [], "AUC_3": [],  # AUC for three ranges
        "Peak_to_Trough_1": [], "Peak_to_Trough_2": [], "Peak_to_Trough_3": [],  # Peak-to-Trough for three ranges
        "STD_1": [], "STD_2": [], "STD_3": [],  # Standard deviation for three ranges
        "Mean_Intensity_1": [], "Mean_Intensity_2": [], "Mean_Intensity_3": []  # Mean intensity for three ranges
    }
    labels = []

    # Iterate through each sample in x_train
    for i, sample in enumerate(x_train):
        power_values = {}

        # Compute power for each unique range
        for key, indices in range_indices.items():
            power_values[key] = np.abs(sample[indices]).sum()
        
        # Compute power ratios
        for ratio_name, (num_range, denom_range) in ratio_ranges.items():
            numerator = power_values[num_range]
            denominator = power_values[denom_range]
            ratio = np.round(numerator / denominator, 2) if denominator != 0 else None
            features[ratio_name].append(ratio)

        # Compute additional spectral features
        selected_ranges = list(range_indices.keys())[:3]  # Take first 3 wavelength ranges for extra features
        
        for j, key in enumerate(selected_ranges):
            indices = range_indices[key]
            spectrum_segment = sample[indices]
            #spectrum_segment = (spectrum_segment - np.mean(spectrum_segment)) / np.std(spectrum_segment)

            # AUC (Trapezoidal Integration)
            auc_value = trapz(spectrum_segment, wavelengths[indices])
            additional_features[f"AUC_{j+1}"].append(auc_value)

            # Peak-to-Trough Ratio (Max/Min Intensity)
            peak_to_trough = np.max(spectrum_segment) / np.min(spectrum_segment) if np.min(spectrum_segment) != 0 else None
            additional_features[f"Peak_to_Trough_{j+1}"].append(peak_to_trough)

            # Standard Deviation
            std_dev = np.std(spectrum_segment)
            additional_features[f"STD_{j+1}"].append(std_dev)

            # Mean Intensity
            mean_intensity = np.mean(spectrum_segment)
            additional_features[f"Mean_Intensity_{j+1}"].append(mean_intensity)

        labels.append(y_train.iloc[i])

    # Create DataFrame with all extracted features
    feature_df = pd.DataFrame(features)
    feature_df = pd.concat([feature_df, pd.DataFrame(additional_features)], axis=1)
    feature_df["Label"] = labels
    
    return feature_df


def plot_power_ratio_histograms(power_ratios_train):
    """
    Plots histograms for all power ratios in the dataset.
    
    Parameters:
    - power_ratios_train: DataFrame containing power ratios and labels.
    """
    ratio_columns = [col for col in power_ratios_train.columns if col in ["Ratio 1", "Ratio 2"]]
    # Print found columns
    print("Available columns:", list(power_ratios_train.columns))
    print("Selected ratio columns:", ratio_columns)
    
    # Create subplots dynamically based on the number of ratios
    fig, axes = plt.subplots(1, len(ratio_columns), figsize=(12, 6), sharey=True)

    # If there's only one ratio, `axes` might not be an array, so we convert it
    if len(ratio_columns) == 1:
        axes = [axes]

    # Loop through each ratio column and plot
    for i, ratio in enumerate(ratio_columns):
        sns.histplot(data=power_ratios_train, x=ratio, hue=power_ratios_train["Label"], kde=True, ax=axes[i])
        axes[i].set_title(f"Histogram: {ratio} Power Ratio")
        axes[i].set_xlabel("Ratio Value")
    
    # Adjust layout and show plots
    plt.tight_layout()
    plt.show()