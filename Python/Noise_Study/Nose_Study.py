import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
from itertools import cycle
from scipy.signal import butter, filtfilt, firwin, filtfilt, savgol_filter
from scipy.stats import zscore

darkdata = 'D:/Noise/AB_OFF_04-01_CALYX/1_S3_Alpha1_DARK AB OFF.xlsx';
BSdata = 'D:/Noise/AB_OFF_04-01_CALYX/1_TRL05-MLL-04-01_FBU-04_V-2_Calyx5_S Filtered_20240906T103646.csv';
#BSdata='D:/Noise/1_BS-2 Traget With 200 um Fiber  Xenon_S Filtered_20241009T143601.csv';

# Read data from files
tempdata = pd.read_excel(darkdata, header=None, usecols='B', skiprows=6, nrows=101).values.flatten()
avg = np.mean(tempdata)

# Define a list of colors (you can use color palettes like 'tab10', 'viridis', etc.)
color_cycle = cycle(plt.cm.tab20.colors)

# Read the BS data
# Read BSdata with appropriate options - make sure this is a pandas DataFrame
dataread = pd.read_csv(BSdata, skiprows=5, nrows=2068)
# Select the column to adjust (assuming it's the second column in CSV, index 1)
dataselect = dataread.iloc[:, 1].astype(str)  # Convert to string if it's not already

# Convert string to float using pd.to_numeric
dataselect = pd.to_numeric(dataselect, errors='coerce')  # This will turn any non-numeric values into NaN

# Exclude rows with NaN values in the second column
dataread_clean = dataread[~dataselect.isna()]
#print(dataread_clean)

dataread_clean = dataread_clean.dropna()

# If dataread_clean is a DataFrame, convert the relevant column to numeric
dataread_clean = pd.to_numeric(dataread_clean.iloc[:, 0], errors='coerce')

adjusted = dataselect - avg

# Convert dataread to a numpy array (first column)
dataread_array = dataread.iloc[:, 0].to_numpy()

# Find the minimum length of the two arrays
min_length = min(len(dataread_array), len(adjusted))

# Trim the larger array (dataread_array or adjusted) from the end
dataread_array = dataread_array[:min_length]
adjusted = adjusted[:min_length]

# Now, you can safely combine the two arrays
data = np.column_stack((dataread_array, adjusted))

# Clean Data
wavelength = data[:, 0]
intensity = data[:, 1]
wavelength = pd.to_numeric(wavelength, errors='coerce')  # Convert to numeric, turning non-numeric into NaN
intensity = pd.to_numeric(intensity, errors='coerce')    # Convert to numeric, turning non-numeric into NaN

# Check for NaN values (now safe to use np.isnan)
validIndices = ~np.isnan(wavelength) & ~np.isnan(intensity)
cleanWavelength = wavelength[validIndices]
cleanIntensity = intensity[validIndices]

signalIndices = (cleanWavelength >= 470) & (cleanWavelength <= 680)
signalIntensity = cleanIntensity[signalIndices]

noiseIndices = (cleanWavelength >= 800) & (cleanWavelength <= 900)
noiseIntensity = cleanIntensity[noiseIndices]


def mean_squared_error(original_intensity, filtered_intensity):
    """
    Calculate the Mean Squared Error (MSE) between the original and filtered intensity values.

    Parameters:
    original_intensity (numpy array): The original signal (before filtering)
    filtered_intensity (numpy array): The filtered signal (after applying a filtering technique)

    Returns:
    float: The computed MSE value.
    """
    original_intensity = np.array(original_intensity)
    filtered_intensity = np.array(filtered_intensity)
    
    mse = np.mean((original_intensity - filtered_intensity) ** 2)
    return mse

def signal_distortion(original_intensity, filtered_intensity):
    """
    Calculate the signal distortion metric between the original and filtered intensity values.
    
    Signal distortion is calculated as the ratio of the variance between original and filtered signals
    to the variance of the original signal.

    Parameters:
    original_intensity (numpy array): The original signal (before filtering)
    filtered_intensity (numpy array): The filtered signal (after applying a filtering technique)

    Returns:
    float: The computed signal distortion value.
    """
    original_intensity = np.array(original_intensity)
    filtered_intensity = np.array(filtered_intensity)
    
    # Variance-based signal distortion (Ratio of variances)
    distortion = np.var(filtered_intensity - original_intensity) / np.var(original_intensity)
    return distortion

def calculate_metrics(original_intensity, filtered_intensity, signal_indices, noise_indices):
    """
    Calculate SNR, SPNR, and Pearson's correlation for the given original and filtered intensity signals.

    Parameters:
    original_intensity (numpy array): The original signal (before filtering)
    filtered_intensity (numpy array): The filtered signal (after applying a filtering technique)
    signal_indices (numpy array): Boolean array of indices representing signal region
    noise_indices (numpy array): Boolean array of indices representing noise region

    Returns:
    dict: Dictionary containing the calculated metrics.
    """
    # SNR calculation
    signal_intensity = filtered_intensity[signal_indices]
    noise_intensity = filtered_intensity[noise_indices]
    
    signal_peak = np.max(signal_intensity)
    noise_std_dev = np.std(noise_intensity)
    snr_ratio = signal_peak / noise_std_dev
    
    # SPNR calculation
    noise_peak = np.max(noise_intensity)
    spnr_ratio = signal_peak / noise_peak
    
    # Pearson's correlation coefficient
    pearsons_r = np.corrcoef(original_intensity, filtered_intensity)[0, 1]
    
    return {
        'SNR': snr_ratio,
        'SPNR': spnr_ratio,
        "Pearson's r": pearsons_r
    }

import numpy as np

'''
def calculate_phase_shift(original_intensity, filtered_intensity):
    """
    Calculate the normalized phase shift between the original and filtered intensity signals.
    This method normalizes the signals and uses cross-correlation for better accuracy.
    
    Additionally, the function returns the count of misaligned samples based on the phase shift.

    Parameters:
    original_intensity (numpy array): The original signal (before filtering)
    filtered_intensity (numpy array): The filtered signal (after applying a filtering technique)

    Returns:
    tuple: The computed phase shift in terms of number of samples and the count of misaligned samples.
    """
    # Normalize the signals
    original_norm = (original_intensity - np.mean(original_intensity)) / np.std(original_intensity)
    filtered_norm = (filtered_intensity - np.mean(filtered_intensity)) / np.std(filtered_intensity)

    # Cross-correlation of the normalized signals
    correlation = np.correlate(original_norm, filtered_norm, mode='full')

    # Find the index of the maximum correlation
    max_corr_index = np.argmax(correlation)

    # Phase shift is the shift that maximizes correlation
    phase_shift = max_corr_index - (len(filtered_intensity) - 1)

    # Calculate the misaligned samples based on the phase shift
    if phase_shift > 0:
        # If phase shift is positive, original signal is ahead of filtered signal
        misaligned_samples = filtered_intensity[phase_shift:]
    elif phase_shift < 0:
        # If phase shift is negative, filtered signal is ahead of original signal
        misaligned_samples = original_intensity[-phase_shift:]
    else:
        # If phase shift is zero, there is no misalignment
        misaligned_samples = np.array([])

    # Count the number of misaligned samples
    misaligned_count = len(misaligned_samples)

    return phase_shift, misaligned_count
'''
def calculate_phase_shift(original_intensity, filtered_intensity):
    """
    Calculate the phase shift between the original and filtered intensity signals.
    
    The phase shift is computed by finding the lag that maximizes the cross-correlation
    between the two signals.

    Parameters:
    original_intensity (numpy array): The original signal (before filtering)
    filtered_intensity (numpy array): The filtered signal (after applying a filtering technique)

    Returns:
    float: The computed phase shift in terms of number of samples.
    """
    # Cross-correlation of the original and filtered signals
    correlation = np.correlate(original_intensity, filtered_intensity, mode='full')
    
    # Find the index of the maximum correlation
    max_corr_index = np.argmax(correlation)
    
    # Phase shift is the shift that maximizes correlation
    phase_shift = max_corr_index - (len(filtered_intensity) - 1)
    
    # Calculate the misaligned samples based on the phase shift
    if phase_shift > 0:
        misaligned_samples = filtered_intensity[phase_shift:]
    elif phase_shift < 0:
        misaligned_samples = original_intensity[-phase_shift:]
    else:
        misaligned_samples = np.array([])  # No misalignment if phase shift is zero

    # Count the number of misaligned samples
    misaligned_count = len(misaligned_samples)
   

    return phase_shift, misaligned_count


def plot_box_plots(original_signal, filtered_signal, phase_shift, misaligned_count):
    """
    Plot box plots to visualize the phase shift.
    
    Parameters:
    original_signal (numpy array): The original signal.
    filtered_signal (numpy array): The filtered signal.
    phase_shift (int): The phase shift (in samples).
    """
    # Shift the filtered signal based on the calculated phase shift
    if phase_shift > 0:
        shifted_filtered_signal = np.concatenate((np.zeros(phase_shift), filtered_signal))[:len(original_signal)]
    elif phase_shift < 0:
        shifted_filtered_signal = filtered_signal[-phase_shift:]
    else:
        shifted_filtered_signal = filtered_signal

    # Create a figure for the box plot
    plt.figure(figsize=(10, 6))

    # Combine original signal and shifted filtered signal for boxplot
    data = [original_signal, shifted_filtered_signal]

    # Plot the box plot
    plt.boxplot(data, vert=False, patch_artist=True, labels=['Original signal', 'Filtered data'])

    # Adding labels and title
    plt.xlabel('Signal Value')
    plt.title(f'Box Plot of Original and Filtered Signals with Phase Shift of {phase_shift} for {misaligned_count} data points of total 2068')

    # Display the plot
    plt.grid(True)
    plt.show()

def fir_filter(data, sample_rates, cutoff_freqs, numtaps_list, signal_indices, noise_indices, wavelength):
    """
    Apply FIR filtering with different sample rates, cutoff frequencies, and numtaps values
    and compute the metrics for each.
    """
    results = {}
    plt.figure(figsize=(10, 6))

    # Plot original signal first as a baseline
    plt.plot(wavelength, data, label='Original Signal', color='black', linewidth=1.5)

    for sample_rate in sample_rates:
        nyquist_rate = sample_rate / 2.0
        
        for cutoff_freq in cutoff_freqs:
            for numtaps in numtaps_list:
                # Design the FIR filter using Hamming window
                fir_coeffs = firwin(numtaps, cutoff_freq / nyquist_rate, window='hamming')

                # Apply zero-phase filtering using filtfilt
                fir_filtered_intensity = filtfilt(fir_coeffs, 1.0, data)

                # Calculate MSE
                mse = mean_squared_error(data, fir_filtered_intensity)

                # Calculate signal distortion
                distortion = signal_distortion(data, fir_filtered_intensity)

                # Calculate other metrics (SNR, SPNR, Pearson's r)
                metrics = calculate_metrics(data, fir_filtered_intensity, signal_indices, noise_indices)

                # Calculate the phase shift and misaligned samples
                phase_shift, misaligned_count = calculate_phase_shift(data, fir_filtered_intensity)

                # Plot the box plots for phase shift
                #plot_box_plots(data, fir_filtered_intensity, phase_shift, misaligned_count)

                # Store metrics in results dictionary
                results[(sample_rate, cutoff_freq, numtaps)] = {
                    'MSE': mse,
                    'Signal Distortion': distortion,
                    'SNR': metrics['SNR'],
                    'SPNR': metrics['SPNR'],
                    "Pearson's r": metrics["Pearson's r"],
                    'Phase Shift': phase_shift,  
                    'Misaligned Samples Count': misaligned_count  
                }

                # Plot the FIR-filtered signal with a different color for each configuration
                label = f'SR={sample_rate}, CF={cutoff_freq}, NT={numtaps}'
                # Get the next color in the cycle
                color = next(color_cycle)
            
                # fir_filtered_intensity is the filtered signal, and wavelength is the x-axis
                plt.plot(wavelength, fir_filtered_intensity, label=label, color=color, linewidth=1.5)


    # Adding labels and legend
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity')
    plt.title('FIR Filtered Signals for Different Configurations')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()
    
    return results

def move_mean_filter(data, window_sizes, signal_indices, noise_indices, wavelength):
    """
    Apply moving mean filtering for a list of window sizes and compute metrics for each.

    Parameters:
    data (numpy array): The signal data to filter.
    window_sizes (list): List of window sizes for moving average filtering.
    signal_indices (numpy array): Boolean array representing the signal region.
    noise_indices (numpy array): Boolean array representing the noise region.
    wavelength (numpy array): The corresponding wavelength data for the signal.

    Returns:
    dict: Dictionary with window sizes as keys and corresponding metrics (MSE, SNR, SPNR, etc.) as values.
    """
    results = {}
    plt.figure(figsize=(10, 6))

    # Plot original signal first as a baseline
    plt.plot(wavelength, data, label='Original Signal', color='black', linewidth=2)

    for window_size in window_sizes:
        # Apply moving mean filtering
        denoised_intensity = np.convolve(data, np.ones(window_size) / window_size, mode='same')
        
        # Calculate MSE
        mse = mean_squared_error(data, denoised_intensity)
        
        # Calculate signal distortion
        distortion = signal_distortion(data, denoised_intensity)
        
        # Calculate other metrics (SNR, SPNR, Pearson's r)
        metrics = calculate_metrics(data, denoised_intensity, signal_indices, noise_indices)

        # Calculate the phase shift and misaligned samples
        phase_shift, misaligned_count = calculate_phase_shift(data, denoised_intensity)

        # Plot the box plots for phase shift
        plot_box_plots(data,denoised_intensity, phase_shift, misaligned_count)
        
        # Store metrics in results dictionary
        results[window_size] = {
            'MSE': mse,
            'Signal Distortion': distortion,
            'SNR': metrics['SNR'],
            'SPNR': metrics['SPNR'],
            "Pearson's r": metrics["Pearson's r"],
            'Phase Shift': phase_shift,  
            'Misaligned Samples Count': misaligned_count  
        }
        
        # Plot the denoised signal with a different color for each window size
        plt.plot(wavelength, denoised_intensity, label=f'Window Size {window_size}', linewidth=2)
    
    # Adding labels and legend
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity')
    plt.title('Moving Mean Filtered Signals for Different Window Sizes')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return results

def median_filter(data, window_sizes, signal_indices, noise_indices, wavelength):
    """
    Apply median filtering for a list of window sizes and compute metrics for each.

    Parameters:
    data (numpy array): The signal data to filter.
    window_sizes (list): List of window sizes for median filtering.
    signal_indices (numpy array): Boolean array representing the signal region.
    noise_indices (numpy array): Boolean array representing the noise region.
    wavelength (numpy array): The corresponding wavelength data for the signal.

    Returns:
    dict: Dictionary with window sizes as keys and corresponding metrics (MSE, SNR, SPNR, etc.) as values.
    """
    results = {}
    plt.figure(figsize=(10, 6))

    # Plot original signal first as a baseline
    plt.plot(wavelength, data, label='Original Signal', color='black', linewidth=2)

    for window_size in window_sizes:
        # Apply median filtering (centered)
        denoised_intensity = pd.Series(data).rolling(window=window_size, center=True).median().to_numpy()

        # Remove NaN values from both original and filtered signals for metric calculations
        valid_indices = ~np.isnan(data) & ~np.isnan(denoised_intensity)
        data_cleaned = data[valid_indices]
        denoised_intensity_cleaned = denoised_intensity[valid_indices]

        # Recalculate signal_indices and noise_indices for cleaned data
        signal_indices_cleaned = signal_indices[valid_indices]
        noise_indices_cleaned = noise_indices[valid_indices]

        # Calculate MSE
        mse = mean_squared_error(data_cleaned, denoised_intensity_cleaned)

        # Calculate signal distortion
        distortion = signal_distortion(data_cleaned, denoised_intensity_cleaned)

        # Calculate other metrics (SNR, SPNR, Pearson's r)
        metrics = calculate_metrics(data_cleaned, denoised_intensity_cleaned, signal_indices_cleaned, noise_indices_cleaned)

        phase_shift, misaligned_count = calculate_phase_shift(data_cleaned, denoised_intensity_cleaned)

        #if (phase_shift != 0):
        #plot_box_plots(data_cleaned, denoised_intensity_cleaned, phase_shift, misaligned_count)

        # Store metrics in results dictionary
        results[window_size] = {
            'MSE': mse,
            'Signal Distortion': distortion,
            'SNR': metrics['SNR'],
            'SPNR': metrics['SPNR'],
            "Pearson's r": metrics["Pearson's r"],
            'Phase Shift': phase_shift,  
            'Misaligned Samples Count': misaligned_count  
        }

        # Plot the median-filtered signal with a different color for each window size
        label = f'Window Size {window_size}'
        plt.plot(wavelength, denoised_intensity, label=label, linewidth=2)

    # Adding labels and legend
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity')
    plt.title('Median Filtered Signals for Different Window Sizes')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

    return results


def butter_lowpass_filter(data, fs, cutoff_freq, order):
    """
    Apply a low-pass Butterworth filter to the data.
    
    Parameters:
    data (numpy array): The signal data to filter.
    fs (int or float): The sampling frequency in Hz.
    cutoff_freq (float): The cutoff frequency in Hz.
    order (int): The order of the filter.
    
    Returns:
    numpy array: The filtered signal.
    """
    # Nyquist frequency is half of the sampling rate
    nyquist = fs / 2.0
    
    # Normalize the cutoff frequency by dividing by the Nyquist frequency
    normalized_cutoff = cutoff_freq / nyquist
    
    # Design Butterworth filter
    b, a = butter(order, normalized_cutoff, btype='low')
    
    # Apply the filter using zero-phase filtering (filtfilt)
    return filtfilt(b, a, data)


def apply_butter_filter_for_multiple_params(data, fs_list, cutoff_freqs, orders, signal_indices, noise_indices, wavelength):
    """
    Apply the low-pass Butterworth filter with multiple fs, cutoff frequencies, and orders, and calculate metrics for each combination.

    Parameters:
    data (numpy array): The signal data to filter.
    fs_list (list): List of sampling frequencies.
    cutoff_freqs (list): List of cutoff frequencies in Hz.
    orders (list): List of filter orders.
    signal_indices (numpy array): Boolean array representing the signal region.
    noise_indices (numpy array): Boolean array representing the noise region.
    wavelength (numpy array): The corresponding wavelength data for the signal.
    
    Returns:
    dict: Dictionary with filter configurations as keys and corresponding metrics as values.
    """
    results = {}
    plt.figure(figsize=(10, 6))

    # Plot original signal first as a baseline
    plt.plot(wavelength, data, label='Original Signal', color='black', linewidth=2)

    # Loop over all combinations of fs, cutoff_freq, and order
    for fs in fs_list:
        for cutoff_freq in cutoff_freqs:
            for order in orders:
                # Apply Butterworth low-pass filter
                denoised_intensity = butter_lowpass_filter(data, fs, cutoff_freq, order)
                
                # Calculate MSE
                mse = mean_squared_error(data, denoised_intensity)
                
                # Calculate signal distortion
                distortion = signal_distortion(data, denoised_intensity)
                
                # Calculate other metrics (SNR, SPNR, Pearson's r)
                metrics = calculate_metrics(data, denoised_intensity, signal_indices, noise_indices)

                phase_shift, misaligned_count = calculate_phase_shift(data, denoised_intensity)

                
                #if (phase_shift != 0):
                plot_box_plots(data, denoised_intensity, phase_shift, misaligned_count)
                
                # Store metrics in results dictionary with configuration as the key
                filter_config = f"fs={fs}, cutoff={cutoff_freq}Hz, order={order}"
                results[filter_config] = {
                    'MSE': mse,
                    'Signal Distortion': distortion,
                    'SNR': metrics['SNR'],
                    'SPNR': metrics['SPNR'],
                    "Pearson's r": metrics["Pearson's r"],
                    'Phase Shift': phase_shift,  
                    'Misaligned Samples Count': misaligned_count 
                }

                # Plot the denoised signal with a different color for each configuration
                label = f'fs={fs}, cutoff={cutoff_freq}Hz, order={order}'
                plt.plot(wavelength, denoised_intensity, label=label, linewidth=2)

    # Adding labels and legend
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity')
    plt.title('Low-Pass Butterworth Filtered Signals for Different Configurations')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

    return results

def savgol_filter_fun(data, frame_lengths, poly_orders, signal_indices, noise_indices, wavelength):
    """
    Apply Savitzky-Golay filtering for a list of frame lengths and polynomial orders, 
    and compute metrics for each.

    Parameters:
    data (numpy array): The signal data to filter.
    frame_lengths (list): List of frame lengths for Savitzky-Golay filtering.
    poly_orders (list): List of polynomial orders for Savitzky-Golay filtering.
    signal_indices (numpy array): Boolean array representing the signal region.
    noise_indices (numpy array): Boolean array representing the noise region.
    wavelength (numpy array): The corresponding wavelength data for the signal.

    Returns:
    dict: Dictionary with (frame_length, poly_order) pairs as keys and corresponding metrics (MSE, SNR, SPNR, etc.) as values.
    """
    results = {}
    plt.figure(figsize=(10, 6))

    # Plot original signal first as a baseline
    plt.plot(wavelength, data, label='Original Signal', color='black', linewidth=2)

    for frame_length in frame_lengths:
                # Ensure frame_length is less than or equal to the length of data
        if frame_length > len(data):
            print(f"Skipping frame length {frame_length} because it's larger than the data size.")
            continue
        #print(f"Data shape: {data.shape}")
        for poly_order in poly_orders:
            # Apply Savitzky-Golay filter to the cleaned intensity data
            sg_filtered_intensity = signal.savgol_filter(data, frame_length, poly_order)

            # Remove NaN values from both original and filtered signals for metric calculations
            valid_indices = ~np.isnan(data) & ~np.isnan(sg_filtered_intensity)
            data_cleaned = data[valid_indices]
            sg_filtered_intensity_cleaned = sg_filtered_intensity[valid_indices]

            # Recalculate signal_indices and noise_indices for cleaned data
            signal_indices_cleaned = signal_indices[valid_indices]
            noise_indices_cleaned = noise_indices[valid_indices]

            # Calculate MSE
            mse = mean_squared_error(data_cleaned, sg_filtered_intensity_cleaned)

            # Calculate other metrics (SNR, SPNR, Pearson's r)
            metrics = calculate_metrics(data_cleaned, sg_filtered_intensity_cleaned, signal_indices_cleaned, noise_indices_cleaned)

            # Calculate signal distortion
            distortion = signal_distortion(data, sg_filtered_intensity_cleaned)

            phase_shift, misaligned_count = calculate_phase_shift(data, sg_filtered_intensity_cleaned)

            #if (phase_shift!=0):
            #plot_box_plots(data, sg_filtered_intensity_cleaned, phase_shift, misaligned_count)

            # Store metrics in results dictionary
            results[(frame_length, poly_order)] = {
                'MSE': mse,
                'Signal Distortion': distortion,
                'SNR': metrics['SNR'],
                'SPNR': metrics['SPNR'],
                "Pearson's r": metrics["Pearson's r"],
                'Phase Shift': phase_shift,  
                'Misaligned Samples Count': misaligned_count 
            }

            # Plot the Savitzky-Golay filtered signal with a label indicating frame length and polynomial order
            label = f'Frame Length {frame_length}, Poly Order {poly_order}'
            plt.plot(wavelength, sg_filtered_intensity, label=label, linewidth=2)

    # Adding labels and legend
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity')
    plt.title('Savitzky-Golay Filtered Signals for Different Frame Lengths and Polynomial Orders')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

    return results
    
##########################MOVE MEAN####################################
#window_sizes = [5, 15, 25, 35, 50, 80, 100, 150]  # List of window sizes to apply moving mean filter

# Apply moving mean filter and calculate metrics for each window size
#metrics_results = move_mean_filter(cleanIntensity, window_sizes, signalIndices, noiseIndices, cleanWavelength)

###############################FIR FILTER#########################

sample_rates = [100, 500, 1000, 2047]  # List of sampling frequencies to test
cutoff_freqs = [10, 30, 45]     # List of cutoff frequencies (normalized)
numtaps_list = [35, 75, 100, 150, 200] 

#Apply FIR filter and calculate metrics for different configurations
metrics_results = fir_filter(cleanIntensity, sample_rates, cutoff_freqs, numtaps_list, signalIndices, noiseIndices, cleanWavelength)


##################################MEDIAN FILTER###########################
#window_sizes = [5, 15, 25, 35, 50, 80, 100, 150]

# Apply median filter and calculate metrics for different window sizes

#metrics_results = median_filter(cleanIntensity, window_sizes, signalIndices, noiseIndices, cleanWavelength)

##################################LPF configurations##############################
'''
fs_list = [2048]
cutoff_freqs = [10]  # Different cutoff frequencies in Hz
orders = [4]

# Assuming 'cleanIntensity', 'signalIndices', 'noiseIndices', and 'wavelength' are already defined
metrics_results = apply_butter_filter_for_multiple_params(
    cleanIntensity, fs_list, cutoff_freqs, orders, signalIndices, noiseIndices, cleanWavelength)

'''

#########################################S Golay Filter###################################

#frame_lengths = [5, 11, 21, 31]  # Example frame lengths
#poly_orders = [2, 3]  # Example polynomial orders

# Call the function
#metrics_results = savgol_filter_fun(cleanIntensity, frame_lengths, poly_orders, signalIndices, noiseIndices, wavelength)


# Print out the metrics results
for window_size, metrics in metrics_results.items():
    print(f"Window Size {window_size} Results:")
    print(f"  MSE: {metrics['MSE']}")
    print(f"  Signal Distortion: {metrics['Signal Distortion']}")
    print(f"  SNR: {metrics['SNR']}")
    print(f"  SPNR: {metrics['SPNR']}")
    print(f"  Pearson's r: {metrics['Pearson\'s r']}")
    print(f"  Phase Shift (samples): {metrics['Phase Shift']}")
    print(f"  Misaligned Samples Count: {metrics['Misaligned Samples Count']}")
    print('---')
    