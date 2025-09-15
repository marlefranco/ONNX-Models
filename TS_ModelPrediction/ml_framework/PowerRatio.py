
import scipy.signal as signal

from scipy.signal import firwin, lfilter

import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import concurrent.futures
from pathlib import Path
def parse_text_file(file_path):
    data_lines = []
    found_data = False
    with open(file_path, 'r') as file:
        for line in file:
            if found_data:
                columns = line.strip().split(',')
                if len(columns) > 1:
                    parsed_line = columns
                    data_lines.append(parsed_line)
                else:
                    found_data = False  # Assuming we're done parsing lines starting with 'data'
                    break
            elif line.strip().startswith('Position'):
                found_data = True
    return data_lines

def categorize_file(filename):
    if 'UA' in filename or 'COM' in filename or 'BEG' in filename or '15P' in filename or 'CHPD' in filename or 'CYS' in filename or 'CYS' in filename or 'MAGPH' in filename:
        return 'stone'
    elif 'Calyx' in filename or 'Ureter' in filename or 'Calx' in filename:
        return 'tissue'
    else:
        return 'unknown'


def find_csv_files(folder_path, number):

    csv_files = []
    # Convert the number to string for pattern matching
    number_pattern = re.escape(str(number))
    # Compile the regex to match files starting with the number followed by '_'
    regex = re.compile(rf"^{number_pattern}_.*\.csv$")

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if regex.match(file):  # Check if the file matches the pattern
                full_path = os.path.join(root, file)
                category = categorize_file(file)  # You can define your logic for categorization
                csv_files.append((full_path, category))
    return csv_files


def apply_fir_low_pass_filter(df, sample_rate, cutoff_freq, numtaps=101):
    """
    Applies a FIR low-pass filter to each column in the DataFrame.

    Parameters:
    df (pd.DataFrame): Input DataFrame where each column is a time series.
    sample_rate (float): Sample rate of the time series.
    cutoff_freq (float): Cutoff frequency for the low-pass filter.
    numtaps (int): Number of taps (coefficients) in the FIR filter. Default is 101.

    Returns:
    pd.DataFrame: DataFrame with filtered time series.
    """
    # Design the FIR filter
    nyquist_rate = sample_rate / 2.0
    fir_coeff = firwin(numtaps, cutoff_freq / nyquist_rate)

    # Apply the filter to each column in the DataFrame
    filtered_df = df.apply(lambda col: lfilter(fir_coeff, 1.0, col), axis=0)

    return filtered_df
def plot_columns(df, x_column,num_columns,sample_type,ratio_dict,plot=False):
    # Convert DataFrame columns to numeric (if necessary)
    df = df.apply(pd.to_numeric)

    # Check if DataFrame has any columns
    if num_columns == 0:
        print("DataFrame has no columns to plot.")
        return

    # Check if the specified x_column exists in the DataFrame
    if x_column not in df.columns:
        print(f"Column '{x_column}' does not exist in the DataFrame.")
        return

        # Calculate ratio for each column
    column_ratios = []
    # Step 1: Filter the wavelength values between 400 and 530
    tissue_df = df[(df[x_column] >= 400) & (df[x_column] <= 530)]
    tissue_ratios=[]
    max_values = []
    #df = apply_fir_low_pass_filter(df, 2047, 20, numtaps=101)
    for i in range(1, num_columns):
        column_power = df.iloc[:, i].abs().sum()  # Calculate absolute total power of the column
        total_elements = df.iloc[:, i].count()
        ratio = round((df.iloc[:, i])*total_elements / column_power ,2)
        column_ratios.append(ratio)
        # Store ratio in the dictionary
        if sample_type not in ratio_dict:
            ratio_dict[sample_type] = []
        ratio_dict[sample_type].append(ratio.tolist())
        total_power = df.iloc[:, i].abs().sum()
        tissue_power = tissue_df.iloc[:, i].abs().sum()
        max_power =  ratio.max()
        tissue_total_ratio = round(tissue_power * 100 / total_power ,2)
        tissue_ratios.append(tissue_total_ratio)
    mean_tissue_ratio = round(np.mean(tissue_ratios) ,2)
    max_values.append(max_power)
    # Compute the average of ratios across all columns
    avg_ratio = np.mean(column_ratios, axis=0)
    if plot:
        # Plot the averaged ratio waveform
        plt.plot(df[x_column], avg_ratio, label=' AUC Ratio ')
        plt.xlabel("wavelength")
        plt.ylabel('Normalized Value' )
        plt.title(sample_type + " Spectrogram " )
        plt.legend()
        plt.grid(True)
        plt.show()
    return tissue_ratios ,max_values

def plot_histogram(ratios):
    plt.hist(ratios['stone'], bins=10, alpha=0.5, label='Stone')
    plt.hist(ratios['tissue'], bins=10, alpha=0.5, label='Tissue')
    plt.xlabel('Band Ratio')
    plt.ylabel('Frequency')
    plt.title('Distribution of max values')
    plt.legend()
    plt.show()

def apply_low_pass_filter(ratio_dict, cutoff_frequency, sample_rate):
    # Design the low-pass FIR filter
    nyquist_rate = sample_rate / 2.0
    numtaps = 101  # Number of taps in the filter
    fir_coeff = signal.firwin(numtaps, cutoff_frequency / nyquist_rate)

    # Apply the filter to each ratio in the ratio_dict
    filtered_ratio_dict = {}
    for sample_type, ratios in ratio_dict.items():
        filtered_ratios = []
        for ratio in ratios:
            filtered_ratio = signal.lfilter(fir_coeff, 1.0, ratio)
            filtered_ratios.append(filtered_ratio)
        filtered_ratio_dict[sample_type] = filtered_ratios

    return filtered_ratio_dict
def calculate_power_ratios(df, x_column, category, file_name, normalized, apply_filter=True,plot=True):
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    # Split file path and name
    path, name = os.path.split(file_name)

    # Convert DataFrame columns to numeric
    df = df.apply(pd.to_numeric, errors='coerce')
    df.iloc[:, 1:] = df.iloc[:, 1:].replace([np.inf, -np.inf], np.nan)
    df.iloc[:, 1:] = (df.iloc[:, 1:] * 1000).astype(int)

    # Apply low-pass filter to the DataFrame if the flag is set
    if apply_filter:
        df_filtered = apply_fir_low_pass_filter(df, sample_rate=2047, cutoff_freq=10, numtaps=101)
    else:
        df_filtered = df

    # Filter DataFrame for the x-axis range 400 to 900
    df_filtered = df_filtered[(df_filtered[x_column] >= 400) & (df_filtered[x_column] <= 900)]

    # Initialize list to store power ratios
    power_ratios = {
        "440-490 / 515-540": [],
        "550-680 / 515-540": []
    }

    # Iterate through each column (skip the first column as it is the x_column)
    for col in df_filtered.columns[1:]:
        # Filter the DataFrame for x_column ranges
        range_440_490 = df_filtered[(df_filtered[x_column] >= 440) & (df_filtered[x_column] <= 490)]
        range_515_540 = df_filtered[(df_filtered[x_column] >= 515) & (df_filtered[x_column] <= 540)]
        range_550_680 = df_filtered[(df_filtered[x_column] >= 550) & (df_filtered[x_column] <= 680)]
        #range_515_540 = df_filtered[(df_filtered[x_column] >= 515) & (df_filtered[x_column] <= 540)]

        # Calculate power for the specific column in the ranges
        power_440_490 = range_440_490[col].abs().sum() if not range_440_490.empty else 0
        power_515_540 = range_515_540[col].abs().sum() if not range_515_540.empty else 0
        power_550_680 = range_550_680[col].abs().sum() if not range_550_680.empty else 0
        
        # Calculate power ratios
        if power_515_540 != 0:  # Avoid division by zero
            ratio_460_480_560_580 = np.round(power_440_490 / power_515_540, 2)
            power_ratios["440-490 / 515-540"].append(ratio_460_480_560_580)
            #print(f"Column: {col}, Power Ratio (460-480 / 560-580): {ratio_460_480_560_580}")
        else:
            power_ratios["440-490 / 515-540"].append(None)
            #print(f"Column: {col}, Power Ratio (460-480 / 560-580): Division by zero")

        if power_515_540 != 0:  # Avoid division by zero
            ratio_560_580_660_680 = np.round(power_550_680 / power_515_540, 2)
            power_ratios["550-680 / 515-540"].append(ratio_560_580_660_680)
            #print(f"Column: {col}, Power Ratio (560-580 / 660-680): {ratio_560_580_660_680}")
        else:
            power_ratios["550-680 / 515-540"].append(None)
            #print(f"Column: {col}, Power Ratio (560-580 / 660-680): Division by zero")

    # Placeholder for further processing
    #print("Completed index and value extraction.")
    if plot:
        # Plot all columns against the specified x_column
        fig, axs = plt.subplots(1, 2, figsize=(20, 10))

        # Line plot with normalized values
        for col in df_filtered.columns[1:50]:  # Skip the first column (x_column)
            total_power = df_filtered[col].abs().sum()
            if total_power != 0:  # Avoid division by zero
                normalized_values = df_filtered[col] / total_power
                axs[0].plot(df_filtered[x_column], df_filtered[col], label=f'ADC Counts {col}')
            else:
                axs[0].plot(df_filtered[x_column], df_filtered[col], label=f'{col} (Unnormalized)')

        # Set labels, title, and grid for line plot
        axs[0].set_xlabel("Wavelength")
        axs[0].set_ylabel("ADC Counts" if normalized else "Values")
        axs[0].set_title(f'Plot of sample variations for category: {category}\nFile: {name}')
        axs[0].grid(True)
        axs[0].set_xlim(400, 900)
        axs[0].set_xticks(np.arange(400, 901, step=20))

        # Histogram plot for both power ratios
        for label, ratios in power_ratios.items():
            axs[1].hist([r for r in ratios if r is not None], bins=20, alpha=0.5, label=label)
        axs[1].set_title('Histogram of Power Ratios')
        axs[1].set_xlabel('Power Ratio')
        axs[1].set_ylabel('Frequency')
        axs[1].legend()

        # Save the plot to a PDF file
        plt.tight_layout()
        plt.savefig(f"{file_name}.pdf")
        plt.show()

        print(f"Plot saved as {file_name}.pdf")

    # Return the calculated power ratios
    return power_ratios


def generate_ratio_histograms(ratios, output_file, file_names):
    """
    Generates histograms for tissue and stone ratios.

    Parameters:
    - ratios: Dictionary containing tissue and stone ratio data.
    - output_file: Path to save the generated histogram plot.
    - file_names: List of file names analyzed.
    """
    # Extract data for stone
    stone_ratio_1 = []
    stone_ratio_2 = []
    for element in ratios.get('stone', []):
        stone_ratio_1.extend([float(x) for x in element.get("460-490 / 515-540", [])])
        stone_ratio_2.extend([float(y) for y in element.get("550-680 / 515-540", [])])

    # Extract data for tissue
    tissue_ratio_1 = []
    tissue_ratio_2 = []
    if 'tissue' in ratios and ratios['tissue']:
        for element in ratios['tissue']:
            tissue_ratio_1.extend([float(x) for x in element.get("460-490 / 515-540", [])])
            tissue_ratio_2.extend([float(y) for y in element.get("550-680 / 515-540", [])])

    # Create subplots for histograms
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    # Histogram for 460-480 / 560-580
    axes[0].hist(stone_ratio_1, bins=20, alpha=0.7, label='Stone', color='red', edgecolor='black')
    axes[0].hist(tissue_ratio_1, bins=20, alpha=0.7, label='Tissue', color='green', edgecolor='black')
    axes[0].set_title('Histogram: 460-480 / 560-580')
    axes[0].set_xlabel('Ratio Value')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()

    # Histogram for 560-580 / 660-680
    axes[1].hist(stone_ratio_2, bins=20, alpha=0.7, label='Stone', color='red', edgecolor='black')
    axes[1].hist(tissue_ratio_2, bins=20, alpha=0.7, label='Tissue', color='green', edgecolor='black')
    axes[1].set_title('Histogram: 560-580 / 660-680')
    axes[1].set_xlabel('Ratio Value')
    axes[1].legend()

    # Add file names analyzed on the side
    file_names_text = "\n".join(file_names)
    plt.gcf().text(0.02, 0.5, f"Files Analyzed:\n{file_names_text}", fontsize=8, va='center', wrap=True)

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.show()
    plt.savefig(output_file)
    plt.close()

def generate_ratio_scatter_plot(ratios, output_file,file_names):

    # Extract data for stone
    stone_x = []
    stone_y = []
    for element in ratios['stone']:
        stone_x.extend([float(x) for x in element.get("460-490 / 515-540", [])])
        stone_y.extend([float(y) for y in element.get("550-680 / 515-540", [])])

    # Extract data for tissue
    tissue_x = []
    tissue_y = []
    if 'tissue' in ratios and ratios['tissue']:
        for element in ratios['tissue']:
            tissue_x.extend([float(x) for x in element.get("460-490 / 515-540", [])])
            tissue_y.extend([float(y) for y in element.get("550-680 / 515-540", [])])

    # Create scatter plot
    plt.figure(figsize=(10, 6))
    if stone_x and stone_y:  # Check if there's data for stone
        plt.scatter(stone_x, stone_y, alpha=0.7, label='Stone', color='red')
    if tissue_x and tissue_y:  # Check if there's data for tissue
        plt.scatter(tissue_x, tissue_y, alpha=0.7, label='Tissue', color='green')

    # Add labels and title
    plt.title('Scatter Plot of Power Ratios\nFiles Analyzed')
    plt.xlabel('460-490 / 515-540')
    plt.ylabel('550-680 / 515-540')
    plt.legend()
    # Display file names on the side
    # Display file names on the side
    file_names_text = "\n".join(file_names)
    plt.gcf().text(0.02, 0.5, f"Files Analyzed:\n{file_names_text}", fontsize=8, va='center', wrap=True)
    plt.show()
    # Save and close the plot
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

folder_path = 'D:\\TRL5-TissueSense-script\\data\\TRL5-Xenon-AB-OFF'
csv_files = find_csv_files(folder_path,1)
ratios ={'stone':[],'tissue':[],}

# Initialize the ratio dictionary
ratio_dict = {}
names =[]
for file, category in csv_files:
        print(f"File: {file}, Category: {category}")
        if category == "unknown":
            print(f"Skipping file: {file} with category: {category}")
            continue
        extracted_name = file.split("_S")[0].split("_")[-1]  # Extract name before _S and _ before that
        names.append(extracted_name)
        parsed_lines = parse_text_file(file)

        # Create DataFrame from parsed lines
        df = pd.DataFrame(parsed_lines)
        num_rows, num_columns = df.shape
        print("Number of rows:", num_rows)
        print("Number of columns:", num_columns)
        # Determine the total number of columns
        num_columns = len(df.columns)
        power_ratios = calculate_power_ratios(df, 0,category,file,True, True, plot=False)

        ratios[category].append(power_ratios)

generate_ratio_scatter_plot(ratios, "power_ratios_histogram.pdf",names)
generate_ratio_histograms(ratios, "power_ratios_histogram.pdf",names)

