import pandas as pd
import numpy as np
from scipy.interpolate import interp1d 
from ops.movmean import moving_mean
import os
from scipy.signal import firwin, lfilter


class Import:
    
    @staticmethod
    def normalize_spectra(interpolated_intensity, wavelength):
            """
            Performs offset correction and normalizes spectral data using 630 nm as the reference.

            Parameters:
            - interpolated_intensity (numpy array): Spectral intensity matrix (samples x wavelengths)
            - wavelength (numpy array): Corresponding wavelength values (1D array)

            Returns:
            - normalized_spectra (numpy array): Offset-corrected and normalized spectra
            - offset_corrected (numpy array): Offset-corrected spectra
            - normalization_factor (numpy array): Normalization factor used for each spectrum
            - largest_negative (numpy array): Largest negative value in each spectrum before correction
            """

            # Initialize arrays
            num_samples, num_wavelengths = interpolated_intensity.shape
            largest_negative = np.min(interpolated_intensity, axis=1)  # Find min value per row
            offset_corrected = interpolated_intensity + (0.1 - largest_negative[:, np.newaxis])  # Offset correction

            # Find index closest to 630 nm
            min_index = np.argmin(np.abs(wavelength - 630))  # Index of the closest wavelength to 630 nm
            normalization_factor = offset_corrected[:, min_index]  # Extract values at 630 nm

            # Normalize spectra
            normalized_spectra = offset_corrected / normalization_factor[:, np.newaxis]

            return normalized_spectra

    
    @staticmethod
    def apply_fir_filter(data, sample_rate, cutoff_freq, numtaps=35):
        """Applies an FIR filter to smooth spectral data."""
        nyquist_rate = sample_rate / 2.0
        fir_coeff = firwin(numtaps, cutoff_freq / nyquist_rate)
        
        # Apply FIR filter row-wise
        filtered_data = np.apply_along_axis(lambda row: lfilter(fir_coeff, 1.0, row), axis=1, arr=data)
        return filtered_data
    
    @staticmethod
    def read_csv(ds_location, darkReference, options, filter_type="movemean", sample_rate=None, cutoff_freq=None, numtaps=None):
        """
        Reads CSV files and processes spectroscopy and metadata information.

        Parameters:
        ds_location (str): Directory containing the CSV files.
        options (dict): Dictionary of options including:
            - FileIndex (int): Index of the files to read, or inf to read all files.
            - SpectrometerType (str): Type of spectrometer ("XL" or "CL").
            - IncludeSubfolders (bool): Whether to include subfolders.

        Returns:
        value (pd.DataFrame): Combined DataFrame of data from all files.
        summary (pd.DataFrame): Summary of metadata.
        """
        # Specify the range for dark reference data
       # dark_reference_range = 'B7:CW2074'  # Columns B to CW contain the 100 dark reference samples

        # Read dark reference intensity data
        dark_intensities = pd.read_excel(darkReference, sheet_name=0, usecols="B:CW", skiprows=6, nrows=2074-7+1, header=None)
        dark_reference_avg = np.random.rand(2068, 1)
       
        # Compute the average across the 100 columns
        dark_reference_avg = np.mean(dark_intensities.values, axis=1)  # Compute the mean for each row (wavelength)
        dark_reference_avg = dark_reference_avg[np.newaxis, :]
        # Transpose the result (optional if needed as row vector)
        #dark_reference = dark_reference_avg.T
        # Input validation
        file_index = options.get("FileIndex", float("inf"))
        spectrometer_type = options.get("SpectrometerType", "XL")
        include_subfolders = options.get("IncludeSubfolders", False)

        # Get all files present in location
        if include_subfolders:
            files = [os.path.join(root, file)
                     for root, _, filenames in os.walk(ds_location)
                     for file in filenames if file.endswith('.csv')]
        else:
            files = [os.path.join(ds_location, file) for file in os.listdir(ds_location)
                     if file.endswith('.csv')]
        
        
        # Determine which files to read
        file_indices = range(len(files)) if file_index == float("inf") else [file_index]

        # Initialize data container
        value = []

        # Loop through every file
        for curr_file_idx in file_indices:
            try:
                fn = files[curr_file_idx]
                print(f"Processing file: {fn}")

                # Read spectroscopy data
                #data = pd.read_csv(fn, skiprows=4, usecols="A:EU2074")

                data = pd.read_csv(fn, skiprows=4, usecols=range(151), nrows=2070, header=None)
                #data.iloc[:2, :] = data.iloc[:2, :].astype(int)
                # Extract wavelength values and consider only the first 2048 readings for uniformity
                wavelength = data.iloc[2:, 0].to_frame().T.reset_index(drop=True)
                wavelength = wavelength.iloc[:, :2048]

                # Remove wavelength column from data
                data = data.iloc[:, 1:].T.reset_index(drop=True)

                # Extract rotation values
                Rotation = data.iloc[:, 0].round().astype(int)
                data.drop(data.columns[0], axis=1, inplace=True)  # Remove first column

                # Extract position values
                position = data.iloc[:, 0].astype(int)
                data.drop(data.columns[0], axis=1, inplace=True)  # Remove first column

                # Consider only the first 2048 readings
                data = data.iloc[:, :2048]

                # Subtract dark reference from each row in data
                # Ensure dark_reference_avg is the correct shape
                dark_reference_avg_trimmed = dark_reference_avg[:, :2048]  # Trim to 2048 columns

                # Subtract row-wise
                data_subtracted = data.to_numpy() - dark_reference_avg_trimmed

                
                if filter_type == "movemean":
                    # Moving mean settings
                    window_size = 35  # Define window size (3-point moving average)
                    #data_subtracted_df = pd.DataFrame(data_subtracted)
                    intensities_smooth = moving_mean(data_subtracted, window_size)

                elif filter_type == "fir":
                    # Set default FIR parameters if not provided
                    sample_rate = sample_rate or 2047  # Default 1000 Hz
                    cutoff_freq = cutoff_freq or 10    # Default 50 Hz
                    numtaps = numtaps or 101            # Default 35 taps

                   # print(f"Applying FIR Filter (Cutoff = {cutoff_freq} Hz)...")
                    intensities_smooth = Import.apply_fir_filter(data_subtracted, sample_rate, cutoff_freq, numtaps)

                else:
                    raise ValueError("Invalid filter_type! Choose 'movemean' or 'fir'.")

                #intensities_smooth = intensities_smooth.to_numpy()
                # Combine Rotation, Position, and Intensities into one DataFrame
                combined_data = pd.DataFrame({
                    'Rotation': Rotation,
                    'Position': position,
                })
                combined_data = pd.concat([combined_data, pd.DataFrame(intensities_smooth)], axis=1)

                # Combine Rotation and Position as unique pairs
                unique_pairs = combined_data[['Rotation', 'Position']].drop_duplicates().to_numpy()

                # Find group indices for each (Rotation, Position) combination
                combined_data['Group'] = combined_data.groupby(['Rotation', 'Position']).ngroup()

                # Initialize an array to store the average spectra for each (Rotation, Position)
                n_features = intensities_smooth.shape[1]  # Number of features (columns in intensities_smooth)
                avg_spectra = np.full((len(unique_pairs), n_features), np.nan)

                # Calculate the mean spectrum for each unique group
                for i, group in enumerate(unique_pairs):
                    # Extract indices for the current group
                    group_indices = combined_data['Group'] == i

                    # Extract the corresponding spectra for this group
                    spectra_group = intensities_smooth[group_indices, :]

                    # Compute the mean spectrum for the group
                    avg_spectra[i, :] = np.nanmean(spectra_group, axis=0)

                # Extract wavelength and intensity
                wavelength = wavelength.iloc[0, :].to_numpy()  # Extract the numeric array from the DataFrame
                intensity = avg_spectra[:, :2048]  # Extract the first 2048 features
                wavelength = np.array(wavelength, dtype=np.float64)
                # Create a new wavelength range
                new_wavelength_range = np.linspace(400, 940, 2048)

                # Initialize a matrix to store interpolated intensities
                interpolated_intensity = np.zeros_like(intensity)

                # Interpolate each row separately
                for i in range(intensity.shape[0]):
                    interp_func = interp1d(wavelength, intensity[i, :], kind='linear', fill_value="extrapolate")
                    interpolated_intensity[i, :] = interp_func(new_wavelength_range)
                    
               # normalized_spectra = Import.normalize_spectra(interpolated_intensity, wavelength)

                # Convert interpolated_intensity into a DataFrame and copy it to data
                data = pd.DataFrame(interpolated_intensity)

                # Rename columns to "Feature 1", "Feature 2", ..., "Feature n"
                data.columns = [f"Feature {i+1}" for i in range(data.shape[1])]

                # Assign response data (Rotation and Position) to the DataFrame
                data['Rotation'] = unique_pairs[:, 0]  # Add Rotation
                data['Position'] = unique_pairs[:, 1]  # Add Position

                # Read metadata from rows 2081 to 2138
                meta_data = pd.read_csv(fn, header=None, skiprows=2080, nrows=58, usecols=[0, 1], names=["Attribute", "Value"])

               # Extract metadata and assign it to the data
                metadata_dict = meta_data.set_index('Attribute')['Value'].to_dict()
                data.loc[:, 'LightSource'] = metadata_dict.get("Light Source Type", "")
                data.loc[:, 'ProbeSize'] = metadata_dict.get("Fiber Type", "")
                data.loc[:, 'AimingBeam'] = metadata_dict.get("Aiming Beam Status", "")
                data.loc[:, 'TargetType'] = metadata_dict.get("Target Type", "")
                data.loc[:, 'TargetNumber'] = metadata_dict.get("Target Number", "")
                data.loc[:, 'IntegrationTimeUsed'] = metadata_dict.get("Integration Time Used (mS)", "")
                data.loc[:, 'SpectrometerUsed'] = metadata_dict.get("Spec Used", "")
                data.loc[:, 'FileName'] = fn

                # Append to value list
                value.append(data)

            except Exception as e:
                print(f'Unable to read from file: {fn}. Error: {e}')
                continue

        # Combine all data into a single DataFrame
        value = pd.concat(value, ignore_index=True) if value else pd.DataFrame()

        # Summary of metadata
        meta_data_vars = ["LightSource", "ProbeSize", "AimingBeam", "TargetType",
                          "TargetNumber", "IntegrationTimeUsed", "SpectrometerUsed", "FileName"]
        summary = value.groupby(meta_data_vars).size().reset_index(name='Count')

        return value, summary, new_wavelength_range

