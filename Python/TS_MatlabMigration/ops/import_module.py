import pandas as pd
import os


class Import:
    @staticmethod
    def read_csv(ds_location, options):
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

        # Filter files based on spectrometer type
        if spectrometer_type == "CL":
            files = [f for f in files if "CL" in f]
        elif spectrometer_type == "XL":
            files = [f for f in files if "XL" in f]

        # Determine which files to read
        file_indices = range(len(files)) if file_index == float("inf") else [file_index]

        # Initialize data container
        value = []

        # Loop through every file
        for curr_file_idx in file_indices:
            try:
                fn = files[curr_file_idx]
                print(f"Processing file: {fn}")

                # Read in spectroscopy data (starting from row 61 onwards)
                data = pd.read_csv(fn, skiprows=60, header=None)
                data.drop(data.columns[0], axis=1, inplace=True)  # Remove the first column (wavelength)
                data = data.T.reset_index(drop=True)
                #data = data.T  # Transpose equivalent to rows2vars

                # Assign 'ResponseRegression' from the first column and round it
                response_regression = data.iloc[:, 0].round()

                data.drop(data.columns[0], axis=1, inplace=True)  # Remove first column
                data = data.iloc[:, :2048]  # Consider only first 2048 readings for uniformity

                # Rename the columns to 'Feature 1', 'Feature 2', ..., 'Feature 2048'
                data.columns = [f"Feature {i+1}" for i in range(data.shape[1])]
                data['ResponseRegression'] = response_regression.round()

                # Read metadata from rows 1 to 58
                meta_data = pd.read_csv(fn, header=None, nrows=58, usecols=[0, 1], names=["Attribute", "Value"])
                
                # Extract metadata and assign it to the data
                metadata_dict = meta_data.set_index('Attribute')['Value'].to_dict()
                data.loc[:, 'LightSource'] = metadata_dict.get("Light Source", "")
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

        return value, summary