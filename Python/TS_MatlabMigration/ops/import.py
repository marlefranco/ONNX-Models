import os
import pandas as pd
import numpy as np
import glob


class Import:

    @staticmethod
    def read(dslocation="", class_option="All"):
        # Validate inputs
        if not os.path.isdir(dslocation) and not os.path.isfile(dslocation):
            raise FileNotFoundError("Location of datastore not found.")

        # Get files with .xlsx extension
        files = glob.glob(os.path.join(dslocation, '*.xlsx'))

        if class_option == "ABC":
            files = [f for f in files if "all_data" in f]
        elif class_option == "D":
            files = [f for f in files if "Class_D" in f]

        # Create a pandas dataframe for storing data
        data_frames = []

        for file in files:
            ds = pd.read_excel(file)
            data_frames.append(ds)

        data = pd.concat(data_frames, ignore_index=True)

        # Process the data
        data = data.rename(columns=lambda x: "Feature " + x if x not in ["Sample ID", "Response"] else x)
        data["Response"] = pd.Categorical(data["Response"])

        # Generate summary
        summary = data.groupby("Response").size().reset_index(name="count")

        return data, summary

    @staticmethod
    def readCSV(ds_location="", file_index=float('inf'), spectrometer_type="XL", include_subfolders=True):
        # Get all files with .csv extension
        if include_subfolders:
            files = glob.glob(os.path.join(ds_location, '**/*.csv'), recursive=True)
        else:
            files = glob.glob(os.path.join(ds_location, '*.csv'))

        if not files:
            print("No files to read.")
            return [], []

        # Filter files based on spectrometer type
        if spectrometer_type == "CL":
            files = [f for f in files if "CL" in f]
        elif spectrometer_type == "XL":
            files = [f for f in files if "XL" in f]

        if not files:
            print(f"No files to read. You selected {spectrometer_type} spectrometer type.")
            return [], []

        # Select files to read based on file index
        if file_index == float('inf'):
            file_indices = range(len(files))
        else:
            file_indices = [file_index]

        all_data = []

        for idx in file_indices:
            try:
                file_path = files[idx]
                # Read spectroscopy data, skipping first rows (like A64 in MATLAB)
                data = pd.read_csv(file_path, skiprows=63)
                data = data.iloc[:, 1:].T  # remove wavelength column and transpose

                # Use only first 2048 readings for uniformity
                data = data.iloc[:, :2048]

                # Rename columns as "Feature 1", "Feature 2", ..., "Feature 2048"
                data.columns = [f"Feature {i + 1}" for i in range(data.shape[1])]

                # Metadata extraction (adjust this based on your metadata structure)
                meta_data = pd.read_csv(file_path, nrows=58)
                light_source = meta_data.loc[0, "Light Source Type"]
                probe_size = meta_data.loc[0, "Fiber Type"]
                aiming_beam = meta_data.loc[0, "Aiming Beam Status"]
                target_type = meta_data.loc[0, "Target Type"]
                target_number = meta_data.loc[0, "Target Number"]
                integration_time_used = meta_data.loc[0, "Integration Time Used (mS)"]
                spectrometer_used = meta_data.loc[0, "Spec Used"]

                # Add metadata columns to the data
                data["LightSource"] = light_source
                data["ProbeSize"] = probe_size
                data["AimingBeam"] = aiming_beam
                data["TargetType"] = target_type
                data["TargetNumber"] = target_number
                data["IntegrationTimeUsed"] = integration_time_used
                data["SpectrometerUsed"] = spectrometer_used
                data["FileName"] = os.path.basename(file_path)

                all_data.append(data)

            except Exception as e:
                print(f"Unable to read file: {file_path}. Error: {e}")
                continue

        final_data = pd.concat(all_data, ignore_index=True)

        # Metadata columns used for grouping
        meta_data_vars = [col for col in final_data.columns if not col.startswith("Feature")]
        summary = final_data.groupby(meta_data_vars).size().reset_index(name="count")

        return final_data, summary
