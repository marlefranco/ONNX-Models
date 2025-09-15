import os
import pandas as pd
from ops.import_module import Import   # Assuming the Import class is in import_module.py

# Initialize empty DataFrames for metrics
sensitivity = pd.DataFrame()
specificity = pd.DataFrame()
accuracytb = pd.DataFrame()
precision = pd.DataFrame()

# Set integra and distanceRange variables
integra = 1  # Currently fixed to 1 as in the MATLAB code; can be a range or other values
distanceRange = "ZeroToFour"

# Define the data location
ds_location = os.path.join(os.getcwd(), "data", "XL_Xenon_ABOFF")

# Initialize a description of the dataset (optional, similar to datadescription in MATLAB)
data_description = 'XenonABOFF-5Avg'

# Options for reading data
options = {
    "FileIndex": float("inf"),      # or a specific index like [0, 1] for the first two files
    "SpectrometerType": "XL",        # Change to "CL" if needed
    "IncludeSubfolders": True         # Set to False if subfolders should not be included
}

# Create an instance of Import and call the read_csv method
data_all, data_summary = Import.read_csv(ds_location=ds_location, options=options)

# Print the current timestamp (equivalent to MATLAB's datestr(now, 'dd/mm/yy-HH:MM'))
print(pd.Timestamp.now().strftime('%d/%m/%y-%H:%M'))

# Show the first few rows of the data (equivalent to MATLAB's head(dataAll))
print(data_all.head())