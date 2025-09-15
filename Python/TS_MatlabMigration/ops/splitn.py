# import os
# import pandas as pd
# import numpy as np

# def partition_data_based_on_filenames(data, train_percent):
#     # Loop through every class
#     classes = data['TargetType'].unique()

#     train_indices = []
#     test_indices = []

#     for cls in classes:
#         this_class_mask = data['TargetType'] == cls
#         this_class_types = data.loc[this_class_mask, 'TargetType'] + data.loc[this_class_mask, 'TargetNumber']
#         num_class_types = len(this_class_types.unique())
#         num_train_types = round(train_percent / 100 * num_class_types)

#         this_class_train_idx = np.random.choice(this_class_types.unique(), num_train_types, replace=False)
#         this_class_test_idx = this_class_types.unique()[~np.isin(this_class_types.unique(), this_class_train_idx)]

#         train_indices.extend(data.index[this_class_mask & (this_class_types.isin(this_class_train_idx))].tolist())
#         test_indices.extend(data.index[this_class_mask & (this_class_types.isin(this_class_test_idx))].tolist())

#     return data.loc[train_indices], data.loc[test_indices], train_indices, test_indices

import pandas as pd
import numpy as np

def partition_data_based_on_filenames(data, train_percent):
    # Loop through every class
    classes = data['TargetType'].unique()

    train_indices = []
    test_indices = []

    # Go through each TargetType (COM 1, COM 2, etc.)
    for cls in classes:
        # Filter data for the specific target type
        this_class_mask = data['TargetType'] == cls
        this_class_data = data[this_class_mask]
        
        # Create a combination of TargetType + TargetNumber for unique samples
        this_class_types = this_class_data['TargetType'] + this_class_data['TargetNumber'].astype(str)
        
        # Get the unique combinations of TargetType + TargetNumber
        unique_class_types = this_class_types.unique()
        
        # Calculate number of train samples based on train_percent
        num_train_samples = int(train_percent / 100 * len(unique_class_types))
        
        # Randomly select train samples from the unique combinations
        train_class_samples = np.random.choice(unique_class_types, num_train_samples, replace=False)
        
        # The remaining samples will go to the test set
        test_class_samples = np.setdiff1d(unique_class_types, train_class_samples)
        
        # Add the corresponding indices to train and test sets
        train_indices.extend(this_class_data[this_class_types.isin(train_class_samples)].index)
        test_indices.extend(this_class_data[this_class_types.isin(test_class_samples)].index)

    # Split the data into train and test based on indices
    train_data = data.loc[train_indices]
    test_data = data.loc[test_indices]

    return train_data, test_data, train_indices, test_indices