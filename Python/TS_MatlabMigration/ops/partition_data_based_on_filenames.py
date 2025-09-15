import os
import pandas as pd
import numpy as np

def partition_data_based_on_filenames(data, train_percent):
    # Loop through every class
    classes = data['TargetType'].unique()

    train_indices = []
    test_indices = []

    for cls in classes:
        this_class_mask = data['TargetType'] == cls
        this_class_types = data.loc[this_class_mask, 'TargetType'] + data.loc[this_class_mask, 'TargetNumber']
        num_class_types = len(this_class_types.unique())
        num_train_types = round(train_percent / 100 * num_class_types)

        this_class_train_idx = np.random.choice(this_class_types.unique(), num_train_types, replace=False)
        this_class_test_idx = this_class_types.unique()[~np.isin(this_class_types.unique(), this_class_train_idx)]

        train_indices.extend(data.index[this_class_mask & (this_class_types.isin(this_class_train_idx))].tolist())
        test_indices.extend(data.index[this_class_mask & (this_class_types.isin(this_class_test_idx))].tolist())

    return data.loc[train_indices], data.loc[test_indices], train_indices, test_indices