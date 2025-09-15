import pandas as pd
from sklearn.model_selection import train_test_split

def partition(data, holdout=0.3, random_state=0):
    """
    Partition a dataframe into training and testing sets with an additional column indicating the partition.
    
    Parameters:
    - data: pandas DataFrame
    - holdout: float, proportion of the dataset to be held out for testing (default: 0.3)
    - random_state: int, random seed for reproducibility (default: 0)
    
    Returns:
    - result: pandas DataFrame with an additional 'Partition' column (categorical 'Train' or 'Test')
    """
    # Default if holdout is 1, assign everything to Test
    if holdout == 1:
       data.loc[:, 'Partition'] = 'Test'
    else:
        # Split the data into train and test sets
        train_data, test_data = train_test_split(data, test_size=holdout, random_state=random_state)
        
        # Add a 'Partition' column
        data.loc[:, 'Partition'] = 'Train'
        data.loc[test_data.index, 'Partition'] = 'Test'
    
    # Convert 'Partition' column to a categorical type
    data['Partition'] = pd.Categorical(data['Partition'], categories=['Train', 'Test'])
    
    return data
