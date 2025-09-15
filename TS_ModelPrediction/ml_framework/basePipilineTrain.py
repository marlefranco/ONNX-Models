import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from ml_utility.baseml import partition

def basePipelineTrain(data, response_col="Response", normalization="zscore", 
                        feature_selection="pca", feature_extraction="none"):
    """
    A function to simulate the basePipelineTrain process in MATLAB.

    Parameters:
    - data: pandas DataFrame
    - response_col: The name of the target column (response).
    - normalization: "none", "zscore", or "range".
    - feature_selection: "none", "fscchi2", or "pca".
    - feature_extraction: Currently not implemented, placeholder for future.

    Returns:
    - data_transformed: pandas DataFrame with transformed features and target.
    - feature_names: List of selected feature names.
    """

    # Extract feature names and response name
    features = data.drop(columns=[response_col])
    responsename = data[response_col]

    # Normalize the features
    if normalization == "zscore":
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
    elif normalization == "range":
        scaler = MinMaxScaler()
        features = scaler.fit_transform(features)

    # Apply feature selection
    if feature_selection == "pca":
        pca = PCA(n_components=0.95)  # 95% variance retained
        features = pca.fit_transform(features)
        feature_names = [f"PC{i+1}" for i in range(features.shape[1])]
    elif feature_selection == "fscchi2":
        selector = SelectKBest(score_func=chi2, k='all')
        features = selector.fit_transform(features, responsename)
        feature_names = [f"Feature_{i+1}" for i in range(features.shape[1])]
    else:
        feature_names = list(data.drop(columns=[response_col]).columns)

    # Partition the data into training and test sets
    data = partition(data)
    
    # Optionally convert response to categorical for classification tasks
    data[responsename] = pd.Categorical(data[responsename])
    
    # Store features, response, partition info in a dict (mimicking MATLAB custom properties)
    data.custom_properties = {
        'Features': featurenames,
        'Response': responsename,
        'TrainingObservations': data['Partition'] == 'Train'
    }
    
    return data
