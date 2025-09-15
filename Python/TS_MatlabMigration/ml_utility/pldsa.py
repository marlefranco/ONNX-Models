import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import LabelEncoder

# --- PLS-DA for important peak selection ---
def pls_important_peaks(xtrain, ytrain, wavelengths, top_n=10):
    """
    Perform PLS-DA to identify the most important spectral peaks.
    """
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(ytrain)

    pls = PLSRegression(n_components=2)
    pls.fit(xtrain, y_encoded)

    feature_importance = np.abs(pls.coef_).flatten()
    important_indices = np.argsort(feature_importance)[-top_n:]
    
    return wavelengths[important_indices], important_indices
