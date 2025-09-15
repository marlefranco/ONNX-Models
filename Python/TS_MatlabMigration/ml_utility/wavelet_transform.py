import numpy as np
import pywt

def wavelet_transform(X, wavelet="db4", level=4):
    transformed_data = []
    
    for i in range(X.shape[0]):  # Loop through each sample (row)
        coeffs = pywt.wavedec(X.iloc[i, :], wavelet, level=level)
        transformed_sample = np.concatenate(coeffs)  # Flatten coefficients
        transformed_data.append(transformed_sample)

    return np.array(transformed_data)

