import pandas as pd
import numpy as np
from scipy.ndimage import uniform_filter1d
""" def moving_mean(arr, window_size):
    result = np.convolve(arr, np.ones(window_size) / window_size, mode='same')
    return result
 """

def moving_mean(arr, window_size):
    return uniform_filter1d(arr, size=window_size, mode='reflect')
 