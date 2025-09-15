import numpy as np

data = np.array([
    9.7933, 8.5633, 10.9633, 3.0233, -0.0767, 10.0733, 5.9433, 12.1233, 5.5133,
    5.3133, 0.7033, 3.4833, 1.2533, 7.3533, 3.0933, 3.3733, 7.3733, -0.8367,
    -6.2967, 3.0633, 2.7433, 5.0933, 4.7433, 4.2633, 2.7233, -1.6067, 2.5533,
    0.4133, -0.4267, 2.8033, 1.1533, 1.5433, 0.5233, 4.9633, 3.9833
])
import numpy as np

def moving_mean(arr, window_size):
    # Calculate half window size
    half_window = window_size // 2

    # Symmetric padding for edge handling
    padded = np.pad(arr, (half_window, half_window), mode='edge')

    # Perform the moving mean
    result = np.convolve(padded, np.ones(window_size) / window_size, mode='valid')
    return result
# Define the window size
window_size = 35

# Calculate the moving mean
result = moving_mean(data, window_size)

# Print the first 40 values
print(result[:40])