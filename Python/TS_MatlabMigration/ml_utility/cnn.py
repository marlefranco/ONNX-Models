
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from keras.models import Model
import tensorflow as tf
import torch
import torch.nn.functional as F

def create_cnn_model(X_train_combined):
    input_shape = X_train_combined.shape[1:]  # Automatically infer input shape (height, width, channels)
    model = Sequential()

    # First Convolutional Layer
    model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train_combined.shape[1], 2)))  # 2 channels (intensity + wavelength)
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.2))

    # Second Convolutional Layer
    model.add(Conv1D(128, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.2))

    # Flattening the output from Conv layers
    model.add(Flatten())

    # Fully Connected Layer
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))

    # Output Layer (for binary classification)
    model.add(Dense(1, activation='sigmoid'))  # Sigmoid for binary classification

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

def preprocess_labels(Y_train):
    # Convert 'Tissue' and 'Non-tissue' labels to binary (1 and 0)
    Y_train_numeric = np.where(Y_train == 'Tissue', 1, 0)
    return Y_train_numeric

def visualize_filters(model):
    # Visualize filters of the first convolutional layer (1D convolution)
    filters = model.layers[0].get_weights()[0]
    
    # Plot the first few filters (assuming the filters are 1D)
    num_filters = filters.shape[2]  # Filters are of shape (kernel_size, input_channels, num_filters)
    
    plt.figure(figsize=(10, 10))
    for i in range(min(5, num_filters)):
        plt.subplot(1, 5, i+1)
        plt.plot(filters[:, 0, i])  # Plot each filter's 1D weights
        plt.title(f'Filter {i+1}')
        plt.axis('off')
    plt.show()


def visualize_feature_importance(X, model, layer_name="conv2d"):
    # Ensure the model is built by calling it with an input shape (this is required if the model hasn't been called yet)
    if not model.built:
        model.build(input_shape=(None, X.shape[1], X.shape[2], 1))  # Adjust input shape if necessary

    # Create a model that outputs the activations of the intermediate layer
    model.predict(X[:1])  # Use a single batch or a small sample to trigger model initialization
    model(X[:1])
    try:
        intermediate_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    except Exception as e:
        print(f"Error accessing layer {layer_name}: {e}")
        return
    #intermediate_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

    # Get the activations for the input data
    activations = intermediate_model.predict(X)

    # Take the average activation across all samples (to get a "feature importance" estimate)
    avg_activations = np.mean(activations, axis=0)

    # Plot the activations
    num_features = avg_activations.shape[-1]  # Number of features in the last layer
    plt.figure(figsize=(12, 6))

    for i in range(num_features):
        plt.subplot(1, num_features, i + 1)
        plt.plot(avg_activations[:, i])
        plt.title(f"Feature {i+1}")
        plt.xlabel("Wavelength Index")
        plt.ylabel("Activation Value")
    
    plt.tight_layout()
    plt.show()



def grad_cam(model, X, class_index):
    last_conv_layer = model.get_layer("conv2d")  # Update with your last convolutional layer name
    grad_model = Model(inputs=model.input, outputs=[last_conv_layer.output, model.output])
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(X)
        loss = predictions[:, class_index]
    
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # Pool gradients for each feature map
    
    heatmap = np.mean(conv_outputs[0] * pooled_grads, axis=-1)  # Weight feature maps by gradients
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    return heatmap

def display_grad_cam(X, heatmap, wavelengths):
    plt.imshow(X[0], cmap='viridis', aspect='auto', interpolation='nearest')  # Show input spectrum
    plt.imshow(heatmap, cmap='jet', alpha=0.5, aspect='auto')  # Overlay Grad-CAM heatmap
    plt.colorbar()
    plt.xlabel('Wavelength (nm)')
    plt.title('Grad-CAM Visualization for Wavelength Regions')
    plt.show()
    
def compute_saliency_map(model, X, y):
    X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
    y_tensor = tf.convert_to_tensor(y, dtype=tf.int32)
    y_tensor = tf.reshape(y_tensor, (-1, 1))  # Reshape to (batch_size, 1) if it's not
    with tf.GradientTape() as tape:
        tape.watch(X_tensor)
        predictions = model(X_tensor)
        print(predictions.shape)
        loss = tf.keras.losses.binary_crossentropy(y_tensor, predictions)

    gradients = tape.gradient(loss, X_tensor)
    saliency_map = tf.reduce_max(tf.abs(gradients), axis=-1)  # Get max gradient across the spectrum
    return saliency_map.numpy()

# def plot_saliency_map(X, saliency_map, wavelengths):
#     # Debug prints to check the shapes
#     print("Shape of saliency_map:", saliency_map.shape)
#     print("Shape of wavelengths:", wavelengths.shape)
    
#     # Check shape of saliency_map[0]
#     print("Shape of saliency_map[0]:", saliency_map[0].shape)

#     plt.figure(figsize=(10, 6))

#     # Compute the average saliency map across all samples (axis 0)
#     average_saliency_map = np.mean(saliency_map, axis=0)
#     print("Shape of average_saliency_map:", average_saliency_map.shape)

#     # Check if the shape of average_saliency_map matches the number of wavelengths (2048)
#     assert average_saliency_map.shape[0] == wavelengths.shape[0], \
#         f"Shape mismatch: expected {wavelengths.shape[0]} wavelengths, but got {average_saliency_map.shape[0]} features."

#     # Plot the average saliency map
#     plt.plot(wavelengths, average_saliency_map, color='red')  # Plot the average saliency map   
#     plt.xlabel('Wavelength (nm)')
#     plt.ylabel('Saliency')
#     plt.title('Saliency Map for Wavelength Regions')
#     plt.show()
# def plot_saliency_map(saliency_map, wavelengths, threshold_percentile=97):
#     # Average the saliency map across all samples
#     average_saliency_map = np.mean(saliency_map, axis=0)

#     # Ensure wavelengths is a 1D array
#     if wavelengths.ndim > 1:
#         wavelengths = wavelengths[0, :]

#     # Determine threshold for highlighting important wavelengths
#     threshold = np.percentile(average_saliency_map, threshold_percentile)

#     # Mask values below the threshold
#     significant_wavelengths = wavelengths[average_saliency_map >= threshold]
#     significant_saliency = average_saliency_map[average_saliency_map >= threshold]

#     plt.figure(figsize=(10, 6))
    
#     # Plot all wavelengths with low opacity
#     plt.plot(wavelengths, average_saliency_map, color='gray', alpha=0.5, label="All Wavelengths")

#     # Highlight only the most important wavelengths in red
#     plt.scatter(significant_wavelengths, significant_saliency, color='red', label=f"Top {100 - threshold_percentile}% Salient Regions")
    
#     plt.xlabel('Wavelength (nm)')
#     plt.ylabel('Saliency')
#     plt.title('Saliency Map for Wavelength Regions')
#     plt.legend()
#     plt.show()

import numpy as np
import matplotlib.pyplot as plt

def plot_saliency_map(saliency_map, wavelengths, y_train_labels, threshold_percentile=98):
    """
    Plots the saliency map highlighting the most important wavelength regions separately 
    for Tissue and Non-Tissue categories.

    Args:
        saliency_map (numpy.ndarray): Array of shape (num_samples, num_features).
        wavelengths (numpy.ndarray): Array of shape (num_features,).
        y_train_labels (numpy.ndarray): Array of shape (num_samples,), containing 1 for 'Tissue' and 0 for 'Non-Tissue'.
        threshold_percentile (float): Percentile threshold for highlighting important wavelengths.
    """

    # Convert numeric labels to boolean masks
    tissue_mask = (y_train_labels == 1)  # Tissue
    non_tissue_mask = (y_train_labels == 0)  # Non-Tissue

    # Compute average saliency for each class
    tissue_saliency = np.mean(saliency_map[tissue_mask], axis=0)
    non_tissue_saliency = np.mean(saliency_map[non_tissue_mask], axis=0)

    # Apply threshold to highlight most important wavelengths
    tissue_threshold = np.percentile(tissue_saliency, threshold_percentile)
    non_tissue_threshold = np.percentile(non_tissue_saliency, threshold_percentile)

    important_tissue = wavelengths[tissue_saliency >= tissue_threshold]
    important_non_tissue = wavelengths[non_tissue_saliency >= non_tissue_threshold]

    # Plot the saliency map
    plt.figure(figsize=(12, 6))
    plt.plot(wavelengths, tissue_saliency, color='red', label='Tissue')
    plt.plot(wavelengths, non_tissue_saliency, color='blue', label='Non-Tissue')

    # Highlight important regions
    plt.scatter(important_tissue, [tissue_threshold] * len(important_tissue), color='red', marker='o', label='Important Tissue Regions')
    plt.scatter(important_non_tissue, [non_tissue_threshold] * len(important_non_tissue), color='blue', marker='o', label='Important Non-Tissue Regions')

    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Saliency')
    plt.title(f'Saliency Map Highlighting Important Wavelengths ({threshold_percentile}th percentile)')
    plt.legend()
    plt.show()
    


def compute_grad_cam(model, X_input, target_class):
    """
    Compute Grad-CAM heatmap for a given sample.

    Parameters:
        model (tf.keras.Model): Trained Keras model.
        X_input (np.array): Single sample input.
        target_class (int): Class index (0 for Non-Tissue, 1 for Tissue).

    Returns:
        np.array: Grad-CAM heatmap.
    """
    # Get the last convolutional layer
    model(X_input)  # Ensure model is called at least once
    last_conv_layer = model.get_layer('conv1d_1')  # Check layer name from `model.summary()`
    
    grad_model = tf.keras.models.Model(
        [model.input], [last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(X_input)
        loss = predictions[:, target_class]

    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))

    conv_output = conv_output.numpy()
    pooled_grads = pooled_grads.numpy()

    for i in range(conv_output.shape[-1]):
        conv_output[0, :, i] *= pooled_grads[i]

    heatmap = np.mean(conv_output, axis=-1)
    heatmap = np.maximum(heatmap, 0)  # ReLU
    heatmap /= np.max(heatmap)  # Normalize

    return heatmap

def compute_grad_cam_for_all(model, X_data, Y_labels):
    """
    Compute Grad-CAM for all samples and separate by Tissue and Non-Tissue.

    Parameters:
        model (tf.keras.Model): Trained Keras model.
        X_data (np.array): Input dataset.
        Y_labels (np.array): Corresponding labels (0 or 1).

    Returns:
        dict: Dictionary containing Grad-CAM heatmaps for Tissue and Non-Tissue.
    """
    tissue_grad_cams = []
    non_tissue_grad_cams = []
    
    for i in range(len(X_data)):
        X_sample = X_data[i]
        target_class = Y_labels[i]
        
        grad_cam = compute_grad_cam(model, X_sample, target_class)
        
        if target_class == 1:
            tissue_grad_cams.append(grad_cam)
        else:
            non_tissue_grad_cams.append(grad_cam)

    return {"tissue": tissue_grad_cams, "non_tissue": non_tissue_grad_cams}

def plot_grad_cam(heatmap, wavelengths, title="Grad-CAM"):
    """
    Plot Grad-CAM heatmap.

    Parameters:
        heatmap (np.array): Computed Grad-CAM heatmap.
        wavelengths (np.array): Wavelength values for x-axis.
        title (str): Plot title.
    """
    plt.figure(figsize=(10, 4))
    plt.plot(wavelengths, heatmap, label=title, color="blue" if "Non-Tissue" in title else "red")
    plt.title(title)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Activation")
    plt.legend()
    plt.show()

def plot_all_grad_cams(grad_cams, wavelengths):
    """
    Plot Grad-CAM heatmaps for both Tissue and Non-Tissue.

    Parameters:
        grad_cams (dict): Dictionary with 'tissue' and 'non_tissue' heatmaps.
        wavelengths (np.array): Wavelength values.
    """
    for i, heatmap in enumerate(grad_cams["tissue"]):
        plot_grad_cam(heatmap, wavelengths, title=f"Tissue Grad-CAM {i+1}")

    for i, heatmap in enumerate(grad_cams["non_tissue"]):
        plot_grad_cam(heatmap, wavelengths, title=f"Non-Tissue Grad-CAM {i+1}")
