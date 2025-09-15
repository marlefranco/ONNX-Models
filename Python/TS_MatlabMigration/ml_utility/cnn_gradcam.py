import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, MaxPooling1D, Dropout, Flatten, Dense

def preprocess_labels(Y_train):
    # Convert 'Tissue' and 'Non-tissue' labels to binary (1 and 0)
    Y_train_numeric = np.where(Y_train == 'Tissue', 1, 0)
    return Y_train_numeric


def create_cnn_model(input_shape):
    inputs = Input(shape=input_shape)  # Functional API Input Layer

    # First Convolutional Layer
    x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)

    # Second Convolutional Layer
    x = Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)

    # Flatten and Fully Connected Layers
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    # Output Layer (Binary Classification)
    outputs = Dense(1, activation='sigmoid')(x)

    # Define Model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model


### ---- GRAD-CAM IMPLEMENTATION ---- ###
def compute_grad_cam(model, input_data, target_class):
    """
    Compute Grad-CAM heatmap for a given sample and target class.
    
    :param model: Trained Keras model
    :param input_data: Single sample input (shape: (1, sequence_length, channels))
    :param target_class: The actual class index (0 or 1)
    :return: Grad-CAM heatmap
    """
    # Ensure input data has batch dimension
    if len(input_data.shape) == 2:
        input_data = np.expand_dims(input_data, axis=0)  

    # Find last Conv1D layer dynamically
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv1D):
            last_conv_layer = layer
            break
    if last_conv_layer is None:
        raise ValueError("No Conv1D layer found in model")

    # Create a model that maps input to last conv layer + output
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[last_conv_layer.output, model.output]
    )

    # Record gradients of the target class w.r.t. last conv layer
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_data)
        loss = predictions[:, target_class]  # Loss for the given class

    # Compute gradients
    grads = tape.gradient(loss, conv_outputs)

    # Global average pooling to get importance weights
    pooled_grads = tf.reduce_mean(grads, axis=1)

    # Weight the conv layer output
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    # Normalize the heatmap
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap)

    return heatmap.numpy().squeeze()
