import numpy as np
import tensorflow as tf
import glob
import matplotlib.pyplot as plt
import os
import keras.backend as K
import math

from model.settings import HEIGHT, WIDTH

def f1_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    actual_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (actual_positives + K.epsilon())
    
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val


def visualize_layer_output(model, layer_num, input_img):
    intermediate_model = tf.keras.models.Model(inputs=model.input, outputs=model.layers[layer_num].output)
    intermediate_output = intermediate_model.predict(input_img)
    
    # Assuming the output shape is (None, height, width, num_filters)
    num_filters = intermediate_output.shape[-1]
    
    fig, axes = plt.subplots(1, num_filters, figsize=(20, 20))
    for i in range(num_filters):
        axes[i].imshow(intermediate_output[0, :, :, i], cmap='gray')
        axes[i].set_title(f'Filter {i}')
        axes[i].axis('off')
    plt.show()
    
    return intermediate_output

def apply_max_pooling(intermediate_output):
    # Assuming output shape is (None, height, width, num_filters)
    pooled_output = tf.nn.max_pool(intermediate_output, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    return pooled_output

def visualize_output(intermediate_output):
    num_filters = intermediate_output.shape[-1]
    max_filters_to_show = 25  # You can change this as per your need
    
    # If there are more filters than the max allowed, sample from them
    if num_filters > max_filters_to_show:
        step = num_filters // max_filters_to_show
        indices = np.arange(0, num_filters, step)[:max_filters_to_show]
    else:
        indices = np.arange(num_filters)
    
    grid_size = math.ceil(math.sqrt(len(indices)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(5 * grid_size, 5 * grid_size))
    
    for i in range(grid_size * grid_size):
        row = i // grid_size
        col = i % grid_size
        
        ax = axes[row, col]
        
        if i < len(indices):
            ax.imshow(intermediate_output[0, :, :, indices[i]], cmap='gray')
            ax.set_title(f'Filter {indices[i]}')
        else:
            ax.axis('off')  # Turn off axes for empty subplots.
        
    plt.show()

def recursive_visualization(model, input_img, layer_num=0):
    if layer_num >= len(model.layers):
        return
    
    # Get the current layer.
    layer = model.layers[layer_num]
    layer_type = layer.__class__.__name__
    
    # Create a model to get the output of the layer.
    intermediate_model = tf.keras.models.Model(inputs=layer.input, outputs=layer.output)
    
    if layer_type == 'Conv2D':
        print(f"Visualizing layer {layer_num}: {layer_type}")
        intermediate_output = intermediate_model.predict(input_img)
        visualize_output(intermediate_output)
        
    elif layer_type == 'MaxPooling2D':
        print(f"Applying Max Pooling for layer {layer_num}")
        intermediate_output = intermediate_model.predict(input_img)
        
    else:
        print(f"Skipping layer {layer_num}: {layer_type}")
        intermediate_output = input_img  # Keep the input the same for the next layer.
        
    # Call recursively to move to the next layer.
    recursive_visualization(model, intermediate_output, layer_num + 1)



path = './car_make_images/train/AlfaRomeo/44_AlfaRomeo.jpg'
raw = tf.io.read_file(path)
img = tf.image.decode_image(raw)
img = tf.image.resize(img, (HEIGHT, WIDTH))
img = tf.image.rgb_to_grayscale(img)
plt.imshow(img/255.)
plt.show()

input_img = tf.expand_dims(img, axis = 0)
input_img = tf.cast(input_img, tf.float32)

latest_model = max(glob.glob("models/model_e*.h5"), default=None, key=os.path.getctime)
custom_metrics = [f1_score]

if latest_model is not None:
    print(f"Loading model: {latest_model}")
    model = tf.keras.models.load_model(latest_model, custom_objects={metric.__name__: metric for metric in custom_metrics})

def conv2D(image, kernel):
    # Get dimensions
    i_height, i_width = image.shape
    k_height, k_width = kernel.shape
    
    # Calculate dimensions of the output image
    o_height = i_height - k_height + 1
    o_width = i_width - k_width + 1
    
    # Initialize the output image
    output = np.zeros((o_height, o_width))
    
    # Apply convolution
    for i in range(o_height):
        for j in range(o_width):
            output[i, j] = np.sum(image[i:i+k_height, j:j+k_width] * kernel)
            
    return output

if latest_model is not None:
    print(f"Loading model: {latest_model}")
    model = tf.keras.models.load_model(latest_model, custom_objects={metric.__name__: metric for metric in custom_metrics})

# Start the recursive visualization
recursive_visualization(model, input_img)