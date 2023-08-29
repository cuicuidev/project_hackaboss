import numpy as np
import tensorflow as tf
import glob
import matplotlib.pyplot as plt
import os
import keras.backend as K

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
    num_filters = min(intermediate_output.shape[-1], 3)  # Get up to 3 filters.
    fig, axes = plt.subplots(1, num_filters, figsize=(5 * num_filters, 5))
    
    for i in range(num_filters):
        if num_filters == 1:
            ax = axes
        else:
            ax = axes[i]
        ax.imshow(intermediate_output[0, :, :, i], cmap='gray')
        ax.set_title(f'Filter {i}')
        ax.axis('off')

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



path = './car_make_images/train/AlfaRomeo/1_AlfaRomeo.jpg'
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

# Squeeze the last dimension to convert the image to 2D
img_2d = tf.squeeze(img).numpy()

filters = model.layers[0].weights[0].numpy()
filters = filters - filters.min()
filters = filters / filters.max()

# Get the number of output channels (filters)
n_filters = filters.shape[3]

fig, axes = plt.subplots(1, n_filters, figsize=(20, 20))
for i in range(n_filters):
    axes[i].imshow(filters[:, :, 0, i], cmap='gray')
    axes[i].set_title(f'Filter {i}')
    axes[i].axis('off')
plt.show()

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