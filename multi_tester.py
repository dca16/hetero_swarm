import pathlib
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TFSMLayer

# Define the path to the folder containing the images
train_images_path = pathlib.Path('/Users/domalberts/Documents/GitHub/hetero_swarm/train_images')

# Define the target image size
img_height = 224  # or the height used in your model training
img_width = 224   # or the width used in your model training

# Load your trained model
model_path = '/Users/domalberts/Documents/GitHub/hetero_swarm/d_model'
testing_model = TFSMLayer(model_path, call_endpoint='serving_default')

# List the class names in the order they were trained
class_names = ['blue_cyl', 'red_sphere', 'green_cube']  # replace with your actual class names

# Initialize the counter for correct predictions
correct_predictions = 0
total_predictions = 0

# Iterate over all images in the train_images folder
for image_path in train_images_path.glob('**/*.png'):  # or use '*.jpg' if your images are in jpg format
    # Extract the true class from the folder name
    true_class_name = image_path.parent.name

    img = tf.keras.utils.load_img(
        image_path, target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    # Convert the input tensor to uint8
    img_array = tf.cast(img_array, tf.uint8)

    predictions = testing_model(img_array, training=False)

    # Extract the predicted class index from 'detection_classes'
    detection_classes = predictions['detection_classes'].numpy()
    predicted_class_index = int(detection_classes[0][0]) - 1  # assuming class indices start from 1

    # Debugging statements to check the predicted class index
    print(f"Predicted class index: {predicted_class_index}")
    print(f"True class name: {true_class_name}")

    # Ensure the predicted class index is within the range of class_names
    if 0 <= predicted_class_index < len(class_names):
        if class_names[predicted_class_index] == true_class_name:
            correct_predictions += 1
    else:
        print(f"Warning: Predicted class index {predicted_class_index} is out of range")

    total_predictions += 1

# Calculate and print the percentage of correct predictions
accuracy = (correct_predictions / total_predictions) * 100
print(f"Model accuracy: {accuracy:.2f}% ({correct_predictions}/{total_predictions} correct predictions)")
