import tensorflow as tf
import numpy as np
import os
import cv2

# Load the trained model and weights
model_path = '/Users/domalberts/Documents/GitHub/hetero_swarm/shape_classifier_model.keras'
weights_path = '/Users/domalberts/Documents/GitHub/hetero_swarm/model_weights.weights.h5'
class_names_path = '/Users/domalberts/Documents/GitHub/hetero_swarm/class_names.txt'

print(f"Loading model from: {model_path}")
model = tf.keras.models.load_model(model_path)
print("Model loaded successfully.")
print(f"Loading model weights from: {weights_path}")
model.load_weights(weights_path)
print("Model weights loaded successfully.")

# Load class names
print(f"Loading class names from: {class_names_path}")
with open(class_names_path, 'r') as f:
    class_names = f.read().splitlines()
print(f"Class names: {class_names}")
class_indices = {name: idx for idx, name in enumerate(class_names)}
print(f"Class indices: {class_indices}")

# Directory containing verification images
verification_images_dir = '/Users/domalberts/Documents/GitHub/hetero_swarm/verification_images'

# Function to classify shape in an image
def classify_shape(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")
    print(f"Original image shape: {image.shape}")
    image_resized = cv2.resize(image, (150, 150))  # Resize to (150, 150) as expected by the model
    image_array = np.expand_dims(image_resized, axis=0) / 255.0  # Normalizing the image
    predictions = model.predict(image_array)
    predicted_class_index = np.argmax(predictions)
    return class_names[predicted_class_index]

# Iterate through all images in the verification_images directory and count predictions
prediction_counts = {name: 0 for name in class_names}

for root, dirs, files in os.walk(verification_images_dir):
    for file in files:
        if file.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(root, file)
            try:
                predicted_class = classify_shape(image_path)
                prediction_counts[predicted_class] += 1
                print(f"The shape for image {file} is: {predicted_class}")
            except ValueError as e:
                print(e)

# Print the count of predictions for each class
print("\nPrediction counts:")
for class_name, count in prediction_counts.items():
    print(f"{class_name}: {count}")
