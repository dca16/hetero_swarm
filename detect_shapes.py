import tensorflow as tf
import numpy as np
import os
import cv2
import random
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

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

# Directory containing test images
test_images_dir = '/Users/domalberts/Documents/GitHub/hetero_swarm/test_images'

# Function to classify shape in an image
def classify_shape(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")
    image_resized = cv2.resize(image, (150, 150))  # Resize to (150, 150) as expected by the model
    image_array = np.expand_dims(image_resized, axis=0) / 255.0  # Normalizing the image
    predictions = model.predict(image_array)
    predicted_class_index = np.argmax(predictions)
    return class_names[predicted_class_index], predictions[0]

# Iterate through all images in the test_images directory and count predictions
y_true = []
y_pred = []

for root, dirs, files in os.walk(test_images_dir):
    for file in files:
        if file.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(root, file)
            true_class = os.path.basename(root)
            y_true.append(class_indices[true_class])
            predicted_class, _ = classify_shape(image_path)
            y_pred.append(class_indices[predicted_class])
            print(f"The shape for image {file} is: {predicted_class} (True: {true_class})")

# Generate confusion matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot()
plt.show()

# Randomly select six images from the test set and display predictions and true labels
random.seed(123)
random_images = random.sample([os.path.join(root, file) for root, _, files in os.walk(test_images_dir) for file in files if file.endswith(('.png', '.jpg', '.jpeg'))], 6)

plt.figure(figsize=(15, 10))
for i, image_path in enumerate(random_images):
    true_class = os.path.basename(os.path.dirname(image_path))
    predicted_class, probabilities = classify_shape(image_path)
    ax = plt.subplot(2, 3, i + 1)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.title(f"Pred: {predicted_class} (True: {true_class})\nProb: {probabilities[class_indices[predicted_class]]:.2f}")
    plt.axis("off")
plt.show()
