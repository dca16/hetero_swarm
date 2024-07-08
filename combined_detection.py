import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model('shape_classifier_model.keras')

# Load class names
with open('class_names.txt', 'r') as f:
    class_names = f.read().splitlines()

# Define class indices based on the class names loaded
class_indices = {class_name: idx for idx, class_name in enumerate(class_names)}

def classify_shape(image):
    # Preprocess the image
    image = cv2.resize(image, (150, 150))
    image = np.expand_dims(image, axis=0)
    image = image / 255.0

    # Predict the class
    prediction = model.predict(image)
    class_index = np.argmax(prediction)
    class_name = class_names[class_index]
    return class_name

def detect_objects(image_path):
    # Load the image
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define color ranges for all colors in HSV
    color_ranges = {
        'blue': ([100, 100, 100], [130, 255, 255]),
        'green': ([35, 100, 100], [85, 255, 255]),
        'red': ([0, 100, 100], [10, 255, 255])
    }

    for color, (lower, upper) in color_ranges.items():
        lower = np.array(lower)
        upper = np.array(upper)

        # Threshold the HSV image to get only the specified color
        mask = cv2.inRange(hsv, lower, upper)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Get the bounding box for each contour
            x, y, w, h = cv2.boundingRect(contour)
            object_roi = image[y:y+h, x:x+w]

            # Classify the shape of the object
            shape = classify_shape(object_roi)

            # Draw bounding box and label
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(image, shape, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the image with detected objects
    cv2.imshow('Detected Objects', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    image_path = '/Users/domalberts/Documents/GitHub/hetero_swarm/full_env_verify/step_33.png'
    detect_objects(image_path)
