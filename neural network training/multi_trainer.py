import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pathlib
from PIL import Image, ImageDraw
import tensorflow_hub as hub

# Define the path to your dataset
data_dir = pathlib.Path('/Users/domalberts/Documents/GitHub/hetero_swarm/train_images').with_suffix('')

# Count images
image_count = len(list(data_dir.glob('*/*.png')))
print(image_count)

# Output individual classes
green_cyl = list(data_dir.glob('green_cyl/*'))
Image.open(str(green_cyl[0]))
Image.open(str(green_cyl[1]))

blue_cube = list(data_dir.glob('blue_cube/*'))
Image.open(str(blue_cube[0]))
Image.open(str(blue_cube[1]))
# Note: won't show unless you call ".show()" at the end of each Image.open line

# Define parameters for loader
batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_names = train_ds.class_names
print(class_names)

# Visualize data
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

# Configure dataset for performance
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Standardize data
normalization_layer = tf.keras.layers.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

# Load a pre-trained object detection model from TensorFlow Hub
detector = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")

def detect_objects(image):
    # Convert image to uint8 and ensure shape is dynamic for height and width
    converted_img = tf.image.convert_image_dtype(image, tf.uint8)[tf.newaxis, ...]
    result = detector(converted_img)
    result = {key: value.numpy() for key, value in result.items()}
    return result

# Visualize detections
def draw_boxes(image, boxes, class_ids, scores, threshold=0.5):
    image_with_boxes = (image * 255).astype(np.uint8)
    image_with_boxes = Image.fromarray(image_with_boxes)
    draw = ImageDraw.Draw(image_with_boxes)
    
    for box, class_id, score in zip(boxes, class_ids, scores):
        if score[0] < threshold:  # Directly index into the score array
            continue
        ymin, xmin, ymax, xmax = box
        (left, right, top, bottom) = (xmin * img_width, xmax * img_width, ymin * img_height, ymax * img_height)
        draw.rectangle([(left, top), (right, bottom)], outline="red", width=2)
        class_name = f"class_id: {class_id}"  # Or use a dictionary to map class_ids to class_names if available
        draw.text((left, top), f"{class_name}: {score[0]:.2f}", fill="red")
    
    return image_with_boxes

# Process an example image
test_path = pathlib.Path('/Users/domalberts/Documents/GitHub/hetero_swarm/train_images/blue_cyl/step_144.png')
img = tf.keras.utils.load_img(test_path, target_size=(img_height, img_width))
img_array = tf.keras.utils.img_to_array(img)

# Detect objects
result = detect_objects(img_array)

# Print the keys in the result to identify the correct key names
print("Detection result keys:", result.keys())

# Visualize detections
boxes = result["detection_boxes"]
# Find the correct key for class names
if "detection_classes" in result:
    class_ids = result["detection_classes"]
elif "detection_class_entities" in result:
    class_ids = result["detection_class_entities"]
else:
    raise KeyError("No class names key found in the detection results")

scores = result["detection_scores"]
img_with_boxes = draw_boxes(img_array / 255.0, boxes, class_ids, scores)
plt.imshow(img_with_boxes)
plt.show()

# Save the model if needed
tf.saved_model.save(detector, '/Users/domalberts/Documents/GitHub/hetero_swarm/d_model')
