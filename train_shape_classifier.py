import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input, Dropout
import os
import shutil
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import PIL

# Set the path to the dataset
dataset_path = '/Users/domalberts/Documents/GitHub/hetero_swarm/verification_images'
train_dir = '/Users/domalberts/Documents/GitHub/hetero_swarm/train_images'
val_dir = '/Users/domalberts/Documents/GitHub/hetero_swarm/val_images'
test_dir = '/Users/domalberts/Documents/GitHub/hetero_swarm/test_images'

# Function to create directories
def create_dir(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

# Create train, validation, and test directories
create_dir(train_dir)
create_dir(val_dir)
create_dir(test_dir)

# Split dataset into training, validation, and test sets
def split_data(dataset_path, train_dir, val_dir, test_dir, train_split=0.7, val_split=0.15, test_split=0.15, seed=123):
    assert train_split + val_split + test_split == 1, "Splits must add up to 1"
    np.random.seed(seed)
    class_names = os.listdir(dataset_path)
    for class_name in class_names:
        class_dir = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_dir):
            images = os.listdir(class_dir)
            np.random.shuffle(images)
            train_count = int(len(images) * train_split)
            val_count = int(len(images) * val_split)
            train_images = images[:train_count]
            val_images = images[train_count:train_count + val_count]
            test_images = images[train_count + val_count:]
            for image in train_images:
                src_path = os.path.join(class_dir, image)
                dst_dir = os.path.join(train_dir, class_name)
                os.makedirs(dst_dir, exist_ok=True)
                shutil.copy(src_path, dst_dir)
            for image in val_images:
                src_path = os.path.join(class_dir, image)
                dst_dir = os.path.join(val_dir, class_name)
                os.makedirs(dst_dir, exist_ok=True)
                shutil.copy(src_path, dst_dir)
            for image in test_images:
                src_path = os.path.join(class_dir, image)
                dst_dir = os.path.join(test_dir, class_name)
                os.makedirs(dst_dir, exist_ok=True)
                shutil.copy(src_path, dst_dir)

split_data(dataset_path, train_dir, val_dir, test_dir)

# Set image size and batch size
img_height, img_width = 150, 150
batch_size = 8

# Load dataset and split into training, validation, and test sets
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Get class names before applying any transformations
class_names = train_ds.class_names
num_classes = len(class_names)

# Apply data augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
])

train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

# Print class names and their counts in the dataset
class_counts = {class_name: 0 for class_name in class_names}
for _, labels in train_ds:
    for label in labels.numpy():
        class_counts[class_names[label]] += 1

print("Class distribution in the training set:")
for class_name, count in class_counts.items():
    print(f"{class_name}: {count}")

# Get steps per epoch and validation steps before applying repeat
steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
validation_steps = tf.data.experimental.cardinality(val_ds).numpy()

# Apply repeat after getting class names
train_ds = train_ds.repeat()
val_ds = val_ds.repeat()

# Define the model using Functional API
inputs = Input(shape=(img_height, img_width, 3))
x = tf.keras.layers.Rescaling(1./255)(inputs)
x = Conv2D(32, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)  # Adding dropout for regularization
outputs = Dense(num_classes, activation='softmax')(x)

model = Model(inputs, outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
'''
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps
)

# Save the model
model_path = 'shape_classifier_model.keras'
model.save(model_path)
print(f"Model saved at: {os.path.abspath(model_path)}")

# Save model weights
weights_path = 'model_weights.weights.h5'
model.save_weights(weights_path)
print(f"Model weights saved at: {os.path.abspath(weights_path)}")

# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Save class names
with open('class_names.txt', 'w') as f:
    for class_name in class_names:
        f.write(f"{class_name}\n")

# Display some validation images with their predicted labels
plt.figure(figsize=(15, 15))
val_images = list(val_ds.take(validation_steps).as_numpy_iterator())

for i, (images, labels) in enumerate(val_images):
    for j in range(images.shape[0]):
        ax = plt.subplot(validation_steps, batch_size, i * batch_size + j + 1)
        plt.imshow(images[j].astype("uint8"))
        plt.title(class_names[np.argmax(model.predict(images[j:j+1]))])
        plt.axis("off")
plt.show()
'''