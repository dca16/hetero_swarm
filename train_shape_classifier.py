import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input, Dropout
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

# Set the path to the dataset
dataset_path = '/Users/domalberts/Documents/GitHub/hetero_swarm/verification_images'

# Set image size and batch size
img_height, img_width = 150, 150
batch_size = 8

# Set training and validation directories
train_dir = dataset_path
validation_dir = dataset_path

# Load dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    validation_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
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

# Generate confusion matrix
y_true = []
y_pred = []

for images, labels in val_ds.take(validation_steps):
    predictions = model.predict(images)
    predicted_labels = np.argmax(predictions, axis=1)
    y_true.extend(labels.numpy())
    y_pred.extend(predicted_labels)

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot()
plt.show()

# Save class names
with open('class_names.txt', 'w') as f:
    for class_name in class_names:
        f.write(f"{class_name}\n")

# Display all validation images with their predicted labels
plt.figure(figsize=(15, 15))
val_images = list(val_ds.take(validation_steps).as_numpy_iterator())

for i, (images, labels) in enumerate(val_images):
    for j in range(images.shape[0]):
        ax = plt.subplot(validation_steps, batch_size, i * batch_size + j + 1)
        plt.imshow(images[j].astype("uint8"))
        plt.title(class_names[np.argmax(model.predict(images[j:j+1]))])
        plt.axis("off")
plt.show()
