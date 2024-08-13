import numpy as np
import tensorflow as tf
import keras
import pathlib

testing_model = keras.models.load_model('/Users/domalberts/Documents/GitHub/hetero_swarm/dom_model.keras')

# get dataset
data_dir = pathlib.Path('/Users/domalberts/Documents/GitHub/hetero_swarm/train_images').with_suffix('')

# define parameters for loader
batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names

# predict on new data
# Define the path to the folder containing the images
train_images_path = pathlib.Path('/Users/domalberts/Documents/GitHub/hetero_swarm/train_images')

correct_predictions = 0
total_predictions = 0

'''
# Iterate over all images in the train_images folder
for image_path in train_images_path.glob('**/*.png'):  # or use '*.jpg' if your images are in jpg format
    # Extract the true class from the folder name
    true_class_name = image_path.parent.name

    img = tf.keras.utils.load_img(
        image_path, target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = testing_model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])

    if class_names[predicted_class_index] == true_class_name:
        correct_predictions += 1
    total_predictions += 1

# Calculate and print the percentage of correct predictions
accuracy = (correct_predictions / total_predictions) * 100
print(f"Model accuracy: {accuracy:.2f}% ({correct_predictions}/{total_predictions} correct predictions)")
'''

# Check multi-images
print("TESTING: RED CYL AND GREEN CUBE")
img = tf.keras.utils.load_img(
pathlib.Path('/Users/domalberts/Documents/GitHub/hetero_swarm/full_env_verify/step_16.png'), target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = testing_model.predict(img_array)
score = tf.nn.softmax(predictions[0])
print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

# Check multi-images
print("TESTING: BLUE CUBE AND GREEN CYL")
img = tf.keras.utils.load_img(
pathlib.Path('/Users/domalberts/Documents/GitHub/hetero_swarm/full_env_verify/step_33.png'), target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = testing_model.predict(img_array)
score = tf.nn.softmax(predictions[0])
print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

# Check multi-images
print("TESTING: RED CUBE AND GREEN CUBE")
img = tf.keras.utils.load_img(
pathlib.Path('/Users/domalberts/Documents/GitHub/hetero_swarm/full_env_verify/step_57.png'), target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = testing_model.predict(img_array)
score = tf.nn.softmax(predictions[0])
print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)


# check individual images
blue_cube = []
blue_cube.append(pathlib.Path('/Users/domalberts/Documents/GitHub/hetero_swarm/train_images/blue_cube/step_50.png'))
blue_cube.append(pathlib.Path('/Users/domalberts/Documents/GitHub/hetero_swarm/train_images/blue_cube/step_122.png'))
blue_cube.append(pathlib.Path('/Users/domalberts/Documents/GitHub/hetero_swarm/train_images/blue_cube/step_240.png'))
blue_cube.append(pathlib.Path('/Users/domalberts/Documents/GitHub/hetero_swarm/train_images/blue_cube/step_897.png'))

print("TESTING: BLUE CUBE")

for bcu in blue_cube:
    img = tf.keras.utils.load_img(
    bcu, target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = testing_model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )

green_cube = []
green_cube.append(pathlib.Path('/Users/domalberts/Documents/GitHub/hetero_swarm/train_images/green_cube/step_64.png'))
green_cube.append(pathlib.Path('/Users/domalberts/Documents/GitHub/hetero_swarm/train_images/green_cube/step_118.png'))
green_cube.append(pathlib.Path('/Users/domalberts/Documents/GitHub/hetero_swarm/train_images/green_cube/step_238.png'))
green_cube.append(pathlib.Path('/Users/domalberts/Documents/GitHub/hetero_swarm/train_images/green_cube/step_310.png'))

print("TESTING: GREEN CUBE")

for gcu in green_cube:
    img = tf.keras.utils.load_img(
    gcu, target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = testing_model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )

red_cube = []
red_cube.append(pathlib.Path('/Users/domalberts/Documents/GitHub/hetero_swarm/train_images/red_cube/step_21.png'))
red_cube.append(pathlib.Path('/Users/domalberts/Documents/GitHub/hetero_swarm/train_images/red_cube/step_124.png'))
red_cube.append(pathlib.Path('/Users/domalberts/Documents/GitHub/hetero_swarm/train_images/red_cube/step_306.png'))
red_cube.append(pathlib.Path('/Users/domalberts/Documents/GitHub/hetero_swarm/train_images/red_cube/step_861.png'))

print("TESTING: RED CUBE")

for rcu in red_cube:
    img = tf.keras.utils.load_img(
    rcu, target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = testing_model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )

blue_cyl = []
blue_cyl.append(pathlib.Path('/Users/domalberts/Documents/GitHub/hetero_swarm/train_images/blue_cyl/step_9.png'))
blue_cyl.append(pathlib.Path('/Users/domalberts/Documents/GitHub/hetero_swarm/train_images/blue_cyl/step_102.png'))
blue_cyl.append(pathlib.Path('/Users/domalberts/Documents/GitHub/hetero_swarm/train_images/blue_cyl/step_243.png'))
blue_cyl.append(pathlib.Path('/Users/domalberts/Documents/GitHub/hetero_swarm/train_images/blue_cyl/step_561.png'))

print("TESTING: BLUE CYL")

for bcy in blue_cyl:
    img = tf.keras.utils.load_img(
    bcy, target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = testing_model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )

green_cyl = []
green_cyl.append(pathlib.Path('/Users/domalberts/Documents/GitHub/hetero_swarm/train_images/green_cyl/step_13.png'))
green_cyl.append(pathlib.Path('/Users/domalberts/Documents/GitHub/hetero_swarm/train_images/green_cyl/step_103.png'))
green_cyl.append(pathlib.Path('/Users/domalberts/Documents/GitHub/hetero_swarm/train_images/green_cyl/step_639.png'))
green_cyl.append(pathlib.Path('/Users/domalberts/Documents/GitHub/hetero_swarm/train_images/green_cyl/step_801.png'))

print("TESTING: GREEN CYL")

for gcy in green_cyl:
    img = tf.keras.utils.load_img(
    gcy, target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = testing_model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )

red_cyl = []
red_cyl.append(pathlib.Path('/Users/domalberts/Documents/GitHub/hetero_swarm/train_images/red_cyl/step_35.png'))
red_cyl.append(pathlib.Path('/Users/domalberts/Documents/GitHub/hetero_swarm/train_images/red_cyl/step_115.png'))
red_cyl.append(pathlib.Path('/Users/domalberts/Documents/GitHub/hetero_swarm/train_images/red_cyl/step_324.png'))
red_cyl.append(pathlib.Path('/Users/domalberts/Documents/GitHub/hetero_swarm/train_images/red_cyl/step_699.png'))

print("TESTING: RED CYL")

for rcy in red_cyl:
    img = tf.keras.utils.load_img(
    rcy, target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = testing_model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )

empty = []
empty.append(pathlib.Path('/Users/domalberts/Documents/GitHub/hetero_swarm/train_images/empty/step_39.png'))
empty.append(pathlib.Path('/Users/domalberts/Documents/GitHub/hetero_swarm/train_images/empty/step_123.png'))
empty.append(pathlib.Path('/Users/domalberts/Documents/GitHub/hetero_swarm/train_images/empty/step_462.png'))
empty.append(pathlib.Path('/Users/domalberts/Documents/GitHub/hetero_swarm/train_images/empty/step_894.png'))

print("TESTING: EMPTY")

for em in empty:
    img = tf.keras.utils.load_img(
    em, target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = testing_model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )