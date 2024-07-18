import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib

# set dataset path
data_dir = tf.keras.utils.image_dataset_from_directory('/Users/domalberts/Documents/GitHub/hetero_swarm/verification_images')
data_dir = pathlib.Path(data_dir).with_suffix('')

image_count = len(list(data_dir.glob('*/*.png')))
print(image_count)

