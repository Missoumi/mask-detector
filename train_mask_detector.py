import tensorflow as tf
import pathlib

IMG_HEIGHT = 160
IMG_WIDTH = 160
DATASET_PATH = "./Face-Mask-Detection-master/dataset/"
# Load datasets
train_ds = tf.keras.preprocessing.image_dataset_from_directory(DATASET_PATH,
                                                               validation_split=0.2,
                                                               subset="training",
                                                               seed=10,
                                                               image_size=(IMG_HEIGHT, IMG_WIDTH),
                                                               batch_size=32)


val_ds = tf.keras.preprocessing.image_dataset_from_directory(DATASET_PATH,
                                                               validation_split=0.2,
                                                               subset="validation",
                                                               seed=10,
                                                               image_size=(IMG_HEIGHT, IMG_WIDTH),
                                                               batch_size=32)

val_batches = tf.data.experimental.cardinality(val_ds)
test_dataset = val_ds.take(val_batches // 5)
val_ds = val_ds.skip(val_batches // 5)

# Standardize the data
rescaling_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset=-1)


# Data augmentation
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
  tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])


# AUTOTUNE = tf.data.experimental.AUTOTUNE

# train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
# val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# Transfer learning
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH) + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(1)


inputs = tf.keras.Input(shape=(160, 160, 3))
x = data_augmentation(inputs)
x = rescaling_layer(inputs)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)
