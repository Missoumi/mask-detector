import tensorflow as tf


image_url = "https://www.thestatesman.com/wp-content/uploads/2017/08/1493458748-beauty-face-517.jpg"
class_names = ["with mask", "without mask"]

# Load image from web

image_path = tf.keras.utils.get_file('Test3', origin=image_url)
img = tf.keras.preprocessing.image.load_img(
    image_path, target_size=(160, 160)
)

img_array = tf.keras.preprocessing.image.img_to_array(img)

import matplotlib.pyplot as plt
plt.imshow(img_array.astype("uint8"))
plt.show()

# Batch
img_array = tf.expand_dims(img_array, 0)
model = tf.keras.models.load_model('./saved_model/my_model')

predictions = model.predict_on_batch(img_array).flatten()
predictions = tf.nn.sigmoid(predictions)
print(predictions[0])
predictions = tf.where(predictions < 0.5, 0, 1)
print(class_names[predictions[0]])
