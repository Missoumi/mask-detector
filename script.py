import cv2
import tensorflow as tf

model = tf.keras.models.load_model('./saved_model/my_model')
class_names = ["with mask", "without mask"]


def transform_img_gray(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def predict_mask(frame):
    image = tf.image.resize(frame, (160, 160))
    image = tf.expand_dims(image, 0)
    predictions = model.predict_on_batch(image).flatten()
    predictions = tf.nn.sigmoid(predictions)
    prediction_probability = predictions[0]
    predictions = tf.where(predictions < 0.5, 0, 1)
    return predictions[0] == 0, 1-prediction_probability


classifier = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")


def detect(frame):
    gray = transform_img_gray(frame)
    faces = classifier.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        mask_exist, probability = predict_mask(frame[y:y + h, x:x + w])
        if mask_exist:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (36, 255, 12), 2)
            cv2.putText(frame, f'Mask exist {probability}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12),
                        2)
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, f'No Mask exist {probability*100}%', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0),
                        2)
    return frame


video_capture = cv2.VideoCapture(0)

while True:
    _, frame = video_capture.read()
    canvas = detect(frame)
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()

cv2.destroyAllWindows()
