import cv2


def transform_img_gray(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


classifier = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")


def detect(frame):
    gray = transform_img_gray(frame)
    faces = classifier.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
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
