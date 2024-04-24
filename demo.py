import cv2
import numpy as np
from keras.models import model_from_json
from google.colab.patches import cv2_imshow

emotion_dict = {0: "Confused", 1: "Not Confused"}

# json_file = open('/content/drive/MyDrive/GGH/model/confused_expression_model.json', 'r')
json_file = open('confused_expression_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# emotion_model.load_weights("/content/drive/MyDrive/GGH/model/confused_expression_model.h5")
emotion_model.load_weights("confused_expression_model.h5")
print("Loaded model from disk")

image = cv2.imread("/content/drive/MyDrive/GGH/confused7.jpeg")

frame = cv2.resize(image, (1280, 720))
gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

face_detector = cv2.CascadeClassifier('/content/drive/MyDrive/GGH/haarcascades/haarcascade.xml')
num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

# Take each face available on the camera and Preprocess it
for (x, y, w, h) in num_faces:
    cv2.rectangle(image, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
    roi_gray_frame = gray_frame[y:y + h, x:x + w]
    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

    # Predict the emotions
    emotion_prediction = emotion_model.predict(cropped_img)
    maxindex = int(np.argmax(emotion_prediction))
    cv2.putText(image, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

cv2_imshow(image)
cv2.waitKey(0)
cv2.destroyAllWindows()
