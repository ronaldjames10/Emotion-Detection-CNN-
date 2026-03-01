import cv2
import numpy as np
from keras.models import model_from_json, Sequential

# -------------------------
# Load model architecture & weights
# -------------------------
with open("facialemotionmodel.json", "r") as json_file:
    model_json = json_file.read()

# Fix: tell Keras how to handle Sequential models from JSON
model = model_from_json(model_json, custom_objects={"Sequential": Sequential})
model.load_weights("facialemotionmodel.h5")

# -------------------------
# Load Haar Cascade for face detection
# -------------------------
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# -------------------------
# Feature extraction helper
# -------------------------
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# -------------------------
# Camera Auto-Detection
# -------------------------
backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_VFW]
camera = None

for backend in backends:
    for index in range(5):  # try camera indexes 0-4
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            print(f"✅ Camera found at index {index} with backend {backend}")
            camera = cap
            break
    if camera:
        break

if not camera:
    print("❌ No accessible camera found")
    exit()

# -------------------------
# Labels for emotions
# -------------------------
labels = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprise'
}

# -------------------------
# Emotion Detection Loop
# -------------------------
while True:
    ret, im = camera.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (p, q, r, s) in faces:
        image = gray[q:q + s, p:p + r]
        cv2.rectangle(im, (p, q), (p + r, q + s), (255, 0, 0), 2)
        image = cv2.resize(image, (48, 48))
        img = extract_features(image)

        # Prediction
        pred = model.predict(img, verbose=0)
        prediction_label = labels[pred.argmax()]

        # Display label on video frame
        cv2.putText(im, prediction_label, (p, q - 10),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (0, 0, 255), 2)

    cv2.imshow("Emotion Detection", im)

    # ESC key to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources

camera.release()
cv2.destroyAllWindows()
