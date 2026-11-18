# ============================================================
# LIVE EMOTION DETECTION USING SAVED KERAS CNN MODEL
# RAF-DB | 100x100 RGB
# ============================================================

import cv2
import numpy as np
import tensorflow as tf


# ============================================================
# 1. LOAD TRAINED MODEL
# ============================================================

MODEL_PATH = r"C:\Users\Asus User\OneDrive\Desktop\App\Emotion_Recognition\models\raf_3cnn_model.keras"

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")
print("Model input shape:", model.input_shape)
print("Model output shape:", model.output_shape)


# ============================================================
# 2. EMOTION LABELS
# ============================================================

class_names = [
    "Surprise",   # 1
    "Fear",       # 2
    "Disgust",    # 3
    "Happy",      # 4
    "Sad",        # 5
    "Angry",      # 6
    "Neutral"     # 7
]


# ============================================================
# 3. CHECK MODEL OUTPUT
# ============================================================

num_classes = model.output_shape[-1]

if num_classes != len(class_names):

    raise ValueError(
        f"Model has {num_classes} outputs, "
        f"but {len(class_names)} class names were provided."
    )


# ============================================================
# 4. LOAD HAAR CASCADE FACE DETECTOR
# ============================================================

cascade_path = cv2.data.haarcascades + \
               "haarcascade_frontalface_default.xml"

face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():

    raise RuntimeError(
        "Could not load Haar Cascade face detector."
    )


print("Face detector loaded successfully.")


# ============================================================
# 5. OPEN WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    raise RuntimeError(
        "Could not open webcam."
    )


# ============================================================
# 6. SET CAMERA RESOLUTION
# ============================================================

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


print()
print("================================================")
print("LIVE EMOTION DETECTION STARTED")
print("Press Q to quit.")
print("================================================")
print()


# ============================================================
# 7. MAIN LOOP
# ============================================================

while True:

    # --------------------------------------------------------
    # Read camera frame
    # --------------------------------------------------------

    ret, frame = cap.read()

    if not ret:

        print("Could not read frame from webcam.")
        break


    # --------------------------------------------------------
    # Convert frame to grayscale for face detection
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    # --------------------------------------------------------
    # Detect faces
    # --------------------------------------------------------

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )


    # --------------------------------------------------------
    # Process detected faces
    # --------------------------------------------------------

    for (x, y, w, h) in faces:

        # ----------------------------------------------------
        # Extract face from COLOR frame
        # ----------------------------------------------------

        face = frame[
            y:y+h,
            x:x+w
        ]


        if face.size == 0:
            continue


        # ----------------------------------------------------
        # Resize to model input size
        # 100 x 100
        # ----------------------------------------------------

        face = cv2.resize(
            face,
            (100, 100),
            interpolation=cv2.INTER_AREA
        )


        # ----------------------------------------------------
        # BGR -> RGB
        # ----------------------------------------------------

        face = cv2.cvtColor(
            face,
            cv2.COLOR_BGR2RGB
        )


        # ----------------------------------------------------
        # Convert to float32
        # ----------------------------------------------------

        face = face.astype(
            np.float32
        )


        # ----------------------------------------------------
        # NORMALIZATION
        #
        # Use this ONLY if your training used:
        #
        # rescale=1./255
        #
        # or:
        #
        # image / 255.0
        # ----------------------------------------------------

        face = face / 255.0


        # ----------------------------------------------------
        # Add batch dimension
        #
        # (100,100,3)
        #
        # becomes
        #
        # (1,100,100,3)
        # ----------------------------------------------------

        face = np.expand_dims(
            face,
            axis=0
        )


        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(
            face,
            verbose=0
        )


        # ----------------------------------------------------
        # Find predicted class
        # ----------------------------------------------------

        predicted_index = np.argmax(
            prediction[0]
        )


        # ----------------------------------------------------
        # Get confidence
        # ----------------------------------------------------

        confidence = float(
            prediction[0][predicted_index]
        )


        confidence_percent = (
            confidence * 100
        )


        # ----------------------------------------------------
        # Get emotion name
        # ----------------------------------------------------

        emotion = class_names[
            predicted_index
        ]


        # ----------------------------------------------------
        # Draw face rectangle
        # ----------------------------------------------------

        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )


        # ----------------------------------------------------
        # Create label
        # ----------------------------------------------------

        label = (
            f"{emotion}: "
            f"{confidence_percent:.1f}%"
        )


        # ----------------------------------------------------
        # Calculate text size
        # ----------------------------------------------------

        (
            text_width,
            text_height
        ), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            2
        )


        # ----------------------------------------------------
        # Position text
        # ----------------------------------------------------

        text_y = y - 10


        if text_y - text_height < 0:

            text_y = (
                y + h + text_height + 10
            )


        # ----------------------------------------------------
        # Text background
        # ----------------------------------------------------

        cv2.rectangle(
            frame,
            (
                x,
                text_y - text_height - baseline
            ),
            (
                x + text_width,
                text_y + baseline
            ),
            (0, 255, 0),
            -1
        )


        # ----------------------------------------------------
        # Emotion text
        # ----------------------------------------------------

        cv2.putText(
            frame,
            label,
            (x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
            cv2.LINE_AA
        )


    # ========================================================
    # DISPLAY CAMERA
    # ========================================================

    cv2.imshow(
        "RAF-DB Live Emotion Detection",
        frame
    )


    # ========================================================
    # PRESS Q TO EXIT
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        break


# ============================================================
# 8. RELEASE CAMERA
# ============================================================

cap.release()

cv2.destroyAllWindows()

print()
print("Webcam stopped.")