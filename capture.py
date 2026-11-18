# ============================================================
# RAF-DB PHOTO EMOTION DETECTION
# 100 x 100 RGB
# 7 EMOTION CLASSES
#
# Controls:
#   SPACE = Capture photo and detect emotion
#   Q / ESC = Quit
# ============================================================

import cv2
import numpy as np
import tensorflow as tf
import os
from datetime import datetime


# ============================================================
# 1. MODEL PATH
# ============================================================

MODEL_PATH = r"C:\Users\Asus User\OneDrive\Desktop\App\Emotion_Recognition\models\raf_3cnn_model.keras"


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
# 3. LOAD MODEL
# ============================================================

print("Loading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")
print("Input shape :", model.input_shape)
print("Output shape:", model.output_shape)


# ============================================================
# 4. CHECK MODEL
# ============================================================

num_classes = model.output_shape[-1]

if num_classes != 7:
    raise ValueError(
        f"Expected 7 output classes, but model has {num_classes}."
    )


# ============================================================
# 5. LOAD FACE DETECTOR
# ============================================================

cascade_path = cv2.data.haarcascades + \
               "haarcascade_frontalface_default.xml"

face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    raise RuntimeError(
        "Could not load Haar Cascade face detector."
    )


# ============================================================
# 6. CREATE PHOTO FOLDER
# ============================================================

save_folder = "captured_photos"

os.makedirs(
    save_folder,
    exist_ok=True
)


# ============================================================
# 7. OPEN CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError(
        "Could not open webcam."
    )


# ============================================================
# 8. CAMERA RESOLUTION
# ============================================================

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1280
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    720
)


print()
print("==============================================")
print(" RAF-DB PHOTO EMOTION DETECTION")
print("==============================================")
print("SPACE  -> Capture photo")
print("Q      -> Quit")
print("ESC    -> Quit")
print("==============================================")
print()


# ============================================================
# 9. MAIN CAMERA LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:

        print("Could not read camera frame.")

        break


    # --------------------------------------------------------
    # Mirror camera
    # --------------------------------------------------------

    frame = cv2.flip(
        frame,
        1
    )


    # --------------------------------------------------------
    # Convert to grayscale for face detection
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
    # Draw detected faces
    # --------------------------------------------------------

    for (x, y, w, h) in faces:

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )


    # --------------------------------------------------------
    # Instructions
    # --------------------------------------------------------

    cv2.rectangle(
        frame,
        (10, 10),
        (350, 55),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        "Press SPACE to capture",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


    # --------------------------------------------------------
    # Display camera
    # --------------------------------------------------------

    cv2.imshow(
        "RAF-DB Photo Emotion Detection",
        frame
    )


    # --------------------------------------------------------
    # Keyboard
    # --------------------------------------------------------

    key = cv2.waitKey(1) & 0xFF


    # ========================================================
    # CAPTURE PHOTO
    # ========================================================

    if key == 32:   # SPACE

        print()
        print("Photo captured.")

        # ----------------------------------------------------
        # Check whether a face was detected
        # ----------------------------------------------------

        if len(faces) == 0:

            print("No face detected.")

            cv2.putText(
                frame,
                "NO FACE DETECTED",
                (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3,
                cv2.LINE_AA
            )

            cv2.imshow(
                "RAF-DB Photo Emotion Detection",
                frame
            )

            cv2.waitKey(1500)

            continue


        # ----------------------------------------------------
        # Process each detected face
        # ----------------------------------------------------

        for face_number, (x, y, w, h) in enumerate(faces):

            # IMPORTANT:
            # Correct crop:
            # y:y+h
            # x:x+w

            face = frame[
                y:y+h,
                x:x+w
            ]


            if face.size == 0:
                continue


            # ------------------------------------------------
            # Resize to model input
            # 100 x 100
            # ------------------------------------------------

            face = cv2.resize(
                face,
                (100, 100),
                interpolation=cv2.INTER_AREA
            )


            # ------------------------------------------------
            # BGR -> RGB
            # ------------------------------------------------

            face = cv2.cvtColor(
                face,
                cv2.COLOR_BGR2RGB
            )


            # ------------------------------------------------
            # Normalize
            # ------------------------------------------------

            face = face.astype(
                np.float32
            ) / 255.0


            # ------------------------------------------------
            # Add batch dimension
            #
            # (100,100,3)
            #
            # ->
            #
            # (1,100,100,3)
            # ------------------------------------------------

            face_input = np.expand_dims(
                face,
                axis=0
            )


            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            prediction = model.predict(
                face_input,
                verbose=0
            )[0]


            # ------------------------------------------------
            # Predicted emotion
            # ------------------------------------------------

            predicted_index = np.argmax(
                prediction
            )

            emotion = class_names[
                predicted_index
            ]


            # ------------------------------------------------
            # Confidence
            # ------------------------------------------------

            confidence = float(
                prediction[predicted_index]
            )

            confidence_percent = (
                confidence * 100
            )


            print(
                f"Face {face_number + 1}: "
                f"{emotion} "
                f"({confidence_percent:.2f}%)"
            )


            # =================================================
            # SAVE ORIGINAL CAPTURED IMAGE
            # =================================================

            timestamp = datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

            filename = (
                f"{timestamp}_"
                f"{emotion}_"
                f"{confidence_percent:.0f}percent.jpg"
            )

            save_path = os.path.join(
                save_folder,
                filename
            )


            # Draw prediction on image before saving
            result_frame = frame.copy()


            cv2.rectangle(
                result_frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                3
            )


            label = (
                f"{emotion}: "
                f"{confidence_percent:.1f}%"
            )


            # ------------------------------------------------
            # Text background
            # ------------------------------------------------

            (text_w, text_h), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                2
            )


            label_y = y - 10

            if label_y - text_h < 0:
                label_y = y + h + text_h + 10


            cv2.rectangle(
                result_frame,
                (
                    x,
                    label_y - text_h - baseline
                ),
                (
                    x + text_w + 10,
                    label_y + baseline
                ),
                (0, 255, 0),
                -1
            )


            # ------------------------------------------------
            # Emotion text
            # ------------------------------------------------

            cv2.putText(
                result_frame,
                label,
                (x + 5, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 0),
                2,
                cv2.LINE_AA
            )


            # ------------------------------------------------
            # Save result
            # ------------------------------------------------

            cv2.imwrite(
                save_path,
                result_frame
            )


            print(
                "Saved:",
                save_path
            )


            # ------------------------------------------------
            # Show result
            # ------------------------------------------------

            cv2.imshow(
                "Captured Result",
                result_frame
            )


            cv2.waitKey(2500)

            cv2.destroyWindow(
                "Captured Result"
            )


    # ========================================================
    # QUIT
    # ========================================================

    elif key == ord("q") or key == 27:

        break


# ============================================================
# 10. CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

print()
print("Camera stopped.")
print("Program finished.")