
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf

from PIL import Image
from utils.model_utils import MODEL_CONFIG, load_model, get_label_map


st.set_page_config(
    page_title="Emotion Prediction",
    page_icon="😊",
    layout="wide",
)

st.title("😊 RAF-DB Emotion Prediction")
st.write("Compare softmax predictions from two trained CNN models.")

MODEL_NAMES = ["raf_3cnn", "raf_extendedcnn"]
IMAGE_SIZE = (100, 100)


# --------------------------------------------------
# LOAD BOTH MODELS WHEN PAGE OPENS
# --------------------------------------------------

@st.cache_resource(show_spinner="Loading RAF-DB models...")
def load_raf_models():
    loaded_models = {}

    for name in MODEL_NAMES:
        config = MODEL_CONFIG[name]

        if config["dataset"] != "RAF-DB":
            raise ValueError(
                f"{name} is configured for {config['dataset']}, not RAF-DB."
            )

        if tuple(config["input_size"]) != IMAGE_SIZE:
            raise ValueError(
                f"{name} expects {config['input_size']}, not {IMAGE_SIZE}."
            )

        if config["channels"] != 3:
            raise ValueError(f"{name} must accept RGB images.")

        model = load_model(name)

        if model is None:
            raise RuntimeError(f"Could not load {name}.")

        loaded_models[name] = model

    return loaded_models


try:
    models = load_raf_models()
    st.success("Both RAF-DB models loaded successfully.")

except Exception as error:
    st.error("Unable to load the RAF-DB models.")
    st.exception(error)
    st.stop()


# --------------------------------------------------
# EMOTION LABELS
# --------------------------------------------------

try:
    label_map = get_label_map("raf_3cnn")

    if isinstance(label_map, dict):
        labels = [
            str(label_map.get(i, f"Class {i}"))
            for i in range(MODEL_CONFIG["raf_3cnn"]["classes"])
        ]
    else:
        labels = list(label_map)

except Exception:
    labels = [
        "Surprise",
        "Fear",
        "Disgust",
        "Happiness",
        "Sadness",
        "Anger",
        "Neutral",
    ]


# --------------------------------------------------
# UPLOAD IMAGE
# --------------------------------------------------

st.subheader("Upload Image")

uploaded_file = st.file_uploader(
    "Choose a face image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        image = image.resize(IMAGE_SIZE)

        st.image(
            image,
            caption="Input image — 100 × 100",
            width=250,
        )

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        st.caption("Input shape: (1, 100, 100, 3)")

        if st.button("Predict and Compare", type="primary"):
            predictions = {}

            with st.spinner("Predicting with both models..."):
                for model_name in MODEL_NAMES:
                    model = models[model_name]

                    output = np.asarray(
                        model.predict(image_array, verbose=0)
                    )[0]

                    # Convert logits to softmax only when necessary.
                    if (
                        np.any(output < 0)
                        or not np.isclose(
                            np.sum(output), 1.0, atol=1e-3
                        )
                    ):
                        output = tf.nn.softmax(output).numpy()

                    predictions[model_name] = output

            # --------------------------------------------------
            # PREDICTED EMOTIONS
            # --------------------------------------------------

            st.subheader("Prediction Results")

            col1, col2 = st.columns(2)

            for column, model_name in zip(
                [col1, col2], MODEL_NAMES
            ):
                probabilities = predictions[model_name]
                predicted_index = int(np.argmax(probabilities))

                emotion = (
                    labels[predicted_index]
                    if predicted_index < len(labels)
                    else f"Class {predicted_index}"
                )

                with column:
                    st.markdown(f"### {model_name}")
                    st.metric("Predicted Emotion", emotion)
                    st.metric(
                        "Confidence",
                        f"{probabilities[predicted_index] * 100:.2f}%",
                    )

            # --------------------------------------------------
            # SOFTMAX OUTPUTS SIDE BY SIDE
            # --------------------------------------------------

            st.subheader("Softmax Probability Comparison")

            col1, col2 = st.columns(2)

            for column, model_name in zip(
                [col1, col2], MODEL_NAMES
            ):
                with column:
                    st.markdown(f"### {model_name}")

                    probabilities = predictions[model_name]

                    for index, probability in enumerate(probabilities):
                        emotion = (
                            labels[index]
                            if index < len(labels)
                            else f"Class {index}"
                        )

                        st.write(
                            f"**{emotion}** — {probability * 100:.2f}%"
                        )

                        st.progress(
                            float(np.clip(probability, 0.0, 1.0))
                        )

            # --------------------------------------------------
            # NUMERIC COMPARISON TABLE
            # --------------------------------------------------

            st.subheader("Softmax Values")

            class_count = len(predictions[MODEL_NAMES[0]])

            probability_df = pd.DataFrame({
                "Emotion": [
                    labels[i] if i < len(labels) else f"Class {i}"
                    for i in range(class_count)
                ],
                MODEL_NAMES[0]: predictions[MODEL_NAMES[0]],
                MODEL_NAMES[1]: predictions[MODEL_NAMES[1]],
            })

            st.dataframe(
                probability_df.style.format({
                    MODEL_NAMES[0]: "{:.2%}",
                    MODEL_NAMES[1]: "{:.2%}",
                }),
                use_container_width=True,
                hide_index=True,
            )

            # --------------------------------------------------
            # DIFFERENCE BETWEEN MODELS
            # --------------------------------------------------

            st.subheader("Probability Difference")

            difference_df = pd.DataFrame({
                "Emotion": probability_df["Emotion"],
                "Difference (percentage points)": (
                    predictions[MODEL_NAMES[0]]
                    - predictions[MODEL_NAMES[1]]
                ) * 100,
            })

            difference_df["Higher Probability"] = (
                difference_df["Difference (percentage points)"]
                .apply(
                    lambda value: (
                        MODEL_NAMES[0] if value > 0
                        else MODEL_NAMES[1] if value < 0
                        else "Tie"
                    )
                )
            )

            st.dataframe(
                difference_df.style.format({
                    "Difference (percentage points)": "{:+.2f}"
                }),
                use_container_width=True,
                hide_index=True,
            )

    except Exception as error:
        st.error("Could not process the uploaded image.")
        st.exception(error)