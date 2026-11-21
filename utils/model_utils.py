
import os
import numpy as np
import tensorflow as tf

from PIL import Image
from tensorflow.keras.models import load_model as keras_load_model
from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
    Dense,
    Conv2D,
)

from tensorflow.keras.utils import register_keras_serializable


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_DIR = os.path.join(BASE_DIR, "models")


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_CONFIG = {
    "ck_bpm": {
        "path": os.path.join(MODEL_DIR, "ck_bpm_model_.keras"),
        "input_size": (48, 48),
        "channels": 1,
        "classes": 7,
        "dataset": "CK+",
    },

    "ck_cbam": {
        "path": os.path.join(MODEL_DIR, "ck_cbam_model_.keras"),
        "input_size": (48, 48),
        "channels": 1,
        "classes": 7,
        "dataset": "CK+",
    },

    "fer_2013": {
        "path": os.path.join(MODEL_DIR, "fer_2013_model_.keras"),
        "input_size": (48, 48),
        "channels": 1,
        "classes": 7,
        "dataset": "FER-2013",
    },

    "fer_cbam": {
        "path": os.path.join(MODEL_DIR, "fer_CBAM_model_.keras"),
        "input_size": (48, 48),
        "channels": 1,
        "classes": 7,
        "dataset": "FER-2013",
    },

    "raf_3cnn": {
        "path": os.path.join(MODEL_DIR, "raf_3cnn_model.keras"),
        "input_size": (100, 100),
        "channels": 3,
        "classes": 7,
        "dataset": "RAF-DB",
    },

    "raf_extendedcnn": {
        "path": os.path.join(MODEL_DIR, "best_cbam_cnn.keras"),
        "input_size": (100, 100),
        "channels": 3,
        "classes": 7,
        "dataset": "RAF-DB",
    },
}


# ============================================================
# BACKWARD COMPATIBILITY
# ============================================================

MODEL_PATHS = {
    name: config["path"]
    for name, config in MODEL_CONFIG.items()
}

MODEL_NAMES = list(MODEL_CONFIG.keys())

MODEL_DATASET = {
    name: config["dataset"]
    for name, config in MODEL_CONFIG.items()
}


# ============================================================
# LABEL MAPS
# IMPORTANT: ORDER MUST MATCH TRAINING CLASS INDICES
# ============================================================

LABEL_MAPS = {
    "ck_bpm": {
        0: "Angry",
        1: "Disgust",
        2: "Fear",
        3: "Happy",
        4: "Sad",
        5: "Surprise",
        6: "Neutral",
    },

    "ck_cbam": {
        0: "Angry",
        1: "Disgust",
        2: "Fear",
        3: "Happy",
        4: "Sad",
        5: "Surprise",
        6: "Neutral",
    },

    "fer_2013": {
        0: "Angry",
        1: "Disgust",
        2: "Fear",
        3: "Happy",
        4: "Sad",
        5: "Surprise",
        6: "Neutral",
    },

    "fer_cbam": {
        0: "Angry",
        1: "Disgust",
        2: "Fear",
        3: "Happy",
        4: "Sad",
        5: "Surprise",
        6: "Neutral",
    },

    "raf_3cnn": {
        0: "Surprise",
        1: "Fear",
        2: "Disgust",
        3: "Happiness",
        4: "Sadness",
        5: "Anger",
        6: "Neutral",
    },

    "raf_extendedcnn": {
        0: "Surprise",
        1: "Fear",
        2: "Disgust",
        3: "Happiness",
        4: "Sadness",
        5: "Anger",
        6: "Neutral",
    },
}


# ============================================================
# SUBTLE CBAM
# Compatible with the provided custom-layer implementation
# ============================================================

@register_keras_serializable(
    package="Custom",
    name="SubtleCBAM",
)
class SubtleCBAM(tf.keras.layers.Layer):

    def __init__(
        self,
        reduction_ratio=16,
        attention_strength=0.15,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.reduction_ratio = reduction_ratio
        self.attention_strength = attention_strength

    def build(self, input_shape):

        channels = int(input_shape[-1])

        reduced_channels = max(
            channels // self.reduction_ratio,
            4,
        )

        # Channel attention
        self.avg_pool = GlobalAveragePooling2D()
        self.max_pool = GlobalMaxPooling2D()

        # Shared MLP
        self.fc1 = Dense(
            reduced_channels,
            activation="relu",
        )

        self.fc2 = Dense(
            channels,
            activation="sigmoid",
        )

        # Spatial attention
        self.spatial_conv = Conv2D(
            filters=1,
            kernel_size=7,
            padding="same",
            activation="sigmoid",
        )

        super().build(input_shape)

    def call(self, inputs):

        # --------------------------------------------
        # CHANNEL ATTENTION
        # --------------------------------------------

        avg = self.avg_pool(inputs)
        max_pool = self.max_pool(inputs)

        avg = self.fc2(self.fc1(avg))
        max_pool = self.fc2(self.fc1(max_pool))

        channel_attention = tf.keras.layers.Add()(
            [avg, max_pool]
        )

        channel_attention = tf.nn.sigmoid(
            channel_attention
        )

        channel_attention = tf.keras.layers.Reshape(
            (1, 1, int(inputs.shape[-1]))
        )(channel_attention)

        # Keep attention close to 1
        channel_attention = (
            1.0
            + self.attention_strength
            * (channel_attention - 0.5)
        )

        x = tf.keras.layers.Multiply()(
            [inputs, channel_attention]
        )

        # --------------------------------------------
        # SPATIAL ATTENTION
        # --------------------------------------------

        avg_spatial = tf.reduce_mean(
            x,
            axis=-1,
            keepdims=True,
        )

        max_spatial = tf.reduce_max(
            x,
            axis=-1,
            keepdims=True,
        )

        spatial_features = tf.keras.layers.Concatenate(
            axis=-1
        )(
            [avg_spatial, max_spatial]
        )

        spatial_attention = self.spatial_conv(
            spatial_features
        )

        # Keep attention subtle
        spatial_attention = (
            1.0
            + self.attention_strength
            * (spatial_attention - 0.5)
        )

        output = tf.keras.layers.Multiply()(
            [x, spatial_attention]
        )

        return output

    def get_config(self):

        config = super().get_config()

        config.update({
            "reduction_ratio": self.reduction_ratio,
            "attention_strength": self.attention_strength,
        })

        return config


# ============================================================
# CUSTOM OBJECTS
# ============================================================

CUSTOM_OBJECTS = {
    "SubtleCBAM": SubtleCBAM,
    "Custom>SubtleCBAM": SubtleCBAM,
}


# ============================================================
# LOAD A COMPLETE SAVED MODEL
# ============================================================

def load_normal_model(model_path):

    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"\nModel file not found:\n{model_path}\n"
            "Check the models directory and configured filename."
        )

    try:
        model = keras_load_model(
            model_path,
            custom_objects=CUSTOM_OBJECTS,
            compile=False,
            safe_mode=False,
        )

    except Exception as exc:
        raise RuntimeError(
            f"\nFailed to load the complete saved model:\n"
            f"{model_path}\n\n"
            "Verify that the file is a valid, complete Keras model "
            "and that its custom layers match the training version.\n"
            f"Original error: {exc}"
        ) from exc

    return model


# ============================================================
# RAF EXTENDED CNN LOADER
# IMPORTANT: LOAD THE COMPLETE MODEL, NOT REBUILT WEIGHTS
# ============================================================

def load_raf_extended_model():

    model_path = MODEL_CONFIG["raf_extendedcnn"]["path"]

    print("\n" + "=" * 65)
    print("LOADING RAF EXTENDED CNN")
    print("=" * 65)
    print(f"Model path: {model_path}")

    # best_cbam_cnn.keras is loaded as a complete saved model.
    # Do not rebuild the CNN and call load_weights() on this file.
    model = load_normal_model(model_path)

    # Validate compatibility with the configured prediction pipeline.
    expected_shape = (100, 100, 3)
    actual_shape = tuple(model.input_shape[1:])

    if actual_shape != expected_shape:
        raise ValueError(
            "\nRAF Extended CNN input shape mismatch.\n"
            f"Expected: {expected_shape}\n"
            f"Saved model: {actual_shape}\n"
            "Check the model's original training input dimensions."
        )

    if model.output_shape[-1] != 7:
        raise ValueError(
            "\nRAF Extended CNN output mismatch.\n"
            f"Expected 7 classes, got {model.output_shape[-1]}."
        )

    print("RAF Extended CNN loaded successfully!")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")

    return model


# ============================================================
# MAIN MODEL LOADER
# ============================================================

def load_model_by_name(model_name):

    if model_name not in MODEL_CONFIG:
        raise ValueError(
            f"\nUnknown model: {model_name}\n"
            f"Available models: {MODEL_NAMES}"
        )

    # All models are loaded from their complete saved files.
    # The RAF Extended CNN receives additional shape validation.
    if model_name == "raf_extendedcnn":
        return load_raf_extended_model()

    model_path = MODEL_CONFIG[model_name]["path"]

    return load_normal_model(model_path)


# ============================================================
# BACKWARD-COMPATIBLE LOADER FUNCTIONS
# ============================================================

def load_cnn_model(model_name):
    return load_model_by_name(model_name)


def load_model(model_name):
    """Compatibility wrapper for existing Streamlit pages."""
    return load_model_by_name(model_name)


# ============================================================
# LABEL FUNCTIONS
# ============================================================

def get_labels(model_name):

    if model_name not in LABEL_MAPS:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    return LABEL_MAPS[model_name]


def get_label_map(model_name):
    return get_labels(model_name)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image, model_name):
    """
    Preprocess a single image for the selected model.

    Output:
        Float32 NumPy array with shape (1, H, W, C).

    Assumption:
        Training images were normalized to [0, 1].

    PIL images and NumPy images are supported.
    NumPy inputs should be grayscale, RGB, or RGBA.
    """

    if model_name not in MODEL_CONFIG:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    config = MODEL_CONFIG[model_name]

    height, width = config["input_size"]
    channels = config["channels"]

    color_mode = "L" if channels == 1 else "RGB"

    # --------------------------------------------
    # CONVERT INPUT TO PIL IMAGE
    # --------------------------------------------

    if isinstance(image, Image.Image):

        pil_image = image.copy()

    else:

        array = np.asarray(image)

        if array.size == 0:
            raise ValueError("Input image is empty.")

        if array.ndim == 3 and array.shape[-1] == 4:
            array = array[..., :3]

        if array.ndim not in (2, 3):
            raise ValueError(
                "Image must have shape (H, W), "
                "(H, W, 3), or (H, W, 4)."
            )

        if array.ndim == 3 and array.shape[-1] != 3:
            raise ValueError(
                "Color image must have three RGB channels."
            )

        if not np.all(np.isfinite(array)):
            raise ValueError(
                "Input image contains NaN or infinite values."
            )

        if array.dtype != np.uint8:

            if (
                np.issubdtype(array.dtype, np.floating)
                and array.size > 0
                and array.min() >= 0
                and array.max() <= 1
            ):
                array = array * 255.0

            array = np.clip(
                array, 0, 255
            ).astype(np.uint8)

        if array.ndim == 2:
            pil_image = Image.fromarray(array)

        else:
            pil_image = Image.fromarray(array)

    # --------------------------------------------
    # COLOR CONVERSION AND RESIZING
    # --------------------------------------------

    pil_image = pil_image.convert(color_mode)

    pil_image = pil_image.resize(
        (width, height),
        Image.Resampling.BILINEAR,
    )

    # --------------------------------------------
    # NORMALIZATION
    # --------------------------------------------

    image_array = np.asarray(
        pil_image,
        dtype=np.float32,
    )

    image_array = image_array / 255.0

    if channels == 1:
        image_array = np.expand_dims(
            image_array,
            axis=-1,
        )

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0,
    )

    return image_array.astype(np.float32)


# ============================================================
# IMAGE PREDICTION
# ============================================================

def predict_image(model, image, model_name):

    processed_image = preprocess_image(
        image,
        model_name,
    )

    predictions = model.predict(
        processed_image,
        verbose=0,
    )

    if predictions.ndim != 2 or predictions.shape[0] != 1:
        raise ValueError(
            "Expected a prediction array with shape (1, classes). "
            f"Received: {predictions.shape}"
        )

    probabilities = predictions[0]

    labels = get_labels(model_name)

    if len(probabilities) != len(labels):
        raise ValueError(
            f"Model returned {len(probabilities)} class scores, "
            f"but {len(labels)} labels are configured for "
            f"{model_name}."
        )

    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_label = labels[predicted_index]

    confidence = float(
        probabilities[predicted_index]
    )

    return {
        "label": predicted_label,
        "class_index": predicted_index,
        "confidence": confidence,
        "probabilities": probabilities,
    }


# ============================================================
# MODEL INFORMATION
# ============================================================

def get_model_info(model_name):

    if model_name not in MODEL_CONFIG:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    config = MODEL_CONFIG[model_name]

    return {
        "name": model_name,
        "path": config["path"],
        "dataset": config["dataset"],
        "input_size": config["input_size"],
        "channels": config["channels"],
        "classes": config["classes"],
        "labels": get_labels(model_name),
    }


# ============================================================
# CHECK MODEL FILES
# ============================================================

def check_model_files():

    results = {}

    for model_name, config in MODEL_CONFIG.items():

        path = config["path"]

        results[model_name] = {
            "exists": os.path.isfile(path),
            "path": path,
        }

    return results