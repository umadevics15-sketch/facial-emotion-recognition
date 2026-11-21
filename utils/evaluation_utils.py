
import os
import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from .model_utils import (
    MODEL_CONFIG,
    MODEL_DATASET,
    get_label_map,
    BASE_DIR
)


# --------------------------------------------------
# DATASET PATHS
# --------------------------------------------------

DATASETS_DIR = os.path.join(BASE_DIR, "datasets")


def get_test_directory(dataset_name):
    """
    Locate the test directory for a dataset.

    Supported examples:
        datasets/FER-2013/test
        datasets/CK+/test
        datasets/RAF-DB/test
    """

    dataset_path = os.path.join(DATASETS_DIR, dataset_name)

    possible_paths = [
        os.path.join(dataset_path, "test"),
        os.path.join(dataset_path, "Test"),
        os.path.join(dataset_path, "testing"),
        os.path.join(dataset_path, "Testing"),
        dataset_path
    ]

    for path in possible_paths:
        if not os.path.isdir(path):
            continue

        # Check whether the directory contains class subdirectories.
        subdirectories = [
            name for name in os.listdir(path)
            if os.path.isdir(os.path.join(path, name))
        ]

        if subdirectories:
            return path

    raise FileNotFoundError(
        f"Could not find a test directory for '{dataset_name}'. "
        f"Checked: {possible_paths}"
    )


# --------------------------------------------------
# MODEL CONFIGURATION
# --------------------------------------------------

def get_model_input_config(model, model_name):
    """
    Get image dimensions and channels from MODEL_CONFIG,
    then validate against the loaded model's input shape.
    """

    if model_name not in MODEL_CONFIG:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available models: {list(MODEL_CONFIG.keys())}"
        )

    config = MODEL_CONFIG[model_name]

    image_size = tuple(config["input_size"])
    channels = config["channels"]

    expected_shape = tuple(model.input_shape[1:])

    configured_shape = (
        image_size[0],
        image_size[1],
        channels
    )

    if expected_shape != configured_shape:
        raise ValueError(
            f"Model configuration mismatch for '{model_name}'. "
            f"MODEL_CONFIG specifies {configured_shape}, "
            f"but the loaded model expects {expected_shape}."
        )

    color_mode = "grayscale" if channels == 1 else "rgb"

    return image_size, color_mode


# --------------------------------------------------
# TEST DATA GENERATOR
# --------------------------------------------------

def create_test_generator(
    dataset_name,
    image_size=None,
    batch_size=32,
    color_mode=None,
    model_name=None
):
    """
    Create a test image generator.

    When model_name is supplied, use that model's configured
    dimensions and channel format.
    """

    if model_name is not None:
        if model_name not in MODEL_CONFIG:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Available models: {list(MODEL_CONFIG.keys())}"
            )

        config = MODEL_CONFIG[model_name]

        # Always take the dimensions and channels from the
        # selected model configuration.
        image_size = tuple(config["input_size"])
        color_mode = (
            "grayscale"
            if config["channels"] == 1
            else "rgb"
        )

        configured_dataset = config["dataset"]

        if dataset_name != configured_dataset:
            raise ValueError(
                f"Dataset mismatch: model '{model_name}' uses "
                f"'{configured_dataset}', but '{dataset_name}' "
                f"was requested."
            )

    if image_size is None:
        image_size = (100, 100)

    if color_mode is None:
        color_mode = "rgb"

    image_size = tuple(image_size)

    if color_mode not in ("grayscale", "rgb"):
        raise ValueError(
            "color_mode must be 'grayscale' or 'rgb'."
        )

    test_directory = get_test_directory(dataset_name)

    datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    generator = datagen.flow_from_directory(
        directory=test_directory,
        target_size=image_size,
        color_mode=color_mode,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )

    if generator.samples == 0:
        raise ValueError(
            f"No test images found in '{test_directory}'."
        )

    print(f"Dataset: {dataset_name}")
    print(f"Test directory: {test_directory}")
    print(f"Image size: {image_size}")
    print(f"Color mode: {color_mode}")
    print(f"Number of images: {generator.samples}")
    print(f"Detected classes: {generator.class_indices}")

    return generator


# --------------------------------------------------
# LABEL MAPPING
# --------------------------------------------------

def get_generator_label_mapping(generator):
    """
    Return the class-to-index mapping detected by Keras.
    """

    return generator.class_indices


# --------------------------------------------------
# MODEL EVALUATION
# --------------------------------------------------

def evaluate_model(model, generator, model_name):
    """
    Evaluate a loaded CNN model on a test generator.

    Returns a dictionary containing metrics,
    classification report, confusion matrix and predictions.
    """

    if model_name not in MODEL_CONFIG:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available models: {list(MODEL_CONFIG.keys())}"
        )

    config = MODEL_CONFIG[model_name]

    expected_size = tuple(config["input_size"])
    expected_channels = config["channels"]

    expected_color_mode = (
        "grayscale"
        if expected_channels == 1
        else "rgb"
    )

    model_shape = tuple(model.input_shape[1:])
    expected_model_shape = (
        expected_size[0],
        expected_size[1],
        expected_channels
    )

    # Check the loaded model shape.
    if model_shape != expected_model_shape:
        raise ValueError(
            f"Model input mismatch. Expected model shape "
            f"{expected_model_shape}, but received {model_shape}."
        )

    # Check the generator dimensions.
    generator_size = tuple(generator.target_size)

    if generator_size != expected_size:
        raise ValueError(
            f"Image size mismatch. Model expects {expected_size}, "
            f"but generator uses {generator_size}. "
            f"Recreate the generator with model_name='{model_name}'."
        )

    if generator.color_mode != expected_color_mode:
        raise ValueError(
            f"Color mode mismatch. Model requires "
            f"'{expected_color_mode}', but generator uses "
            f"'{generator.color_mode}'."
        )

    # Validate model output dimensions.
    number_of_outputs = model.output_shape[-1]

    if number_of_outputs != config["classes"]:
        raise ValueError(
            f"Model has {number_of_outputs} output classes, "
            f"but MODEL_CONFIG specifies {config['classes']}."
        )

    # Predict probabilities.
    generator.reset()
    predictions = model.predict(generator, verbose=1)

    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = generator.classes

    if len(predicted_classes) != len(true_classes):
        raise ValueError(
            "Prediction count does not match the number of test labels."
        )

    # Use the model's label map.
    label_map = get_label_map(model_name)

    # Keep class IDs aligned with the model's output indices.
    class_ids = list(range(config["classes"]))

    target_names = [
        label_map.get(index, f"Class_{index}")
        for index in class_ids
    ]

    # Metrics.
    accuracy = accuracy_score(true_classes, predicted_classes)

    precision = precision_score(
        true_classes,
        predicted_classes,
        labels=class_ids,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        true_classes,
        predicted_classes,
        labels=class_ids,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        true_classes,
        predicted_classes,
        labels=class_ids,
        average="weighted",
        zero_division=0
    )

    report = classification_report(
        true_classes,
        predicted_classes,
        labels=class_ids,
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )

    matrix = confusion_matrix(
        true_classes,
        predicted_classes,
        labels=class_ids
    )

    # Build results.
    results = {
        "model_name": model_name,
        "dataset_name": config["dataset"],
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "classification_report": report,
        "confusion_matrix": matrix,
        "predictions": predicted_classes,
        "probabilities": predictions,
        "true_classes": true_classes,
        "label_map": label_map,
        "target_names": target_names,
        "generator": generator
    }

    return results


# --------------------------------------------------
# METRICS DATAFRAME
# --------------------------------------------------

def metrics_dataframe(results):
    """
    Convert overall metrics to a DataFrame.
    """

    return pd.DataFrame([
        {
            "Model": results["model_name"],
            "Dataset": results["dataset_name"],
            "Accuracy": results["accuracy"],
            "Precision": results["precision"],
            "Recall": results["recall"],
            "F1 Score": results["f1_score"]
        }
    ])


# --------------------------------------------------
# CLASS-WISE METRICS DATAFRAME
# --------------------------------------------------

def classwise_dataframe(results):
    """
    Return precision, recall and F1-score for each class.
    """

    report = results["classification_report"]

    rows = []

    for class_name in results["target_names"]:
        if class_name not in report:
            continue

        class_metrics = report[class_name]

        rows.append({
            "Emotion": class_name,
            "Precision": class_metrics["precision"],
            "Recall": class_metrics["recall"],
            "F1 Score": class_metrics["f1-score"],
            "Support": class_metrics["support"]
        })

    return pd.DataFrame(rows)