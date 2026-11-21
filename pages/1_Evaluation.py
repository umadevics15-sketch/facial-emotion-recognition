
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from utils.model_utils import MODEL_CONFIG, load_model
from utils.evaluation_utils import (
    create_test_generator,
    evaluate_model,
    metrics_dataframe,
    classwise_dataframe
)


st.set_page_config(
    page_title="Model Evaluation",
    page_icon="📊",
    layout="wide"
)

st.title("📊 CNN Model Evaluation")

st.write(
    "Evaluate your trained emotion recognition models "
    "using their corresponding test datasets."
)


# --------------------------------------------------
# MODEL SELECTION
# --------------------------------------------------

model_options = list(MODEL_CONFIG.keys())

model_name = st.selectbox(
    "Select a model",
    options=model_options,
    format_func=lambda key: (
        f"{key} — {MODEL_CONFIG[key]['dataset']}"
    )
)

config = MODEL_CONFIG[model_name]

st.info(
    f"Dataset: {config['dataset']} | "
    f"Input size: {config['input_size'][0]} × "
    f"{config['input_size'][1]} | "
    f"Channels: {config['channels']}"
)


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

if st.button("Evaluate Model", type="primary"):

    try:
        with st.spinner("Loading model and evaluating test images..."):

            # Load the selected trained model.
            model = load_model(model_name)

            # Create a generator using the selected model's config.
            generator = create_test_generator(
                dataset_name=config["dataset"],
                image_size=config["input_size"],
                batch_size=32,
                color_mode=(
                    "grayscale"
                    if config["channels"] == 1
                    else "rgb"
                ),
                model_name=model_name
            )

            # Run evaluation.
            results = evaluate_model(
                model,
                generator,
                model_name
            )

        st.success("Model evaluation completed!")

        # ------------------------------------------
        # OVERALL METRICS
        # ------------------------------------------

        st.subheader("Overall Performance")

        metrics = metrics_dataframe(results)

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Accuracy",
            f"{results['accuracy'] * 100:.2f}%"
        )

        col2.metric(
            "Precision",
            f"{results['precision'] * 100:.2f}%"
        )

        col3.metric(
            "Recall",
            f"{results['recall'] * 100:.2f}%"
        )

        col4.metric(
            "F1 Score",
            f"{results['f1_score'] * 100:.2f}%"
        )

        st.dataframe(
            metrics,
            use_container_width=True,
            hide_index=True
        )

        # ------------------------------------------
        # CLASS-WISE RESULTS
        # ------------------------------------------

        st.subheader("Class-wise Performance")

        classwise = classwise_dataframe(results)

        st.dataframe(
            classwise,
            use_container_width=True,
            hide_index=True
        )

        # ------------------------------------------
        # CONFUSION MATRIX
        # ------------------------------------------

        st.subheader("Confusion Matrix")

        matrix = results["confusion_matrix"]
        labels = results["target_names"]

        fig, ax = plt.subplots(figsize=(10, 7))

        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax
        )

        ax.set_xlabel("Predicted Emotion")
        ax.set_ylabel("Actual Emotion")
        ax.set_title(f"Confusion Matrix — {model_name}")

        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

        # ------------------------------------------
        # CLASSIFICATION REPORT
        # ------------------------------------------

        st.subheader("Classification Report")

        report_df = classwise.copy()

        st.dataframe(
            report_df,
            use_container_width=True,
            hide_index=True
        )

    except Exception as error:
        st.error("Evaluation failed.")
        st.exception(error)