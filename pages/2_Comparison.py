
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from utils.model_utils import MODEL_CONFIG, load_model
from utils.evaluation_utils import (
    create_test_generator,
    evaluate_model,
)

st.set_page_config(
    page_title="Model Comparison",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Model Comparison")

# Select dataset and its two models
datasets = sorted({
    config["dataset"] for config in MODEL_CONFIG.values()
})

dataset_name = st.selectbox("Select dataset", datasets)

available_models = [
    name for name, config in MODEL_CONFIG.items()
    if config["dataset"] == dataset_name
]

if len(available_models) < 2:
    st.error("At least two models are required for this dataset.")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    model_a = st.selectbox(
        "Model 1",
        available_models,
        index=0,
    )

with col2:
    model_b = st.selectbox(
        "Model 2",
        available_models,
        index=1,
    )

if model_a == model_b:
    st.warning("Please select two different models.")
    st.stop()


# Evaluate using the SAME pipeline as the Evaluation page
if st.button("Compare Models", type="primary"):
    results = {}
    errors = {}

    progress = st.progress(0)

    for index, model_name in enumerate([model_a, model_b]):
        try:
            with st.spinner(f"Evaluating {model_name}..."):
                config = MODEL_CONFIG[model_name]

                model = load_model(model_name)

                generator = create_test_generator(
                    dataset_name=config["dataset"],
                    image_size=config["input_size"],
                    batch_size=32,
                    color_mode=(
                        "grayscale"
                        if config["channels"] == 1
                        else "rgb"
                    ),
                    model_name=model_name,
                )

                results[model_name] = evaluate_model(
                    model,
                    generator,
                    model_name,
                )

        except Exception as error:
            errors[model_name] = str(error)

        progress.progress((index + 1) / 2)

    st.session_state["comparison_results"] = results
    st.session_state["comparison_errors"] = errors
    st.session_state["comparison_selection"] = (
        dataset_name, model_a, model_b
    )


# Display results
selection = (dataset_name, model_a, model_b)

if st.session_state.get("comparison_selection") == selection:
    results = st.session_state["comparison_results"]
    errors = st.session_state["comparison_errors"]

    for name, error in errors.items():
        st.error(f"{name}: {error}")

    if len(results) != 2:
        st.warning("Both models must evaluate successfully to compare.")
        st.stop()

    result_a = results[model_a]
    result_b = results[model_b]

    metric_info = [
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1-score", "f1_score"),
    ]

    st.subheader("Performance Comparison")

    left, right = st.columns(2)

    with left:
        st.markdown(f"### {model_a}")
        for label, key in metric_info:
            st.metric(label, f"{result_a[key] * 100:.2f}%")

    with right:
        st.markdown(f"### {model_b}")
        for label, key in metric_info:
            st.metric(label, f"{result_b[key] * 100:.2f}%")

    # Side-by-side summary table
    st.subheader("Metric Summary")

    comparison_df = pd.DataFrame({
        model_a: [result_a[key] for _, key in metric_info],
        model_b: [result_b[key] for _, key in metric_info],
    }, index=[label for label, _ in metric_info])

    comparison_df.index.name = "Metric"

    st.dataframe(
        comparison_df.style.format("{:.2%}"),
        use_container_width=True,
    )

    # Grouped bar chart
    st.subheader("Metric Comparison Chart")

    chart_df = pd.DataFrame({
        "Metric": [label for label, _ in metric_info] * 2,
        "Score (%)": (
            [result_a[key] * 100 for _, key in metric_info]
            + [result_b[key] * 100 for _, key in metric_info]
        ),
        "Model": [model_a] * 4 + [model_b] * 4,
    })

    chart_df["Metric"] = pd.Categorical(
        chart_df["Metric"],
        categories=[label for label, _ in metric_info],
        ordered=True,
    )

    fig, ax = plt.subplots(figsize=(9, 4.5))

    sns.barplot(
        data=chart_df,
        x="Metric",
        y="Score (%)",
        hue="Model",
        ax=ax,
    )

    ax.set_ylim(0, 100)
    ax.set_ylabel("Score (%)")
    ax.set_xlabel("")
    ax.legend(title="Model")
    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    # Confusion matrices from the actual evaluation results
    st.subheader("Confusion Matrices")

    left, right = st.columns(2)

    for column, model_name, result in [
        (left, model_a, result_a),
        (right, model_b, result_b),
    ]:
        with column:
            st.markdown(f"#### {model_name}")

            fig, ax = plt.subplots(figsize=(7, 6))

            sns.heatmap(
                result["confusion_matrix"],
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=result["target_names"],
                yticklabels=result["target_names"],
                ax=ax,
            )

            ax.set_xlabel("Predicted Emotion")
            ax.set_ylabel("Actual Emotion")
            ax.tick_params(axis="x", rotation=45)
            ax.tick_params(axis="y", rotation=0)

            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # Best metric per model
    st.subheader("Better Result")

    for label, key in metric_info:
        score_a = result_a[key]
        score_b = result_b[key]

        if abs(score_a - score_b) < 1e-12:
            winner = "Tie"
        else:
            winner = model_a if score_a > score_b else model_b

        st.write(
            f"**{label}:** {winner}"
            if winner == "Tie"
            else f"**{label}:** {winner} "
                 f"({max(score_a, score_b) * 100:.2f}%)"
        )