import streamlit as st


st.set_page_config(
    page_title="Emotion Recognition",
    page_icon="😊",
    layout="wide"
)


st.title(
    "😊 Facial Emotion Recognition"
)


st.markdown(
    """
    ## Welcome

    This application provides three main functions:

    ### 📊 Evaluation
    Evaluate individual CNN models using their test datasets.

    ### 📈 Comparison
    Compare the performance of all six trained models.

    ### 😊 Prediction
    Upload a facial image and predict the emotion using a selected model.

    ---

    ### Available Models

    **CK+**
    - CK+ BPM
    - CK+ CBAM

    **FER-2013**
    - FER-2013
    - FER-2013 CBAM

    **RAF-DB**
    - RAF-DB 3CNN
    - RAF-DB Extended CNN
    """
)


st.info(
    "Select a page from the sidebar to begin."
)