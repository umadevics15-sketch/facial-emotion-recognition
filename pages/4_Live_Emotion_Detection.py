import streamlit as st
import subprocess
import sys
from pathlib import Path

st.set_page_config(page_title="Live Emotion Detection", page_icon="🎥", layout="wide")

# Project structure:
# Emotion_Recognition/
#   app.py
#   live.py
#   models/raf_3cnn_model.keras
#   pages/4_Live_Emotion_Detection.py

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = PROJECT_ROOT / "live.py"

st.title("🎥 Live Emotion Detection")
st.write("Start your webcam to detect facial emotions continuously.")
st.info(
    "The camera runs in an OpenCV window separate from the Streamlit browser. "
    "Press Q in the camera window to stop it."
)

if not SCRIPT_PATH.exists():
    st.error(f"Cannot find live.py at: {SCRIPT_PATH}")
    st.write("Place live.py in the main Emotion_Recognition project folder.")
else:
    if st.button("Start Live Camera", type="primary", use_container_width=True):
        try:
            # Launch the existing script from the project root so relative paths work.
            subprocess.Popen(
                [sys.executable, str(SCRIPT_PATH)],
                cwd=str(PROJECT_ROOT)
            )
            st.success("Live camera started. Switch to the OpenCV camera window.")
        except Exception as exc:
            st.error(f"Could not start live camera: {exc}")

st.caption("If the camera does not open, close other apps using the webcam and try again.")
