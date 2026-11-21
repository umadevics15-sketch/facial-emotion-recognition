import streamlit as st
import subprocess
import sys
from pathlib import Path

st.set_page_config(page_title="Capture Photo", page_icon="📸", layout="wide")

# Project structure:
# Emotion_Recognition/
#   app.py
#   capture.py
#   models/raf_3cnn_model.keras
#   pages/5_Capture_Photo.py

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = PROJECT_ROOT / "capture.py"

st.title("📸 Capture Photo & Detect Emotion")
st.write("Open the camera, position your face, and press SPACE in the camera window.")
st.info(
    "The camera and prediction result appear in an OpenCV window separate from "
    "the Streamlit browser. Press SPACE to capture; press Q or ESC to exit."
)

if not SCRIPT_PATH.exists():
    st.error(f"Cannot find capture.py at: {SCRIPT_PATH}")
    st.write("Place capture.py in the main Emotion_Recognition project folder.")
else:
    if st.button("Open Camera / Capture Photo", type="primary", use_container_width=True):
        try:
            subprocess.Popen(
                [sys.executable, str(SCRIPT_PATH)],
                cwd=str(PROJECT_ROOT)
            )
            st.success("Camera started. Switch to the OpenCV camera window and press SPACE.")
        except Exception as exc:
            st.error(f"Could not start photo capture: {exc}")

st.caption("Captured images are saved by capture.py in its captured_photos folder.")
