import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os

st.set_page_config(page_title="AI-Powered Surveillance", layout="wide")
st.title("🚨 AI-Powered Surveillance System (YOLOv8)")

# Load YOLOv8 model
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")   # change to your trained weights if available

model = load_model()

# File uploader
uploaded_file = st.file_uploader("📂 Upload a video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection
        results = model(frame)

        # Annotate frame
        annotated_frame = results[0].plot()

        # Display frame
        stframe.image(annotated_frame, channels="BGR", use_column_width=True)

    cap.release()
    os.remove(tfile.name)
    st.success("✅ Video processing complete!")
