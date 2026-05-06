import streamlit as st
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO
import av
import cv2
import numpy as np

# Page Configuration
st.set_page_config(page_title="AI Object Tracker", layout="wide")

@st.cache_resource
def load_model():
    # Using yolov8n (nano) for best real-time CPU performance
    return YOLO("yolov8n.pt")

model = load_model()

st.title("🎥 Enhanced Live Object Detection & Tracing")
st.sidebar.header("Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)

st.write("This app detects objects and counts them in real-time using YOLOv8.")

# Enhancement logic: The callback handles the frame processing
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    # Run YOLOv8 tracking
    results = model.track(
        img,
        persist=True,
        conf=conf_threshold,
        verbose=False
    )

    # Process results
    annotated_frame = results[0].plot()
    
    # ENHANCEMENT: Object Counting Logic
    if results[0].boxes.id is not None:
        count = len(results[0].boxes.id)
    else:
        count = 0

    # Overlay count on the video feed
    cv2.putText(
        annotated_frame, 
        f"Total Objects: {count}", 
        (50, 50), 
        cv2.FONT_HERSHEY_DUPLEX, 
        1.2, 
        (0, 255, 0), 
        2
    )

    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# Layout columns
col1, col2 = st.columns([3, 1])

with col1:
    ctx = webrtc_streamer(
        key="object-detection",
        video_frame_callback=video_frame_callback,
        async_processing=True,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={"video": True, "audio": False},
    )

with col2:
    st.subheader("Activity Status")
    if ctx.state.playing:
        st.success("Camera is Active")
        st.info("The AI is currently tracking and counting objects.")
    else:
        st.warning("Camera is Offline")

    # ENHANCEMENT: Instructions for Report
    st.markdown("""
    **To complete your report:**
    1. Take a screenshot when objects are detected.
    2. Check the counter at the top-left of the video.
    3. Note if detection slows down when more objects appear.
    """)
