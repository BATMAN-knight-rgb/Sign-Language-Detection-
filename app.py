import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import requests   
import io

# Streamlit config
st.set_page_config(
    page_title="LingunaMansu - Sign to Text",
    layout="centered",
    page_icon="🤟"
)

# Custom CSS for styling
st.markdown("""
    <style>
    body {background-color: #f0f4f8;}
    .main {background-color: #ffffff; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);}
    h1 {color: #3f51b5; text-align: center;}
    .predict-box {background-color: #e3f2fd; padding: 1rem; border-radius: 10px; text-align: center; font-size: 24px; color: #0d47a1; margin-top: 20px;}
    </style>
""", unsafe_allow_html=True)

# App header
st.markdown('<div class="main">', unsafe_allow_html=True)
st.markdown("<h1>LingunaMansu 🤟</h1>", unsafe_allow_html=True)
st.write("Real-time Sign Language to Text Translator")

# Webcam section
frame_window = st.empty()
predict_box = st.empty()
translate_box = st.empty()

# Start/stop buttons
start = st.button("📷 Start Webcam")
stop = st.button("❌ Stop Webcam")

if start:
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FPS, 15)  # Set the frame rate to 15 FPS to reduce load
    st.info("Webcam started. Show a sign!")
    
    frame_counter = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to capture frame.")
            break

        frame_counter += 1
        
        # Process every 5th frame to reduce load
        if frame_counter % 5 == 0:
            # Resize the frame to lower resolution
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(frame_rgb, (640, 480))  # Lower resolution for faster processing
            frame_window.image(resized, channels="RGB")

            # Convert to image for backend prediction
            img = Image.fromarray(resized)
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            byte_img = buf.getvalue()

            # Send to backend for prediction
            try:
                response = requests.post("http://localhost:8000/predict", data=byte_img)
                if response.status_code == 200:
                    prediction = response.json()["gesture"]
                    predict_box.markdown(f'<div class="predict-box">Predicted: <b>{prediction}</b></div>', unsafe_allow_html=True)

                    # Translation part
                    try:
                        translate_response = requests.post(
                            "http://localhost:8000/translate",
                            json={"text": prediction}
                        )
                        if translate_response.status_code == 200:
                            translated_text = translate_response.json()["translated_text"]
                            translate_box.markdown(f"### 📝 Translated Text: {translated_text}")
                        else:
                            translate_box.warning("Translation service failed.")
                    except Exception as e:
                        translate_box.warning("Translation backend not available.")

            except Exception as e:
                predict_box.markdown('<div class="predict-box">🔌 Backend not connected</div>', unsafe_allow_html=True)

        # Break if stop button is clicked
        if stop:
            cap.release()
            st.success("Webcam stopped.")
            break

st.markdown("</div>", unsafe_allow_html=True)
