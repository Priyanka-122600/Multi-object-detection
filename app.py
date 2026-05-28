from io import BytesIO
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

@st.cache_resource
def load_model():
    return YOLO('yolov5su.pt')

model = load_model()

st.set_page_config(page_title="Multi-Object Detection App", layout="centered")
st.title("🔍 Multi-Object Detection Web App")
st.write("Upload an image to detect multiple objects using the YOLOv5 model.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Detecting objects..."):
        results = model.predict(source=np.array(image), conf=0.3)
        result_img = results[0].plot()

    st.image(result_img, caption="Detected Objects", use_container_width=True)

    # Download button — all inside the if block
    result_pil = Image.fromarray(result_img)
    img_buffer = BytesIO()
    result_pil.save(img_buffer, format="JPEG")
    img_bytes = img_buffer.getvalue()

    st.download_button(
        label="📥 Download Detected Image",
        data=img_bytes,
        file_name="detected_image.jpg",
        mime="image/jpeg"
    )