import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os

# =========================
# LOAD MODEL (once)
# =========================
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# =========================
# PROCESS IMAGE
# =========================
def process_image(image):
    img = np.array(image)

    results = model.predict(source=img, conf=0.4, save=False)

    free = 0
    occupied = 0

    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()

    for box, cls in zip(boxes, classes):
        x1, y1, x2, y2 = map(int, box)

        if int(cls) == 0:
            color = (0, 255, 0)
            free += 1
        else:
            color = (0, 0, 255)
            occupied += 1

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    total = free + occupied

    return img, free, occupied, total


# =========================
# UI
# =========================
st.set_page_config(page_title="Smart Parking", layout="wide")

st.title("🚗 Smart Parking Slot Detection")

uploaded_file = st.file_uploader("Upload Parking Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)

    st.subheader("Original Image")
    st.image(image, use_container_width=True)

    with st.spinner("Detecting..."):
        output_img, free, occupied, total = process_image(image)

    st.subheader("Detection Result")
    st.image(output_img, channels="BGR", use_container_width=True)

    # Stats
    col1, col2, col3 = st.columns(3)

    col1.metric("🟢 Free Slots", free)
    col2.metric("🔴 Occupied", occupied)
    col3.metric("📊 Total", total)
