import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠", layout="centered")

# ✅ Auto download model from Google Drive
@st.cache_resource
def load_model():
    if not os.path.exists('brain_tumor_model.h5'):
        with st.spinner("Downloading model... please wait"):
            gdown.download(
                'https://drive.google.com/uc?id=1y-rbKy63uxRosSV5Ci8CMJ1nCcDMpJLn',  
                'brain_tumor_model.h5',
                quiet=False
            )
    return tf.keras.models.load_model('brain_tumor_model.h5')

model = load_model()
def is_likely_mri(img):
    gray = np.array(img.convert('L'))
    return gray.mean() < 110 and gray.std() > 35

def preprocess(img):
    img = img.resize((224, 224))
    arr = np.array(img.convert('RGB')) / 255.0
    return np.expand_dims(arr, axis=0)

st.title("🧠 Brain Tumor Detection")
st.write("Upload an MRI scan image to detect brain tumor.")
st.divider()

uploaded = st.file_uploader("📤 Upload MRI Image", type=["jpg","jpeg","png"])
if uploaded is not None:
    img = Image.open(uploaded)
    if not is_likely_mri(img):
        st.error("❌ Not a valid MRI scan. Please upload a brain MRI image.")
        st.stop()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📷 Uploaded MRI")
        st.image(img, use_column_width=True)
    with col2:
        st.subheader("🔍 Result")
        with st.spinner("Analyzing..."):
            arr = preprocess(img)
            score = model.predict(arr)[0][0]
        if score > 0.5:
            st.error("⚠️ Tumor Detected")
            confidence = score * 100
        else: 
          st.success("✅ No Tumor Detected")
          confidence = (1 - score) * 100
          st.metric("Confidence", f"{confidence:.1f}%")
        st.progress(float(confidence / 100))
        st.divider()
        if score > 0.5:
            st.warning("⚠️ Please consult a doctor immediately.")
        else:
            st.info("✅ Stay healthy!")

st.caption("⚠️ For educational purposes only.")
