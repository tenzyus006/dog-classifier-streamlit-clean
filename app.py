import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageFilter, ImageOps
import pandas as pd
import altair as alt
import os
import random

# ============================================
# 1) PAGE CONFIG + ACCESSIBILITÉ (WCAG)
# ============================================
st.set_page_config(
    page_title="Dog Breed Classifier",
    layout="wide"
)

# Larger base font for accessibility
st.markdown("""
<style>
html, body, [class*="css"] {
    font-size: 18px !important;
}
</style>
""", unsafe_allow_html=True)

# High-contrast tables
st.markdown("""
<style>
thead th {background-color: #000000 !important; color: white !important;}
tbody td {background-color: #f5f5f5 !important;}
</style>
""", unsafe_allow_html=True)


# ============================================
# 2) LOAD MODELS (RELATIVE PATHS)
# ============================================
@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path)

MODEL_PATH_B0 = "EfficientNetB0_model.keras"
MODEL_PATH_V2 = "EfficientNetV2S_model.keras"

model_b0 = load_model(MODEL_PATH_B0)
model_v2 = load_model(MODEL_PATH_V2)

class_names = ["Irish_Terrier", "Tibetan_Terrier", "Boxer"]


# ============================================
# 3) EDA — EXPLORATION DES DONNÉES
# ============================================
st.header("📊 Exploratory Data Analysis (EDA)")

DATASET_PATH = "dataset"  # ✔️ Folder in the repo

if os.path.exists(DATASET_PATH):
    st.subheader("📌 Class Distribution")

    class_counts = {}
    for cls in class_names:
        folder = os.path.join(DATASET_PATH, cls)
        class_counts[cls] = len(os.listdir(folder)) if os.path.exists(folder) else 0

    df_counts = pd.DataFrame(list(class_counts.items()), columns=["Class", "Count"])

    chart = alt.Chart(df_counts).mark_bar().encode(
        x="Class",
        y="Count",
        tooltip=["Class", "Count"]
    )
    st.altair_chart(chart, use_container_width=True)

    # Show sample images
    st.subheader("🖼️ Sample Images")
    cols = st.columns(3)

    for i, cls in enumerate(class_names):
        folder = os.path.join(DATASET_PATH, cls)
        if os.path.exists(folder) and len(os.listdir(folder)) > 0:
            img_path = os.path.join(folder, random.choice(os.listdir(folder)))
            img = Image.open(img_path)
            cols[i].image(img, caption=f"{cls} (sample)", use_container_width=True)

    # Image Transformations
    st.subheader("🎨 Image Transformations")

    sample_cls = class_names[0]
    sample_folder = os.path.join(DATASET_PATH, sample_cls)
    sample_img = Image.open(os.path.join(sample_folder, os.listdir(sample_folder)[0]))

    colA, colB, colC = st.columns(3)
    colA.image(sample_img, caption="Original", use_container_width=True)
    colB.image(ImageOps.equalize(sample_img), caption="Equalized", use_container_width=True)
    colC.image(sample_img.filter(ImageFilter.GaussianBlur(3)), caption="Blurred", use_container_width=True)

else:
    st.warning("⚠️ Dataset folder not found. Place a small dataset in /dataset.")


# ============================================
# 4) MAIN CLASSIFICATION APP
# ============================================
st.header("🐶 Dog Breed Classifier")

uploaded_file = st.file_uploader("Upload a dog image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # -------------------------------
    # EfficientNetB0 Prediction
    # -------------------------------
    img_b0 = img.resize((224, 224))
    arr_b0 = tf.keras.applications.efficientnet.preprocess_input(
        np.expand_dims(image.img_to_array(img_b0), axis=0)
    )
    preds_b0 = model_b0.predict(arr_b0)
    idx_b0 = np.argmax(preds_b0)

    # -------------------------------
    # EfficientNetV2 Prediction
    # -------------------------------
    img_v2 = img.resize((384, 384))
    arr_v2 = tf.keras.applications.efficientnet.preprocess_input(
        np.expand_dims(image.img_to_array(img_v2), axis=0)
    )
    preds_v2 = model_v2.predict(arr_v2)
    idx_v2 = np.argmax(preds_v2)

    # -------------------------------
    # Display predictions
    # -------------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("EfficientNetB0")
        st.success(f"Prediction: {class_names[idx_b0]} — {preds_b0[0][idx_b0]*100:.2f}%")
        st.table(pd.DataFrame(preds_b0[0], index=class_names, columns=["Probability"]))

    with col2:
        st.subheader("EfficientNetV2-S")
        st.success(f"Prediction: {class_names[idx_v2]} — {preds_v2[0][idx_v2]*100:.2f}%")
        st.table(pd.DataFrame(preds_v2[0], index=class_names, columns=["Probability"]))
