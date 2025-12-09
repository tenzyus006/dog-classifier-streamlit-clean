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
# 1) Configuration accessibilité (WCAG)
# ============================================
st.set_page_config(
    page_title="Dog Breed Classifier",
    layout="wide"
)

st.markdown("""
<style>
html, body, [class*="css"] {
    font-size: 18px !important;
}
</style>
""", unsafe_allow_html=True)

# High contrast for tables
st.markdown("""
<style>
thead th {background-color: #000000 !important; color: white !important;}
tbody td {background-color: #f5f5f5 !important;}
</style>
""", unsafe_allow_html=True)


# ============================
# 2) Load pretrained models
# ============================
@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path)

MODEL_PATH_B0 = r"C:\Users\tenzi\Desktop\fastapi_app\EfficientNetB0_model.keras"
MODEL_PATH_V2 = r"C:\Users\tenzi\Desktop\fastapi_app\EfficientNetV2S_model.keras"

model_b0 = load_model(MODEL_PATH_B0)
model_v2 = load_model(MODEL_PATH_V2)

class_names = ["Irish_Terrier", "Tibetan_Terrier", "Boxer"]


# ============================================
# 3) Dataset exploration (EDA)
# ============================================
st.header("📊 Exploratory Data Analysis (EDA)")

DATASET_PATH = r"C:\Users\tenzi\Desktop\fastapi_app\dataset\train"


if os.path.exists(DATASET_PATH):
    st.subheader("📌 Distribution des classes")

    class_counts = {cls: len(os.listdir(os.path.join(DATASET_PATH, cls))) for cls in class_names}
    df_counts = pd.DataFrame(list(class_counts.items()), columns=["Class", "Count"])

    chart = alt.Chart(df_counts).mark_bar().encode(
        x="Class",
        y="Count",
        tooltip=["Class", "Count"]
    )
    st.altair_chart(chart, use_container_width=True)

    # Show sample images
    st.subheader("🖼️ Exemples d’images du dataset")
    cols = st.columns(3)

    for i, cls in enumerate(class_names):
        img_path = os.path.join(DATASET_PATH, cls, random.choice(os.listdir(os.path.join(DATASET_PATH, cls))))
        img = Image.open(img_path)
        cols[i].image(img, caption=f"{cls} (exemple)", use_container_width=True)

    # Transformations
    st.subheader("🎨 Transformations d’images (Equalization & Blur)")

    sample_cls = class_names[0]
    sample_path = os.path.join(DATASET_PATH, sample_cls, random.choice(os.listdir(os.path.join(DATASET_PATH, sample_cls))))
    sample_img = Image.open(sample_path)

    colA, colB, colC = st.columns(3)
    colA.image(sample_img, caption="Original", use_container_width=True)

    equalized = ImageOps.equalize(sample_img)
    colB.image(equalized, caption="Equalization", use_container_width=True)

    blurred = sample_img.filter(ImageFilter.GaussianBlur(3))
    colC.image(blurred, caption="Blur", use_container_width=True)

else:
    st.warning("⚠️ Dataset non trouvé. Placez votre dataset dans 'dataset/train/*'.")


# ============================================
# 4) Application de prédiction
# ============================================
st.header("🐶 Dog Breed Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Prediction B0
    img_b0 = img.resize((224, 224))
    arr_b0 = tf.keras.applications.efficientnet.preprocess_input(np.expand_dims(image.img_to_array(img_b0), axis=0))
    preds_b0 = model_b0.predict(arr_b0)
    i0 = np.argmax(preds_b0)

    # Prediction V2
    img_v2 = img.resize((384, 384))
    arr_v2 = tf.keras.applications.efficientnet.preprocess_input(np.expand_dims(image.img_to_array(img_v2), axis=0))
    preds_v2 = model_v2.predict(arr_v2)
    i2 = np.argmax(preds_v2)

    # Show predictions
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("EfficientNetB0")
        st.success(f"Prediction: {class_names[i0]} — {preds_b0[0][i0]*100:.2f}%")
        st.table(pd.DataFrame(preds_b0[0], index=class_names, columns=["Probability"]))

    with col2:
        st.subheader("EfficientNetV2-S")
        st.success(f"Prediction: {class_names[i2]} — {preds_v2[0][i2]*100:.2f}%")
        st.table(pd.DataFrame(preds_v2[0], index=class_names, columns=["Probability"]))
