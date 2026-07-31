import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

IMG_SIZE = (224, 224)
THRESHOLD = 0.1
MODEL_PATH = "model/crack_detector.keras"


@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)


def preprocess(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB").resize(IMG_SIZE)
    return np.array(img, dtype=np.float32) / 255.0


def classify(model, image: Image.Image):
    arr = preprocess(image)
    prob = float(model.predict(np.expand_dims(arr, 0), verbose=0)[0][0])
    is_cracked = prob >= THRESHOLD
    label = "Cracked" if is_cracked else "Not Cracked"
    confidence = prob if is_cracked else 1.0 - prob
    return label, confidence, prob


st.set_page_config(
    page_title="Bridge Deck Crack Detector",
    page_icon=":bridge_at_night:",
    layout="centered",
)

st.title("Bridge Deck Crack Detection")
st.write(
    "Upload a photograph of a concrete bridge deck surface. "
    "The model classifies it as **Cracked** or **Not Cracked**."
)

model = get_model()

uploaded = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
)

if uploaded is not None:
    image = Image.open(uploaded)
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded image", use_container_width=True)

    label, confidence, raw_score = classify(model, image)

    with col2:
        if label == "Cracked":
            st.error(f"### {label}")
        else:
            st.success(f"### {label}")

        st.metric("Confidence", f"{confidence:.1%}")
        st.caption(f"Raw score: {raw_score:.4f} | Threshold: {THRESHOLD}")
else:
    st.info("Upload an image to get started.")


with st.sidebar:
    st.header("About")
    st.markdown(
        """
**Model:** MobileNetV2 (transfer learning)
**Input:** 224 x 224 px
**Dataset:** [Surface Crack Detection](https://www.kaggle.com/datasets/arunrk7/surface-crack-detection)
**Test accuracy:** 99.65%
**ROC AUC:** 0.9999

---

GET 324 Mini-Project | Group EE3
Dept. of Electrical & Electronics Engineering
"""
    )
