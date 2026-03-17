import requests
import streamlit as st
from PIL import Image

# ── Config ────────────────────────────────────────────────────
API_URL = "http://plant_api:8000"

# ── Page setup ────────────────────────────────────────────────
st.set_page_config(
    page_title="Plant Seedling Classifier",
    page_icon="🌱",
    layout="centered"
)

# ── Header ────────────────────────────────────────────────────
st.title("🌱 Plant Seedling Classifier")
st.write("Upload a seedling image and the model will identify the species.")

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("About")
    st.write("This app uses a CNN trained on 12 plant species.")
    st.write("Model: PlantCNN (PyTorch)")

    st.divider()

    # Health check
    st.subheader("API Status")
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            st.success("API is online")
        else:
            st.error("API returned an error")
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API")

# ── File uploader ─────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"]
)

# ── Prediction ────────────────────────────────────────────────
if uploaded_file is not None:

    # Show the uploaded image
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Prediction")

        with st.spinner("Classifying..."):
            # Reset file pointer and send to FastAPI
            uploaded_file.seek(0)
            files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
            response = requests.post(f"{API_URL}/predict", files=files)

        if response.status_code == 200:
            result = response.json()

            st.metric(
                label="Species",
                value=result["prediction"]
            )

            st.metric(
                label="Confidence",
                value=f"{result['confidence']}%"
            )

            # Confidence bar
            confidence_decimal = result["confidence"] / 100
            st.progress(confidence_decimal)

            if result["confidence"] < 60:
                st.warning("Low confidence — try a clearer image")
            else:
                st.success("High confidence prediction")

        else:
            st.error(f"Prediction failed: {response.text}")