# ============================================================
# SOCOFing LIVE CNN CLASSIFICATION DEMO
# ============================================================

import streamlit as st
import numpy as np
import cv2

from PIL import Image

import tensorflow as tf

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(

    page_title="SOCOFing AI Classifier",

    page_icon="🖐️",

    layout="wide"
)

# ============================================================
# TITLE
# ============================================================

st.title("🖐️ SOCOFing CNN Fingerprint Classification")

st.markdown("""
### EfficientNetB0 Multi-Task Fingerprint Classifier

This AI system classifies:

- Gender
- Hand
- Finger Type

from fingerprint images using deep learning.
""")

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource

def load_model():

    model = tf.keras.models.load_model(

        "model_optimized.tflite",

        compile=False
    )

    return model

model = load_model()

# ============================================================
# CLASS LABELS
# ============================================================

gender_classes = [

    'Female',

    'Male'
]

hand_classes = [

    'Left',

    'Right'
]

finger_classes = [

    'Index',

    'Little',

    'Middle',

    'Ring',

    'Thumb'
]

# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):

    # Convert PIL image to NumPy
    image = np.array(image)

    # Ensure RGB format
    image = cv2.cvtColor(

        image,

        cv2.COLOR_BGR2RGB
    )

    # Resize image
    image = cv2.resize(

        image,

        (224,224)
    )

    # Normalize pixel values
    image = image / 255.0

    # Expand dimensions
    image = np.expand_dims(

        image,

        axis=0
    )

    return image

# ============================================================
# FILE UPLOADER
# ============================================================

uploaded_file = st.file_uploader(

    "Upload Fingerprint Image",

    type=['bmp', 'png', 'jpg', 'jpeg']
)

# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:

    # ========================================================
    # LOAD IMAGE
    # ========================================================

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    # ========================================================
    # DISPLAY IMAGE
    # ========================================================

    with col1:

        st.image(

            image,

            caption="Uploaded Fingerprint",

            use_container_width=True
        )

    # ========================================================
    # PREPROCESS IMAGE
    # ========================================================

    processed_image = preprocess_image(image)

    # ========================================================
    # MODEL PREDICTION
    # ========================================================

    predictions = model.predict(processed_image)

    gender_pred = predictions[0]
    hand_pred = predictions[1]
    finger_pred = predictions[2]

    # ========================================================
    # GET PREDICTED CLASSES
    # ========================================================

    gender_index = np.argmax(gender_pred)

    hand_index = np.argmax(hand_pred)

    finger_index = np.argmax(finger_pred)

    # ========================================================
    # CONFIDENCE SCORES
    # ========================================================

    gender_conf = np.max(gender_pred) * 100

    hand_conf = np.max(hand_pred) * 100

    finger_conf = np.max(finger_pred) * 100

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    with col2:

        st.success("Prediction Completed")

        st.markdown("## Classification Results")

        # ====================================================
        # GENDER
        # ====================================================

        st.metric(

            label="Gender",

            value=gender_classes[gender_index]
        )

        st.progress(float(gender_conf/100))

        st.write(

            f"Confidence: {gender_conf:.2f}%"
        )

        st.divider()

        # ====================================================
        # HAND
        # ====================================================

        st.metric(

            label="Hand",

            value=hand_classes[hand_index]
        )

        st.progress(float(hand_conf/100))

        st.write(

            f"Confidence: {hand_conf:.2f}%"
        )

        st.divider()

        # ====================================================
        # FINGER
        # ====================================================

        st.metric(

            label="Finger Type",

            value=finger_classes[finger_index]
        )

        st.progress(float(finger_conf/100))

        st.write(

            f"Confidence: {finger_conf:.2f}%"
        )

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("About")

st.sidebar.info("""

SOCOFing CNN Classifier

Research-grade biometric AI system.

Model:
- EfficientNetB0
- Multi-task CNN

Tasks:
- Gender Classification
- Hand Classification
- Finger Classification

Dataset:
- SOCOFing Dataset

""")

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(

    "Developed using EfficientNetB0 + TensorFlow + Streamlit"
)
