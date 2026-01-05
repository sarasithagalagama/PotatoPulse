import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os

# Set page configuration
st.set_page_config(
    page_title="PotatoPulse | Disease Classification",
    page_icon="🥔",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for the "Dark Card" Aesthetic
st.markdown("""
<style>
    /* 1. Global Background & Fonts */
    .stApp {
        background-color: #050505;
        background-image: linear-gradient(rgba(255, 255, 255, 0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255, 255, 255, 0.03) 1px, transparent 1px);
        background-size: 30px 30px;
        font-family: 'Inter', sans-serif;
    }

    /* 2. Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* 3. The Main Card Container */
    .main-card {
        background-color: #121212;
        border: 1px solid #2a2a2a;
        border-radius: 20px;
        padding: 40px;
        max-width: 650px;
        margin: 50px auto; /* Centers the card */
        box-shadow: 0 20px 50px rgba(0,0,0,0.5);
        text-align: center;
    }

    /* 4. Badge Styling */
    .badge {
        background-color: rgba(46, 204, 113, 0.15);
        color: #2ecc71;
        padding: 5px 15px;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 1px;
        display: inline-block;
        margin-bottom: 20px;
        border: 1px solid rgba(46, 204, 113, 0.3);
    }

    /* 5. Typography */
    .title-text {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 5px;
    }
    
    .subtitle-text {
        color: #888888;
        font-size: 0.9rem;
        margin-bottom: 30px;
    }

    /* 6. Customizing the File Uploader to look like the "Drop Zone" */
    /* This overrides Streamlit's default uploader styles */
    .stFileUploader {
        background-color: #0a0a0a;
        border: 2px dashed #333;
        border-radius: 12px;
        padding: 30px;
        text-align: center;
        transition: border-color 0.3s ease;
    }
    
    .stFileUploader:hover {
        border-color: #2ecc71;
    }

    /* 7. Action Button Overrides */
    div.stButton > button {
        width: 100%;
        background-color: #1a1a1a;
        color: #ffffff;
        border: 1px solid #333;
        padding: 15px 0;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 10px;
        transition: all 0.3s ease;
        margin-top: 20px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    div.stButton > button:hover {
        background-color: #2ecc71;
        color: #000;
        border-color: #2ecc71;
        box-shadow: 0 5px 15px rgba(46, 204, 113, 0.3);
    }

    /* Result Animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .result-box {
        animation: fadeIn 0.5s ease-out;
        margin-top: 30px;
        padding: 20px;
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
        border-left: 5px solid #2ecc71;
    }

</style>
""", unsafe_allow_html=True)

# --- APP LOGIC ---

@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'potato_model.keras')
    if not os.path.exists(model_path):
        return None
    return tf.keras.models.load_model(model_path)

model = load_model()
CLASS_NAMES = ['Early Blight', 'Late Blight', 'Healthy']

def predict_image(image, model):
    size = (256, 256)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction[0])
    confidence = np.max(prediction[0]) * 100
    return CLASS_NAMES[class_index], confidence

# --- LAYOUT CONSTRUCTION ---

# Use columns to center the "Card" visually
col_left, col_center, col_right = st.columns([1, 2, 1])

with col_center:
    st.markdown("""
        <div class="main-card">
            <div class="badge">● POTATOPULSE X1</div>
            <div class="title-text">🥔 Potato Disease Analyzer</div>
            <div class="subtitle-text">AI-Powered Detection for Early Blight, Late Blight & Healthy Leaves</div>
    """, unsafe_allow_html=True)
    
    # File Uploader
    file = st.file_uploader("", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    
    # Action Button
    analyze_btn = st.button("Run Analysis Protocol")

    if file is not None:
        if analyze_btn: # Only analyze when button is clicked
            if model is None:
                st.error("Error: Model not found.")
            else:
                image = Image.open(file)
                
                with st.spinner("Processing bio-data..."):
                    class_name, confidence = predict_image(image, model)
                
                # Result Display
                status_color = "#2ecc71" if class_name == "Healthy" else "#e74c3c"
                
                st.markdown(f"""
                <div class="result-box" style="border-left-color: {status_color};">
                    <h3 style="margin:0; color: {status_color};">{class_name} Detected</h3>
                    <p style="margin:5px 0 0 0; color: #aaa;">Confidence Protocol: <strong>{confidence:.2f}%</strong></p>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True) # Close main-card
