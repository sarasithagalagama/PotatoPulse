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
    initial_sidebar_state="expanded"
)

# Custom CSS for premium aesthetics
st.markdown("""
    <style>
    /* Global Styles */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
        font-family: 'Inter', sans-serif;
    }
    
    /* Custom Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    
    /* Titles and Headings */
    h1, h2, h3 {
        color: #2ecc71;
        font-weight: 700;
    }
    
    h1 {
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 0 0 20px rgba(46, 204, 113, 0.3);
    }
    
    /* Glassmorphism Card for Results */
    .result-card {
        background: rgba(22, 27, 34, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(48, 54, 61, 0.5);
        border-radius: 16px;
        padding: 2rem;
        margin-top: 2rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        text-align: center;
    }
    
    /* Upload Widget Styling */
    .stFileUploader {
        padding: 2rem;
        border: 2px dashed #2ecc71;
        border-radius: 12px;
        background-color: rgba(46, 204, 113, 0.05);
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        background-color: rgba(46, 204, 113, 0.1);
        border-color: #27ae60;
    }

    /* Info Box styling */
    .info-box {
        background-color: #1f2937;
        border-left: 4px solid #2ecc71;
        padding: 1rem;
        margin-bottom: 1rem;
        border-radius: 4px;
    }
    </style>
""", unsafe_allow_html=True)

# Application Header
st.markdown("<h1>🥔 PotatoPulse <br><span style='font-size: 1.5rem; color: #a3a3a3; font-weight: 400;'>Advanced Disease Detection System</span></h1>", unsafe_allow_html=True)

# Sidebar Content
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/188/188333.png", width=100)
    st.markdown("### About PotatoPulse")
    st.markdown("""
    PotatoPulse uses advanced Deep Learning (CNN) to detect diseases in potato plants with high accuracy.
    
    **Supported Classes:**
    - 🌿 **Healthy**: The plant is healthy.
    - 🍂 **Early Blight**: Caused by *Alternaria solani*.
    - 🍄 **Late Blight**: Caused by *Phytophthora infestans*.
    """)
    st.markdown("---")
    st.markdown("### Usage Guide")
    st.markdown("1. Upload a clear image of a potato leaf.\n2. The system will analyze the image.\n3. View the prediction and confidence score.")
    
    st.markdown("---")
    st.markdown("<div style='text-align: center; color: #666;'>Powered by TensorFlow & Streamlit</div>", unsafe_allow_html=True)


# Load the Model
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'potato_model.keras')
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
        return None
    return tf.keras.models.load_model(model_path)

model = load_model()

# Class Names (Standard PlantVillage Order)
CLASS_NAMES = ['Early Blight', 'Late Blight', 'Healthy']

def import_and_predict(image_data, model):
    size = (256, 256)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)
    
    # Model expects input in range [0, 255] if Rescaling layer is included in the model itself (which it is)
    # However, let's verify if the notebook's preprocessing handles everything.
    # The notebook model has `layers.Rescaling(1./255)` as the first layer.
    # So passing uint8 [0, 255] is correct.
    
    prediction = model.predict(img_array)
    return prediction

# Main Content Area
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    file = st.file_uploader("Upload Leaf Image", type=["jpg", "png", "jpeg"])

if file is None:
    st.markdown("<div style='text-align: center; margin-top: 2rem; color: #888;'>Please upload an image to start analysis</div>", unsafe_allow_html=True)
else:
    if model is None:
        st.error("Model could not be loaded. Please check the model path.")
    else:
        image = Image.open(file)
        
        # Display the image
        st.markdown("<br>", unsafe_allow_html=True)
        col_img1, col_img2, col_img3 = st.columns([1, 2, 1])
        with col_img2:
            st.image(image, use_container_width=True, caption="Uploaded Image")
        
        # Make Prediction
        with st.spinner('Analyzing...'):
            prediction = import_and_predict(image, model)
            class_index = np.argmax(prediction[0])
            class_name = CLASS_NAMES[class_index]
            confidence = np.max(prediction[0]) * 100

        # Display Result
        st.markdown(f"""
        <div class="result-card">
            <h2 style="margin-bottom: 0;">Prediction Result</h2>
            <div style="font-size: 3rem; margin: 1rem 0; color: #2ecc71;">{class_name}</div>
            <div style="font-size: 1.2rem; color: #aaa;">Confidence: <strong>{confidence:.2f}%</strong></div>
        </div>
        """, unsafe_allow_html=True)
        
        # Additional Details based on prediction
        st.markdown("<br>", unsafe_allow_html=True)
        if class_name == 'Healthy':
            st.success("✅ Great news! The plant appears to be healthy.")
        elif class_name == 'Early Blight':
            st.warning("⚠️ **Early Blight Detected**\n\nSymptoms include small, dark spots with concentric rings on older leaves. It is often caused by warm, humid conditions.")
        elif class_name == 'Late Blight':
            st.error("🚨 **Late Blight Detected**\n\nThis is a serious disease that causes large, irregular, dark/water-soaked spots. It spreads rapidly in cool, wet weather.")
