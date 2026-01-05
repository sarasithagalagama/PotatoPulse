import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os

# Set page configuration
st.set_page_config(
    page_title="PotatoPulse",
    page_icon="🥔",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
        /* Import Google Font */
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        /* Main Background */
        .stApp {
            background-color: #F8FAFC; /* Light gray/white background for cleanliness */
            color: #0F172A;
        }
        
        /* Dark Mode Support (optional adjustments if system prefers dark) */
        @media (prefers-color-scheme: dark) {
            .stApp {
                background-color: #0F172A; /* Slate 900 */
                color: #F8FAFC;
            }
        }

        /* Header Styling */
        .main-header {
            text-align: center;
            padding: 2rem 0;
            margin-bottom: 2rem;
        }
        
        .main-header h1 {
            font-weight: 700;
            color: #166534; /* Green-800 */
            margin-bottom: 0.5rem;
            font-size: 2.5rem;
        }
        
        .main-header p {
            color: #64748B; /* Slate-500 */
            font-size: 1.1rem;
        }

        /* File Uploader Customization */
        [data-testid="stFileUploader"] {
            background-color: white;
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            border: 1px solid #E2E8F0;
        }
        
        @media (prefers-color-scheme: dark) {
            [data-testid="stFileUploader"] {
                background-color: #1E293B;
                border-color: #334155;
            }
            .main-header h1 { color: #4ADE80; } /* Green-400 */
            .main-header p { color: #94A3B8; }
        }

        /* Result Card */
        .prediction-card {
            background: white;
            border-radius: 16px;
            padding: 2rem;
            margin-top: 2rem;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
            text-align: center;
            border-top: 6px solid #22c55e;
            animation: fadeIn 0.5s ease-in;
        }
        
        @media (prefers-color-scheme: dark) {
            .prediction-card {
                background: #1E293B;
                box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
            }
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .confidence-badge {
            display: inline-block;
            padding: 0.5rem 1rem;
            border-radius: 9999px;
            background-color: #DCFCE7;
            color: #166534;
            font-weight: 600;
            font-size: 0.9rem;
            margin-top: 1rem;
        }
        
        @media (prefers-color-scheme: dark) {
            .confidence-badge {
                background-color: #14532D;
                color: #DCFCE7;
            }
        }

        /* Hide Streamlit Menu */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🥔 PotatoPulse</h1>
    <p>Potato Disease Detection & Analysis System</p>
</div>
""", unsafe_allow_html=True)

# Model Loading
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'potato_model.keras')
    if not os.path.exists(model_path):
        st.error("Model not found.")
        return None
    return tf.keras.models.load_model(model_path)

model = load_model()
CLASS_NAMES = ['Early Blight', 'Late Blight', 'Healthy']

def predict(image, model):
    image = image.convert('RGB')
    image = ImageOps.fit(image, (256, 256), Image.Resampling.LANCZOS)
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)
    return model.predict(img_array)

# Main Interface
col1, col2, col3 = st.columns([1, 8, 1])

with col2:
    uploaded_file = st.file_uploader("Drop your leaf image here to analyze", type=['png', 'jpg', 'jpeg'])

    if uploaded_file and model:
        image = Image.open(uploaded_file)
        
        # Grid layout for result
        result_col1, result_col2 = st.columns([1, 1])
        
        with result_col1:
            st.markdown("### 📸 Scan Preview")
            st.image(image, use_container_width=True, caption="Source Image")
            
        with result_col2:
            with st.spinner("Processing bio-markers..."):
                predictions = predict(image, model)
                score = tf.nn.softmax(predictions[0])
                class_index = np.argmax(predictions[0])
                label = CLASS_NAMES[class_index]
                confidence = np.max(predictions[0]) * 100
            
            # Dynamic color based on result
            color_code = "#22c55e" # Green
            if label != "Healthy":
                color_code = "#ef4444" # Red
            
            st.markdown(f"""
            <div class="prediction-card" style="border-top-color: {color_code};">
                <h3 style="margin:0; color: #64748B; font-size: 1rem; text-transform: uppercase; letter-spacing: 1px;">Diagnosis</h3>
                <h2 style="font-size: 2.5rem; font-weight: 800; margin: 0.5rem 0; color: {color_code};">{label}</h2>
                <div class="confidence-badge">
                    Accuracy: {confidence:.2f}%
                </div>
                <p style="margin-top: 1.5rem; color: #64748B; font-size: 0.95rem; line-height: 1.6;">
                    {
                        "The plant specimen shows no signs of disease. Maintain regular watering schedules." if label == "Healthy" else 
                        f"Detected signs of {label}. Immediate attention recommended to prevent crop spread."
                    }
                </p>
            </div>
            """, unsafe_allow_html=True)
