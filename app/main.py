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

    /* 3. The Main Card Container - Refined */
    .main-card {
        background-color: #121212;
        border: 1px solid #2a2a2a;
        border-radius: 20px;
        padding: 40px;
        max-width: 800px;
        margin: 0 auto; 
        box-shadow: 0 20px 50px rgba(0,0,0,0.5);
        text-align: center;
        margin-bottom: 20px;
    }

    /* 4. Badge Styling */
    .badge {
        background-color: rgba(46, 204, 113, 0.1);
        color: #2ecc71;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 1.5px;
        display: inline-block;
        margin-bottom: 25px;
        border: 1px solid rgba(46, 204, 113, 0.2);
        box-shadow: 0 0 15px rgba(46, 204, 113, 0.1);
    }

    /* 5. Typography */
    .title-text {
        color: #ffffff;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 10px;
        letter-spacing: -1px;
    }
    
    .subtitle-text {
        color: #888888;
        font-size: 1rem;
        margin-bottom: 40px;
        font-weight: 300;
    }

    /* 6. File Uploader Styling - Integrated Look */
    [data-testid='stFileUploader'] {
        width: 100%;
        max-width: 600px;
        margin: 0 auto;
    }

    [data-testid='stFileUploader'] section {
        background-color: #0f0f0f !important;
        border: 2px dashed #333 !important;
        border-radius: 12px !important;
        padding: 40px !important; 
        transition: all 0.3s ease;
    }
    
    [data-testid='stFileUploader'] section:hover {
        border-color: #2ecc71 !important;
        background-color: #151515 !important;
    }
    
    [data-testid='stFileUploader'] .uploadedFileName {
        color: #fff !important;
    }

    /* 7. Action Button Styling - Cyberpunk/Tech feel */
    .stButton {
        display: flex;
        justify-content: center;
        margin-top: 25px;
    }

    .stButton > button {
        background: linear-gradient(90deg, #2ecc71 0%, #27ae60 100%);
        color: #000;
        border: none;
        padding: 18px 40px;
        font-size: 1rem;
        font-weight: 800;
        border-radius: 8px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        text-transform: uppercase;
        letter-spacing: 2px;
        width: 100%;
        max-width: 600px;
        box-shadow: 0 4px 15px rgba(46, 204, 113, 0.2);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(46, 204, 113, 0.4);
        letter-spacing: 3px;
        color: #fff;
    }
    
    .stButton > button:active {
        transform: translateY(1px);
    }

    /* 8. Result Card Styling */
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
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

col_spacer1, col_main, col_spacer2 = st.columns([1, 4, 1])

with col_main:
    # 1. Header Card
    st.markdown("""
        <div class="main-card">
            <div class="badge">● POTATOPULSE X1</div>
            <div class="title-text">Potato Disease Analyzer</div>
            <div class="subtitle-text">Advanced AI Diagnostics for Solanum tuberosum</div>
        </div>
    """, unsafe_allow_html=True)
    
    # 2. Upload Zone
    file = st.file_uploader("", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    
    # 3. Action Button
    analyze_btn = st.button("INITIATE DIAGNOSTIC PROTOCOL")

    # 4. Results Section
    if file is not None and analyze_btn:
        if model is None:
            st.error("System Error: Neural Network Model Not Found.")
        else:
            image = Image.open(file)
            
            with st.spinner("Analyzing Bio-Signature..."):
                class_name, confidence = predict_image(image, model)
            
            status_color = "#2ecc71" if class_name == "Healthy" else "#ff4757"
            bg_gradient = f"linear-gradient(135deg, {status_color}22 0%, {status_color}05 100%)"
            
            # Complex HTML Result Card
            st.markdown(f"""
            <div style="
                background: {bg_gradient}; 
                padding: 40px; 
                border-radius: 20px; 
                border: 1px solid {status_color}33; 
                margin-top: 30px; 
                animation: slideUp 0.6s ease-out;
                position: relative;
                overflow: hidden;
            ">
                <!-- Decorative Glow -->
                <div style="
                    position: absolute;
                    top: -50px;
                    right: -50px;
                    width: 150px;
                    height: 150px;
                    background: {status_color};
                    filter: blur(80px);
                    opacity: 0.2;
                "></div>

                <div style="display: flex; align-items: start; justify-content: space-between; flex-wrap: wrap; gap: 30px; position: relative; z-index: 1;">
                    
                    <!-- Left: Diagnosis -->
                    <div style="flex: 2; min-width: 250px;">
                        <div style="color: {status_color}; font-weight: 700; font-size: 0.9rem; letter-spacing: 2px; margin-bottom: 15px; opacity: 0.8;">• DIAGNOSIS REPORT</div>
                        <div style="
                            font-size: 3.5rem; 
                            font-weight: 800; 
                            line-height: 1; 
                            margin-bottom: 20px; 
                            color: #fff;
                            text-shadow: 0 0 30px {status_color}44;
                        ">{class_name}</div>
                        
                        <div style="display: flex; gap: 15px;">
                            <div style="background: rgba(0,0,0,0.3); padding: 15px 25px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1);">
                                <div style="color: #888; font-size: 0.7rem; letter-spacing: 1px; margin-bottom: 5px;">CONFIDENCE SCORE</div>
                                <div style="color: #fff; font-weight: 700; font-size: 1.5rem;">{confidence:.1f}%</div>
                            </div>
                            <div style="background: rgba(0,0,0,0.3); padding: 15px 25px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1);">
                                <div style="color: #888; font-size: 0.7rem; letter-spacing: 1px; margin-bottom: 5px;">MODEL VERSION</div>
                                <div style="color: #fff; font-weight: 700; font-size: 1.5rem;">CNN-X1</div>
                            </div>
                        </div>
                    </div>

                    <!-- Right: Info/Next Steps -->
                    <div style="flex: 1; min-width: 200px; background: rgba(0,0,0,0.2); padding: 20px; border-radius: 15px; border-left: 3px solid {status_color};">
                        <div style="color: #fff; font-weight: 600; margin-bottom: 10px;">Analysis Notes</div>
                        <div style="color: #aaa; font-size: 0.9rem; line-height: 1.6;">
                            { "Specimen appears free of pathogens." if class_name == "Healthy" else "Pathogen detected. Immediate quarantine and treatment recommended to prevent spread." }
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show image below neatly
            st.markdown("<br>", unsafe_allow_html=True)
            st.image(image, caption="Uploaded Specimen Source", width=300)
