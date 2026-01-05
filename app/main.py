import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os

# Set page configuration
st.set_page_config(
    page_title="PotatoPulse",
    page_icon="🥔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Dark Mode + Lime Green Theme
st.markdown("""
    <style>
        /* Import Google Font */
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Space Grotesk', sans-serif;
            color: #e5e5e5;
        }

        /* Main Background */
        .stApp {
            background-color: #0a0a0a;
            background-image: 
                radial-gradient(circle at 50% 0%, #1a2e05 0%, transparent 60%),
                radial-gradient(circle at 100% 100%, #1a2e05 0%, transparent 50%);
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #111111;
            border-right: 1px solid #222;
        }

        /* Custom Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #0a0a0a; 
        }
        ::-webkit-scrollbar-thumb {
            background: #333; 
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #bef264; 
        }

        /* Header Styling */
        .main-header {
            background: #171717;
            border: 1px solid #333;
            padding: 2.5rem;
            border-radius: 24px;
            margin-bottom: 2rem;
            text-align: center;
            position: relative;
            overflow: hidden;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5), 0 8px 10px -6px rgba(0, 0, 0, 1);
        }
        
        .main-header::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 4px;
            background: linear-gradient(90deg, #bef264, #84cc16);
        }
        
        .main-header h1 {
            font-weight: 700;
            margin-bottom: 0.5rem;
            font-size: 3rem;
            color: #ffffff;
            letter-spacing: -1px;
        }
        
        .main-header p {
            font-size: 1.1rem;
            color: #888;
            font-weight: 400;
        }

        /* Info/Success/Warning/Error Boxes (Dark Theme) */
        .info-box {
            background-color: #1e293b;
            border-left: 4px solid #3b82f6;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        .success-box {
            background-color: #14532d; /* Dark Green */
            border: 1px solid #22c55e;
            border-left: 4px solid #22c55e;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        .error-box {
            background-color: #450a0a; /* Dark Red */
            border: 1px solid #ef4444;
            border-left: 4px solid #ef4444;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }

        /* Card Styling */
        .card {
            background-color: #171717;
            border: 1px solid #262626;
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
            margin-bottom: 1rem;
            transition: transform 0.2s, border-color 0.2s;
        }
        .card:hover {
            border-color: #444;
            transform: translateY(-2px);
        }

        /* Metric Card */
        .metric-card {
            background: #171717;
            border-radius: 16px;
            padding: 1.5rem;
            text-align: center;
            border: 1px solid #333;
            border-top: 4px solid #bef264;
        }
        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: #bef264;
        }
        .metric-label {
            font-size: 0.9rem;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        /* Input/Upload Area */
        [data-testid="stFileUploader"] {
            background-color: #171717;
            border: 1px dashed #444;
            border-radius: 16px;
            padding: 2rem;
        }
        [data-testid="stFileUploader"]:hover {
            border-color: #bef264;
        }

        /* Buttons */
        .stButton>button {
            background: #bef264;
            color: #000000;
            border: none;
            border-radius: 12px;
            padding: 0.75rem 2rem;
            font-weight: 700;
            font-size: 1rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            width: 100%;
        }
        .stButton>button:hover {
            background: #d9f99d;
            box-shadow: 0 0 20px rgba(190, 242, 100, 0.4);
            transform: translateY(-2px);
            color: black;
        }

        /* Result Styles */
        .result-header {
            font-size: 1.2rem;
            font-weight: 600;
            color: #888;
            margin-bottom: 1rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .result-label {
            font-size: 2.5rem;
            font-weight: 800;
            margin: 1rem 0;
        }
        .healthy { color: #bef264; }
        .diseased { color: #ef4444; }

        /* Progress Bars */
        .stProgress > div > div > div > div {
            background-color: #bef264;
        }
        .stProgress {
            background-color: #222;
        }

        /* Chat Widget Floating Action Button */
        .chat-widget {
            position: fixed;
            bottom: 30px;
            right: 30px;
            background: #bef264;
            color: black;
            width: 60px;
            height: 60px;
            border-radius: 30px;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
            cursor: pointer;
            z-index: 999;
            transition: all 0.3s;
        }
        .chat-widget:hover {
            transform: scale(1.1);
        }

        /* Hide default Streamlit Styling */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
    </style>
""", unsafe_allow_html=True)

# Floating Chat Widget (Visual Only)
st.markdown("""
<div class="chat-widget" title="Agri-Assistant">
    <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path></svg>
</div>
""", unsafe_allow_html=True)

# Sidebar with Restored Content + Dark Theme
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/potato.png", width=80)
    st.markdown("<h2 style='color: #bef264;'>PotatoPulse</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.subheader("📋 Navigation")
    page = st.radio("", ["🏠 Home", "ℹ️ About", "📊 Model Info"], label_visibility="collapsed")
    
    st.markdown("---")
    st.subheader("🎯 Quick Stats")
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <div style="color: #888; font-size: 0.9rem;">Model Accuracy</div>
        <div style="font-size: 2rem; font-weight: 700; color: #bef264;">97.77%</div>
        <span style="background: rgba(190, 242, 100, 0.1); color: #bef264; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem;">↑ Training</span>
    </div>
    
    <div style="margin-bottom: 20px;">
        <div style="color: #888; font-size: 0.9rem;">Validation Acc</div>
        <div style="font-size: 2rem; font-weight: 700; color: #bef264;">97.40%</div>
        <span style="background: rgba(190, 242, 100, 0.1); color: #bef264; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem;">↑ 0.37% gap</span>
    </div>

    <div style="margin-bottom: 20px;">
        <div style="color: #888; font-size: 0.9rem;">Test Accuracy</div>
        <div style="font-size: 2rem; font-weight: 700; color: #bef264;">94.53%</div>
        <span style="background: rgba(190, 242, 100, 0.1); color: #bef264; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem;">↑ Robust</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("🔬 Detectable Diseases")
    st.markdown("""
    - 🍂 **Early Blight**
    - 🦠 **Late Blight**
    - ✅ **Healthy**
    """)
    
    st.markdown("---")
    st.markdown("Developed by [Sarasitha Galagama](https://sarasitha.me/)")

# Model Loading
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'potato_model.keras')
    if not os.path.exists(model_path):
        st.error("❌ Model not found. Please ensure the model file exists.")
        return None
    return tf.keras.models.load_model(model_path)

model = load_model()
CLASS_NAMES = ['Early Blight', 'Late Blight', 'Healthy']

# Disease Information dictionary
DISEASE_INFO = {
    'Early Blight': {
        'description': 'Caused by *Alternaria solani*, Early Blight affects older leaves first, creating concentric ring patterns.',
        'symptoms': ['Dark brown spots with concentric rings', 'Yellow halo around spots', 'Leaf yellowing and dropping'],
        'treatment': ['Remove infected leaves', 'Apply fungicide', 'Improve air circulation', 'Avoid overhead watering'],
        'severity': 'Moderate'
    },
    'Late Blight': {
        'description': 'Caused by *Phytophthora infestans*, Late Blight is highly destructive and can devastate entire crops rapidly.',
        'symptoms': ['Water-soaked lesions on leaves', 'White fuzzy growth on leaf undersides', 'Rapid plant death'],
        'treatment': ['Immediate fungicide application', 'Remove and destroy infected plants', 'Improve drainage', 'Use resistant varieties'],
        'severity': 'Severe'
    },
    'Healthy': {
        'description': 'The plant shows no signs of disease and appears to be in good health.',
        'symptoms': ['Vibrant green leaves', 'No discoloration or spots', 'Normal growth pattern'],
        'treatment': ['Continue regular care', 'Monitor for changes', 'Maintain proper watering', 'Ensure adequate nutrition'],
        'severity': 'None'
    }
}

def predict(image, model):
    image = image.convert('RGB')
    image = ImageOps.fit(image, (256, 256), Image.Resampling.LANCZOS)
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)
    img_array = img_array / 255.0  # Normalize
    return model.predict(img_array)

# Main Content Logic
if page == "🏠 Home":
    # Header
    st.markdown("""
    <div class="main-header">
        <div style="background: rgba(190, 242, 100, 0.1); color: #bef264; display: inline-block; padding: 4px 12px; border-radius: 99px; font-size: 0.8rem; font-weight: 600; letter-spacing: 1px; margin-bottom: 1rem; border: 1px solid rgba(190, 242, 100, 0.2);">● SYSTEM ONLINE</div>
        <h1>PotatoPulse</h1>
        <p>Potato Disease Detection & Analysis System</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Instructions
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class="info-box">
            <h4 style="margin-top: 0; color: #3b82f6;">📸 How to Use</h4>
            <ol style="margin-bottom: 0;">
                <li>Upload a clear image of a potato leaf</li>
                <li>Wait for the AI model to analyze the image</li>
                <li>Review the diagnosis and recommendations</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    # File Upload Section
    st.markdown("### 📤 Upload Leaf Image")
    uploaded_file = st.file_uploader(
        "Choose an image file (PNG, JPG, JPEG)",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a clear image of a potato leaf for disease detection"
    )
    
    if uploaded_file and model:
        # Create columns for layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📸 Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True, caption=f"Uploaded: {uploaded_file.name}")
            
            # Image info
            st.markdown(f"""
            <div class="card">
                <strong style="color: #bef264">Image Details:</strong><br>
                📏 Size: {image.size[0]} x {image.size[1]} pixels<br>
                🎨 Format: {image.format}<br>
                📦 File Size: {uploaded_file.size / 1024:.2f} KB
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### 🔬 Analysis Results")
            
            # Prediction
            with st.spinner("🔄 Analyzing image... Please wait"):
                predictions = predict(image, model)
                class_index = np.argmax(predictions[0])
                label = CLASS_NAMES[class_index]
                confidence = predictions[0][class_index] * 100
                
                # Get all confidences
                confidences = {CLASS_NAMES[i]: predictions[0][i] * 100 for i in range(len(CLASS_NAMES))}
            
            # Display result
            status_class = "healthy" if label == "Healthy" else "diseased"
            box_class = "success-box" if label == "Healthy" else "error-box"
            
            st.markdown(f"""
            <div class="{box_class}">
                <div class="result-header" style="color: white opacity: 0.8">Diagnosis</div>
                <div class="result-label {status_class}">{label}</div>
                <div style="font-size: 1.2rem; font-weight: 600;">
                    Confidence: {confidence:.2f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence Progress Bar
            st.markdown("**Confidence Score**")
            st.progress(float(confidence / 100))
            
            # All Predictions
            st.markdown("**All Predictions:**")
            for disease, conf in sorted(confidences.items(), key=lambda x: x[1], reverse=True):
                st.markdown(f"- **{disease}**: <span style='color: #bef264'>{conf:.2f}%</span>", unsafe_allow_html=True)
        
        # Detailed Information
        st.markdown("---")
        st.markdown("### 📋 Detailed Analysis")
        
        info = DISEASE_INFO[label]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="card">
                <h4 style="color: #bef264">📖 Description</h4>
                <p>{info['description']}</p>
                <h4 style="color: #bef264">⚠️ Severity Level</h4>
                <p><strong>{info['severity']}</strong></p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="card">
                <h4 style="color: #bef264">🔍 Symptoms</h4>
                <ul>
                    {''.join([f'<li>{symptom}</li>' for symptom in info['symptoms']])}
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="card">
            <h4 style="color: #bef264">💊 Recommended Treatment</h4>
            <ul>
                {''.join([f'<li>{treatment}</li>' for treatment in info['treatment']])}
            </ul>
        </div>
        """, unsafe_allow_html=True)

elif page == "ℹ️ About":
    st.markdown("""
    <div class="main-header">
        <h1>ℹ️ About PotatoPulse</h1>
        <p>AI-Powered Agricultural Disease Detection</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ## 🎯 Project Overview
    
    PotatoPulse is an end-to-end deep learning solution designed to automate the detection of pathological states 
    in potato leaf imagery. Built to address crop yield loss caused by **Phytophthora infestans** (Late Blight) 
    and **Alternaria solani** (Early Blight), this system leverages a custom Deep Convolutional Neural Network (CNN) 
    to achieve high-accuracy classification in real-time.
    
    ## 🔬 Technical Approach
    
    The project demonstrates a full machine learning lifecycle:
    - **Data Pipeline Construction**: Automated data curation and preprocessing
    - **Image Preprocessing**: Resizing, normalization, and augmentation
    - **Model Architecture Design**: Custom 6-layer CNN with 32-64 filters
    - **Deployment**: Scalable inference engine using Streamlit
    
    ## 🎓 Dataset
    
    - **Source**: PlantVillage Dataset
    - **Classes**: Early Blight, Late Blight, Healthy
    - **Preprocessing**: Resizing (256x256), Rescaling (1/255)
    - **Augmentation**: Random flip and rotation
    
    ## 🚀 Technology Stack
    
    - **Deep Learning**: TensorFlow, Keras
    - **Computer Vision**: OpenCV, Pillow (PIL), NumPy
    - **Data Processing**: Pandas, Matplotlib
    - **Deployment**: Streamlit
    - **Version Control**: Git
    
    ## 👨‍💻 Developer
    
    This project serves as a practical implementation of computer vision techniques applied to precision agriculture, 
    showcasing proficiency in tensor operations, neural network design, and production-grade software development.
    """)

elif page == "📊 Model Info":
    st.markdown("""
    <div class="main-header">
        <h1>📊 Model Information</h1>
        <p>Deep Learning Architecture & Performance</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model Architecture
    st.markdown("## 🏗️ Model Architecture")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="card">
            <h4 style="color: #bef264">🧠 Network Structure</h4>
            <ul>
                <li><strong>Type:</strong> Sequential CNN</li>
                <li><strong>Input Shape:</strong> 256 x 256 x 3</li>
                <li><strong>Convolutional Layers:</strong> 6</li>
                <li><strong>Filters:</strong> 32-64</li>
                <li><strong>Activation:</strong> ReLU</li>
                <li><strong>Pooling:</strong> MaxPooling2D (2x2)</li>
                <li><strong>Output:</strong> Softmax (3 classes)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <h4 style="color: #bef264">⚙️ Training Configuration</h4>
            <ul>
                <li><strong>Optimizer:</strong> Adam</li>
                <li><strong>Loss Function:</strong> Sparse Categorical Crossentropy</li>
                <li><strong>Epochs:</strong> 20</li>
                <li><strong>Batch Size:</strong> 32</li>
                <li><strong>Data Augmentation:</strong> Yes</li>
                <li><strong>Validation Split:</strong> 20%</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Performance Metrics
    st.markdown("## 📈 Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Training Accuracy</div>
            <div class="metric-value">97.77%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card" style="border-top-color: #22c55e;">
            <div class="metric-label">Validation Accuracy</div>
            <div class="metric-value" style="color: #22c55e;">97.40%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card" style="border-top-color: #fbbf24;">
            <div class="metric-label">Test Accuracy</div>
            <div class="metric-value" style="color: #fbbf24;">94.53%</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Model Robustness
    st.markdown("## 🛡️ Model Robustness & Generalization")
    
    st.markdown("""
    <div class="success-box">
        <h4 style="color: white;">✅ Excellent Generalization</h4>
        <p style="color: #e5e5e5;">
            The model demonstrates excellent generalization with minimal overfitting:
        </p>
        <ul style="color: #e5e5e5;">
            <li><strong>Small Generalization Gap:</strong> Only 0.37% difference between training and validation accuracy</li>
            <li><strong>Effective Data Augmentation:</strong> Random rotation and flipping prevent overfitting</li>
            <li><strong>Consistent Test Performance:</strong> 94.53% accuracy on unseen data confirms reliability</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Layer Details
    st.markdown("## 🔍 Detailed Layer Architecture")
    
    st.markdown("""
    ```
    Model: "sequential"
    _________________________________________________________________
    Layer (type)                Output Shape              Param #   
    =================================================================
    resizing (Resizing)         (None, 256, 256, 3)       0         
    rescaling (Rescaling)       (None, 256, 256, 3)       0         
    random_flip (RandomFlip)    (None, 256, 256, 3)       0         
    random_rotation (RandomRot) (None, 256, 256, 3)       0         
    conv2d (Conv2D)             (None, 254, 254, 32)      896       
    max_pooling2d (MaxPooling2D)(None, 127, 127, 32)      0         
    conv2d_1 (Conv2D)           (None, 125, 125, 64)      18496     
    max_pooling2d_1 (MaxPooling)(None, 62, 62, 64)        0         
    conv2d_2 (Conv2D)           (None, 60, 60, 64)        36928     
    max_pooling2d_2 (MaxPooling)(None, 30, 30, 64)        0         
    conv2d_3 (Conv2D)           (None, 28, 28, 64)        36928     
    max_pooling2d_3 (MaxPooling)(None, 14, 14, 64)        0         
    conv2d_4 (Conv2D)           (None, 12, 12, 64)        36928     
    max_pooling2d_4 (MaxPooling)(None, 6, 6, 64)          0         
    conv2d_5 (Conv2D)           (None, 4, 4, 64)          36928     
    max_pooling2d_5 (MaxPooling)(None, 2, 2, 64)          0         
    flatten (Flatten)           (None, 256)               0         
    dense (Dense)               (None, 64)                16448     
    dense_1 (Dense)             (None, 3)                 195       
    =================================================================
    Total params: 183,747
    Trainable params: 183,747
    Non-trainable params: 0
    _________________________________________________________________
    ```
    """)
