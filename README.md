# PotatoPulse | Advanced Agricultural Disease Detection

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://potatopulse.streamlit.app/)

PotatoPulse is an end-to-end deep learning solution designed to automate the detection of pathological states in potato leaf imagery. Built to address crop yield loss caused by *Phytophthora infestans* (Late Blight) and *Alternaria solani* (Early Blight), this system leverages a custom Deep Convolutional Neural Network (CNN) to achieve high-accuracy classification in real-time.

The project demonstrates a full machine learning lifecycle—from data pipeline construction and image preprocessing to model architecture design and deployment of a scalable inference engine.

## Technical Architecture

The core classification engine is a sequential CNN trained on the PlantVillage dataset, engineered for robustness and generalization:

*   **Data Pipeline**: Implemented automated data curation, including partitioning (Train/Val/Test), resizing, and normalization (recalling pixel values).
*   **Augmentation**: Integrated `RandomFlip` and `RandomRotation` layers directly into the model to mitigate overfitting and improve invariance to orientation.
*   **Model Architecture**: A 6-block Convolutional Network utilizing:
    *   32-64 filter feature extraction layers with ReLU activation.
    *   MaxPooling for spatial down-sampling and translational invariance.
    *   Flattening and Dense layers with Softmax activation for multi-class probability distribution.
*   **Optimization**: Compiled with the Adam optimizer and Sparse Categorical Crossentropy loss function for efficient convergence.

## Technology Stack

*   **Deep Learning**: TensorFlow, Keras
*   **Computer Vision**: OpenCV, Pillow (PIL), NumPy
*   **Data Processing**: Pandas, Matplotlib (Performance Visualization)
*   **Deployment & UI**: Streamlit (Python-based reactive web framework)
*   **Version Control**: Git

This project serves as a practical implementation of computer vision techniques applied to precision agriculture, showcasing proficiency in tensor operations, neural network design, and production-grade software development.
