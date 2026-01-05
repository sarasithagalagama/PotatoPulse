# PotatoPulse | Advanced Agricultural Disease Detection

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://potatopulse.streamlit.app/)

PotatoPulse is an end-to-end deep learning solution designed to automate the detection of pathological states in potato leaf imagery. Built to address crop yield loss caused by *Phytophthora infestans* (Late Blight) and *Alternaria solani* (Early Blight), this system leverages a custom Deep Convolutional Neural Network (CNN) to achieve high-accuracy classification in real-time.

The project demonstrates a full machine learning lifecycle—from data pipeline construction and image preprocessing to model architecture design and deployment of a scalable inference engine.

## Technical Architecture

The core classification engine is a sequential CNN trained on the PlantVillage dataset, engineered for robustness and generalization:

## Dataset

*   **Source**: [PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)


*   **Data Pipeline**: Implemented automated data curation, including partitioning (Train/Val/Test), resizing, and normalization (recalling pixel values).
*   **Augmentation**: Integrated `RandomFlip` and `RandomRotation` layers directly into the model to mitigate overfitting and improve invariance to orientation.
*   **Model Architecture**: A 6-block Convolutional Network utilizing:
    *   32-64 filter feature extraction layers with ReLU activation.
    *   MaxPooling for spatial down-sampling and translational invariance.
    *   Flattening and Dense layers with Softmax activation for multi-class probability distribution.
## How the Model Was Trained

*   **Dataset Setup**: Loaded using `image_dataset_from_directory()` from the PlantVillage dataset.
*   **Preprocessing**:
    *   Applied `Resizing(256, 256)` for standardized input.
    *   Applied `Rescaling(1./255)` to normalize pixel intensity.
    *   **Augmentation**: Integrated `RandomFlip("horizontal_and_vertical")` and `RandomRotation(0.2)`.
*   **Architecture (Custom CNN)**:
    *   Used a **Sequential** model built from scratch.
    *   **6 Convolutional Layers**:
        *   32 filters (3x3) + MaxPooling2D (2x2)
        *   5 x [64 filters (3x3) + MaxPooling2D (2x2)]
        *   Activation: `relu`
    *   **Dense Layers**:
        *   Flatten layer
        *   Dense (64 units, activation='relu')
        *   Dense (3 units, activation='softmax')
*   **Training Config**:
    *   Optimizer: `adam`
    *   Loss: `SparseCategoricalCrossentropy`
    *   Epochs: 20
*   **Performance Achieved**:
    *   **Training Accuracy**: ~97.77%
    *   **Validation Accuracy**: ~97.40%
    *   **Test Accuracy**: ~94.53%

## Model Robustness & Generalization

This custom CNN demonstrates excellent generalization with minimal overfitting:

*   **Small Generalization Gap**: The difference between **Training Accuracy (97.77%)** and **Validation Accuracy (97.40%)** is roughly **0.37%**. This extremely narrow gap indicates the model is not memorizing the training data but actually learning meaningful disease patterns.
*   **Effective Data Augmentation**: The integrated random rotation and flipping layers force the model to learn features invariant to orientation, preventing it from overfitting to specific image alignments in the training set.
*   **Consistent Test Performance**: Achieving **~94.53%** on a completely unseen test set confirms the model's reliability for deployment in real agricultural settings.

## Technology Stack

*   **Deep Learning**: TensorFlow, Keras
*   **Computer Vision**: OpenCV, Pillow (PIL), NumPy
*   **Data Processing**: Pandas, Matplotlib (Performance Visualization)
*   **Deployment & UI**: Streamlit (Python-based reactive web framework)
*   **Version Control**: Git

This project serves as a practical implementation of computer vision techniques applied to precision agriculture, showcasing proficiency in tensor operations, neural network design, and production-grade software development.
