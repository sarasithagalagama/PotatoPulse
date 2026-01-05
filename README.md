# 🥔 PotatoPulse - Disease Classification

PotatoPulse is an advanced deep learning application designed to detect diseases in potato plants from leaf images. It identifies **Early Blight**, **Late Blight**, and **Healthy** conditions with high accuracy using a Convolutional Neural Network (CNN).

## 🚀 Features

- **Disease Detection**: Classifies images into 'Early Blight', 'Late Blight', or 'Healthy'.
- **Deep Learning**: Powered by a custom trained TensorFlow/Keras model.
- **Instant Analysis**: Get real-time predictions with confidence scores.
- **User-Friendly Interface**: Built with [Streamlit](https://streamlit.io/) for a smooth experience.

## 🛠️ Tech Stack

- **Python**: Core programming language.
- **TensorFlow/Keras**: For loading and running the classification model.
- **Streamlit**: For the web interface.
- **Pillow**: For image processing.

## 📂 Project Structure

```
.
├── app/
│   ├── main.py          # The main Streamlit application
│   └── requirements.txt # Python dependencies
├── model/
│   └── potato_model.keras # Trained model file
├── notebook/
│   └── Potato_Disease_training.ipynb # Training notebook
└── README.md
```

## 🏃‍♂️ How to Run Locally

1. **Clone the repository**:

   ```bash
   git clone <your-repo-url>
   cd PotatoPulse
   ```

2. **Install dependencies**:

   ```bash
   pip install -r app/requirements.txt
   ```

3. **Run the app**:
   ```bash
   streamlit run app/main.py
   ```

## ☁️ Deployment

This app is ready to be deployed on **Streamlit Cloud**:

1. Push this repository to GitHub.
2. Log in to [share.streamlit.io](https://share.streamlit.io/).
3. Click "New App".
4. Select your repository, branch, and set the **Main file path** to `app/main.py`.
5. Click **Deploy**!

---

_Created with ❤️ by PotatoPulse Team_
