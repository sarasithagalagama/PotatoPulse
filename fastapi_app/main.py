import os
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image, ImageOps
import uvicorn
import io

app = FastAPI(title="PotatoPulse API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'potato_model.keras')

if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
else:
    model = None
    print(f"Error: Model not found at {MODEL_PATH}")

CLASS_NAMES = ['Early Blight', 'Late Blight', 'Healthy']

@app.get("/")
async def read_index():
    return FileResponse('fastapi_app/static/index.html')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    # Preprocessing
    image = image.convert('RGB')
    image = ImageOps.fit(image, (256, 256), Image.Resampling.LANCZOS)
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)
    
    # Prediction
    predictions = model.predict(img_array)
    prediction_scores = predictions[0]
    
    # Get top class
    class_index = np.argmax(prediction_scores)
    class_name = CLASS_NAMES[class_index]
    confidence = float(np.max(prediction_scores) * 100)
    
    # Create confidence distribution dict
    confidence_distribution = {
        class_name: float(score * 100) 
        for class_name, score in zip(CLASS_NAMES, prediction_scores)
    }
    
    return {
        "class": class_name,
        "confidence": confidence,
        "distribution": confidence_distribution
    }

# Mount static files (CSS, JS, Images)
app.mount("/static", StaticFiles(directory="fastapi_app/static"), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
