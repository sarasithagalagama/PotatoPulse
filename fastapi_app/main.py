import os
import numpy as np
import tflite_runtime.interpreter as tflite
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image, ImageOps
import uvicorn
import io

app = FastAPI(title="PotatoPulse X1 API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load TFLite Model
# Updated to use tflite model for deployment
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'potato_model.tflite')

interpreter = None
input_details = None
output_details = None

if os.path.exists(MODEL_PATH):
    try:
        interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print("TFLite Model loaded successfully.")
    except Exception as e:
        print(f"Error loading TFLite model: {e}")
else:
    print(f"Error: Model not found at {MODEL_PATH}")

CLASS_NAMES = ['Early Blight', 'Late Blight', 'Healthy']

@app.get("/")
async def read_index():
    return FileResponse('fastapi_app/static/index.html')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if interpreter is None:
        raise HTTPException(status_code=500, detail="Model logic not initialized")

    # Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    # Preprocessing
    image = image.convert('RGB')
    image = ImageOps.fit(image, (256, 256), Image.Resampling.LANCZOS)
    img_array = np.array(image, dtype=np.float32)
    
    # Normalize if previously trained model expected it (Usually /255.0)
    # The original keras model in notebook had a Rescaling(1./255) layer.
    # Keras models INCLUDE that layer in the TFLite conversion usually.
    # But usually TFLite input expects raw float32 input.
    # Let's check expectations. Safest is to pass [0, 255] float32 as before if the Rescaling layer was part of the model map.
    
    img_array = np.expand_dims(img_array, axis=0)
    
    # Prediction Main
    input_index = input_details[0]['index']
    output_index = output_details[0]['index']
    
    interpreter.set_tensor(input_index, img_array)
    interpreter.invoke()
    
    predictions = interpreter.get_tensor(output_index)
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
