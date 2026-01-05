import tensorflow as tf
import os

# Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'model', 'potato_model.keras')
tflite_path = os.path.join(current_dir, 'model', 'potato_model.tflite')

print(f"Loading model from {model_path}...")
model = tf.keras.models.load_model(model_path)

print("Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

print(f"Saving TFLite model to {tflite_path}...")
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)

print("Conversion complete!")
