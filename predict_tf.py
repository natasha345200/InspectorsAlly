from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os
import tensorflow as tf

def load_model_with_retry(model_path):
    try:
        # First try loading as SavedModel
        model = load_model(model_path, compile=False)
        return model
    except:
        try:
            # If SavedModel fails, try loading as .h5 format
            model = load_model(model_path + '.h5', compile=False)
            return model
        except Exception as e:
            raise ValueError(f"Failed to load model from {model_path}. Error: {str(e)}")

# Load model - updated path handling
model_dir = os.path.join("weights", "converted_savedmodel")
model_path = os.path.join(model_dir, "saved_model.pb")  # or model.h5 for h5 format

# Verify model exists
if not os.path.exists(model_dir):
    raise FileNotFoundError(f"Model directory not found at: {model_dir}")

model = load_model_with_retry(model_dir)  # For SavedModel, pass directory not file

# Load labels
labels_path = os.path.join(model_dir, "labels.txt")
if not os.path.exists(labels_path):
    raise FileNotFoundError(f"Labels file not found at: {labels_path}")

with open(labels_path, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

def predict_image(img_pil):
    size = (224, 224)
    image = ImageOps.fit(img_pil, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)

    # Normalize
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    prediction = model.predict(data)
    index = int(np.argmax(prediction))
    class_name = class_names[index].strip().lower()
    confidence_score = round(float(prediction[0][index]) * 100, 2)

    if class_name in ["good/perfect", "perfect", "good"]:
        return f"✅ This is a **Good** product. (Confidence: {confidence_score}%)"
    else:
        return f"⚠️ Detected an **Abnormal** product. (Confidence: {confidence_score}%)"
