from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os
import tensorflow as tf

def get_model_path():
    """Handle path resolution for both local and cloud environments"""
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_path, "weights", "converted_savedmodel")
    
    # Check for SavedModel format
    if os.path.exists(os.path.join(model_dir, "saved_model.pb")):
        return model_dir
    
    # Check for H5 format
    h5_path = os.path.join(model_dir, "model.h5")
    if os.path.exists(h5_path):
        return h5_path
    
    # Check alternative locations (common in Streamlit Cloud)
    cloud_path = os.path.join("/mount/src/inspectorsally", "weights", "converted_savedmodel")
    if os.path.exists(cloud_path):
        return cloud_path
    
    raise FileNotFoundError(f"Could not find model in:\n1. {model_dir}\n2. {cloud_path}")

# Load model
try:
    model_path = get_model_path()
    model = load_model(model_path, compile=False)
    
    # Load labels
    labels_path = os.path.join(os.path.dirname(model_path), "labels.txt")
    with open(labels_path, "r") as f:
        class_names = [line.strip() for line in f.readlines()]
except Exception as e:
    raise RuntimeError(f"Model loading failed: {str(e)}")

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
