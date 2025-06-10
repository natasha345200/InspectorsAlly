import os
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow.keras.models import load_model
import logging

# Set up logging for better debugging on Streamlit Cloud
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_model_path():
    """Resolve model path for local and cloud environments."""
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_path, "weights", "converted_savedmodel")
    h5_path = os.path.join(base_path, "weights", "model.h5")
    cloud_base = "/mount/src/inspectorsally"  # Adjust if repo name differs
    cloud_model_dir = os.path.join(cloud_base, "weights", "converted_savedmodel")
    cloud_h5_path = os.path.join(cloud_base, "weights", "model.h5")

    # Check possible model locations
    for path in [model_dir, h5_path, cloud_model_dir, cloud_h5_path]:
        if os.path.exists(path):
            if path.endswith(".h5") or os.path.exists(os.path.join(path, "saved_model.pb")):
                logger.info(f"Found model at: {path}")
                return path

    # List directory contents for debugging
    logger.error(f"Base path contents: {os.listdir(base_path)}")
    if os.path.exists(os.path.join(base_path, "weights")):
        logger.error(f"Weights dir contents: {os.listdir(os.path.join(base_path, 'weights'))}")
    raise FileNotFoundError(
        f"Model not found in:\n1. {model_dir}\n2. {h5_path}\n3. {cloud_model_dir}\n4. {cloud_h5_path}"
    )

# Load model and labels
try:
    model_path = get_model_path()
    logger.info(f"Loading model from: {model_path}")
    model = load_model(model_path, compile=False)

    # Load labels
    labels_path = os.path.join(os.path.dirname(model_path), "labels.txt")
    if not os.path.exists(labels_path):
        logger.error(f"Labels file not found at: {labels_path}")
        raise FileNotFoundError(f"Labels file not found at: {labels_path}")
    with open(labels_path, "r") as f:
        class_names = [line.strip().lower() for line in f.readlines()]
    logger.info(f"Loaded {len(class_names)} class names: {class_names}")
except Exception as e:
    logger.error(f"Model loading failed: {str(e)}")
    raise RuntimeError(f"Model loading failed: {str(e)}")

def predict_image(img_pil):
    """Predict the class of an input image."""
    try:
        # Ensure image is RGB
        if img_pil.mode != "RGB":
            logger.info(f"Converting image from {img_pil.mode} to RGB")
            img_pil = img_pil.convert("RGB")

        # Resize and preprocess image
        size = (224, 224)
        image = ImageOps.fit(img_pil, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)

        # Normalize (confirm this matches your model's training preprocessing)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array

        # Make prediction
        prediction = model.predict(data, verbose=0)
        index = int(np.argmax(prediction))
        if index >= len(class_names):
            logger.error(f"Prediction index {index} out of bounds for class_names: {class_names}")
            raise ValueError(f"Invalid prediction index: {index}")
        
        class_name = class_names[index]
        confidence_score = round(float(prediction[0][index]) * 100, 2)
        logger.info(f"Prediction: {class_name}, Confidence: {confidence_score}%")

        # Map class names to good/abnormal
        good_classes = {"good", "perfect", "good/perfect"}
        if class_name in good_classes:
            return f"✅ This is a **Good** product. (Confidence: {confidence_score}%)"
        return f"⚠️ Detected an **Abnormal** product. (Confidence: {confidence_score}%)"
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise RuntimeError(f"Prediction failed: {str(e)}")
