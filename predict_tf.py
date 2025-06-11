import os
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
import logging

# Set up logging for better debugging on Streamlit Cloud
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_model_path():
    """Resolve model path for local and cloud environments."""
    base_path = os.path.dirname(os.path.abspath(__file__))

    # Updated to point directly to the model.savedmodel folder
    model_dir = os.path.join(base_path, "weights", "converted_savedmodel", "model.savedmodel")
    cloud_model_dir = "/mount/src/inspectorsally/weights/converted_savedmodel/model.savedmodel"

    for path in [model_dir, cloud_model_dir]:
        if os.path.exists(path) and os.path.isdir(path):
            logger.info(f"Found SavedModel at: {path}")
            return path

    logger.error(f"Model not found in expected directories.")
    raise FileNotFoundError("Model directory not found in expected locations.")

# Load model and labels
try:
    model_path = get_model_path()
    logger.info(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)

    # Try labels.txt from both local and cloud paths
    labels_path = os.path.join(os.path.dirname(os.path.dirname(model_path)), "labels.txt")
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
