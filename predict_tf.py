import os
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
import logging

# Set up logging for better debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_model_path():
    """Finds the model path for local/cloud environments."""
    base_path = os.path.dirname(os.path.abspath(__file__))

    local_path = os.path.join(base_path, "weights", "converted_savedmodel", "model.savedmodel")
    cloud_path = "/mount/src/inspectorsally/weights/converted_savedmodel/model.savedmodel"

    for path in [local_path, cloud_path]:
        if os.path.exists(path):
            logger.info(f"Model found at: {path}")
            return path

    raise FileNotFoundError("Model directory not found.")

# Load model and labels at import
try:
    model_path = get_model_path()
    model = tf.keras.models.load_model(model_path, compile=False)

    labels_file = os.path.join(os.path.dirname(os.path.dirname(model_path)), "labels.txt")
    if not os.path.exists(labels_file):
        raise FileNotFoundError(f"Labels file not found at: {labels_file}")

    with open(labels_file, "r") as f:
        class_names = [line.strip().lower() for line in f.readlines()]
    logger.info(f"Classes loaded: {class_names}")

except Exception as e:
    logger.error(f"Initialization error: {e}")
    raise

def predict_image(img_pil):
    """Predicts the class of the image."""
    try:
        if img_pil.mode != "RGB":
            img_pil = img_pil.convert("RGB")

        size = (224, 224)
        image = ImageOps.fit(img_pil, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)

        normalized = (image_array.astype(np.float32) / 127.5) - 1
        input_data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        input_data[0] = normalized

        predictions = model.predict(input_data)
        index = int(np.argmax(predictions))
        confidence = round(float(predictions[0][index]) * 100, 2)

        if index >= len(class_names):
            raise ValueError("Invalid prediction index.")

        result = f"Prediction: {class_names[index].capitalize()} (Confidence: {confidence}%)"
        logger.info(result)
        return result

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise RuntimeError(f"Prediction failed: {e}")
