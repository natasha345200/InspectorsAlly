import os
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_model_path():
    base = os.path.dirname(os.path.abspath(__file__))
    p1 = os.path.join(base, "weights/converted_savedmodel/model.savedmodel")
    p2 = "/mount/src/inspectorsally/weights/converted_savedmodel/model.savedmodel"
    for p in [p1, p2]:
        if os.path.exists(p):
            logger.info(f"Model found at: {p}")
            return p
    raise FileNotFoundError("Model not found")

try:
    model_path = get_model_path()
    model = tf.keras.models.load_model(model_path, compile=False)

    # Correct label path, inside converted_savedmodel
    labels_file = os.path.join(os.path.dirname(model_path), "labels.txt")
    if not os.path.exists(labels_file):
        raise FileNotFoundError(f"Labels file not found at: {labels_file}")

    with open(labels_file) as f:
        class_names = [line.strip().lower() for line in f]
    logger.info(f"Loaded classes: {class_names}")

except Exception as e:
    logger.error(f"Initialization error: {e}")
    raise

def predict_image(img):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = ImageOps.fit(img, (224, 224), Image.Resampling.LANCZOS)
    data = np.expand_dims(np.asarray(img).astype(np.float32), 0)
    data = (data / 127.5) - 1

    preds = model.predict(data)
    idx = int(np.argmax(preds[0]))
    confidence = round(float(preds[0][idx]) * 100, 2)

    if idx >= len(class_names):
        raise RuntimeError("Prediction index out of range")

    return f"{class_names[idx].capitalize()} — {confidence}%"
