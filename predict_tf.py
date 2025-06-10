from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os
import tensorflow as tf

# Load model
model_path = os.path.join("weights", "converted_savedmodel", "model.savedmodel")
model = load_model(model_path, compile=False)

# Load labels
with open(os.path.join("weights", "converted_savedmodel", "labels.txt"), "r") as f:
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
