import tensorflow as tf
import numpy as np
import cv2
from model import create_model
import matplotlib.pyplot as plt


# Load trained model
model = tf.keras.models.load_model("traffic_sign_model.h5")

# Load test image
def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (32, 32))
    img = np.expand_dims(img, axis=0) / 255.0  # Normalize and reshape

    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    
    print(f"Predicted class: {predicted_class}")

if __name__ == "__main__":
    test_image = "dataset/Test/00000.png"
    predict_image(test_image)
