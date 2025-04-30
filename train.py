import tensorflow as tf
from model import create_model
from data_preprocessing import load_data
import matplotlib.pyplot as plt

# Load the data from the dataset
X_train, X_val, y_train, y_val = load_data()

# Create the cnn model defined in model.py
model = create_model()

# Train the model 
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
    
# Save the model    
model.save("traffic_sign_model.h5")

# Plot the accuracy and loss
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Model Accuracy")

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Model Loss")

plt.show()