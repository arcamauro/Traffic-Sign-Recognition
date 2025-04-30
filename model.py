import tensorflow as tf
from tensorflow.keras import layers, datasets, models, regularizers
import matplotlib.pyplot as plt

def create_model():
    model = models.Sequential()
    
    # Convolutional Layer 1
    model.add(layers.Conv2D(32, (5, 5), input_shape=(32, 32, 3)), kernel_regularizer=regularizers.l2(0.01))
    model.add(layers.BatchNormalization())
    model.add(layers.Activations('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
    # Convolutional Layer 2
    model.add(layers.Conv2D(64, (3, 3)), kernel_regularizer=regularizers.l2(0.01))
    model.add(layers.BatchNormalization())
    model.add(layers.Activations('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
    # Convolutional Layer 3
    model.add(layers.Conv2D(128, (3, 3)), kernel_regularizer=regularizers.l2(0.01))
    model.add(layers.BatchNormalization())
    model.add(layers.Activations('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
    # Flatten and dense layers
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.dropout(0.5))
    
    # Output layer and compile
    model.add(layers.Dense(43, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

if __name__ == "__main__":
    model = create_model()
    model.summary()