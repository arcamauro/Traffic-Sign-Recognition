import os
import cv2
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Parameters
IMG_SIZE = 32
DATASET_PATH = "dataset/"
NUM_CLASSES = 43

def load_data():
    train_df = pd.read_csv(os.path.join(DATASET_PATH, "dataset/Train.csv"))

    data, labels = [], []
    for _, row in train_df.iterrows():
        img_path = os.path.join(DATASET_PATH, "Train", row["Path"])
        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        data.append(img)
        labels.append(row["ClassId"])

    data = np.array(data)
    labels = to_categorical(np.array(labels), NUM_CLASSES)
    
    return data, labels

def data_augment(X_train, y_train):
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=[0.8, 1.2],
        zoom_range=0.2
    )
    datagen.fit(X_train)
    return datagen

def preprocessing_split():
    X, y = load_data()
        
    # Split into training and validation sets (80% train, 20% validation)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Data augmentation
    train_datagen = data_augment(X_train, y_train)

    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    return X_train, X_val, y_train, y_val, train_datagen

    # Run preprocessing
if __name__ == "__main__":
    X_train, X_val, y_train, y_val, train_datagen = preprocessing_split()