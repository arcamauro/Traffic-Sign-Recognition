import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import os
# Function to load the data from a CSV file using pandas function to read CSV and sort it
def load_data(file_path):
    data = pd.read_csv(file_path, sep=",")
    data.sort_values("Path", inplace=True)
    data.reset_index(drop=True, inplace=True)

    return data

df = load_data("dataset/Train.csv")

class TrafficSignRecognitionDataset:
    def __init__(self, dataframe, root_dir, transform=None):
        self.data = dataframe
        self.root_dir = root_dir
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx]['Path']
        label = int(self.dataframe.iloc[idx]['ClassId'])

        # Costruisci path assoluto
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_data(self):
        return self.data