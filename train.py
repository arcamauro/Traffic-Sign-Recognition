import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from data_preprocessing import TrafficSignRecognitionDataset
from data_preprocessing import load_data
from model import TrafficSignRecognitionCNN
# Load the dataset
df = load_data("dataset/Train.csv")
transforms = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

dataset = TrafficSignRecognitionDataset(dataframe=df, root_dir="dataset/Train", transform=transforms)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TrafficSignRecognitionCNN.to(device)
criterion = nn.CrossEntropyLoss()
loss_fn = torch.optim.Adam(model.parameters(), lr=0.001)