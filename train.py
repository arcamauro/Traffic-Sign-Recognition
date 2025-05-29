import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from data_preprocessing import TrafficSignRecognitionDataset
from data_preprocessing import load_data
# Load the dataset
df = load_data("dataset/Train.csv")
transforms = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

dataset = TrafficSignRecognitionDataset(dataframe=df, root_dir="dataset/Train", transform=transforms)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)