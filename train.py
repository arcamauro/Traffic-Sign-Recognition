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

# Instantiate device and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TrafficSignRecognitionCNN().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop for 50 epochs, I use early stopping to prevent overfitting
def train_loop(dataloader, model, loss_fn, optimizer, device, epochs=50, patience=5):
    size = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        model.train()
        running_loss = 0.0
        
        for batch, (X, y) in enumerate(dataloader):

            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            pred = model(X)
            loss = loss_fn(pred, y)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch % 100 == 0:
                current = batch * batch_size + len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1} - Average loss: {avg_loss:>8f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after epoch {epoch+1}")
                break
            
if __name__ == "__main__":
    train_loop(dataloader, model, loss_fn, optimizer, device, epochs=50, patience=5)