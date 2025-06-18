import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from model import LivenessModel
from data_loader import CelebASpoofDataset
import os

def train(model, loader, criterion, optimizer, device):
    model.train()
    total, correct, total_loss = 0, 0, 0
    for i, (img, label) in enumerate(loader):
        img, label = img.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, pred = torch.max(output, 1)
        total += label.size(0)
        correct += (pred == label).sum().item()

        if i % 10 == 0:
            print(f"Batch {i}, Loss: {loss.item():.4f}")
    return total_loss / len(loader), correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    dataset = CelebASpoofDataset("train_list.txt", transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = LivenessModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    for epoch in range(10):
        loss, acc = train(model, loader, criterion, optimizer, device)
        print(f"[Epoch {epoch+1}] Loss: {loss:.4f} | Accuracy: {acc*100:.2f}%")
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/liveness_model.pth")

if __name__ == "__main__":
    main()
