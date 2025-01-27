import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data import get_dataloaders
from model import TrafficCNN

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for profiles, conditions in train_loader:
        profiles, conditions = profiles.to(device), conditions.to(device)
        optimizer.zero_grad()
        outputs = model(profiles)
        loss = criterion(outputs, conditions)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

def evaluate(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for profiles, conditions in test_loader:
            profiles, conditions = profiles.to(device), conditions.to(device)
            outputs = model(profiles)
            loss = criterion(outputs, conditions)
            running_loss += loss.item()
    return running_loss / len(test_loader)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    profiles_dir = "data/profiles"
    conditions_dir = "data/conditions"
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 20

    train_loader, test_loader = get_dataloaders(profiles_dir, conditions_dir, batch_size)

    model = TrafficCNN().to(device)
    criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        test_loss = evaluate(model, test_loader, criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

    print("Training complete.")

if __name__ == "__main__":
    main()