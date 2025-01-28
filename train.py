import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from data import get_dataloaders
from model import TrafficCNN

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for profiles, conditions in train_loader:
        profiles, conditions = profiles.to(device), conditions.to(device)

        # Ensure labels are class indices (fix potential one-hot encoding issue)
        if conditions.dim() > 1:
            conditions = conditions.argmax(dim=1)

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

            if conditions.dim() > 1:
                conditions = conditions.argmax(dim=1)

            outputs = model(profiles)
            loss = criterion(outputs, conditions)
            running_loss += loss.item()

    return running_loss / len(test_loader)

def plot_losses(train_losses, test_losses, num_epochs):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss", marker='o', linestyle='-')
    plt.plot(range(1, num_epochs + 1), test_losses, label="Test Loss", marker='s', linestyle='--')

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Test Loss Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_plot.png")
    plt.show()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    profiles_dir = "data/profiles"
    conditions_dir = "data/conditions"
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 40

    train_loader, test_loader = get_dataloaders(profiles_dir, conditions_dir, batch_size)

    model = TrafficCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        test_loss = evaluate(model, test_loader, criterion, device)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "traffic_cnn.pth")
    print("Model saved as 'traffic_cnn.pth'.")

    # Plot loss curves
    plot_losses(train_losses, test_losses, num_epochs)

    print("Training complete.")

if __name__ == "__main__":
    main()
