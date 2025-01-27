import torch
import torch.nn as nn
import torch.nn.functional as F

class TrafficCNN(nn.Module):
    def __init__(self):
        super(TrafficCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 3 * 15, 512)  # Correct flattened size
        self.fc2 = nn.Linear(512, 6)  # Output 6 classes
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension (1x30x120)
        x = self.pool(F.relu(self.conv1(x)))  # 32x15x60
        x = self.pool(F.relu(self.conv2(x)))  # 64x7x30
        x = self.pool(F.relu(self.conv3(x)))  # 128x3x15
        x = x.view(-1, 128 * 3 * 15)  # Flatten to 128*3*15=5760
        x = F.relu(self.fc1(x))  # 512
        x = self.dropout(x)
        x = self.fc2(x)  # 6
        return x