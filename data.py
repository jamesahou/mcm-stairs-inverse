import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class TrafficDataset(Dataset):
    def __init__(self, profiles_dir, conditions_dir, transform=None):
        self.profiles_dir = profiles_dir
        self.conditions_dir = conditions_dir
        self.transform = transform
        self.indices = range(972)  # Assuming 2000 samples

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        profile_path = os.path.join(self.profiles_dir, f"profile_{idx}.npy")
        condition_path = os.path.join(self.conditions_dir, f"conditions_{idx}.npy")

        profile = np.load(profile_path)
        condition = np.load(condition_path)

        condition = condition[:-1]

        if self.transform:
            profile = self.transform(profile)

        profile = torch.tensor(profile, dtype=torch.float32)
        condition = torch.tensor(condition, dtype=torch.float32)

        return profile, condition

def get_dataloaders(profiles_dir, conditions_dir, batch_size=32, train_split=0.8):
    dataset = TrafficDataset(profiles_dir, conditions_dir)
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader