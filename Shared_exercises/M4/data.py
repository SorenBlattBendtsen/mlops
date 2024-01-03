import torch
from torch.utils.data import Dataset
import os

def mnist(path='./Shared_exercises/M4/data/'):
    """Return train and test dataloaders for MNIST."""
    train_dataset = CustomDataset(path, 'train')
    test_dataset = CustomDataset(path, 'test')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    return train_loader, test_loader

def absoluteFilePaths(directory):
    for dirpath,_,filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))

class CustomDataset(Dataset):
    def __init__(self, path, partition):
        self.images = []
        self.labels = []
        files = absoluteFilePaths(path)
        for file in files:
            if partition in file:
                if 'images' in file:
                    self.images.append(torch.load(file))
                elif 'target' in file:
                    self.labels.append(torch.load(file))
            else:
                continue
        self.images = torch.cat(self.images)
        self.labels = torch.cat(self.labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
