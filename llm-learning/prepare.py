import sys  

import torch
import config
from torch.utils.data import Dataset, DataLoader
from config import data_path, batch_size, block_size

class TextDataset(Dataset):
    def __init__(self, path, block_size):
        text = open(path, 'r', encoding='utf-8').read()
        self.chars = sorted(list(set(text)))
        self.stoi = {ch:i for i,ch in enumerate(self.chars)}
        self.itos = {i:ch for ch,i in self.stoi.items()}
        self.vocab_size = len(self.chars)
        data = [self.stoi[ch] for ch in text]
        self.data = torch.tensor(data, dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.block_size]
        y = self.data[idx+1:idx+self.block_size+1]
        return x, y


def get_dataloaders():
    ds = TextDataset(data_path, block_size)
    config.vocab_size = ds.vocab_size
    train_loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    return train_loader, ds.stoi, ds.itos