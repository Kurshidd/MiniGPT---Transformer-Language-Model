import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, text, tokenizer, block_size):
        self.text = text
        self.tokenizer = tokenizer
        self.block_size = block_size
        
        # Only store the text, encode on-the-fly in __getitem__
        
    def __len__(self):
        return len(self.text) - self.block_size
    
    def __getitem__(self, idx):
        # Encode only the required chunk
        chunk = self.text[idx:idx + self.block_size + 1]
        encoded = self.tokenizer.encode(chunk)
        
        x = torch.tensor(encoded[:-1], dtype=torch.long)
        y = torch.tensor(encoded[1:], dtype=torch.long)
        
        return x, y