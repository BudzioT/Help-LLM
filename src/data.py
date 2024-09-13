import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader


class GPTDataset(Dataset):
    """Dataset for the GPT"""
    def __init__(self, text, tokenizer, length, stride):
        """Create the dataset"""
        self.input_ids = []
        self.target_ids = []
        # Encode text, store the result IDs
        token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

        # Chunk text into overlapping sequences (sliding window)
        for i in range(0, len(token_ids) - length, stride):
            input_chunk = token_ids[i:i + length]
            target_chunk = token_ids[i + 1: i + length + 1]

            self.input_ids.append(torch.tensor(input_chunk))
            self.input_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        """Get length of dataset"""
        return len(self.input_ids)

    def __getitem__(self, item):
        """Get input and target items"""
        return self.input_ids[item], self.target_ids[item]
