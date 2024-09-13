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


def create_dataloader(text, batch_size=4, length=256, stride=128, shuffle=True,
                      drop_last=True, workers=0):
    """Create a GPT dataloader for the dataset"""
    # Create dependencies - tokenizer and dataset
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDataset(text, tokenizer, length, stride)

    # Create the dataloader
    dataloader = DataLoader(dataset, batch_size, shuffle, drop_last=drop_last,
                            num_workers=workers)
    return dataloader
