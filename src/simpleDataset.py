from torch.utils.data import Dataset, DataLoader

class simpleDataset(Dataset):
    """Simple GPT dataset"""
    def __init__(self, txt, tokenizer, max_length, stride):
        """Initialize the dataset"""
        self.input_ids = []
        self.target_ids = []

        # Tokenize the set

