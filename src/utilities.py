import tiktoken
import torch


def text_to_ids(text, tokenizer):
    """Encode text into token IDs"""
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def ids_to_text(ids, tokenizer):
    """Decode IDs back to text"""
    flat_tensor = ids.squeeze(0)
    return tokenizer.decode(flat_tensor.tolist())