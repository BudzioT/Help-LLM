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

def calc_loss_batch(input_batch, target_batch, model, device):
    """Calculate cross-entropy loss of the batch"""
    # Get the correct form of batches
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)

    # Calculate loss from logits
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

def calc_loss_loader(data_loader, model, device, batches=None):
    """Calculate average loss from the data loader"""
    total_loss = 0
    # Safeguard, if data loader doesn't have any info
    if len(data_loader) == 0:
        return float("nan")
    # If batches are no-existent, set number to the length of data from loader
    elif batches is None:
        batches = len(data_loader)
    # Otherwise reduce number of batches to the length of data from loader if needed
    else:
        batches = min(batches, len(data_loader))

    # Go through each entry in data loader
    for i, (input_batch, target_batch) in enumerate(data_loader):
        # If it is in specified batches range, calculate loss and add it to the total
        if i < batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break

    # Return average loss
    return total_loss / batches
