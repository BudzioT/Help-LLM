import tiktoken
import torch

from gpt import GPT
from utilities import *
from data import create_dataloader


def generate_simple_text(model, index, max_new_tokens, context_size):
    """Generate new words for the text"""
    for _ in range(max_new_tokens):
        # Use only given amount of last tokens if current context size exceeds supported one
        index_condition = index[:, -context_size:]

        # Get predictions
        with torch.no_grad():
            logits = model(index_condition)

        # Focus only on the last step
        logits = logits[:, -1, :]

        # Get index of vocabulary element with the hightest logits value
        index_next = torch.argmax(logits, -1, True)
        # Append it to the current sequence
        index = torch.cat((index, index_next), dim=1)

    return index


def main():
    """Main function of GPT program"""
    ##############################################
    # Setup model
    ##############################################
    # Configurations for GPT
    GPT_CONFIG = {
        "vocab_size": 50257,  # Size based on default tokenizer
        "context_length": 1024,
        "embedding_dim": 768,
        "layers": 12,
        "heads": 12,
        "dropout_rate": 0.1,
        "qkv_bias": False
    }
    # Training settings
    TRAIN_SETTINGS = {
        "learning_rate": 5e-4,
        "epochs": 10,
        "batch_size": 2,
        "weight_decay": 0.1
    }

    # Torch seed for randomness
    torch.manual_seed(123)

    # Choose a correct device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model of LLM
    model = GPT(GPT_CONFIG)
    # Assign model to the used device
    model.to(device)

    # Create tokenizer, encode starter text and build a tensor out of it
    tokenizer = tiktoken.get_encoding("gpt2")

    ##############################################
    # Load data
    ##############################################
    # Load the file
    file_path = "../resources/the-verdict.txt"
    with open(file_path, 'r', encoding="utf-8") as file:
        input_text = file.read()

    ##############################################
    # Initialize dataloaders
    ##############################################
    # Set the training ratio and split text into the train/validation blocks
    train_ratio = 0.90
    split_index = int(train_ratio * len(input_text))

    # Data loaders for training and validation
    train_loader = create_dataloader(input_text[:split_index], TRAIN_SETTINGS["batch_size"],
                                     GPT_CONFIG["context_length"], GPT_CONFIG["context_length"],
                                     True, True, 0)
    validation_loader = create_dataloader(input_text[split_index:], TRAIN_SETTINGS["batch_size"],
                                          GPT_CONFIG["context_length"], GPT_CONFIG["context_length"],
                                          False, False, 0)


# Run the main program
if __name__ == "__main__":
    main()
