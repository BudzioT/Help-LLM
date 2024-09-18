import tiktoken
import torch

from gpt import GPT
from utilities import *
from data import create_dataloader


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


def train_model(model, train_loader, validation_loader, optimizer, device, epochs,
                eval_freq, eval_iter, input_text, tokenizer):
    """Train the model on the given input text, with specified parameters"""
    # List to track training parameters
    train_losses, validation_losses, tokens_seen_tracker = [], [], []
    tokens_seen, global_step = 0, -1

    # Training loop
    for epoch in range(epochs):
        # Set model to training mode
        model.train()

        # Go through every batch set in the train loader
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # Reset loss gradient from the previous batch
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            # Calculate loss gradient and update weights using them
            loss.backward()
            optimizer.step()

            # Update global step along with information about what tokens were seen
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, validation_loss = evaluate_model(
                    model, train_loader, validation_loader, device, eval_iter)
                #######################################
                # Finish this
                #######################################


def evaluate_model(model, train_loader, validation_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, eval_iter)
        validation_loss = calc_loss_loader(validation_loader, model, device, eval_iter)
    model.train()

    return train_loss, validation_loss

# Run the main program
if __name__ == "__main__":
    main()
