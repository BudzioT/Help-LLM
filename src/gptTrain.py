import tiktoken
import torch
import matplotlib.pyplot as plt

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
    # Create AdamW optimizer
    optimizer = torch.optim.AdamW(model.parameters(), TRAIN_SETTINGS["learning_rate"],
                                  weight_decay=TRAIN_SETTINGS["weight_decay"])

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

    # Begin the training
    train_losses, validation_losses, tokens_seen = train_model(
        model, train_loader, validation_loader, optimizer, device, TRAIN_SETTINGS["epochs"],
        5, 1, "Every effort moves you", tokenizer)

    return train_losses, validation_losses, tokens_seen


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
            optimizer.zero_grad()  # Reset loss gradient from the previous batch
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            # Calculate loss gradient and update weights using them
            loss.backward()
            optimizer.step()

            # Update global step along with information about what tokens were seen
            tokens_seen += input_batch.numel()
            global_step += 1

            # Evaluation step
            if global_step % eval_freq == 0:
                # Evaluate the model
                train_loss, validation_loss = evaluate_model(
                    model, train_loader, validation_loader, device, eval_iter)
                # Append all results to lists
                train_losses.append(train_loss)
                validation_losses.append(validation_loss)
                tokens_seen_tracker.append(tokens_seen)

                print(f"Epoch {epoch + 1} ({global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Validation loss: {train_loss:.3f}")

        print_sample(model, tokenizer, device, input_text)
    return train_losses, validation_losses, tokens_seen_tracker


def print_sample(model, tokenizer, device, input_text):
    """Generate and print sample"""
    # Evaluate model, calculate context size and encode the text
    model.eval()
    context_size = model.pos_embedding.weight.shape[0]
    encoded = text_to_ids(input_text, tokenizer).to(device)

    # If there isn't gradient existing, generate text
    with torch.no_grad():
        # Get token ids, decode them and print the resulting text
        token_ids = generate_text(model, encoded, 50, context_size)
        decoded = ids_to_text(token_ids, tokenizer)
        print(decoded.replace('\n', ' '))
    # Change back to training mode
    model.train()


def evaluate_model(model, train_loader, validation_loader, device, eval_iter):
    """Evaluate model to training mode"""
    model.eval()
    # If there isn't any gradient, calculate the loss
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, eval_iter)
        validation_loss = calc_loss_loader(validation_loader, model, device, eval_iter)
    model.train()

    return train_loss, validation_loss


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    """Plot the losses"""
    fig, axis1 = plt.subplots()

    # Plot training and validations losses against epochs
    axis1.plot(epochs_seen, train_losses, label="Training loss")
    axis1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    axis1.set_xlabel("Epochs")
    axis1.set_ylabel("Loss")
    axis1.legend(loc="upper right")

    # Second X-Axis for tokens that were seen
    axis2 = axis1.twiny()  # Copy first axis
    axis2.plot(tokens_seen, train_losses, alpha=0)  # Add aligned ticks
    axis2.set_xlabel("Tokens seen")

    # Adjust layout
    fig.tight_layout()


# Run the main program
if __name__ == "__main__":
    main()
