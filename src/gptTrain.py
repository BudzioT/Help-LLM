import math
import time
import os

import tiktoken
import torch
import matplotlib.pyplot as plt

from gpt import GPT
from utilities import *
from data import create_dataloader


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


def main():
    """Main function of GPT program"""
    ##############################################
    # Setup model
    ##############################################
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

    ##############################################
    # Train
    ##############################################
    # Calculate how many warmup steps there are based on total steps
    total_steps = len(train_loader) * TRAIN_SETTINGS["epochs"]
    warmup_steps = int(0.1 * total_steps)

    train_losses, validation_losses, tokens_seen = train_model(
        model, optimizer, device, TRAIN_SETTINGS["epochs"],
        5, 5, "Every effort moves you", tokenizer, warmup_steps, 1e-5, 1e-5)

    ##############################################
    # Plot results and save model weights
    ##############################################
    # Plot results
    epochs_tensor = torch.linspace(0, TRAIN_SETTINGS["epochs"], len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, validation_losses)
    plt.savefig("../resources/loss.pdf")

    # Save model
    torch.save(model.state_dict(), "../resources/model.pth")


def train_model(model, optimizer, device, epochs,
                eval_freq, eval_iter, input_text,
                tokenizer, warmup_steps, initial_lr=3e-05, min_lr=1e-6):
    """Train the model on the given input text, with specified parameters"""
    # List to track training parameters
    train_losses, validation_losses, tokens_seen_tracker, lrs_tracker = [], [], [], []
    tokens_seen, global_step = 0, -1

    # Get max learning rate from optimizer
    peak_lr = optimizer.param_groups[0]["lr"]
    # Learning rate increment during warmup
    lr_increment = (peak_lr - initial_lr) / warmup_steps

    start_time = time.time()

    # Get the files
    data_dir = "../resources/data"
    all_files = [os.path.join(path, name) for path, subdirs, files
                 in os.walk(data_dir) for name in files if name.endswith(".txt")]
    total_files = len(all_files)

    try:
        # Training loop
        for epoch in range(epochs):
            # Go through each train file
            for index, file_path in enumerate(all_files, 1):
                # Start counting time, read the file correctly
                book_start_time = time.time()
                text_data = read_text_file(file_path) + " <|endoftext|> "
                print(f"Working on file {index} of {total_files}")

                # Create dataloaders
                train_loader, validation_loader = create_dataloaders(text_data, 0.9,
                                                                     TRAIN_SETTINGS["batch_size"],
                                                                     GPT_CONFIG["context_length"],
                                                                     GPT_CONFIG["context_length"], 0)

                # Calculate total number of iterations in training process
                total_training_steps = len(train_loader) * epochs

                # Set model to training mode
                model.train()

                # Go through every batch set in the train loader
                for input_batch, target_batch in train_loader:
                    optimizer.zero_grad()  # Reset loss gradient from the previous batch
                    global_step += 1

                    # Adjust learning rate based on the current phase
                    if global_step < warmup_steps:  # Use warmup learning rate if current step is in the warmup zone
                        lr = initial_lr + global_step * lr_increment
                    else:  # Otherwise use cosine annealing
                        progress = ((global_step - warmup_steps) / (total_training_steps - warmup_steps))
                        lr = min_lr + (peak_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

                    # Apply learning rate to the optimizer
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = lr
                    lrs_tracker.append(lr)  # Keep track of learning rates

                    # Calculate loss gradient and update weights using them
                    loss = calc_loss_batch(input_batch, target_batch, model, device)
                    loss.backward()

                    # Apply gradient clipping to avoid exploding gradients (after warmup is over)
                    if global_step > warmup_steps:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    # Step the optimizer
                    optimizer.step()
                    # Update global step along with information about what tokens were seen
                    tokens_seen += input_batch.numel()

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
                print_eta(start_time, book_start_time, index, total_files)

    except KeyboardInterrupt:
        file_name = "../resources/model.pth"
        torch.save(model.state_dict(), file_name)
        print("Saved the model")

    return train_losses, validation_losses, tokens_seen_tracker


def create_dataloaders(text_data, train_ratio, batch_size, max_length, stride, workers=0):
    """Create and return train and validation loaders"""
    split_index = int(train_ratio * len(text_data))
    train_loader = create_dataloader(text_data[:split_index], batch_size, max_length, stride, True,
                                     True, workers)
    validation_loader = create_dataloader(text_data[split_index:], batch_size, max_length, stride, False,
                                          False, workers)
    return train_loader, validation_loader


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


def convert_time(seconds):
    """Convert time from seconds into hours, minutes and seconds"""
    hours, remaining = divmod(seconds, 3600)
    minutes, seconds = divmod(remaining, 60)
    return int(hours), int(minutes), int(seconds)


def print_eta(start_time, book_start_time, index, total_files):
    # Get current time, calculate time that passed from the start
    book_end_time = time.time()
    elapsed_time = book_end_time - book_start_time
    total_elapsed_time = book_end_time - start_time
    # Calculate remaining books, average time spent on them and estimated one
    books_remaining = total_files - index
    average_time = total_elapsed_time / index
    eta = average_time * books_remaining

    # Convert times to hours, minutes and seconds
    book_h, book_m, book_s = convert_time(elapsed_time)
    total_h, total_m, total_s = convert_time(total_elapsed_time)
    eta_h, eta_m, eta_s = convert_time(eta)

    print(f"Books processed {book_h} : {book_m} : {book_s}",
          f"Total time elapsed {total_h} : {total_m} : {total_s}",
          f"ETA for remaining ones {eta_h} : {eta_m} : {eta_s}")


# Run the main program
if __name__ == "__main__":
    main()
