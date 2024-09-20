import math
import os
import json
import urllib.request

import numpy as np
import tiktoken
import torch
from urllib3 import request
from tqdm import tqdm

from gpt import GPT
from utilities import *


def download_file(url, destination):
    """Download file from the given url"""
    # Open the url and download file to the given destination
    with urllib.request.urlopen(url) as response:
        # Get file size
        file_size = int(response.headers.get("Context_length", 0))

        # If this file already exists, check if size of it is the same as requested one
        if os.path.exists(destination):
            local_file_size = os.path.getsize(destination)
            if local_file_size == file_size:  # Return if it is
                print(f"File is already downloaded and up to date: {destination}")
                return

        # Get size of one block (1 MB) and save name of URL file
        block_size = 1024
        progress_bar_desc = os.path.basename(url)

        # Create a progress bar
        with tqdm(total=file_size, unit="iB", unit_scale=True,
                  desc=progress_bar_desc) as progress_bar:
            # Open the file
            with open(destination, "wb") as file:
                # Update progress bar and save file at the same time
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    file.write(chunk)

                    progress_bar.update(len(chunk))


def generate(model, device, indexes, max_tokens, context_size,
             temperature=0.0, top_k=None, eos_id = None):
    """Generate response from GPT"""
    indexes = indexes.to(device)

    # Go through every token
    for _ in range(max_tokens):
        # Get previous responses
        index_condition = indexes[:, -context_size:]

        with torch.no_grad():  # Get predictions
            logits = model(index_condition)
        logits = logits[:, -1, :]  # Focus on the last step

        # Filter logits with top_k sampling
        if top_k is not None and top_k > 0:
            # Ensure that top_k isn't greater than dimension
            top_k = min(top_k, logits.size(-1))
            # Keep only certain amount of top values
            top_logits, _ = torch.topk(logits, math.ceil(top_k))
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val,
                                 torch.tensor(float("-inf")).to(logits.device), logits)
        # Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature

            # Softmax to get probabilities
            probs = torch.softmax(logits, -1)

            # Sample from distribution
            index_next = torch.multinomial(probs, 1)
        # Otherwise use highest logits number
        else:
            index_next = torch.argmax(logits, -1, True)

        # End early if encountered a flag to end
        if index_next == eos_id:
            break

        # Apply index to the running sequence
        indexes = torch.cat((indexes, index_next), 1)

    return indexes


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

    # Torch seed for randomness
    # torch.manual_seed(123)

    # Choose a correct device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Update gpt parameters
    gpt = GPT(GPT_CONFIG)
    # Load model if it already exists
    if os.path.exists("../resources/model.pth"):
        checkpoint = torch.load("../resources/model.pth", map_location=device, weights_only=True)
        gpt.load_state_dict(checkpoint)
    gpt.to(device)
    gpt.eval()

    # Create tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    ##############################################
    # Generate text
    ##############################################
    input_text = "Every effort moves you"

    token_ids = generate(gpt, device, text_to_ids(input_text, tokenizer),
                         25, GPT_CONFIG["context_length"], 1.0, 50)
    print("Output text:", ids_to_text(token_ids, tokenizer))


if __name__ == "__main__":
    main()
