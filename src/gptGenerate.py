import os
import json
import urllib.request

import numpy as np
import tiktoken
import torch
from fsspec.implementations.http import file_size
from sympy.unify.core import index
from urllib3 import request
from tqdm import tqdm


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

def generate(model, indexes, max_tokens, context_size,
             temperature=0.0, top_k=None, eos_id = None):
    """Generate response from GPT"""
    # Go through every token
    for _ in range(max_tokens):
        # Get previous responses
        index_condition = indexes[:, -context_size:]

        with torch.no_grad():  # Get predictions
            logits = model(index_condition)
        logits = logits[:, -1, :]  # Focus on the last step

        # Filter logits with top_k sampling
        if top_k is not None:
            # Keep only certain amount of top values
            top_logits, _ = torch.topk(logits, top_k)
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
