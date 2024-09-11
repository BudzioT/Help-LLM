import os
import re
from itertools import count

# Read raw text from file
with open("../resources/the-verdict.txt", 'r', encoding="utf-8") as file:
    raw_text = file.read()

# Split it into tokens and strip spaces
preprocessed = re.split(r"([.,?!;:_'\"()]|--|\s)", raw_text)
preprocessed = [item for item in preprocessed if item.strip()]

# Create vocabulary online
preprocessed = sorted(set(preprocessed))
vocab = {token:token_id for token_id, token in enumerate(preprocessed)}

for item, i in vocab.items():
    print(item, i)
