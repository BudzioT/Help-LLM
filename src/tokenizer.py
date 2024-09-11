import os
import re


class SimpleTokenizer:
    """Simple tokenizer class to encode/decode text"""
    def __init__(self, vocabulary):
        """Initialize the tokenizer"""
        self.str_to_int = vocabulary
        self.int_to_str = {token_id: token for token, token_id in vocabulary.items()}

    def encode(self, text):
        """Encode text into IDs"""
        preprocessed = re.split(r"([.,?!;:_'\"()]|--|\s)", text)
        # Get rid of spaces
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        # Add unknown special tokens if needed
        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]

        ids = [self.str_to_int[word] for word in preprocessed]
        return ids

    def decode(self, ids):
        """Decode IDs back to text"""
        text = " ".join([self.int_to_str[index] for index in ids])
        # Add spaces
        text = re.sub(r"\s+([.,?!;:_'\"()])", r"\1", text)
        return text

# Read raw text from file
with open("../resources/the-verdict.txt", 'r', encoding="utf-8") as file:
    raw_text = file.read()

# Split it into tokens and strip spaces
preprocessed = re.split(r"([.,?!;:_'\"()]|--|\s)", raw_text)
preprocessed = [item for item in preprocessed if item.strip()]

# Create vocabulary
tokens = sorted(list(set(preprocessed)))
tokens.extend(["<|endoftext|>", "<|unk|>"])

vocab = {token:token_id for token_id, token in enumerate(tokens)}

# Create tokenizer
tokenizer = SimpleTokenizer(vocab)
text = "Hello, do you like tea. Is this-- a test?"
# Encode it
ids = tokenizer.encode(text)
print(ids)
print()
# Decode it
print(tokenizer.decode(ids))
