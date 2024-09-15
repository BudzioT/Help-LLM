import tiktoken
import torch

from gpt import GPT


def text_to_ids(text, tokenizer):
    """Encode text into token IDs"""
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def ids_to_text(ids, tokenizer):
    """Decode IDs back to text"""
    flat_tensor = ids.squeeze(0)
    return tokenizer.decode(flat_tensor.tolist())


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
    torch.manual_seed(123)
    # Model of LLM
    model = GPT(GPT_CONFIG)
    # Disable dropout
    model.eval()

    start_text = "I will make you"

    # Create tokenizer, encode starter text and build a tensor out of it
    tokenizer = tiktoken.get_encoding("gpt2")

    print("\nInput text:", start_text, "\n")

    # Generate the rest of text, decode it
    output = generate_simple_text(model, text_to_ids(start_text, tokenizer),
                                  10, GPT_CONFIG["context_length"])

    print("Output length:", len(output[0]))
    print("Output text:", ids_to_text(output, tokenizer))


# Run the main program
if __name__ == "__main__":
    main()
