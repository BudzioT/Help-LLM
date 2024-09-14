import torch

from transformer import Transformer
from layerNorm import LayerNorm


class GPT(torch.nn.Module):
    """A generative pre-trained transformer class"""
    def __init__(self, cfg):
        """Initialize GPT with given configuration, create proper embeddings, transformer and normalizations"""
        super().__init__()
        # Initialize embeddings
        self.token_embedding = torch.nn.Embedding(cfg["vocab_size"], cfg["embedding_dim"])
        self.pos_embedding = torch.nn.Embedding(cfg["context_length"], cfg["embedding_dim"])
        self.dropout_embedding = torch.nn.Dropout(cfg["dropout_rate"])

        # Create transformer blocks
        self.transformers = torch.nn.Sequential(
            *[Transformer(cfg) for _ in range(cfg["layers"])]
        )

        # Initialize final normalization along with linear output head
        self.final_normalization = LayerNorm(cfg["embedding_dim"])
        self.output_head = torch.nn.Linear(cfg["embedding_dim"], cfg["vocab_size"], False)

    def forward(self, input_index):
        """Calculate logits based on input index"""
        batch_size, sequence_len = input_index.shape
        token_embeddings = self.token_embedding(input_index)
        pos_embeddings = self.pos_embedding(torch.arange(sequence_len, device=input_index.device))

        # Get argument torch, make it go through every last step
        argument = token_embeddings + pos_embeddings
        argument = self.dropout_embedding(argument)
        argument = self.transformers(argument)
        argument = self.final_normalization(argument)

        # Return logits from the calculated torch
        logits = self.output_head(argument)
        return logits
