import torch

from multiHeadAttention import MultiHeadAttention
from feedForward import FeedForward
from layerNorm import LayerNorm


class Transformer(torch.nn.Module):
    """Transformer for the LLM"""
    def __init__(self, cfg):
        """Initialize transformer with the given configurations"""
        super().__init__()
        # Initialize attention system with all given configurations
        self.attention_sys = MultiHeadAttention(
            dim_in=cfg["embedding_dim"],
            dim_out=cfg["embedding_dim"],
            context_length=cfg["context_length"],
            heads=cfg["heads"],
            dropout=cfg["dropout_rate"],
            qkv_bias=cfg["qkv_bias"]
        )

        # Store feed forward system, layer normalizations and dropout shortcut
        self.feed_forward = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["embedding_dim"])
        self.norm2 = LayerNorm(cfg["embedding_dim"])
        self.drop_shortcut = torch.nn.Dropout(cfg["dropout_rate"])

    def forward(self, argument):
        """Transform argument and forward it"""
        # Attention block with shortcut
        shortcut = argument
        argument = self.norm1(argument)
        argument = self.attention_sys(argument)
        argument = self.drop_shortcut(argument)
        argument = argument + shortcut

        # Feed-forward block with shortcut
        shortcut = argument
        argument = self.norm2(argument)
        argument = self.feed_forward(argument)
        argument = self.drop_shortcut(argument)
        argument = argument + shortcut

        return argument
