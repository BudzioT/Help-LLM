import torch


class MultiHeadAttention(torch.nn.Module):
    """Class to manage multi-headed attention system"""
    def __init__(self, dim_in, dim_out, context_length, dropout, heads, qkv_bias=False):
        """Initialize the attention system"""
        super().__init__()
        # Safeguard for out dimension size
        assert dim_out % heads == 0, "Out dimension must be divisible by number of heads"

        # General variables
        self.dim_out = dim_out
        self.heads = heads
        self.head_dim = dim_out // heads

        # Weight matrices
        self.W_query = torch.nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.W_key = torch.nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.W_value = torch.nn.Linear(dim_in, dim_out, bias=qkv_bias)

        # All head outputs combined
        self.result_out = torch.nn.Linear(dim_out, dim_out)
        # Dropout to mask some attention weights - reduces overfitting
        self.dropout = torch.nn.Dropout(dropout)

        # Mask buffer
        self.register_buffer("mask",
                             torch.triu(torch.ones(context_length, context_length), 1))

    def forward(self, argument):
        """Forward argument to get the context vector"""
        pass
