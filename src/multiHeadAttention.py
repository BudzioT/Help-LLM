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

        # Mask buffer to show only previous tokens
        self.register_buffer("mask",
                             torch.triu(torch.ones(context_length, context_length), 1))

    def forward(self, argument):
        """Forward argument to get the context vector"""
        base, num_tokens, dim_in = argument.shape

        # Get proper shapes of weight matrices
        keys = self.W_key(argument)
        queries = self.W_query(argument)
        values = self.W_value(argument)

        # Split matrices to heads
        keys = keys.view(base, num_tokens, self.heads, self.head_dim)
        queries = queries.view(base, num_tokens, self.heads, self.head_dim)
        values = values.view(base, num_tokens, self.heads, self.head_dim)

        # Transpose matrices (swap dimensions: tokens with heads)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Calculate attention scores (by dot product of every head)
        attn_scores = queries @ keys.transpose(2, 3)

        # Get bool mask truncated to number of tokens
        mask = self.mask.bool()[:num_tokens, :num_tokens]

        # Fill attention scores with the mask (-inf to make it stay as 0)
        attn_scores.masked_fill(mask, -torch.inf)

        # Calculate attention weights, apply dropout
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, -1)
        attn_weights = self.dropout(attn_weights)

        # Get the right context vector shape
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine all heads, convert them to linear
        context_vec = context_vec.contiguous().view(base, num_tokens, self.dim_out)
        context_vec = self.result_out(context_vec)

        return context_vec
