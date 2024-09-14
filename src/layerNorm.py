import torch


class LayerNorm(torch.nn.Module):
    """Layer normalization class"""
    def __init__(self, embedding_dim):
        """Initialize layer normalization handler"""
        super().__init__()
        # Very small number, near to 0
        self.eps = 1e-5

        # Variables that will be adjusted in need, when training
        self.scale = torch.nn.Parameter(torch.ones(embedding_dim))
        self.shift = torch.nn.Parameter(torch.zeros(embedding_dim))

    def forward(self, argument):
        """Forward the argument torch"""
        # Values needed for normalization
        mean = argument.mean(dim=-1, keepdim=True)
        var = argument.var(dim=-1, keepdim=True, unbiased=False)

        # Normalize the argument and forward it (eps is a safeguard to not divide by 0)
        normalized_arg = (argument - mean) / torch.sqrt(var + self.eps)
        return self.scale * normalized_arg + self.shift
