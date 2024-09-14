import torch


class GELU(torch.nn.Module):
    """GELU activation class"""
    def __init__(self):
        """Initialize module"""
        super().__init__()

    @staticmethod
    def forward(argument):
        """Forward argument torch after applying GELU"""
        # Some crazy math calculations
        return 0.5 * argument * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) *
                                                (argument + 0.044715 * torch.pow(argument, 3))
                                                ))


class FeedForward(torch.nn.Module):
    """Feed forward network class"""
    def __init__(self, cfg):
        super().__init__()
        # Expand inputs by factor of 4, apply GELU and then shrink it to the original size
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(cfg["embedding_dim"], 4 * cfg["embedding_dim"]),
            GELU(),
            torch.nn.Linear(4 * cfg["embedding_dim"], cfg["embedding_dim"])
        )

    def forward(self, argument):
        """Feed forward the argument torch"""
        return self.layers(argument)
