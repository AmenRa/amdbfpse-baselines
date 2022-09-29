import torch
from torch import Tensor, einsum, nn


class Mean(nn.Module):
    """Mean pooling-based user model."""

    def __init__(self):
        super(Mean, self).__init__()

    def forward(self, embeddings: Tensor, history_mask: Tensor) -> Tensor:
        numerators = einsum("xyz,xy->xyz", embeddings, history_mask).sum(dim=1)

        # Clamp all values in [min, max] to prevent zero division
        denominators = torch.clamp(history_mask.sum(dim=-1), min=1e-9)

        return einsum("xz,x->xz", numerators, 1 / denominators)
