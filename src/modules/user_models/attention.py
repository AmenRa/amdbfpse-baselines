from torch import Tensor, einsum, float32, nn, sqrt, tensor


class ScaledDotProduct(nn.Module):
    """Scaled-Dot Product Alignment Model."""

    def __init__(self, embedding_dim: int = 312):
        super(ScaledDotProduct, self).__init__()

        self.scale = 1.0 / sqrt(tensor(embedding_dim, dtype=float32))

    def forward(self, Q: Tensor, K: Tensor) -> Tensor:
        return einsum("xz,xyz->xy", Q, K).mul(self.scale)


class Attention(nn.Module):
    """Standard Attention Mechanism."""

    def __init__(
        self, embedding_dim: int = 312,
    ):
        super(Attention, self).__init__()

        self.alignment_model = ScaledDotProduct(embedding_dim)

        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self, Q: Tensor, K: Tensor, V: Tensor, attention_mask: Tensor
    ) -> Tensor:
        # Scoring / Alignment --------------------------------------------------
        alignment_scores = self.alignment_model(Q, K)

        # Masking --------------------------------------------------------------
        alignment_scores = alignment_scores.masked_fill(
            attention_mask == 0, -1e4
        )

        # Normalization --------------------------------------------------------
        attention_weights = self.softmax(alignment_scores)

        # Aggregation ----------------------------------------------------------
        return einsum("xy,xyz->xz", attention_weights, V)
