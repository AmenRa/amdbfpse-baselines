from torch import Tensor, nn

from ..user_models import Attention, Mean


class UserEncoder(nn.Module):
    """Aggregates user history to generate user embedding."""

    def __init__(
        self, aggregation_mode: str, embedding_dim: int = 312,
    ):
        super(UserEncoder, self).__init__()

        self.aggregation_mode = aggregation_mode
        self.embedding_dim = embedding_dim

        if aggregation_mode == "mean":
            self.aggregator = Mean()
        elif aggregation_mode == "attention":
            self.aggregator = Attention(embedding_dim)
        else:
            raise NotImplementedError()

    def forward(
        self,
        user_doc_embeddings: Tensor,
        query_embeddings: Tensor,
        history_mask: Tensor,
    ) -> Tensor:
        if self.aggregation_mode == "mean":
            return self.aggregator(user_doc_embeddings, history_mask)
        else:
            return self.aggregator(
                Q=query_embeddings,
                K=user_doc_embeddings,
                V=user_doc_embeddings,
                attention_mask=history_mask,
            )

