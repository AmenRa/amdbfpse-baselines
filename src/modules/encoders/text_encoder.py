import torch
from torch import Tensor, einsum, nn
from transformers import AutoModel


class TextEncoder(nn.Module):
    def __init__(self, model: str = "bert-base-uncased"):
        super(TextEncoder, self).__init__()

        self.NeuralLM = AutoModel.from_pretrained(
            model, add_pooling_layer=False, local_files_only=True,
        )

    def mean_pooling(
        self, embeddings: Tensor, attention_mask: Tensor,
    ) -> Tensor:
        """Computes token embeddings mean pooling.

        Args:
            embeddings (Tensor): [batch_size, n_tokens, embedding_dim]
            attention_mask (Tensor): [batch_size, n_tokens]

        Returns:
            Tensor: [batch_size, embedding_dim]
        """
        # Zero out padding token embeddings and sum over tokens dimension
        numerators = einsum("xyz,xy->xyz", embeddings, attention_mask).sum(
            dim=1
        )

        # Clamp all values in [min, max] to prevent zero division
        denominators = torch.clamp(attention_mask.sum(dim=-1), min=1e-9)

        return einsum("xz,x->xz", numerators, 1 / denominators)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Forward function.

        Args:
            input_ids (Tensor): [batch_size, n_tokens]
            attention_mask (Tensor): [batch_size, n_tokens]

        Returns:
            Tensor: [batch_size, embedding_dim]
        """

        embeddings = self.NeuralLM(
            input_ids=input_ids, attention_mask=attention_mask,
        ).last_hidden_state

        return self.mean_pooling(embeddings, attention_mask)
