import torch
from pytorch_lightning import LightningModule
from torch import Tensor, nn
from torch.nn import CosineSimilarity
from torch.nn.functional import relu
from torchmetrics import (
    Accuracy,
    RetrievalMAP,
    RetrievalMRR,
    RetrievalNormalizedDCG,
)

from .modules import TextEncoder, UserEncoder


class TripletMarginLoss(nn.Module):
    """
    Triplet Margin Loss function.
    """

    def __init__(self, margin=1.0):
        super(TripletMarginLoss, self).__init__()
        self.margin = margin

    def forward(self, pos_scores: Tensor, neg_scores: Tensor) -> Tensor:
        return relu(self.margin - pos_scores + neg_scores).mean()


class PersonalizationModel(LightningModule):
    def __init__(
        self,
        user_encoder_kwargs: dict,
        language_model: str,
        learning_rate: float = 5e-5,
        margin: float = 1.0,
    ):
        super().__init__()

        # Layers ---------------------------------------------------------------
        self.text_encoder = TextEncoder(language_model)
        self.user_encoder = UserEncoder(**user_encoder_kwargs)

        # Scoring function -----------------------------------------------------
        self.scoring_function = CosineSimilarity(dim=1)

        # Loss function --------------------------------------------------------
        self.criterion = TripletMarginLoss(margin)

        # Optimizer ------------------------------------------------------------
        self.optimizer = torch.optim.AdamW
        self.learning_rate = learning_rate

        # Metrics --------------------------------------------------------------
        self.accuracy = Accuracy()
        self.mrr = RetrievalMRR()
        self.ndcg = RetrievalNormalizedDCG()
        self.map = RetrievalMAP()

    def listwise_scoring_function(self, Q_emb: Tensor, D_emb: Tensor) -> Tensor:
        """Applies the scoring function to each query-document pre-defined combination.

        Args:
            Q_emb (Tensor): [batch_size, embedding_dim]
            D_emb (Tensor): [batch_size * n_docs_per_query, embedding_dim]

        Returns:
            Tensor: [batch_size * n_docs_per_query]
        """

        Q_emb = Q_emb.repeat_interleave(
            int(D_emb.shape[0] / Q_emb.shape[0]), dim=0
        )

        return self.scoring_function(Q_emb, D_emb)

    def in_batch_scoring_function(self, Q_emb: Tensor, D_emb: Tensor) -> Tensor:
        """Applies the scoring function to each possible query-document combination in the batch.

        Args:
            Q_emb (Tensor): [batch_size, embedding_dim]
            D_emb (Tensor): [2 * batch_size, embedding_dim]

        Returns:
            Tensor: [2 * batch_size]
        """

        return self.scoring_function(
            Q_emb.repeat_interleave(D_emb.shape[0], dim=0),
            D_emb.repeat((Q_emb.shape[0], 1)),
        )

    # In PyTorch Lightning, training_step is called during training.
    # It is used to separate the training forward from the inference forward.
    def training_step(self, batch, batch_idx) -> float:
        Q, D, history_mask, rand_neg_mask, n_docs, n_user_docs = batch

        # Compute text embeddings ----------------------------------------------
        Q_emb = self.text_encoder(**Q)
        D_emb, U_doc_embs = torch.split(
            self.text_encoder(**D), [n_docs, n_user_docs]
        )

        U_doc_embs = U_doc_embs.reshape(
            Q_emb.shape[0],  # batch size
            int(U_doc_embs.shape[0] / Q_emb.shape[0]),
            U_doc_embs.shape[-1],
        )

        D_pos_emb, D_neg_emb = D_emb.tensor_split(2)

        # Compute user embeddings ----------------------------------------------
        U_emb = self.user_encoder(U_doc_embs, Q_emb, history_mask)

        # Compute positive scores ----------------------------------------------
        pos_scores = self.scoring_function(U_emb, D_pos_emb)

        # HARD NEGATIVES =======================================================
        # Compute hard negative scores -----------------------------------------
        neg_scores = self.scoring_function(U_emb, D_neg_emb)

        # Compute hard negative loss -------------------------------------------
        hard_loss = self.criterion(pos_scores, neg_scores)

        # IN-BATCH RANDOM NEGATIVES ============================================
        # Compute the scores for each possible query-document combination ------
        rand_scores = self.in_batch_scoring_function(Q_emb, D_emb)
        # Remove non-eligible query-document scores
        rand_scores = rand_scores[rand_neg_mask.flatten()]
        repeated_pos_scores = pos_scores.repeat_interleave(
            D_emb.shape[0], dim=0
        )[rand_neg_mask.flatten()]

        # Compute random negative loss -----------------------------------------
        rand_loss = self.criterion(repeated_pos_scores, rand_scores)

        # COMBINE LOSSES =======================================================
        combined_loss = rand_loss + hard_loss

        # METRICS ==============================================================
        hard_accuracy = self.accuracy(
            torch.where(pos_scores > neg_scores, 1, 0),
            torch.ones(
                pos_scores.shape[0], dtype=torch.long, device=self.device
            ),
        )

        rand_accuracy = self.accuracy(
            torch.where(repeated_pos_scores > rand_scores, 1, 0),
            torch.ones(
                repeated_pos_scores.shape[0],
                dtype=torch.long,
                device=self.device,
            ),
        )

        # Logger ---------------------------------------------------------------
        self.log(
            "hard_acc",
            hard_accuracy,
            on_step=True,  # Logs the metric at that step in training
            prog_bar=False,  # Logs to the progress bar
            logger=True,  # Logs to the logger like Tensorboard
            batch_size=len(Q_emb),
        )

        self.log(
            "rand_acc",
            rand_accuracy,
            on_step=True,  # Logs the metric at that step in training
            prog_bar=False,  # Logs to the progress bar
            logger=True,  # Logs to the logger like Tensorboard
            batch_size=len(Q_emb),
        )

        self.log(
            "hard_loss",
            hard_loss,
            on_step=True,  # Logs the metric at that step in training
            prog_bar=False,  # Logs to the progress bar
            logger=True,  # Logs to the logger like Tensorboard
            batch_size=len(Q_emb),
        )

        self.log(
            "rand_loss",
            rand_loss,
            on_step=True,  # Logs the metric at that step in training
            prog_bar=False,  # Logs to the progress bar
            logger=True,  # Logs to the logger like Tensorboard
            batch_size=len(Q_emb),
        )

        return combined_loss

    def training_epoch_end(self, outs):
        # Log epoch metric
        self.log("train_acc_epoch", self.accuracy.compute())

    def validation_step(self, batch, batch_idx) -> float:
        Q, D, U_docs, history_mask, indices, targets = batch

        # Compute text embeddings ----------------------------------------------
        Q_emb = self.text_encoder(**Q)

        U_doc_embs = self.text_encoder(**U_docs)
        U_doc_embs = U_doc_embs.reshape(
            Q_emb.shape[0],  # batch size
            int(U_doc_embs.shape[0] / Q_emb.shape[0]),
            U_doc_embs.shape[-1],
        )

        D_emb = self.text_encoder(**D)

        # Compute user embeddings ----------------------------------------------
        U_emb = self.user_encoder(U_doc_embs, Q_emb, history_mask)

        # Compute scores -------------------------------------------------------
        scores = self.listwise_scoring_function(U_emb, D_emb)

        # Compute metric to use for checkpointing ------------------------------
        targets = targets.flatten()
        indices = indices.flatten()
        ndcg_score = self.ndcg(scores, targets, indices)
        self.log(
            "val_ndcg",
            ndcg_score,
            logger=True,  # Logs to the logger like Tensorboard
            batch_size=len(Q_emb),
        )

        return scores, targets, indices

    def validation_epoch_end(self, outs):
        scores = torch.cat([out[0] for out in outs], dim=0)
        targets = torch.cat([out[1] for out in outs], dim=0)
        indices = torch.cat([out[2] for out in outs], dim=0)

        map_score = self.map(scores, targets, indices)
        mrr_score = self.mrr(scores, targets, indices)
        ndcg_score = self.ndcg(scores, targets, indices)

        self.log(
            "val_map",
            map_score,
            logger=True,  # Logs to the logger like Tensorboard
            on_epoch=True,  # Logs the metric at the end of the validation step
        )

        self.log(
            "val_mrr",
            mrr_score,
            logger=True,  # Logs to the logger like Tensorboard
            on_epoch=True,  # Logs the metric at the end of the validation step
        )

        self.log(
            "val_ndcg",
            ndcg_score,
            logger=True,  # Logs to the logger like Tensorboard
            on_epoch=True,  # Logs the metric at the end of the validation step
        )

    def compute_scores_for_precomputed_embeddings(
        self,
        Q_emb: Tensor,
        D_emb: Tensor,
        U_doc_embs: Tensor,
        history_mask: Tensor,
    ) -> Tensor:
        # Compute user embeddings ----------------------------------------------
        U_emb = self.user_encoder(U_doc_embs, Q_emb, history_mask)

        # Compute scores -------------------------------------------------------
        scores = self.listwise_scoring_function(
            U_emb,
            D_emb.reshape(D_emb.shape[0] * D_emb.shape[1], D_emb.shape[2]),
        )

        return scores.reshape(Q_emb.shape[0], -1)

    # THIS IS JUST FOR ANALYSIS PURPOSE
    def compute_scores_for_precomputed_embeddings_analysis(
        self,
        Q_emb: Tensor,
        D_emb: Tensor,
        U_doc_embs: Tensor,
        history_mask: Tensor,
    ) -> Tensor:
        # Compute user embeddings ----------------------------------------------
        U_emb = self.user_encoder(U_doc_embs, Q_emb, history_mask)
        n_zero_users = torch.sum(~U_emb.any(1))

        # Compute scores -------------------------------------------------------
        scores = self.listwise_scoring_function(
            U_emb,
            D_emb.reshape(D_emb.shape[0] * D_emb.shape[1], D_emb.shape[2]),
        )

        return scores.reshape(Q_emb.shape[0], -1), n_zero_users

    # THIS IS JUST FOR ANALYSIS PURPOSE
    def compute_scores_for_precomputed_embeddings_analysis_2(
        self, Q_emb: Tensor, U_doc_embs: Tensor, history_mask: Tensor,
    ) -> Tensor:
        # Compute user embeddings ----------------------------------------------
        attention_weights = self.user_encoder.aggregator.get_attention_weights(
            Q_emb, U_doc_embs, history_mask
        )
        return torch.count_nonzero(attention_weights, dim=1)

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.learning_rate)
