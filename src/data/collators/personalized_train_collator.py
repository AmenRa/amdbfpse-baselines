from typing import Callable, Union

import torch
from torch import Tensor

from .train_collator import TrainCollator


class PersonalizedTrainCollator(TrainCollator):
    """Collator providing batches with in-batch random negatives for training personalized retrieval models."""

    def __init__(
        self,
        query_tokenizer: Union[object, Callable],
        doc_tokenizer: Union[object, Callable],
    ):
        super().__init__(query_tokenizer, doc_tokenizer)

    # def sample_positives(
    #     self,
    #     batch_rel_doc_ids: list[list[str]],
    #     batch_retrieved_doc_ids: list[list[str]],
    # ) -> list[str]:
    #     """For each query, sample a positive (relevant) document.

    #     Args:
    #         batch_rel_doc_ids (list[list[str]]): Relevant document ids for the queries in the batch.

    #     Returns:
    #         list[str]: list of postive documents.
    #     """
    #     positives = [None] * len(batch_rel_doc_ids)

    #     for i, (rel_doc_ids, retrieved_doc_ids) in enumerate(
    #         zip(batch_rel_doc_ids, batch_retrieved_doc_ids)
    #     ):
    #         try:
    #             positives[i] = random.choice(
    #                 [x for x in retrieved_doc_ids if x in rel_doc_ids]
    #             )
    #         except Exception:
    #             # If there are no positives, sample from all relevant documents
    #             positives[i] = random.choice(rel_doc_ids)

    #     return positives

    # def sample_hard_negatives(
    #     self,
    #     batch_rel_doc_ids: list[list[str]],
    #     batch_retrieved_doc_ids: list[list[str]],
    # ) -> list[str]:
    #     """For each query, sample a negative (non-relevant) document from the BM25 results.

    #     Args:
    #         batch_rel_doc_ids (list[list[str]]): Relevant document ids for the queries in the batch.

    #         batch_retrieved_doc_ids (list[list[str]]): Document previously retrieved for the queries in the batch.

    #     Returns:
    #         list[str]: list of hard negatives.
    #     """

    #     hard_negatives = [None] * len(batch_rel_doc_ids)

    #     for i, (rel_doc_ids, retrieved_doc_ids) in enumerate(
    #         zip(batch_rel_doc_ids, batch_retrieved_doc_ids)
    #     ):
    #         try:
    #             hard_negatives[i] = random.choice(
    #                 [x for x in retrieved_doc_ids if x not in rel_doc_ids]
    #             )
    #         except:
    #             # If there are no hard negatives, use a fake document
    #             hard_negatives[i] = "fake_doc"

    #     return hard_negatives

    # def sample_user_docs(
    #     self, batch_user_doc_ids: list[list[str]]
    # ) -> list[list[str]]:
    #     """Samples user document sets."""
    #     return [
    #         random.sample(x, self.n_user_docs)
    #         if len(x) >= self.n_user_docs
    #         else x + ["fake_doc"] * (self.n_user_docs - len(x))  # Padding
    #         for x in batch_user_doc_ids
    #     ]

    # def generate_history_mask(
    #     self, batch_user_doc_ids: list[list[str]]
    # ) -> list[list[str]]:
    #     """Generates a mask to keep track of user history padding."""
    #     return np.where(np.asarray(batch_user_doc_ids) != "fake_doc", 1, 0)

    def __call__(
        self, batch: list[str]
    ) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Tensor], Tensor]:
        """Call method.

        Args:
            query_ids (list[str]): List of Query IDs.

        Returns:
            tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Tensor], Tensor]: The first element of the tuple is the output of the tokenizer for the queries, the second element is the output of the tokenizer for the positive documents, the second element is the output of the tokenizer for the hard negatives, and the last element is the in-batch random negatives mask.
        """

        batch_query = [x["query"] for x in batch]
        batch_rel_doc_ids = [x["rel_doc_ids"] for x in batch]
        batch_pos_doc = [x["pos_doc"] for x in batch]
        batch_neg_doc = [x["neg_doc"] for x in batch]
        batch_user_docs = [x["user_docs"] for x in batch]
        batch_history_mask = [x["history_mask"] for x in batch]

        # In-batch Random Negative Mask ----------------------------------------
        rand_neg_mask = self.compute_in_batch_random_negative_mask(
            batch_rel_doc_ids, batch_pos_doc, batch_neg_doc
        )
        rand_neg_mask = torch.tensor(rand_neg_mask)

        # History Mask ---------------------------------------------------------
        history_mask = torch.tensor(batch_history_mask)

        # Encode texts ---------------------------------------------------------
        encoded_queries = self.tokenize_queries(batch_query)

        docs = (
            batch_pos_doc
            + batch_neg_doc
            + [y for x in batch_user_docs for y in x]
        )
        encoded_docs = self.tokenize_docs(docs)

        return (
            encoded_queries,
            encoded_docs,
            history_mask,
            rand_neg_mask,
            2 * len(batch_pos_doc),
            len(docs) - 2 * len(batch_pos_doc),
        )
