from typing import Callable, Union

import torch
from torch import Tensor


class TrainCollator:
    """Collator providing batches with in-batch random negatives for training retrieval models."""

    def __init__(
        self,
        query_tokenizer: Union[object, Callable],
        doc_tokenizer: Union[object, Callable],
    ):
        self.query_tokenizer = query_tokenizer
        self.doc_tokenizer = doc_tokenizer

    def tokenize_queries(self, queries: list[str]) -> dict[str, Tensor]:
        return self.query_tokenizer(queries)

    def tokenize_docs(self, docs: list[str]) -> dict[str, Tensor]:
        return self.doc_tokenizer(docs)

    def compute_in_batch_random_negative_mask(
        self,
        batch_rel_doc_ids: list[list[str]],
        pos_doc_ids: list[str],
        neg_doc_ids: list[str],
    ) -> list[list[bool]]:
        """Compute in-batch random negative mask.

        Args:
            batch_rel_doc_ids (list[list[str]]): Relevant document ids for the queries in the batch.

            pos_doc_ids (list[str]): Positive document ids for the current batch.

            neg_doc_ids (list[str]): Hard negative document ids for the current batch.

        Returns:
            list[list[bool]]: In-batch random negative mask. Each element is a list of booleans indicating the eligibility of the training documents as random negatives for a specific query of the batch.
        """
        batch_doc_ids = pos_doc_ids + neg_doc_ids

        rand_neg_mask = [None] * len(batch_rel_doc_ids)
        for i, _ in enumerate(batch_rel_doc_ids):
            not_eligible = [
                *batch_rel_doc_ids[i],
                pos_doc_ids[i],
                neg_doc_ids[i],
            ]
            rand_neg_mask[i] = [id not in not_eligible for id in batch_doc_ids]

        return rand_neg_mask

    def __call__(self, batch: list[str]):
        """Call method."""

        batch_query = [x["query"] for x in batch]
        batch_rel_doc_ids = [x["rel_doc_ids"] for x in batch]
        batch_pos_doc = [x["pos_doc"] for x in batch]
        batch_neg_doc = [x["neg_doc"] for x in batch]

        # In-batch Random Negative Mask ----------------------------------------
        rand_neg_mask = self.compute_in_batch_random_negative_mask(
            batch_rel_doc_ids, batch_pos_doc, batch_neg_doc
        )
        rand_neg_mask = torch.tensor(rand_neg_mask)

        # Encode texts ---------------------------------------------------------
        encoded_queries = self.tokenize_queries(batch_query)
        encoded_docs = self.tokenize_docs(batch_pos_doc + batch_neg_doc)

        return encoded_queries, encoded_docs, rand_neg_mask
