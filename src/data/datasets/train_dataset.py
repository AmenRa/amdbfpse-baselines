import random

from indxr import Indxr
from src.oneliner_utils import join_path
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

        train_queries_path = join_path(data_dir, "train", "queries.jsonl")
        docs_path = join_path(data_dir, "collection.jsonl")

        self.train_index = Indxr(train_queries_path)
        self.doc_index = Indxr(docs_path)

    def sample_positive(self, rel_doc_ids: list, retrieved_doc_ids: list):
        """Sample a positive (relevant) document."""
        try:
            return random.choice(
                [x for x in retrieved_doc_ids if x in rel_doc_ids]
            )
        except Exception:
            # If there are no positives, sample from all relevant documents
            return random.choice(rel_doc_ids)

    def sample_negative(self, rel_doc_ids: list, retrieved_doc_ids: list):
        """Sample a negative (non-relevant) document from BM25 results."""
        try:
            return random.choice(
                [x for x in retrieved_doc_ids if x not in rel_doc_ids]
            )
        except Exception:
            # If there are no hard negatives, use a fake document
            return "fake_doc"

    def get_documents(self, pos_doc_id, neg_doc_id):
        pos_doc = self.doc_index.get(pos_doc_id)["title"]
        neg_doc = (
            self.doc_index.get(neg_doc_id)["title"]
            if neg_doc_id != "fake_doc"
            else "[PAD]"
        )

        return pos_doc, neg_doc

    # Support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index: int) -> str:
        query = self.train_index[index]

        rel_doc_ids = query["rel_doc_ids"]
        retrieved_doc_ids = query["bm25_doc_ids"]

        # Sampling -------------------------------------------------------------
        pos_doc_id = self.sample_positive(rel_doc_ids, retrieved_doc_ids)
        neg_doc_id = self.sample_negative(rel_doc_ids, retrieved_doc_ids)

        # Get docs -------------------------------------------------------------
        pos_doc, neg_doc = self.get_documents(pos_doc_id, neg_doc_id)

        return {
            "query": query["text"],
            "rel_doc_ids": rel_doc_ids,
            "pos_doc": pos_doc,
            "neg_doc": neg_doc,
        }

    # This allows to call len(dataset) to get the dataset size
    def __len__(self) -> int:
        return len(self.train_index)
