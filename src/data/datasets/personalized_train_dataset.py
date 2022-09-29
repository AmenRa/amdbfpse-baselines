import random

from .train_dataset import TrainDataset


class PersonalizedTrainDataset(TrainDataset):
    def __init__(
        self, data_dir: str, n_user_docs: int = 20,
    ):
        super().__init__(data_dir=data_dir)

        self.n_user_docs = n_user_docs

    def sample_user_docs(self, user_doc_ids: list):
        """Samples user document set."""
        return random.sample(user_doc_ids, self.n_user_docs)

    def generate_history_mask(self, user_doc_ids: list):
        """Generates a mask to keep track of user history padding."""
        return [1 for _ in user_doc_ids] + [0] * (
            self.n_user_docs - len(user_doc_ids)
        )

    def get_documents(self, user_doc_ids, pos_doc_id, neg_doc_id):
        user_docs = [x["title"] for x in self.doc_index.mget(user_doc_ids)]
        pos_doc = self.doc_index.get(pos_doc_id)["title"]
        neg_doc = (
            self.doc_index.get(neg_doc_id)["title"]
            if neg_doc_id != "fake_doc"
            else "[PAD]"
        )

        return user_docs, pos_doc, neg_doc

    # Support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index: int) -> str:
        query = self.train_index[index]

        rel_doc_ids = query["rel_doc_ids"]
        retrieved_doc_ids = query["bm25_doc_ids"]
        user_doc_ids = query["user_doc_ids"]

        # Sampling -------------------------------------------------------------
        user_doc_ids = self.sample_user_docs(user_doc_ids)
        pos_doc_id = self.sample_positive(rel_doc_ids, retrieved_doc_ids)
        neg_doc_id = self.sample_negative(rel_doc_ids, retrieved_doc_ids)

        # History Mask ---------------------------------------------------------
        history_mask = self.generate_history_mask(user_doc_ids)

        # Get docs -------------------------------------------------------------
        user_docs, pos_doc, neg_doc = self.get_documents(
            user_doc_ids, pos_doc_id, neg_doc_id
        )

        return {
            "query": query["text"],
            "rel_doc_ids": rel_doc_ids,
            "pos_doc": pos_doc,
            "neg_doc": neg_doc,
            "user_docs": user_docs,
            "history_mask": history_mask,
        }

    # This allows to call len(dataset) to get the dataset size
    def __len__(self) -> int:
        return len(self.train_index)
