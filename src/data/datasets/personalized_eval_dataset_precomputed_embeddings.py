import numpy as np

from .eval_dataset_precomputed_embeddings import (
    EvalDatasetPrecomputedEmbeddings,
)


class PersonalizedEvalDatasetPrecomputedEmbeddings(
    EvalDatasetPrecomputedEmbeddings
):
    def __init__(self, dataset_dir: str, data_dir: str, split: str):
        super().__init__(dataset_dir, data_dir, split)

    # Support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index: int) -> str:
        query_id = self.query_ids[index]
        bm25_doc_ids = self.bm25_results[query_id]["bm25_doc_ids"]
        user_doc_ids = self.bm25_results[query_id]["user_doc_ids"]

        retrieved_doc_ids = self.pad_bm25_results(bm25_doc_ids)

        query_embedding = np.array(
            self.get_query_embedding(query_id), dtype=np.float16,
        )

        retrieved_doc_embeddings = np.array(
            [self.get_doc_embedding(doc_id) for doc_id in retrieved_doc_ids],
            dtype=np.float16,
        )

        user_doc_embeddings = [
            self.get_doc_embedding(doc_id) for doc_id in user_doc_ids
        ]

        return {
            "query_id": query_id,
            "bm25_doc_ids": bm25_doc_ids,
            "q_emb": query_embedding,
            "d_emb": retrieved_doc_embeddings,
            "u_doc_embs": user_doc_embeddings,
        }

    # This allows to call len(dataset) to get the dataset size
    def __len__(self) -> int:
        return len(self.query_ids)
