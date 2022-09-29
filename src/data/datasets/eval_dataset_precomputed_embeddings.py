import h5py
import numpy as np
from src.oneliner_utils import join_path, read_jsonl, read_list
from torch.utils.data import Dataset


class EvalDatasetPrecomputedEmbeddings(Dataset):
    def __init__(self, dataset_dir: str, data_dir: str, split: str):
        self.dataset_dir = dataset_dir
        self.query_ids = read_list(
            join_path(dataset_dir, split, "query_ids.txt")
        )

        queries_path = join_path(dataset_dir, split, "queries.jsonl")
        embeddings_path = join_path(data_dir, "embeddings.h5")
        doc_mapping_path = join_path(data_dir, "doc_emb_map.jsonl")
        query_mapping_path = join_path(data_dir, f"{split}_query_emb_map.jsonl")

        self.query_embeddings = h5py.File(embeddings_path, "r")[
            f"{split}_queries"
        ][...]
        self.query_mapping = {
            query["id"]: query["index"]
            for query in read_jsonl(query_mapping_path)
        }

        self.doc_embeddings = h5py.File(embeddings_path, "r")["docs"][...]
        self.doc_embeddings = np.vstack(
            [
                self.doc_embeddings,
                # Padding doc
                np.array(
                    [0.0] * self.doc_embeddings.shape[-1], dtype=np.float16
                ),
            ]
        )
        self.doc_mapping = {
            doc["id"]: doc["index"] for doc in read_jsonl(doc_mapping_path)
        }
        self.doc_mapping["fake_doc"] = -1  # Padding doc

        self.bm25_results = {
            q["id"]: {
                "user_doc_ids": q["user_doc_ids"],
                "bm25_doc_ids": q["bm25_doc_ids"],
            }
            for q in read_jsonl(queries_path)
        }

    def get_doc_embedding(self, id: str) -> np.ndarray:
        return self.doc_embeddings[self.doc_mapping[id]]

    def get_query_embedding(self, id: str) -> np.ndarray:
        return self.query_embeddings[self.query_mapping[id]]

    def pad_bm25_results(self, retrieved_doc_ids: list[str]) -> list[str]:
        """Pads BM25 results to a fixed length."""
        return (
            retrieved_doc_ids[:1000]
            if len(retrieved_doc_ids) >= 1000
            else retrieved_doc_ids
            + ["fake_doc"] * (1000 - len(retrieved_doc_ids))
        )

    # Support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index: int) -> str:
        query_id = self.query_ids[index]
        bm25_doc_ids = self.bm25_results[query_id]["bm25_doc_ids"]

        retrieved_doc_ids = self.pad_bm25_results(bm25_doc_ids)

        query_embedding = np.array(
            self.get_query_embedding(query_id), dtype=np.float16,
        )

        retrieved_doc_embeddings = np.array(
            [self.get_doc_embedding(doc_id) for doc_id in retrieved_doc_ids],
            dtype=np.float16,
        )

        return {
            "query_id": query_id,
            "bm25_doc_ids": bm25_doc_ids,
            "q_emb": query_embedding,
            "d_emb": retrieved_doc_embeddings,
        }

    # This allows to call len(dataset) to get the dataset size
    def __len__(self) -> int:
        return len(self.query_ids)
