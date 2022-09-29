import numpy as np
import torch


class PersonalizedEvalCollatorPrecomputedEmbeddings:
    def pad_user_docs(self, U_doc_embs: list):
        """Pads user document sets to max len in the batch."""
        max_len = max(len(x) for x in U_doc_embs)
        pad_emb = np.array([0.0] * len(U_doc_embs[0][0]))

        history_mask = [
            [1] * len(x) + [0] * (max_len - len(x)) for x in U_doc_embs
        ]
        U_doc_embs = [x + [pad_emb] * (max_len - len(x)) for x in U_doc_embs]

        return U_doc_embs, history_mask

    def __call__(self, batch: list):
        bm25_doc_ids = {x["query_id"]: x["bm25_doc_ids"] for x in batch}
        Q_emb = torch.tensor(np.array([x["q_emb"] for x in batch]))
        D_emb = torch.tensor(np.array([x["d_emb"] for x in batch]))
        U_doc_embs = [x["u_doc_embs"] for x in batch]
        U_doc_embs, history_mask = self.pad_user_docs(U_doc_embs)

        return {
            "bm25_doc_ids": bm25_doc_ids,
            "Q_emb": Q_emb,
            "D_emb": D_emb,
            "U_doc_embs": torch.tensor(np.array(U_doc_embs, dtype=np.float16)),
            "history_mask": torch.tensor(history_mask),
        }

