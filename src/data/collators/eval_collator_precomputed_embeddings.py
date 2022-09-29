from typing import Dict

import numpy as np
import torch


class EvalCollatorPrecomputedEmbeddings:
    def __call__(self, batch: list):
        bm25_doc_ids = {x["query_id"]: x["bm25_doc_ids"] for x in batch}
        Q_emb = torch.tensor(np.array([x["q_emb"] for x in batch]))
        D_emb = torch.tensor(np.array([x["d_emb"] for x in batch]))

        return {
            "bm25_doc_ids": bm25_doc_ids,
            "Q_emb": Q_emb,
            "D_emb": D_emb,
        }
