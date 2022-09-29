import json
import os

import h5py
import hydra
import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything
from src.biencoder import BiEncoder
from src.oneliner_utils import count_lines, join_path, setup_logger
from src.personalization_model import PersonalizationModel
from src.tokenizer import Tokenizer
from tqdm import tqdm


def query_generator(path: str, tokenizer, batch_size: int):
    ids, texts = [], []

    with open(path, "r") as f:
        for line in f:
            query = json.loads(line)
            ids.append(query["id"])
            texts.append(query["text"])

            if len(ids) == batch_size:
                encoded_texts = tokenizer(texts)
                input_ids = encoded_texts["input_ids"]
                attention_mask = encoded_texts["attention_mask"]

                yield (ids, input_ids, attention_mask)

                ids, texts = [], []

        if len(ids) != 0:
            encoded_texts = tokenizer(texts)
            input_ids = encoded_texts["input_ids"]
            attention_mask = encoded_texts["attention_mask"]

            yield (ids, input_ids, attention_mask)


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Setup logger -------------------------------------------------------------
    setup_logger(
        logger, dir=cfg.general.logs_dir, filename="3_embed_queries.log"
    )

    # Set seeds for reproducibility --------------------------------------------
    logger.info("Random seeds")
    seed_everything(42)

    # Load model ===============================================================
    logger.info("Model")
    if cfg.model.kind == "personalized":
        model = PersonalizationModel.load_from_checkpoint(
            join_path(cfg.general.model_dir, "model.ckpt"), **cfg.model.init
        ).cuda()
    else:
        model = BiEncoder.load_from_checkpoint(
            join_path(cfg.general.model_dir, "model.ckpt"), **cfg.model.init
        ).cuda()
    model.eval()

    # Load tokenizer ===========================================================
    logger.info("Tokenizer")
    tokenizer = Tokenizer(**cfg.tokenizer.init)

    # Compute embeddings =======================================================
    for split in ["val", "test"]:
        # I/O Paths ------------------------------------------------------------
        os.makedirs(cfg.general.data_dir, exist_ok=True)

        queries_path = join_path(cfg.dataset.data_dir, split, "queries.jsonl")
        embeddings_path = join_path(cfg.general.data_dir, "embeddings.h5")
        mapping_path = join_path(
            cfg.general.data_dir, f"{split}_query_emb_map.jsonl"
        )

        n_queries = count_lines(queries_path)
        pbar = tqdm(
            total=n_queries,
            desc=f"Embedding {split} queries",
            position=0,
            dynamic_ncols=True,
            mininterval=1.0,
        )

        with torch.cuda.amp.autocast(), torch.no_grad(), h5py.File(
            embeddings_path, "a"
        ) as h5, open(mapping_path, "w") as mapping:
            if f"{split}_queries" in h5:
                del h5[f"{split}_queries"]
            query_dataset = h5.create_dataset(
                f"{split}_queries",
                shape=(0, cfg.language_model.embedding_dim),
                dtype=np.float16,
                maxshape=(None, cfg.language_model.embedding_dim),
            )
            offset = 0

            for query_ids, input_ids, attention_mask in query_generator(
                queries_path, tokenizer=tokenizer, batch_size=64
            ):
                # Compute embeddings -------------------------------------------
                embeddings = model.text_encoder(
                    input_ids=input_ids.cuda(),
                    attention_mask=attention_mask.cuda(),
                )

                # Compute new offset -------------------------------------------
                new_offset = offset + embeddings.shape[0]

                # Save embeddings in HDF5 file ---------------------------------
                query_dataset.resize(new_offset, axis=0)
                query_dataset[offset:new_offset] = (
                    embeddings.detach().cpu().numpy()
                )

                # Save mapping -------------------------------------------------
                for i, id in enumerate(query_ids):
                    mapping.write(json.dumps({"id": id, "index": offset + i,}))
                    mapping.write("\n")

                # Update offset ------------------------------------------------
                offset = new_offset

                pbar.update(len(query_ids))

        pbar.close()


if __name__ == "__main__":
    with logger.catch(message="Error catched..."):
        main()
