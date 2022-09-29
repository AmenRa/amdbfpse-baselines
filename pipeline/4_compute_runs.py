import contextlib
import os

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything
from src.biencoder import BiEncoder
from src.data import (EvalCollatorPrecomputedEmbeddings,
                      EvalDatasetPrecomputedEmbeddings,
                      PersonalizedEvalCollatorPrecomputedEmbeddings,
                      PersonalizedEvalDatasetPrecomputedEmbeddings)
from src.oneliner_utils import join_path, setup_logger, write_json
from src.personalization_model import PersonalizationModel
from torch import no_grad
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader


def compute_run(model, dataloader):
    run = {}

    for batch in dataloader:
        with autocast(), no_grad():
            batch_scores = (
                model.compute_scores_for_precomputed_embeddings(
                    Q_emb=batch["Q_emb"].cuda(), D_emb=batch["D_emb"].cuda(),
                )
                .detach()
                .cpu()
                .numpy()
                .astype("float64")
            )

        for i, q_id in enumerate(batch["bm25_doc_ids"].keys()):
            run[q_id] = dict(zip(batch["bm25_doc_ids"][q_id], batch_scores[i]))
            with contextlib.suppress(Exception):
                del run[q_id]["fake_doc"]
    return run


def compute_personalized_run(model, dataloader):
    run = {}

    for batch in dataloader:
        with autocast(), no_grad():
            batch_scores = (
                model.compute_scores_for_precomputed_embeddings(
                    Q_emb=batch["Q_emb"].cuda(),
                    D_emb=batch["D_emb"].cuda(),
                    U_doc_embs=batch["U_doc_embs"].cuda(),
                    history_mask=batch["history_mask"].cuda(),
                )
                .detach()
                .cpu()
                .numpy()
                .astype("float64")
            )

        for i, q_id in enumerate(batch["bm25_doc_ids"].keys()):
            run[q_id] = dict(zip(batch["bm25_doc_ids"][q_id], batch_scores[i]))
            with contextlib.suppress(Exception):
                del run[q_id]["fake_doc"]
    return run


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Setup logger -------------------------------------------------------------
    setup_logger(
        logger, dir=cfg.general.logs_dir, filename="4_compute_runs.log"
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

    # Collator -----------------------------------------------------------------
    logger.info("Collator")
    if cfg.model.kind == "personalized":
        eval_collator = PersonalizedEvalCollatorPrecomputedEmbeddings()
    else:
        eval_collator = EvalCollatorPrecomputedEmbeddings()

    # Compute runs =============================================================
    for split in ["val", "test"]:
        # Dataset --------------------------------------------------------------
        logger.info(f"{split.capitalize()} set")
        if cfg.model.kind == "personalized":
            dataset = PersonalizedEvalDatasetPrecomputedEmbeddings(
                dataset_dir=cfg.dataset.data_dir,
                data_dir=cfg.general.data_dir,
                split=split,
            )
        else:
            dataset = EvalDatasetPrecomputedEmbeddings(
                dataset_dir=cfg.dataset.data_dir,
                data_dir=cfg.general.data_dir,
                split=split,
            )

        logger.info(f"Computing {split} run...")
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=512,
            shuffle=False,
            num_workers=4,
            collate_fn=eval_collator,
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True,
        )

        if cfg.model.kind == "personalized":
            run = compute_personalized_run(model=model, dataloader=dataloader)
        else:
            run = compute_run(model=model, dataloader=dataloader)

        # Save run
        os.makedirs(cfg.general.runs_dir, exist_ok=True)
        write_json(
            run,
            join_path(cfg.general.runs_dir, f"{cfg.model.name}_{split}.json"),
        )

        del run  # Release memory

    logger.success("Runs computations complete\n")


if __name__ == "__main__":
    with logger.catch(message="Error catched..."):
        main()
