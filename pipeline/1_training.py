import os

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from src.biencoder import BiEncoder
from src.data import (
    DataModule,
    PersonalizedTrainCollator,
    PersonalizedTrainDataset,
    TrainCollator,
    TrainDataset,
)
from src.oneliner_utils import join_path, setup_logger
from src.personalization_model import PersonalizationModel
from src.tokenizer import Tokenizer
from src.utils import load_obj

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_pl_loggers(cfg: DictConfig) -> list:
    loggers = []

    if cfg.logging.do_logs:
        loggers.extend(
            load_obj(logger.class_name)(**logger.params)
            for logger in cfg.logging.pl_loggers
        )

    return loggers


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Setup logger -------------------------------------------------------------
    setup_logger(logger, dir=cfg.general.logs_dir, filename="1_training.log")

    # Set seeds for reproducibility --------------------------------------------
    logger.info("Random seeds")
    seed_everything(cfg.general.seed)

    # Dataset ------------------------------------------------------------------
    logger.info("Dataset")
    if cfg.model.kind == "personalized":
        dataset = PersonalizedTrainDataset(
            data_dir=cfg.dataset.data_dir, n_user_docs=cfg.training.n_user_docs
        )
    else:
        dataset = TrainDataset(data_dir=cfg.dataset.data_dir)

    # Tokenizer ----------------------------------------------------------------
    logger.info("Tokenizer")
    tokenizer = Tokenizer(**cfg.tokenizer.init)

    # Collator -----------------------------------------------------------------
    logger.info("Collator")
    if cfg.model.kind == "personalized":
        train_collator = PersonalizedTrainCollator(
            query_tokenizer=tokenizer, doc_tokenizer=tokenizer
        )
    else:
        train_collator = TrainCollator(
            query_tokenizer=tokenizer, doc_tokenizer=tokenizer
        )

    # DataModule ---------------------------------------------------------------
    logger.info("DataModule")
    datamodule = DataModule(
        train_dataset=dataset,
        train_dataloader_params=cfg.datamodule.train_dataloader_params,
        train_collate_fn=train_collator,
    )

    # Trainer ------------------------------------------------------------------
    logger.info("Trainer")
    trainer = Trainer(logger=get_pl_loggers(cfg), **cfg.trainer)

    # Model --------------------------------------------------------------------
    logger.info("Model")
    if cfg.model.kind == "personalized":
        model = PersonalizationModel(**cfg.model.init)
    else:
        model = BiEncoder(**cfg.model.init)

    # Training -----------------------------------------------------------------
    logger.info("Training")
    trainer.fit(model, datamodule=datamodule)

    # Save trained model -------------------------------------------------------
    logger.info("Save model")
    trainer.save_checkpoint(join_path(cfg.general.model_dir, "model.ckpt"))

    logger.success("Training complete!")


if __name__ == "__main__":
    with logger.catch(message="Error catched..."):
        main()
