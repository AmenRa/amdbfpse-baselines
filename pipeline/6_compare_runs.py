import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from ranx import Qrels, Run, compare
from src.oneliner_utils import join_path, setup_logger


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Setup logger -------------------------------------------------------------
    setup_logger(
        logger, dir=cfg.general.logs_dir, filename="6_compare_runs.log"
    )

    logger.info("Loading test qrels...")
    qrels = Qrels.from_file(
        join_path(cfg.dataset.data_dir, "test", "qrels.json")
    )

    logger.info("Loading BM25 test run...")
    bm25_run = Run.from_file(
        join_path(cfg.dataset.data_dir, "test", "bm25_run.json")
    )
    bm25_run.name = "BM25"
    runs = [bm25_run]

    models = [
        "BM25_BiEnc",
        "BM25_Mean",
        "BM25_QA",
        "BM25_BiEnc_Mean",
        "BM25_BiEnc_QA",
    ]

    for model in models:
        model_name = " + ".join(model.split("_"))

        logger.info(f"Loading {model_name} test run...")
        run = Run.from_file(
            join_path(cfg.general.runs_dir, f"{model}_test.json")
        )
        run.name = model_name

        runs.append(run)

    report = compare(
        qrels,
        runs,
        metrics=["map@100", "mrr@10", "ndcg@10"],
        stat_test="student",
        max_p=0.001,
        rounding_digits=4,
        show_percentages=True,
    )

    logger.info("\n" + str(report))

    logger.success("Comparison done!")


if __name__ == "__main__":
    with logger.catch(message="Error catched..."):
        main()
