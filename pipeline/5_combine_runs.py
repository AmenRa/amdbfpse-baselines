import os

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from ranx import Qrels, Run, fuse, optimize_fusion
from src.oneliner_utils import join_path, setup_logger, write_json


def fusion(val_qrels, val_runs, test_runs, norm="max", method="wsum") -> Run:
    # Optimize fusion ----------------------------------------------------------
    best_params = optimize_fusion(
        qrels=val_qrels,
        runs=val_runs,
        norm=norm,
        method=method,
        metric="ndcg@100",
    )

    # Fuse test runs -----------------------------------------------------------
    combined_run = fuse(
        runs=test_runs, norm=norm, method=method, params=best_params
    )
    combined_run.name = "_".join([run.name for run in test_runs])

    return best_params, combined_run


def save_fusion_output(cfg, best_params, combined_run):
    os.makedirs(cfg.general.hyperparams_dir, exist_ok=True)
    write_json(
        best_params,
        join_path(cfg.general.hyperparams_dir, f"{combined_run.name}.json"),
    )
    combined_run.save(
        join_path(cfg.general.runs_dir, f"{combined_run.name}_test.json")
    )


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Setup logger -------------------------------------------------------------
    setup_logger(
        logger, dir=cfg.general.logs_dir, filename="5_combine_runs.log"
    )

    # VAL QRELS ----------------------------------------------------------------
    logger.info("Loading val qrels...")
    val_qrels = Qrels.from_file(
        join_path(cfg.dataset.data_dir, "val", "qrels.json")
    )

    # VAL RUNS -----------------------------------------------------------------
    logger.info("Loading val runs...")
    bm25_val_run = Run.from_file(
        join_path(cfg.dataset.data_dir, "val", "bm25_run.json")
    )
    bienc_val_run = Run.from_file(
        join_path(cfg.general.runs_dir, "BiEnc_val.json")
    )
    mean_val_run = Run.from_file(
        join_path(cfg.general.runs_dir, "Mean_val.json")
    )
    qa_val_run = Run.from_file(join_path(cfg.general.runs_dir, "QA_val.json"))

    # TEST RUNS ----------------------------------------------------------------
    logger.info("Loading test runs...")
    bm25_test_run = Run.from_file(
        join_path(cfg.dataset.data_dir, "test", "bm25_run.json")
    )
    bm25_test_run.name = "BM25"

    bienc_test_run = Run.from_file(
        join_path(cfg.general.runs_dir, "BiEnc_test.json")
    )
    bienc_test_run.name = "BiEnc"

    mean_test_run = Run.from_file(
        join_path(cfg.general.runs_dir, "Mean_test.json")
    )
    mean_test_run.name = "Mean"

    qa_test_run = Run.from_file(join_path(cfg.general.runs_dir, "QA_test.json"))
    qa_test_run.name = "QA"

    # BM25 + Other =============================================================
    # BM25 + BiEnc -------------------------------------------------------------
    logger.info("Combining BM25 + BiEnc...")
    best_params, combined_run = fusion(
        val_qrels=val_qrels,
        val_runs=[bm25_val_run, bienc_val_run],
        test_runs=[bm25_test_run, bienc_test_run],
    )
    save_fusion_output(cfg, best_params, combined_run)

    # BM25 + Mean --------------------------------------------------------------
    logger.info("Combining BM25 + Mean...")
    best_params, combined_run = fusion(
        val_qrels=val_qrels,
        val_runs=[bm25_val_run, mean_val_run],
        test_runs=[bm25_test_run, mean_test_run],
    )
    save_fusion_output(cfg, best_params, combined_run)

    # BM25 + QA ----------------------------------------------------------------
    logger.info("Combining BM25 + QA...")
    best_params, combined_run = fusion(
        val_qrels=val_qrels,
        val_runs=[bm25_val_run, qa_val_run],
        test_runs=[bm25_test_run, qa_test_run],
    )
    save_fusion_output(cfg, best_params, combined_run)

    # BiEnc + Personalized =====================================================
    # BiEnc + Mean -------------------------------------------------------------
    logger.info("Combining BiEnc + Mean...")
    best_params, combined_run = fusion(
        val_qrels=val_qrels,
        val_runs=[bienc_val_run, mean_val_run],
        test_runs=[bienc_test_run, mean_test_run],
    )
    save_fusion_output(cfg, best_params, combined_run)

    # BiEnc + QA ---------------------------------------------------------------
    logger.info("Combining BiEnc + QA...")
    best_params, combined_run = fusion(
        val_qrels=val_qrels,
        val_runs=[bienc_val_run, qa_val_run],
        test_runs=[bienc_test_run, qa_test_run],
    )
    save_fusion_output(cfg, best_params, combined_run)

    # BM25 + BiEnc + Personalized ==============================================
    # BM25 + BiEnc + Mean ------------------------------------------------------
    logger.info("Combining BM25 + BiEnc + Mean...")
    best_params, combined_run = fusion(
        val_qrels=val_qrels,
        val_runs=[bm25_val_run, bienc_val_run, mean_val_run],
        test_runs=[bm25_test_run, bienc_test_run, mean_test_run],
    )
    save_fusion_output(cfg, best_params, combined_run)

    # BM25 + BiEnc + QA --------------------------------------------------------
    logger.info("Combining BM25 + BiEnc + QA...")
    best_params, combined_run = fusion(
        val_qrels=val_qrels,
        val_runs=[bm25_val_run, bienc_val_run, qa_val_run],
        test_runs=[bm25_test_run, bienc_test_run, qa_test_run],
    )
    save_fusion_output(cfg, best_params, combined_run)

    logger.success("Fusion complete!")


if __name__ == "__main__":
    with logger.catch(message="Error catched..."):
        main()
