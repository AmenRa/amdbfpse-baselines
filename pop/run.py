import click
from ranx import Qrels, Run, evaluate, fuse, optimize_fusion
from src.oneliner_utils import join_path, read_jsonl


@click.command()
@click.option("--dataset", required=True)
def main(dataset):
    print(f"Computing Pop run for {dataset}")
    dataset_path = join_path("datasets", dataset)
    in_refs_path = join_path(dataset_path, "in_refs.jsonl")

    val_queries_path = join_path(dataset_path, "val", "queries.jsonl")
    val_bm25_path = join_path(dataset_path, "val", "bm25_run.json")
    val_qrels_path = join_path(dataset_path, "val", "qrels.json")

    test_queries_path = join_path(dataset_path, "test", "queries.jsonl")
    test_bm25_path = join_path(dataset_path, "test", "bm25_run.json")

    val_queries = read_jsonl(val_queries_path)
    test_queries = read_jsonl(test_queries_path)

    val_qrels = Qrels.from_file(val_qrels_path)

    val_bm25_run = Run.from_file(val_bm25_path)
    test_bm25_run = Run.from_file(test_bm25_path)

    pop_dict = {
        x["doc_id"]: len(x["in_refs"]) for x in read_jsonl(in_refs_path)
    }

    best_score = 0.0
    best_params = None
    best_root = 0

    for root in range(1, 11):
        smoothing_factor = 1 / root

        val_pop_run = Run(
            {
                q["id"]: {
                    doc_id: pop_dict.get(doc_id, 0) ** smoothing_factor
                    for doc_id in q["bm25_doc_ids"]
                }
                for q in val_queries
            }
        )

        params = optimize_fusion(
            qrels=val_qrels,
            runs=[val_bm25_run, val_pop_run],
            norm="max",
            method="wsum",
            metric="ndcg@100",
        )

        score = evaluate(
            qrels=val_qrels,
            run=fuse(
                [val_bm25_run, val_pop_run],
                norm="max",
                method="wsum",
                params=params,
            ),
            metrics="ndcg@100",
        )

        if score > best_score:
            best_score = score
            best_params = params
            best_root = root

    test_pop_run = Run(
        {
            q["id"]: {
                doc_id: pop_dict.get(doc_id, 0) ** (1 / best_root)
                for doc_id in q["bm25_doc_ids"]
            }
            for q in test_queries
        }
    )

    test_pop_run.save(join_path(dataset_path, "Pop_test.json"))
    combined_run = fuse([test_bm25_run, test_pop_run], params=best_params)
    combined_run.save(join_path(dataset_path, "BM25_Pop_test.json"))


if __name__ == "__main__":
    # execute only if run as a script
    main()
