# !/bin/bash

datasets="computer_science physics political_science psychology"
models="biencoder mean qa"

for DATASET in $datasets;
do
    for MODEL in $models;
    do
        python -m pipeline.1_training model=$MODEL dataset=$DATASET
        python -m pipeline.2_embed_docs model=$MODEL dataset=$DATASET
        python -m pipeline.3_embed_queries model=$MODEL dataset=$DATASET
        python -m pipeline.4_compute_runs model=$MODEL dataset=$DATASET

    done

    python -m pipeline.5_combine_runs dataset=$DATASET
    python -m pipeline.6_compare_runs dataset=$DATASET
done