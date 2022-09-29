# !/bin/bash

datasets="computer_science physics political_science psychology"

for DATASET in $datasets;
do
    python -m pop.run --dataset=$DATASET
done