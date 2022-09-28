# A Multi-Domain Benchmark for Personalized Search Evaluation Baselines
Baselines from the paper A Multi-Domain Benchmark for Personalized Search Evaluation.


## Step 1 - Create and activate the CONDA env
```sh
conda env create -f env.yml
```
```sh
conda activate amdbfpse-baselines
```

## Step 2 - Download language model locally
Install [Git LFS](https://git-lfs.github.com):
```sh
sudo apt-get install git-lfs
git lfs install
```

Clone the model:
```sh
git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
```