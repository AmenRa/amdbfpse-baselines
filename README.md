# A Multi-Domain Benchmark for Personalized Search Evaluation Baselines
Baselines from the paper A Multi-Domain Benchmark for Personalized Search Evaluation.

The following guide assumes you are using a [Linux](https://en.wikipedia.org/wiki/Linux) distro, such as [Ubuntu](https://ubuntu.com).

## Step 1 - Download the datasets
Install [wget](https://www.gnu.org/software/wget/):
```sh
sudo apt install wget
```
Install [7-zip](https://www.7-zip.org):
```sh
sudo apt install p7zip-full p7zip-rar
```
Download the benchmark datasets with the provided script.  
It will download, decompress, and move the datasets in the appropriate folders.
```sh
sh download.sh
```
If you want to get rid of the original archives exectute the following command:
```sh
rm -rf ./tmp
```

## Step 2 - Create and activate the CONDA env
```sh
conda env create -f env.yml
```
```sh
conda activate amdbfpse-baselines
```

## Step 3 - Clone the language model locally
Install [Git LFS](https://git-lfs.github.com):
```sh
sudo apt-get install git-lfs
git lfs install
```

Clone the model:
```sh
git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
```

## Step 4 - Train & Evaluate
```sh
sh run.sh
```