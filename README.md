<p align="center">
  <img src="./figures/overview.jpg" alt="Overview of the PPCIR Framework" width="100%"/>
</p>

# 🤖PPCIR: Precise Personalized Conversational IR via Fine-Grained Fusion 
<p>
<a href="https://github.com/DaoD/INTERS/blob/main/LICENSE">
<img src="https://img.shields.io/badge/MIT-License-blue" alt="license">
</a>
</p> 
This is the repository for the paper "Precise Personalized Conversational Information Retrieval via Fine-Grained Fusion" submitted to SIGIR 2025. To falilitate follow-up research on Personalized Conversational IR, we release in this repo

- Our solid codebase for a personalized conversational RAG pipeline, with flexible choices of various retrievers, rerankers, as well as response generators.
- Detailed hands-on guidance of index building, enveronment setting for dense, spasre and splade retrieval, as well as necessary data preprocessing for TREC iKAT 2023 & 2024 datasets
- All the prompts and few-shot examples used in the paper.

Let us get started!

## 📚 Environment Setup & Index Building 
### Conda Python Environment
Please follow the steps below to create a conda environment with all the necessary packages.

Option 1: Use the provided environment.yml file
```bash
# Create a new conda environment
conda env create -n <your desired env name> -f environment.yml
```
Option 2: Manually install the packages
```bash
# Create a new conda environment
conda create -p <thefolder where you store the environment> python=3.12
conda activate <the folder where you store the environment>
# Install torch
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
# FAISS for dense retrieval
conda install -c conda-forge faiss-gpu
# For rankGPT:
conda install -c laura-dietz cbor=1.0.0
# Install other packages with pip
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```
### Index Building
1. 
