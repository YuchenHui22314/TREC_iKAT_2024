<p align="center">
  <img src="./figures/overview.jpg" alt="Overview of the PPCIR Framework" width="100%"/>
</p>

# 🤖PPCIR: Precise Personalized Conversational IR via Fine-Grained Fusion 
<p>
<a href="https://github.com/DaoD/INTERS/blob/main/LICENSE">
<img src="https://img.shields.io/badge/MIT-License-blue" alt="license">
</a>
</p> 
This is the repository for the paper "Precise Personalized Conversational Information Retrieval via Fine-Grained Fusion" submitted to SIGIR 2025. To falilitate follow-up research on Personalized Conversational IR, we release the following resources:

- Our solid codebase for a personalized conversational RAG pipeline, with flexible choices of various retrievers, rerankers, as well as response generators.
- Detailed hands-on guidance of index building, enveronment setting for dense, spasre and splade retrieval, as well as necessary data preprocessing for TREC iKAT 2023 & 2024 datasets.
- All the [prompts](#prompts), few-shot examples, and manually curated Chain-of-Thought reasoning used in the paper.

Let us get started!

## 📚 Environment Setup & Index Building 
### Conda Python Environment
Please follow the steps below to create a conda environment with all the necessary packages.

[Option 1] Use the provided environment.yml file
```bash
# Create a new conda environment
conda env create -n <your desired env name> -f environment.yml
```
[Option 2] Manually install the packages
```bash
# Create a new conda environment
conda create -p <the_folder_where_you_store_the_environment> python=3.12
conda activate <the_folder_where_you_store_the_environment>
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

#### 1. Sparse index building
For index building, one should first download the collection following the [instructions](https://www.trecikat.com/data/#how-do-i-access-these-resources) provided by TREC iKAT organizers. Due to license issues, we cannot provide the collection directly. After downloading the collection (in `.jsonl` format), one can build lucene index using this script
`/src/indexing/ikat_23_jsonl_sparse_indexing.sh` 

#### 2. Dense index building
First, download the collection as mentioned above. We should then transform the `.jsonl` file to `.tsv` format using `/data_preprocessing_scripts/jsonl_to_tsv.py`. Finally, run the multi GPU dense index building script as follows:

```bash
cd /src/indexing/dense
torchrun --nproc_per_node 4 distributed_dense_index.py \
  --local-rank -1 \
  --n_gpu 4 \
  --model_type ance \   
  --collection_path <your tsv file path> \
  --pretrained_doc_encoder_path <Huggingface_repo_local_path> \
  --output_index_dir_path <output index path> \
  --seed 42 \
  --per_gpu_index_batch_size 700 \
  --num_docs_per_block 10000000 \
  --max_doc_length 256
```
Here you should modify the batch size according to your GPU memory. On the other hand, due to RAM restriction, we cannot load all embeddings for the whole collection into memory in one go. The program will therefore encode `num_docs_per_block` at a time then save it to the disk before continuing with the next. We call this a "embeddgng block" of the whole collection. For instance, if we set `num_docs_per_block` to 10M, then since ClueWeb22B has 116M documents, the program will yield 116M//10M + 1 = 12 embedding blocks. Please review the annotated code for more details.

#### 3. Splade indexing
First, download the collection as mentioned above. We should then change the `.jsonl` file to `.tsv` format using `/data_preprocessing_scripts/jsonl_to_tsv.py`. Finally, run the single GPU Splade index building script as follows:
```bash
cd /src/indexing/splade
python index.py \
  --collection_path <your tsv file path> \
  --pretrained_doc_encoder_path <Huggingface_repo_local_path> \
  --output_index_dir_path <output index path> \
  --seed 42 \
  --per_gpu_index_batch_size 190 \
  --max_doc_length 256
```
You should modify the batch size according to your GPU memory.

## 📝 Download TREC iKAT topics and relevance judgement

The next step is downloading the official TREC iKAT topics and relevance judgement (qrel files) from the [iKAT track official website](https://github.com/irlabamsterdam/iKAT/tree/main) and [TREC website](https://trec.nist.gov/data/ikat2023.html),  then proceeding to preprocess them. 

### Topics file preprocessing 
In our pipeline, we use the `Turn` object as the internal representation of each conversation turn (see `/src/topics.py`), so the preprocessing step involves reading the official iKAT topic file, flattening all the conversations to turn level, then converting each turn to a `Turn` object. Specifically, suppose that the downloaded topic file is stored at `./data/topics/2023_ikat_test_topics.json` (also apply for iKAT 24), then use the following code to preprocess trec topics. This would generate all `Trun` objects and transcript them to a `json` file. Note that the resulting `ikat_2023_test.json` file will be used as the input for the evaluation pipeline.
```python
import sys
sys.path.append('./src/')
from topics import (
    load_turns_from_ikat_topic_files, 
    save_turns_to_json, 
    )
import json

# downloaded from ikat website, in ikat format
ikat_topic_file = "./data/topics/2023_ikat_test_topics.json"
where_to_save = "./data/topics/ikat_2023_test.json"
list_of_turns = load_turns_from_ikat_topic_files(ikat_topic_file)
turn_dict_list = save_turns_to_json(list_of_turns, where_to_save)
print(json.dumps(turn_dict_list[0], indent=4))
```
the corresponding output should look like:
<details>
  <summary> 🖱️ Click to unfold</summary>
  <p>

```json
{
    "turn_id": "9-1-1",
    "conversation_id": "9-1",
    "title": "Finding a diet",
    "current_utterance": "Can you help me find a diet for myself?",
    "current_response": "Sure, these diets fit your condition and preference: 1) Vegan Mediterranean combines the principles of the Mediterranean diet with veganism and emphasizes whole grains, vegetables, beans, nuts, seeds, herbs, and spices. It excludes animal foods while incorporating plant-based alternatives, offering a sustainable and compassionate approach. (2) Vegan Keto: is a high-fat, low-carbohydrate, and moderate-protein diet which is also exclusively plant-based. It suits individuals following a vegan lifestyle who wish to adopt a ketogenic eating plan. (3) Eco-Atkins is a vegetarian variation of the Atkins diet which focuses on whole grains, vegetables, tofu, legumes, nuts, and seeds while minimizing animal products. By reducing carbohydrate intake and incorporating plant-based protein sources, it promotes sustainability and healthier eating. (4) The Ornish Diet (with limited exercise) is designed to improve overall health, support weight loss, and reverse cardiovascular disease progression, the Ornish diet is low in fat. It emphasizes exercise, stress management, and healthy relationships alongside a low-fat eating plan. It is worth noting that vegan diets, in general, have been linked to a lower risk of chronic health conditions, including heart disease. These diet options provide alternatives for individuals with specific dietary preferences and goals.",
    "oracle_utterance": "Can you help me find a diet for myself considering that I'm vegetarian, allergic to soybeans, lactose intolerant, can't exercise too much, and should drink water regularly?",
    "response_provenance": [
        "clueweb22-en0035-25-01897:1",
        "clueweb22-en0004-30-08099:2",
        "clueweb22-en0038-84-16253:4",
        "clueweb22-en0020-69-12751:1",
        "clueweb22-en0007-46-12888:5"
    ],
    "context_utterances": [],
    "ptkb": {
        "1": "I don't like the new spin-off; because I keep comparing the two and it has lower quality.",
        "2": "Because of my kidney problem, I have to drink water frequently to stay hydrated.",
        "3": "I'm going to change my phone.",
        "4": "I can't exercise too much because of the heart problem that I have.",
        "5": "I'm vegetarian.",
        "6": "I'm lactose intolerant.",
        "7": "I'm allergic to soybeans.",
        "8": "I just finished watching the Game of Thrones.",
        "9": "I didn't like how the series ended, especially the war scenes.",
        "10": "I'm an Android user."
    },
    "ptkb_provenance": [
        5,
        4,
        2
    ],
    "reformulations": [],
    "results": []
}
```
  </p>
</details>

### Qrels file preprocessing 
We use `/data_preprocessing_scripts/preprocess_qrel.py` to preprocess qrel file downloaded from ikat website. This just replaces _ with - for unifroming iKAT 23 and 24 qrel files. 
## ✍ Query Rewrite
Once the topics have been processed, we are ready for getting query reformulations. Please view files in `/src/rewrite` to run query rewriting. Specifically, you can run
```bash
cd ./src/rewrite
./rewrite_ikat_23.sh # or ./rewrite_ikat_24.sh
```
to rewrite queries for TREC iKAT 2023 or 2024 respectively. The program will first read the `json` file for loading `Turn` objects to memory, then put the rewritten queries to each `Turn` object then retranscripted them back to a `json` file. You should carefully review the code to understand the rewriting process and determine which  `reformulation_name` and `rewrite_model` to specify in the script. Here is a brief summary of available reformulation names:
- `gpt-4o_judge_and_rewrite`: The personalized reformulation method used in the paper, which first dynimically decide the persaonlization need for each query then rewrite and response accordingly.
- `gpt-4o_rar`: The non-personalized query reformulation method used in the paper. Will get a de-contextualized query and pseudo-response.
- ... and so on. Please review the code for more details.

### Prompts
An example of the prompt used in the paper is available at `/src/rewrite/judge_and_rewrite_prompt_example.txt`. The corresponding few-shot CoT examples are available at `/data/topics/ikat23/demonstration_using_ikat23_level.json` and `/data/topics/ikat24/demonstration_using_ikat24_level.json`


## 🚀 Running the evaluation Pipeline
Before running the pipeline, please make sure that you have built the indexes, downloaded relevance judgements, and got qeury reformulations as mentioned above. Then, you should specify all the required evaluation parameters in `/src/evaluate/eval_pipeline_config.yaml`. then run
```bash
cd ./src/evaluate
python run_experiments.py
```
this will yield the evaluation metrics for our methods in the "Aligned Comparison" part of the paper (w/. reranker). 


### How to modify the yaml file
The yaml file has 3 main zones, as shown in the following example. Here is a bref summary of each zone:
- `"iterate"`: This zone is where you define the combination of experiments to run. For instance, in the example below, we are running experiments on 3 retrieval models, 2 topics (two datsets), and 2 rerankers on the human rewrite then generate an answer using gpt-4o, which results in 3x2x2=12 experiments. The program `run_experiments.py` will iterate over all the combinations of these parameters and run the experiments (sequencially). 
- `"param_mapping"`: This zone is where you define some fixed relationships between the parameters. For instance, if you specify that you want to run experiments for iKAT 23 dataset in the first zone, then the path for both input query and qrels file should be those for iKAT23. In the following example, we define such a mapping between values of "topics" and values of "input_query_path" and "qrel_file_path". 
- `"fixed"`: This zone is where you define all other parameters that should remain unchanged for all the experiments defined in the first zone. For the full parameter list, please refer to the annotated code in `/src/evaluate/evaluation.py`. We will also explain each of these parameters in the following section. Basically, the parameter values defined by the first zone and the second zone will overwrite the values defined in this zone. For the details about this mechanism, please refer to the annotated code in `/src/evaluate/run_experiments.py`.
```yaml

###############################################################
##### First zone: Define the combination of experiments to run
###############################################################
iterate:
  topics: ["ikat_23_test", "ikat_24_test"]
  retrieval_model: ["BM25", "ance", "splade_v3"]
  retrieval_query_type: ["oracle"]
  reranking_query_type: ["oracle"]
  reranker : ["monot5_base_10k","rankllama"]
  generation_model: ['gpt-4o-2024-08-06']

###############################################################
##### Second zone: define some fixed relationships between the parameters
###############################################################
param_mapping:
  topics:
    ikat_24_test:
      input_query_path: "../../data/topics/ikat24/ikat_2024_test.json"
      qrel_file_path: "../../data/qrels/ikat_24_qrel.txt"             # not available for non-participants
    ikat_23_test:
      input_query_path: "../../data/topics/ikat23/ikat_2023_test.json"
      qrel_file_path: "../../data/qrels/ikat_23_qrel.txt"
  machine: ..................

###############################################################
##### Third zone: All other parameters that should remaining unchanged for all the exmperiments defined in the first zone.
###############################################################
fixed:
  collection: "ClueWeb_ikat"  # Dataset collection
  topics: "ikat_23_test"  # Test topics
  input_query_path: "../../data/topics/ikat23/ikat_2023_test.json"  # Path to input query file
  sparse_index_dir_path: "/part/01/Tmp/yuchen/indexes/clueweb22b_ikat23_official_sparse_index/"  # Use local disk index for fastest access
  output_dir_path: "../../results"  # Path to save results
  qrel_file_path: "../../data/qrels/ikat_23_qrel.txt"  # Qrel file path
  seed: 42  # Random seed for reproducibility

  # rertrieval parameters ......................
  # reranking parameters ......................
  # response generation parameters ......................
  # Fusion parameters ......................
```
### Evaluation Parameters
Now we will explain all the parameters that you can specify in the yaml file. Please refer to the annotated code in `/src/evaluate/evaluation.py` for further details. The behavior of the file  `evaluation.py` is as follows:
1. get all information about this run and generate an identifiable name for this run, e.g. name = "S1[raw]-S2[none]-g[raw]-[BM25]-[monot5_base_10k_4_1_none]-[s2_top50]", this means use BM25 to search with raw utterance, then use monot5_base_10k to rerank with raw utterance on top50 docs, then use raw utterance to generate response. 4_1_ refers to rankGPT window size and step size, and since we do not use rankGPT here, that is not relevant.
2. initialize a wandb run if `save_to_wandb` is true
3. if `run_rag` = false, try to load the ranking list from `output_dir_path`/`collection`/`topics`/ranking/name.txt, if not found, raise an error 
4. if `run_rag` = true, continue with search:
5. search: if `retrieval_model` = "none", load the ranking list from `given_ranking_list_path`, if not found, raise an error 
5. search: if `fusion_type` is not "none", do retrieval + fusion, using `QRs_to_rank`
6. search: if no fusion, do retrieval.
7. search: rerank based on the result of 6 or 7
12. search: save the ranking list if `save_ranking_list` is true
8. generate response
9. if `run_eval` = true, evaluate the results
11. save the results to `Turn` object if `save_results_to_object` is true
10. save the evaluation metrics to wandb if `save_to_wandb` is true
12. save the evaluation metrics to output_dir_path/collection/topics/metrics/name.json


Now let us continue with all the parameters~
#### 1. General parameters
-   `collection`: The name of the collection. This is used to generate an identifiable name for each run and to determine where to save the evaluation results. (e.g., "ClueWeb_ikat")
-   `topics`: The name of the dataset. This is used to generate an identifiable name for each run and to determine where to save the evaluation results. (e.g., "ikat_23_test")
-   `input_query_path`: Path to the input query file. (e.g., "../../data/topics/ikat23/ikat_2023_test.json")
-   `sparse_index_dir_path`: Path to the sparse index directory. Use a local disk index for the fastest access. 
-   `output_dir_path`: Path to save the results. Default to "../../results", then the results will be saved to "../../results/[collection]/[topics]/". 
-   `qrel_file_path`: Path to the qrel file. (e.g., "../../data/qrels/ikat_23_qrel.txt")
-   `seed`: Random seed for reproducibility. (e.g., 42)
-   `save_results_to_object`: Whether to save results to a `Turn` object. (e.g., true)
-   `save_ranking_list`: Whether to save the ranking list to a file located at "../../results/[collection]/[topics]/ranking". (e.g., true)
-   `run_rag`: Whether to run the Retrieval-Augmented Generation pipeline. (e.g., true). 
-   `run_eval`: Whether to run the evaluation. (e.g., true) Set to false if you just want to get something to submit to TREC. 

#### 2. Retrieval parameters

-   `retrieval_model`: The retrieval model to use. Can be "none", "BM25", "ance", "dpr", "splade_v3". If "none", read the ranking list from `given_ranking_list_path`. 
-   `retrieval_top_k`: The number of documents to retrieve. (e.g., 1000)
-   `personalization_group`: The personalization group to use. Can be "a", "b", "c", "all". You should always use "all" for the experiments in this paper.
-   `query_gpu_id`: The GPU ID for dense/splade query encoder. (e.g., 2)
-   `query_encoder_batch_size`: The batch size for query encoding. (e.g., 200)

##### Splade parameters

-   `splade_query_encoder_path`: Path to the Splade query encoder. 
-   `splade_index_dir_path`: Path to the Splade index. 

##### Dense parameters

-   `use_pyserini_dense_search`: Whether to use Pyserini's dense search. (e.g., false) Please set to false for this paper.
-   `dense_query_encoder_path`: Path to the dense query encoder. 
-   `dense_index_dir_path`: Path to the dense index. 
-   `faiss_n_gpu`: Number of GPUs for FAISS. (e.g., 4)
-   `use_gpu_for_faiss`: Whether to use GPU for FAISS. (e.g., true)
-   `embed_dim`: Embedding dimension. (e.g., 768)
-   `tempmem`: Temporary memory setting. (e.g., -1)
-   `passage_block_num`: Number of passage blocks. (e.g., 12)

##### BM25 parameters

-   `bm25_k1`: BM25 k1 parameter. (e.g., 0.9)
-   `bm25_b`: BM25 b parameter. (e.g., 0.4)

##### Query Expansion (QE) parameters

-   `qe_type`: The type of query expansion to use. (e.g., "none")
-   `fb_terms`: Number of feedback terms. (e.g., 20)
-   `fb_docs`: Number of feedback documents. (e.g., 10)
-   `original_query_weight`: Weight of the original query. (e.g., 0.5)

#### 3. Fusion parameters

-   `fusion_type`: The type of fusion to use. Can be 'none', 'round_robin', 'linear_combination', 'linear_weighted_score', 'per_query_personalize_level'. 
-   `QRs_to_rank`: The list of QRs to fuse. (e.g., ["mistral_rar_rw", "mistral_rar_rwrs", "mistral_judge_and_rewrite_rw"])
-   `fuse_weights`: Weights for linear combination fusion. (e.g., [0.1, 0.4])
-   `fusion_normalization`: Normalization method for fusion. Can be "none", "max", "min-max". 
-   `level_type`: The type of personalization level. Can be 'per_query_personalize_level', "gpt-4o_judge_and_rewrite"(used by this paper), "gpt-3.5_judge_and_rewrite". 
-   `per_query_weight_max_value`: Maximum weight for per-query optimization. (e.g., 1.2) Not relevant to this paper.
-   `optimize_level_weights`: Strategy for optimizing level weights. Can be "no level", "2+1", "group".  Please use "group" for the main method of this paper. "no level" means find the best weight without considering levels, 2+1 means first determine the weights for the first two QR, then determine the weight for the third QR. "group" means per-level optimization as described in the paper.
-   `target_metrics`: Metrics to optimize using grid search. (e.g., "mrr,ndcg@3,recall@10,recall@100")
-   `optimize_step`: Grid search step size. (e.g., 0.01)

#### 4. Reranking parameters

-   `reranker`: The reranker to use. Can be "none", "rankllama", "rankgpt", "monot5_base", "monot5_base_10k", "monot5_large", "monot5_large_10k", "monot5_3b", "monot5_3b_10k". 
-   `rerank_top_k`: The number of top documents to rerank. (e.g., 50)
-   `cache_dir`: Cache directory for Hugging Face models. 
-   `rerank_batch_size`: Batch size for reranking. (e.g., 67)
-   `rerank_quant`: Quantization level for reranking. Can be "none", "8b", "4b". 
-   `rankgpt_llm`: LLM to use for rankGPT. (e.g., "gpt-3.5-turbo")
-   `window_size`: Window size for rankGPT. (e.g., 4)
-   `step`: Step size for rankGPT. (e.g., 1)

#### 5. Response Generation parameters

-   `generation_model`: The model to use for response generation. (e.g., "gpt-4o-2024-08-06")
-   `generation_prompt`: The prompt to use for response generation. Can be "none", "raw"
-   `generation_top_k`: The number of documents to use for response generation. (e.g., 3)

#### 6. Metrics parameters

-   `save_to_wandb`: Whether to save the results to Weights & Biases. (e.g., true)
-   `metrics`: The metrics to compute. (e.g., "map,ndcg,ndcg_cut.1,ndcg_cut.3,ndcg_cut.5,ndcg_cut.10,P.1,P.3,P.5,P.10,P.20,recall.5,recall.10,recall.20,recall.50,recall.100,recall.1000,recip_rank")
-   `metrics_to_print`: The metrics to print to the console. (e.g., ["recip_rank", "ndcg_cut_3", "recall_10", "recall_100"])
-   `given_ranking_list_path`: Path to a given ranking list for evaluation. (e.g., "/results/ClueWeb_ikat/ikat_24_test/ranking/S1[gpt-4o_rar_rw_fuse_rar_rwrs_fuse_non_personalized_cot1_rw]-S2[gpt-4o_rar_personalized_cot1_rw]-g[gpt-4o_rar_personalized_cot1_rw]-[none]-[monot5_base_10k_4_1_none]-[s2_top50].txt")

#### 7. iKAT Project Specific parameters

-   `run_name`: A name for the run (for iKAT submission). (e.g., "none")
-   `retrieval_query_type`: The type of query used for retrieval. (See evaluation.py for all possible values)
-   `reranking_query_type`: The type of query used for reranking. (See evaluation.py for all possible values)
-   `generation_query_type`: The type of query used for generation. (See evaluation.py for all possible values)


# Citation
If you find this project useful, please cite our paper: 
```
xxxxxx
```

# Acknowledgement
We would like to thank xxx, xxx, and xxx.

