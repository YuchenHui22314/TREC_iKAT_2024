<p align="center">
  <img src="./figures/overview.jpg" alt="Overview of the PPCIR Framework" width="100%"/>
</p>

# 🤖PPCIR: Precise Personalized Conversational IR via Fine-Grained Fusion 
<p>
<a href="https://github.com/DaoD/INTERS/blob/main/LICENSE">
<img src="https://img.shields.io/badge/MIT-License-blue" alt="license">
</a>
</p> 
This is the repository for the paper "Precise Personalized Conversational Information Retrieval via Fine-Grained Fusion" submitted to SIGIR 2025. To facilitate follow-up research on Personalized Conversational IR, we release the following resources:

- Our solid codebase for a personalized conversational RAG pipeline, with flexible choices of various retrievers, rerankers, and response generators.
- Detailed hands-on guidance for index building and environment setup (dense, sparse, and Splade retrieval), as well as the data preprocessing for the TREC iKAT 2023 & 2024 datasets.
- All the prompts, few-shot examples, and manually curated Chain-of-Thought reasoning used in the paper.

### 🆕 Extensions beyond the paper

Since the SIGIR submission, the codebase has grown into a general bench for **(personalized) conversational dense retrieval**. The additions below are fully wired into the same config-driven pipeline and are documented in their own sections:

- **Qwen3-Embedding dense retrieval** — `qwen3` (`Qwen3-Embedding-0.6B`, last-token pooling + L2, instruction-aware queries) as a first-class retriever, plus two **ConvDR-style fine-tuned** query encoders, `conv-qwen3` and `conv-ance`. These fine-tune only the *query* encoder and keep the *document* encoder **frozen**, so they **reuse the existing base ClueWeb22-B index** as-is. (See [Retrieval Models](#-retrieval-models).)
- **New conversational query-reformulation (QR) options** — feed a whole dialogue directly to the dense encoder with no human rewrite: `full_conversation` (interleaved user/system turns), `qwen_conversation` (+ `_ptkb` / `_ptkb_previous_conv_as_ptkb` persona variants), and `oracle_qwen_instruct`. (See [Query Reformulation](#-query-reformulation).)
- **TREC iKAT 2025 (offline / passage-ranking)** — topics + NIST qrels preprocessed into the exact same internal schema as 23/24. (See [`IKAT_2025_DATA_REPORT.md`](IKAT_2025_DATA_REPORT.md).)
- **Shared-corpus "stream once, fan out" evaluation** — a refactor that searches the ~478 GB ClueWeb index **once** for a whole batch of experiments that share a doc index, instead of re-streaming it per experiment. (See [Shared-Corpus Evaluation](#-shared-corpus-evaluation-stream-once-fan-out).)

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
For index building, one should first download the collection following the [instructions](https://www.trecikat.com/data/#how-do-i-access-these-resources) provided by the TREC iKAT organizers. Due to license issues, we cannot provide the collection directly. After downloading the collection (in `.jsonl` format), one can build a Lucene index using this script:
`/src/indexing/ikat_23_jsonl_sparse_indexing.sh` 

#### 2. Dense index building
First, download the collection as mentioned above. We then transform the `.jsonl` file to `.tsv` format using `/data_preprocessing_scripts/jsonl_to_tsv.py`. Finally, run the multi-GPU dense index building script as follows:

```bash
cd /src/indexing/dense
python distributed_dense_index.py \
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
Modify the batch size according to your GPU memory. Because of RAM limits, we cannot load the embeddings of the whole collection into memory at once. The program therefore encodes `num_docs_per_block` documents at a time and saves them to disk before continuing with the next chunk. We call this an **embedding block** of the collection. For instance, with `num_docs_per_block = 10M` and ClueWeb22-B's 116M documents, the program yields `116M // 10M + 1 = 12` embedding blocks. Please review the annotated code for more details.

> **`qwen3` / ConvDR encoders.** Pass `--model_type qwen-embedding` to encode the documents with `Qwen3-Embedding-0.6B` (last-token pooling + L2, dim **1024**); ANCE is `--model_type ance` (dim **768**). The `conv-qwen3` / `conv-ance` retrievers keep the doc encoder frozen, so **no new document index is needed** — they reuse the base Qwen3 / ANCE index.

> **Merging per-rank blocks (required before search).** Multi-GPU encoding writes per-rank shards `doc_emb_block.rank_{r}.{b}.pb`, but the searcher reads consecutively numbered `doc_emb_block.{id}.pb`. Merge them once with:
> ```bash
> cd /src
> python -m apcir.indexing.dense.merge_clueweb_index \
>   --input  <per-rank index dir> \
>   --output <merged index dir> \
>   --expected <total #docs>          # asserts (#docs per block) is a clean multiple
> ```
> Choose the block size from your RAM budget — one block must fit in memory during search. See [`CLUEWEB_QWEN_INDEXING_REPORT.md`](CLUEWEB_QWEN_INDEXING_REPORT.md) for the block-size formula and the encoding details.

#### 3. Splade indexing
First, download the collection as mentioned above. We then change the `.jsonl` file to `.tsv` format using `/data_preprocessing_scripts/jsonl_to_tsv.py`. Finally, run the single-GPU Splade index building script as follows:
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
Modify the batch size according to your GPU memory.

## ✍ Query Reformulation

In a conversational setting, the user's current utterance is usually under-specified (ellipsis, coreference), so we rewrite it into a self-contained search query before retrieval. In this pipeline the rewrite is selected by the **`--retrieval_query_type`** flag (with parallel `--reranking_query_type` / `--generation_query_type` for the later stages). Each value maps to one branch of `Turn.query_type_2_query()` in [`src/apcir/functional/topics.py`](src/apcir/functional/topics.py), which builds the actual query string from the `Turn` object.

The paper ships a large family of LLM-based rewrites — manual `oracle`, `rar_*` (rewrite / rewrite+summary, with/without CoT), personalized `*_personalized_cot*`, `*_judge_and_rewrite_*`, and their `_fuse_` combinations. To **add your own** option, touch four wiring points: the `query_type_2_query` dispatch in `topics.py`, the `--retrieval_query_type` argparse `choices`, `evaluation_util.get_query_list` (if it needs per-retriever routing), and the YAML `choices` / `param_mapping`.

**Conversational (no-rewrite) options added for dense retrieval.** These feed the dialogue *directly* to a dense encoder, with no LLM rewrite:

| `retrieval_query_type` | Builds | Intended encoder |
| --- | --- | --- |
| `full_conversation` | Interleaved `[q1, r1, …, q_cur]` (user utterances + system responses). A **logical** value resolved at runtime to `full_conversation_sparse` (BM25, `[SEP]`-joined) or `full_conversation_dense` (ConvDR token build: per-turn 64/64 caps, 512 total). | `ance` / `conv-ance` |
| `qwen_conversation` | Qwen instruction + interleaved `User:` / `System:` turns, ending in `User's last question:` (v3 template). | `qwen3` / `conv-qwen3` |
| `qwen_conversation_ptkb` | `qwen_conversation` + the full numbered **user profile** (PTKB), profile placed *before* the conversation. | `qwen3` / `conv-qwen3` |
| `qwen_conversation_ptkb_previous_conv_as_ptkb` | The above + the same persona's **previous conversation** prepended (iKAT-25 only). | `qwen3` / `conv-qwen3` |
| `oracle_qwen_instruct` | The human `oracle` rewrite wrapped in the MSMARCO Qwen instruction (`Instruct: Given a web search query, retrieve relevant passages…`). | `qwen3` only |

> **Encoding consistency.** A dense query rewrite is only meaningful if the query is encoded the *same way* as the documents in its index — same pooling (last-token for Qwen, the model's native pooling for ANCE) and the same L2 normalization. The `qwen_*` options apply the instruction to the **query side only**, never to the documents.

## 🔎 Retrieval Models

`--retrieval_model` selects the first-stage retriever. Sparse and dense paths share the same `search()` entry point; dense models additionally need a matching embedding index (`--dense_index_dir_path`, `--embed_dim`, `--passage_block_num`).

| Model | Type | Notes |
| --- | --- | --- |
| `BM25` | sparse | Lucene; optional RM3 pseudo-relevance feedback. |
| `splade_v3` | learned sparse | Splade-v3 index. |
| `ance`, `dpr`, `repllama` | dense | Single-vector dense encoders (dim 768 / …). |
| **`qwen3`** | dense | `Qwen3-Embedding-0.6B`, last-token pooling + L2, **dim 1024**, instruction-aware queries. |
| **`conv-qwen3`** | dense | ConvDR-style fine-tuned **query** encoder over `qwen3`; doc encoder frozen → **reuses the base `qwen3` index**. |
| **`conv-ance`** | dense | ConvDR-style fine-tuned **query** encoder over `ance`; doc encoder frozen → **reuses the base `ance` index**. |

Because the ConvDR fine-tunes register as their own model names (`conv-qwen3` / `conv-ance`), their result files carry a distinct stem and never collide with the base-model runs — base and fine-tuned can be compared side by side. To wire in a new retriever, touch five places (the `--retrieval_model` argparse `choices`, the `search.py` dispatch, the `dense_search.py` query-encoding branch, the YAML `param_mapping`, and the YAML `iterate`) and mind the dense traps: merge the index into `doc_emb_block.{id}.pb`, set `embed_dim` correctly, and keep query↔corpus encoding consistent.

## 🚀 Running the evaluation Pipeline
Before running the pipeline, please make sure you have built the indexes as mentioned above. Then download the official TREC iKAT topics and relevance judgements (qrel files):

| Dataset | Topics | Qrels |
| --- | --- | --- |
| TREC iKAT 2023 | [topics](https://github.com/irlabamsterdam/iKAT/tree/main/2023/data) | [qrels](https://www.trecikat.com/data/2023/qrels) |
| TREC iKAT 2024 | [topics](https://www.trecikat.com/data/#topics-for-ikat-year-2-2024) | Released after the embargo |
| TREC iKAT 2025 (offline) | [topics + qrels](https://github.com/irlabamsterdam/ikat2025) (`offline/`) | NIST `qrels-nist.trec` (passage ranking) |

### Topics file preprocessing 
In our pipeline, we use the `Turn` object as the internal representation of each conversation turn (see `/src/topics.py`), so the preprocessing step reads the official iKAT topic file, flattens all the conversations to the turn level, and converts each turn to a `Turn` object. Specifically, suppose the downloaded topic file is stored at `./data/topics/2023_ikat_test_topics.json` (the same applies to iKAT 24); then use the following code to preprocess the TREC topics. Note that the resulting `ikat_2023_test.json` is used as the input for the evaluation pipeline.
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

**iKAT 2025.** The 2025 offline topics use a few renamed fields (`turns`→`responses`, `utterance`→`user_utterance`, `ptkb` dict→array, etc.). They are handled by [`data_preprocessing_scripts/preprocess_ikat25.py`](data_preprocessing_scripts/preprocess_ikat25.py), which emits the **same internal schema** as 23/24. See [`IKAT_2025_DATA_REPORT.md`](IKAT_2025_DATA_REPORT.md) for the full field-difference table and the offline-vs-interactive track discussion.

### Qrels file preprocessing 
We use `/data_preprocessing_scripts/preprocess_qrel.py` to preprocess the qrel file downloaded from the iKAT website. It simply replaces `_` with `-` to unify the iKAT 23 / 24 / 25 qrel files with the `{number}-{turn_id}` qid convention.

### Launching experiments
Experiments are config-driven. `run_experiments.py` reads a `fuse_then_eval` YAML, takes the **Cartesian product** of everything under `iterate:`, overlays `param_mapping:` (per machine / model / topic / QR type) onto a copy of `fixed:`, and launches one `apcir.evaluate.evaluation` process per combination.

```bash
cd src
python -m apcir.evaluate.run_experiments --config ./apcir/evaluate/fuse_then_eval_config_23.yaml
```

Control switches: `reranker` / `fusion_type` (`none` ⇒ retrieval only), `generation_model` (non-`none` ⇒ RAG answer), `run_eval` (compute metrics), and automatic result reuse (a run whose `ranking_list_path` already exists is loaded, not recomputed). Results land under `results/<collection>/<topics>/{ranking,metrics,per_query_metrics,ikat_format_output}/`; the filename stem encodes every key parameter — `S1[<retrieval_qr>]-S2[<rerank_qr>]-g[<gen_qr>]-[<retrieval_model>]-[<reranker>_<window>_<step>_<quant>]-[s2_top<K>]`.

## ⚡ Shared-Corpus Evaluation (stream once, fan out)

For dense retrieval the **corpus is the embedding index**, which for ClueWeb22-B is ~**478 GB** — larger than RAM. The legacy path runs each experiment as its own process, and each process **re-streams the whole index from disk block-by-block**. When you sweep many query types / reranker settings over the *same* doc index (e.g. `qwen3` + `conv-qwen3` × several QRs × 23/24/25, all sharing the one qwen index), you pay that disk read **N times** for nothing — the run is disk-bound.

The `run_experiments_grouped` entry point fixes this with a **"stream once, fan out"** loop:

1. **Group** all experiments by `CorpusKey` (`index_dir`, `embed_dim`, `block_num`) — the *encoder is deliberately not part of the key*, so different query encoders over the same doc index group together.
2. **Phase A** — encode every job's queries (cheap, GPU) and vertically **stack** them into one query matrix `Q`.
3. **Phase B** — stream the corpus **once**: for each block, search *all* stacked queries in a single FAISS call, then merge across blocks.
4. **Phase C** — slice the merged results back per job and write the exact same ranking / metrics files as the legacy path.

```bash
cd src
python -m apcir.evaluate.run_experiments_grouped \
  --config ./apcir/evaluate/fuse_then_eval_config_qwen_conv.yaml \
  --merge compat      # compat = byte-faithful to legacy (validation); topk = fast numpy (production)
```

It consumes the **same YAML** as `run_experiments.py` and is fully opt-in — the legacy path is untouched and remains the reference. On the qwen sweep this turns N disk passes into one (≈ N× on the disk-dominant term).

**Validation.** The grouped path was checked byte-for-byte against the legacy path (`--merge compat`): for `conv-ance × full_conversation × {23,25}` the ranking, per-query, and averaged metrics are **bit-identical**; for `24` the only difference is float32 LSB jitter in FAISS-GPU scores from the larger stacked GEMM (every reported metric Δ = 0; MAP Δ = 3e-6, i.e. unchanged at table precision). So the framework is numerically faithful within GPU floating-point noise. Design and validation notes: [`SHARED_CORPUS_FRAMEWORK_DESIGN.md`](SHARED_CORPUS_FRAMEWORK_DESIGN.md), [`GROUPED_CODE_REVIEW.md`](GROUPED_CODE_REVIEW.md). Code: [`src/apcir/search/grouped/`](src/apcir/search/grouped/) (`spec.py`, `block_source.py`, `runner.py`, `merge.py`).

## ⚙️ Evaluation Parameters

All knobs are documented inline in the `search()` docstring of [`src/apcir/search/search.py`](src/apcir/search/search.py) and exposed as `argparse` flags in [`src/apcir/evaluate/evaluation.py`](src/apcir/evaluate/evaluation.py). The most relevant groups:

| Group | Key parameters |
| --- | --- |
| General | `collection`, `topics`, `input_query_path`, `qrel_file_path`, `output_dir_path`, `run_rag`, `run_eval`, `save_ranking_list` |
| Retrieval | `retrieval_model`, `retrieval_query_type`, `retrieval_top_k`, `personalization_group` |
| Dense / FAISS | `dense_query_encoder_path`, `dense_index_dir_path`, `embed_dim`, `passage_block_num`, `faiss_n_gpu`, `use_gpu_for_faiss`, `query_encoder_batch_size` |
| Fusion | `fusion_type`, `QRs_to_rank`, `fuse_weights`, `fusion_normalization`, `optimize_level_weights` |
| Reranking | `reranker`, `rerank_top_k`, `reranking_query_type`, `window_size`, `step`, `rerank_quant` |
| Generation | `generation_model`, `generation_prompt`, `generation_top_k`, `generation_query_type` |
| Metrics | `metrics`, `metrics_to_print`, `target_metrics` |

See the YAML files under `src/apcir/evaluate/` (e.g. `fuse_then_eval_config_23.yaml`, `fuse_then_eval_config_qwen_conv.yaml`) for complete, runnable examples.
