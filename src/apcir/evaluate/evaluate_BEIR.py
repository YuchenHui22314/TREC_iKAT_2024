import os
import sys
import types
import pickle

import torch
from transformers import AutoTokenizer
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

from convdr.drivers.gen_passage_embeddings import load_model
from apcir.functional.llm import BeirConvdrEncoder

args = types.SimpleNamespace()

args.model_type = "dpr"  
args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args.cache_dir = None  
args.local_rank = -1  

config, tokenizer, model = load_model(args, "/data/rech/huiyuche/huggingface/convdr/convdr-multi-orquac.cp")    

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") 
tokenizer.model_max_length = 512

convdr_model = BeirConvdrEncoder(
    model,
    tokenizer,
    device=args.device
    )

custom_model = DRES(
    convdr_model, batch_size=256
)

retriever = EvaluateRetrieval(custom_model, score_function="dot")

base_path = "/data/rech/huiyuche/beir"

dataset_list = [
    "msmarco",
    "trec-covid",
    "nfcorpus",
    # "bioasq",     xxxxxx
    "nq",
    "hotpotqa",
    "fiqa",
    # "signal1m",xxxxxx
    # "trec-news",xxxxxx
    "arguana",
    "webis-touche2020",
    "cqadupstack",
    "quora",
    "dbpedia-entity",
    "scidocs",
    "fever",
    "climate-fever",
    "scifact"
]

result_dict = {}

for data_path in dataset_list:
    data_path = "scifact"
    corpus, queries, qrels = GenericDataLoader(os.path.join(base_path,data_path,data_path)).load(split="test") # or split = "train" or "dev"

    #### Retrieve dense results (format of results is identical to qrels)
    results = retriever.retrieve(corpus, queries)
    result_dict[data_path] = results

    with open(f"/data/rech/huiyuche/TREC_iKAT_2024/results/beir/ranking/convdr_results_up_to_{data_path}.pkl", "wb") as f:
        pickle.dump(result_dict, f)
    
    
    print("###############################")
    print("###############################")
    print("Results for dataset: {}".format(data_path))
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
    
    with open(f"/data/rech/huiyuche/TREC_iKAT_2024/results/beir/metrics/convdr_metrics.txt", "a") as f:
        print("###############################")
        print("###############################")
        f.write("Results for dataset: {}\n".format(data_path))
        f.write("NDCG: {}\n".format(ndcg))
        f.write("MAP: {}\n".format(_map))
        f.write("Recall: {}\n".format(recall))
        f.write("Precision: {}\n\n".format(precision))



