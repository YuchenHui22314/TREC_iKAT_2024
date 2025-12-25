import os
import sys
import types
import pickle
import argparse
import json

import torch
from transformers import AutoTokenizer

from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval import models

from convdr.drivers.gen_passage_embeddings import load_model
from apcir.functional.llm import BeirConvdrEncoder,BeirCLSEncoder, BeirMPoolingEncoder, BeirANCEEncoder

parser = argparse.ArgumentParser(description="Evaluate BEIR datasets with ance model")
parser.add_argument(
    "--split",
    type=int,
    default=0,
    help="the split of BEIR datasets to evaluate: 0 - first split, 1 - second split, 2 - third split, 3 - fourth split",
)
args = parser.parse_args()


args.device = torch.device(f"cuda:{int(args.split) % 4}" if torch.cuda.is_available() else "cpu")
args.embedding_dir = "/data/rech/huiyuche/beir/embeddings/ance"


### ance
encoder = BeirANCEEncoder(
    model_path="/data/rech/huiyuche/huggingface/models--castorini--ance-msmarco-passage/snapshots/6d7e7d6b6c59dd691671f280bc74edb4297f8234",
    device=args.device,
    max_length_query=512,
    max_length_doc=512
)

# Then plug into BEIR

model = DRES(encoder, batch_size=1400)

retriever = EvaluateRetrieval(model, score_function="dot")

base_path = "/data/rech/huiyuche/beir"


dataset_list_01 = [
    #"cqadupstack",
    "scifact",
    "trec-covid",
    "nfcorpus",
    # "bioasq",     xxxxxx
    "fiqa",
    # "signal1m",xxxxxx
    # "trec-news",xxxxxx
    "arguana",
    "webis-touche2020",
    "quora",
    "scidocs"
    # "msmarco",
    # "nq",
    # "hotpotqa",
    # "dbpedia-entity",
    # "fever",
    # "climate-fever"
]

# dataset_list_01 = [
#     "cqadupstack"
# ]

dataset_list_02 = [
    # "scifact",
    # "trec-covid",
    # "nfcorpus",
    # "bioasq",     xxxxxx
    # "fiqa",
    # "signal1m",xxxxxx
    # "trec-news",xxxxxx
    # "arguana",
    # "webis-touche2020",
    # "cqadupstack",
    # "quora",
    "msmarco",
    "dbpedia-entity"
]

dataset_list_03 = [
    "fever",
    "climate-fever"
    ]

dataset_list_04 = [
    "nq",
    "hotpotqa"
    ]

dataset_list_05 = [
    "msmarco",
    "scifact",
    "trec-covid",
    "nfcorpus",
    # "bioasq",     xxxxxx
    "fiqa",
    # "signal1m",xxxxxx
    # "trec-news",xxxxxx
    "arguana",
    "webis-touche2020",
    "quora",
    "scidocs",
    "nq",
    "hotpotqa",
    "dbpedia-entity",
    "fever",
    "climate-fever",
    "cqadupstack",
]

dataset_list_06 = [
    "trec-covid",
]

if args.split == 0:
    dataset_list = dataset_list_01
elif args.split == 1:
    dataset_list = dataset_list_02
elif args.split == 2:
    dataset_list = dataset_list_03
elif args.split == 3:
    dataset_list = dataset_list_04  
elif args.split == 4:
    dataset_list = dataset_list_05
elif args.split == 5:
    dataset_list = dataset_list_06
else:
    raise ValueError("Invalid split value. It should be 0, 1, 2, or 3.")

result_dict = {}

for data_path in dataset_list:

    args.embedding_dir = os.path.join(args.embedding_dir,data_path)

    if data_path == "cqadupstack":

        base_path = "/data/rech/huiyuche/beir/cqadupstack/cqadupstack"

        query_num_metric_dict = {}
        sub_datasets = [os.path.join(base_path, d) for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
        
        print(sub_datasets[0])

        for sub_data_path in sub_datasets:
            args.embedding_dir = os.path.join("/data/rech/huiyuche/beir/embeddings/ance/cqadupstack", os.path.basename(sub_data_path))


            print("Loading dataset: {}".format(sub_data_path))
            corpus, queries, qrels = GenericDataLoader(sub_data_path).load(split="test") # or split = "train" or "dev"

            #### Retrieve dense results (format of results is identical to qrels)
            #### ( and save to embedding directory)

            # results = retriever.retrieve(corpus, queries)

            results = retriever.encode_and_retrieve(
                corpus=corpus,
                queries=queries,
                encode_output_path=args.embedding_dir,
                overwrite=False,  # Set to True if you want to overwrite existing embeddings
            )

            result_dict[sub_data_path] = results

            print("###############################")
            print("###############################")
            print("Results for dataset: cqadupstack_{}".format(sub_data_path))
            ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
            
            query_num_metric_dict[sub_data_path] = {
                "num_queries": len(queries),
                "metrics":{
                    **ndcg,
                    **_map,
                    **recall,
                    **precision
                }
            }
        
        # weighted average of metrics according to number of queries
        info_dicts = list(query_num_metric_dict.values())
        total_query_number = sum([info_dict["num_queries"] for info_dict in info_dicts])

        all_metric_keys = set()
        for info_dict in info_dicts:
            all_metric_keys.update(info_dict["metrics"].keys())
            
        weighted_metrics = {}

        for key in sorted(all_metric_keys):
            weighted_sum = sum(
                info_dict["metrics"].get(key, 0) * info_dict["num_queries"]
                for info_dict in info_dicts
            )
            weighted_metrics[key] = weighted_sum / total_query_number


        with open(f"/data/rech/huiyuche/TREC_iKAT_2024/results/beir/metrics/beir_{args.split}_ance_metrics.txt", "a") as f:
            print("###############################")
            print("###############################")
            f.write("Results for dataset: {}\n".format(data_path))
            f.write("Weighted Average Metrics for cqadupstack:\n")
            for metric_name, metric_value in weighted_metrics.items():
                f.write(f"{metric_name}: {metric_value}\n")
            f.write("\n")   
            
        
        continue

    else: 

        base_path = "/data/rech/huiyuche/beir"
    
        corpus, queries, qrels = GenericDataLoader(os.path.join(base_path,data_path,data_path)).load(split="test") # or split = "train" or "dev"

        #### Retrieve dense results (format of results is identical to qrels)
        #### ( and save to embedding directory)
        # results = retriever.retrieve(corpus, queries)
        results = retriever.encode_and_retrieve(
            corpus=corpus,
            queries=queries,
            encode_output_path=args.embedding_dir,
            overwrite=False,  # Set to True if you want to overwrite existing embeddings
        )
        # print(results.keys())

        # save results for debugging:
        
        result_dict[data_path] = results

        with open(f"/data/rech/huiyuche/TREC_iKAT_2024/results/beir/ranking/beir_{args.split}_ance_results_up_to_{data_path}.pkl", "wb") as f:
            pickle.dump(result_dict, f)

        
        
        print("###############################")
        print("###############################")
        print("Results for dataset: {}".format(data_path))
        ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
        
        with open(f"/data/rech/huiyuche/TREC_iKAT_2024/results/beir/metrics/beir_{args.split}_ance_metrics.txt", "a") as f:
            print("###############################")
            print("###############################")
            f.write("Results for dataset: {}\n".format(data_path))
            f.write("NDCG: {}\n".format(ndcg))
            f.write("MAP: {}\n".format(_map))
            f.write("Recall: {}\n".format(recall))
            f.write("Precision: {}\n\n".format(precision))


# python -m apcir.evaluate.evaluate_BEIR_ance --split 0
# python -m apcir.evaluate.evaluate_BEIR_ance --split 1
# python -m apcir.evaluate.evaluate_BEIR_ance --split 2
# python -m apcir.evaluate.evaluate_BEIR_ance --split 3
# python -m apcir.evaluate.evaluate_BEIR_ance --split 4
# python -m apcir.evaluate.evaluate_BEIR_ance --split 5
