import pickle
import sys
import logging
import argparse
import os
import numpy as np
import json
from typing import Mapping, Tuple, List, Optional, Union, Any, Dict
from tqdm import tqdm
from dataclasses import asdict
import random
import pickle

from pyserini.search import get_topics, get_qrels
import pytrec_eval

sys.path.append('/data/rech/huiyuche/TREC_iKAT_2024/src/')
#sys.path.append('../')


from topics import (
    Turn, 
    Result,
    Reformulation,
    save_turns_to_json, 
    load_turns_from_json,
    filter_ikat_23_evaluated_turns,
    get_turn_by_qid
    )

from constants import IKAT_AUTOMATIC_RUN_TEMPLATE_DICT



# Get the filename without extension nor parent directory
def extract_filename(path):
    # Extract the filename with extension
    filename_with_ext = os.path.basename(path)
    # Remove the extension
    filename, ext = os.path.splitext(filename_with_ext)
    return filename

# get query list & qid list for pyserirni batch search.
def get_query_list(args):

    turn_list = []
    query_list = []
    qid_list_string = []
    reranking_query_list = []
    generation_query_list = []

    '''
    Arguments:
    args.topics: str
    args.input_query_path: str
    args.retrieval_query_type: str
    args.reranking_query_type: str
    args.generation_query_type: str
    args.fb_terms: int
    args.original_query_weight: float
    args.fusion_type: str
    args.QRs_to_rank: List[str]
    args.fuse_weights: List[float]
    args.fusion_query_lists: List[List[str]]

    Returns:
    retrieval_query_list: List[str]
    reranking_query_list: List[str]
    generation_query_list: List[str]
    fusion_query_lists: List[List[str]]
    qid_list_string: List[str]
    turn_list: List[Turn]

    '''

    # apply topic specific processing
    if "ikat" in args.topics:
        turn_list = load_turns_from_json(
            input_topic_path=args.input_query_path,
            range_start=0,
            range_end=-1
            )
        
        # filter out the non-evaluated turns for ikat 23
        if args.topics == "ikat_23_test":
            evaluated_turn_list = filter_ikat_23_evaluated_turns(turn_list)
        elif args.topics == "ikat_24_test":
            evaluated_turn_list = turn_list

        qid_list_string = [str(turn.turn_id) for turn in evaluated_turn_list]


        # load query/reformulated query according to query type.
        # possible to call a llm to rewrite the query at this step.
        retrieval_query_list = [turn.query_type_2_query(args.retrieval_query_type, args.fb_terms, args.original_query_weight) for turn in evaluated_turn_list]
        reranking_query_list = [turn.query_type_2_query(args.reranking_query_type , args.fb_terms, args.original_query_weight) for turn in evaluated_turn_list]
        generation_query_list = [turn.query_type_2_query(args.generation_query_type, args.fb_terms, args.original_query_weight) for turn in evaluated_turn_list]
    

        if args.fusion_type != "none":
            fusion_query_lists = []
            for QR_name in args.QRs_to_rank:
                fusion_query_lists.append([turn.query_type_2_query(QR_name, args.fb_terms, args.original_query_weight) for turn in evaluated_turn_list])
        else:
            fusion_query_lists = None


    assert len(retrieval_query_list) != 0, "No queries found, args.topics may be wrong"
    assert len(retrieval_query_list) == len(qid_list_string), "Number of queries and qid_list_string not match"

    
    return retrieval_query_list, reranking_query_list, generation_query_list, fusion_query_lists, qid_list_string, turn_list



def evaluate(
    run: dict,
    qrel_file_path: str,
    ranking_list_path: str,
    metrics_list: List[str],
    metrics_list_key_form: List[str]
):

    '''
    Evaluate the ranking list using pytrec_eval.
    Args:
        run (dict): ranking list in dictionary format required by pytrec_eval. If None, the ranking list will be read from ranking_list_path.
            - example:     
                     run = {qid: {doc.docid: doc.score for doc in docs} for qid, docs in hits.items()}
        qrel_file_path (str): path to the trec format qrel file
        ranking_list_path (str): path to the trec format ranking list file
        metrics_list (List[str]): list of metrics to evaluate
            - example: ["map", "ndcg_cut.10", "P.5"]
        metrics_list_key_form (List[str]): list of metrics in key form (change . to _)
            - example: ["map", "ndcg_cut_10", "P_5"]
    
    Returns:
        query_metrics_dic (dict): metrics for each query
            - example:          
                query_metrics_dic = {       
                    "qid1" : {"ndcg_cut_10" : 0.1, "map" : 0.2},
                    "qid2" : {"ndcg_cut_10" : 0.2, "map" : 0.3},
                    "qid3" : {"ndcg_cut_10" : 0.3, "map" : 0.4},
                }
        averaged_metrics (dict): averaged metrics
            - example:
                averaged_metrics = {
                    "ndcg_cut_10" : 0.1,
                    "map" : 0.2,
                }
    '''

    # read qrels
    with open(qrel_file_path, 'r') as f_qrel:
        qrel = pytrec_eval.parse_qrel(f_qrel)
    # read ranking list
    if run is None:
        with open(ranking_list_path, 'r') as f_run:
            run = pytrec_eval.parse_run(f_run)

    #  evaluate
    print("trec_eval evaluating...")
    evaluator = pytrec_eval.RelevanceEvaluator(qrel, set(metrics_list))
    query_metrics_dic = evaluator.evaluate(run)

    # average metrics
    '''
    example:
    metrics = { 
        "ndcg_cut_10" : [0.1, 0.2, 0.3, 0.4, 0.5], 
        "map" : [0.1, 0.2, 0.3, 0.4, 0.5],
            }
    '''
    metrics = {metric : [metrics[metric] for metrics in query_metrics_dic.values()] for metric in metrics_list_key_form}   

    averaged_metrics = { metric : np.average(metric_list) for metric, metric_list in metrics.items() }

    # for each query, add the number of relevant documents in query_metrics_dic
    # why not use sum(list(qrel[qid].values()))? Because relevance judgement may be graded instead of binary.
    for qid in query_metrics_dic.keys():
        query_metrics_dic[qid]["num_rel"] = sum([1 for doc in qrel[qid].values() if doc > 0])

    print("################# Retrieval eval Results #################")
    print(json.dumps(averaged_metrics, indent=4))
    print("##########################################################")

    return query_metrics_dic, averaged_metrics


def print_formatted_latex_metrics(metrics_dict, metrics_list):
    '''
    usage example:
    print_formateed_latex_metrics(
        {"ndcg_cut_10": 0.1, "map": 0.2}, 
        ["ndcg_cut_10", "map"]
        )
    '''

    # Calculate the metrics and format them
    result = []
    for metric in metrics_list:
        value = metrics_dict.get(metric, 0) * 100
        result.append(f"{value:.1f}")
    
    # Print the result separated by tab
    print(" & ".join(result))


def generate_and_save_ikat_submission(
    ikat_output_path: str,
    run_name: str,
    reformulation_name: str,
    hits: Dict[str, List[Any]],
    turn_list : List[Turn],
    response_dict: Dict[str, List[str]],
    top_k: int
    ) -> None:

    # resulting dictionary
    result_dict = IKAT_AUTOMATIC_RUN_TEMPLATE_DICT
    result_dict["run_name"] = run_name

    for qid, ordered_doc_object_list in hits.items():

        three_chiffres = qid.split("-")
        # adapt turn_id format. work for both ikat23 and ikat24
        conversation_id = "-".join(three_chiffres[:-1])
        turn_id = f"{conversation_id}_{three_chiffres[-1]}"
        
        # get ptkb_provenance, which should be from the reformulation.
        turn_object = get_turn_by_qid(qid,turn_list)

        #ptkb_provenance = turn_object.get_ptkb_provenance(reformulation_name)
        # TODO: add ptkb_provenance
        ptkb_provenance = []

        responses = []
        for rank, response in enumerate(response_dict[qid]):
            real_rank = rank + 1
            responses.append(
                {
                    "rank": real_rank,
                    "text": response,
                    "ptkb_provenance": ptkb_provenance,
                    "passage_provenance": [
                        {
                            "id": doc_object.docid,
                            "score": doc_object.score,
                            "used": False if i >= top_k else True
                        } for i, doc_object in enumerate(ordered_doc_object_list)
                    ]
                }
            )
        

        result_dict["turns"].append(
            {
                "turn_id": turn_id,
                "responses": responses
            }
            )
        
    with open(ikat_output_path, "w") as f:
        json.dump(result_dict, f, indent=4)



