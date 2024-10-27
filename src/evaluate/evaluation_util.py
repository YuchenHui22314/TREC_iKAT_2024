import pickle
import sys
import logging
import argparse
import os
import numpy as np
import json
from functools import reduce
from collections import defaultdict
import time
from progressbar import *
from typing import Mapping, Tuple, List, Optional, Union, Any, Dict
from tqdm import tqdm
from dataclasses import asdict
import random
import pickle

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

from rank_gpt import run_retriever, sliding_windows

from rerank import (
    load_rankllama, 
    rerank_rankllama,
    load_t5_DDP,
    rerank_t5_DDP,
    load_t5_DP,
    rerank_t5_DP,
    hits_2_rankgpt_list
    )

from constants import IKAT_AUTOMATIC_RUN_TEMPLATE_DICT
    
from response_generation import (
    generate_responses
    ) 

from pyserini.search.lucene import LuceneSearcher
from pyserini.search import FaissSearcher
from pyserini.search import get_topics, get_qrels
import pytrec_eval

class PyScoredDoc:
    def __init__(self, docid: str, score: float):
        self.docid = docid
        self.score = score
    
    def __repr__(self):
        return f"docid: {self.docid}, score: {self.score}"


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
    load turns from json file. Output format:
    retrieval_query_list: List[str] 
    reranking_query_list: List[str]
    generation_query_list: List[str]
    qid_list_string: List[str]
    turn_list: List[Turn] 
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


# 1st stage retrieval + 2nd stage reranking
def search(
    args: Any
    ) -> Dict[str, List[PyScoredDoc]]:
    """
    Perform search and retrieval using different models and rerankers.

    Args:
        args (Any): Additional arguments.

    Returns:
        Dict[str, List[Any]]: Pyserini hits object, where "PyScoredDoc" similar to an Anserini hit object. The Lists would be reranked if reranker is used. 
    """

    '''
    All required arguments are:
        - args.retrieval_query_list: List[str]: List of retrieval queries.
        - args.reranking_query_list: List[str]: List of reranking queries.
        - args.qid_list_string: List[str]: List of query IDs.
        - args.run_name: str
        - args.file_name_stem: str
        - args.ranking_list_path: str
        - args.save_ranking_list: bool
        - args.given_ranking_list_path: str
        - args.index_dir_path: str
    # Fusion
        - args.fusion_type: str
        - args.QRs_to_rank: List[str]
        - args.fuse_weights: List[float]
        - args.fusion_query_lists: List[List[str]]
    # Sparse
        - args.retrieval_model: str
        - args.retrieval_top_k: int
        - args.bm25_k1: float
        - args.bm25_b: float
        - args.qe_type: str
        - args.fb_terms: int
        - args.fb_docs: int
        - args.original_query_weight: float
    # Dense
        - args.dense_query_encoder_path: str
        - args.index_dir_path: str
    # Reranker
        - args.reranker: str
        - args.rerank_top_k: int
        # RankGPT
            - args.step: int
            - args.window_size: int
            - args.rankgpt_llm: str
        # Rankllama
            - args.rerank_quant: str
            - args.cache_dir: str

    
    '''

    ##########################################################################################
    # we have the possibility to load a custom ranking list instead of searching or reranking
    ##########################################################################################

    if args.retrieval_model == "none":
        assert args.given_ranking_list_path != "none", " --given_ranking_list_path should be provided when --run_from_rerank or --run_from_generate is true, because we do not do retrieval/retrieval+reranking in these cases."

        # even we do not search, we have to get access to the index (raw documents via a searcher)
        searcher = LuceneSearcher(args.index_dir_path)
        # load the ranking list
        with open(args.given_ranking_list_path, "r") as f:
            run = pytrec_eval.parse_run(f)
            hits = {qid: [PyScoredDoc(docid, score) for docid, score in docs.items()] for qid, docs in run.items()}
        #sort the hits by score
        for qid in hits.keys():
            hits[qid] = sorted(hits[qid], key=lambda x: x.score, reverse=True)

        

    #######################
    # First stage retrieval
    #######################

    # No fusion
    if args.fusion_type == "none":
        hits = Retrieval(args)
    # fusion 1: linear weighted score
    elif args.fusion_type == "linear_weighted_score":
        assert len(args.QRs_to_rank) -1 == len(args.fuse_weights), "The number of QRs to fuse should be one more than the number of weights."
        fuse_weights = [None] + args.fuse_weights       # the first weight would not be used, just a dummy value.
        print("fusing ranking lists with linear weighted score...")
        
        def linear_weighted_score_fusion_reduce_function(hits_0_and_weight, hits_1_and_weight):
            hits_0 = hits_0_and_weight[0] # 0 is the query, 1 is the weight
            hits_1 = hits_1_and_weight[0] # 0 is the query, 1 is the weight
            hits_and_weight_new = linear_weighted_score_fusion(
                hits_0, 
                hits_1, 
                args.retrieval_top_k, 
                None,
                hits_1_and_weight[1],  # the weight associated with the second query is the right one to use (just a design choice)
                args.run_name)
            return hits_and_weight_new
        
        # search for all queries to get the hits
        hits_list = []
        for QR in args.fusion_query_lists:
            args.retrieval_query_list = QR
            hits_list.append(Retrieval(args))

        # make a list of tuples, each tuple contains a query and a weight
        hits_and_weights = list(zip(hits_list, fuse_weights))

        # get the fused ranking list
        hits = reduce(linear_weighted_score_fusion_reduce_function, hits_and_weights)[0]
    
    # fusion2: lottery fusion
    elif args.fusion_type == "round_robin":
        print("fusing ranking lists with round robin...")
        # first search.
        hits_list = []
        for QR in args.fusion_query_lists:
            args.retrieval_query_list = QR
            hits_list.append(Retrieval(args))
        
        # round robin fusion. Random selet at each run.
        hits = round_robin_fusion(hits_list, args.retrieval_top_k, 42)
    
    # fusion3: linear combination.
    # difference with linear_weighted_score is that now we can specify the weight of the 1st query. More general than linear_weighted_score. 
    # I should have directly imiplemented this. 
    elif args.fusion_type == "linear_combination":
        print("fusing ranking lists with linear combination...")
        # first search.
        hits_list = []
        for QR in args.fusion_query_lists:
            args.retrieval_query_list = QR
            hits_list.append(Retrieval(args))
        
        # check weight length
        assert len(args.fuse_weights) == len(args.fusion_query_lists), "The number of weights should be equal to the number of query lists."

        # use same weights for all queries
        qid_weights_dict = {qid: args.fuse_weights for qid in args.qid_list_string}

        # linear combination fusion
        hits = per_query_linear_combination(hits_list, qid_weights_dict, args.retrieval_top_k)



    ##############################
    # reranking
    ##############################

    if not args.reranker == "none":
        hits = rerank(hits, args)

    ##############################
    # save ranking list 
    ##############################


    #sort the hits by score
    for qid in hits.keys():
        hits[qid] = sorted(hits[qid], key=lambda x: x.score, reverse=True)

    # generate run dictionary required by pytrec_eval
    run = {qid: {doc.docid: doc.score for doc in docs} for qid, docs in hits.items()}


    # save ranking list
    # format: query-id Q0 document-id rank score run_name
    if args.save_ranking_list:
        if args.run_name == "none":
            run_name = args.file_name_stem
        else:
            run_name = args.run_name 

        with open(args.ranking_list_path, "w") as f:
            for qid in args.qid_list_string:
                for i, item in enumerate(hits[qid]):
                    f.write("{} {} {} {} {} {}".format(
                        qid,
                        "Q0",
                        item.docid,
                        i+1,
                        item.score,
                        run_name
                        ))
                    f.write('\n')

    return hits, run

############## First stage retrieval ###############
def Retrieval(args):

    '''
    All required arguments are:
        - args.retrieval_query_list: List[str]
        - args.qid_list_string: List[str]
        - args.index_dir_path: str
    # Sparse
        - args.retrieval_model: str
        - args.retrieval_top_k: int
        - args.bm25_k1: float
        - args.bm25_b: float
        - args.qe_type: str
        - args.fb_terms: int
        - args.fb_docs: int
        - args.original_query_weight: float
    # Dense
        - args.dense_query_encoder_path: str
    '''

    # sparse search
    if args.retrieval_model == "BM25":
        print("BM 25 searching...")
        searcher = LuceneSearcher(args.index_dir_path)
        searcher.set_bm25(args.bm25_k1, args.bm25_b)

        # rm3 pseudo relevance feedback
        if args.qe_type == "rm3":
            searcher.set_rm3(
                fb_terms = args.fb_terms,
                fb_docs = args.fb_docs,
                original_query_weight = args.original_query_weight
            )
                
        print("the length of retrieval_query_list is ", len(args.retrieval_query_list))
        hits = searcher.batch_search(args.retrieval_query_list, args.qid_list_string, k = args.retrieval_top_k, threads = 40)

    # dense search
    elif args.retrieval_model in ["ance", "dpr"]:
        print(f"{args.retrieval_model} searching...")
        searcher = FaissSearcher(
            args.index_dir_path,
            args.dense_query_encoder_path 
        )
        hits = searcher.batch_search(args.retrieval_query_list, args.qid_list_string, k = args.retrieval_top_k, threads = 40)
    

    ##############################
    # TODO: add splade
    ##############################

    return hits

###################### Reranking ######################
def rerank(hits, args):

    """
    Perform reranking on hits objects.

    Args:
        hits (Dict[str, List[Any]): Pyserini hits object, or a "PyScoredDoc" similar to an Anserini hit object. Must include .docid and .score.
        args (Any): Additional arguments.

    Returns:
        hits (Dict[str, List[Any]): Pyserini hits object, or a "PyScoredDoc" similar to an Anserini hit object. Must include .docid and .score. The result list should be sorted 
    """

    '''
    All required arguments are:
        - args.reranking_query_list: List[str]: List of reranking queries.
        - args.reranker: str
        - args.rerank_top_k: int
        - args.qid_list_string: List[str]: List of query IDs.
        - args.index_dir_path: str path to the pyserini index.
        # RankGPT
            - args.step: int
            - args.window_size: int
            - args.rankgpt_llm: str
        # Rankllama
            - args.rerank_quant: str
            - args.cache_dir: str  (also applied for T5)

    
    '''

    print(f"{args.reranker} reranking top {args.rerank_top_k}...")

    # generate a qid-reranking_query dictionary
    reranking_query_dic = {qid: reranking_query for qid, reranking_query in zip(args.qid_list_string, args.reranking_query_list)}

    searcher = LuceneSearcher(args.index_dir_path)

    if args.reranker == "rankgpt":


        # generate input format required by rankgpt
        rank_gpt_list, _ = hits_2_rankgpt_list(searcher, reranking_query_dic, hits)

        # get hyperparameters
        llm_name = args.rankgpt_llm
        rank_end = args.rerank_top_k
        step = args.step
        window_size = args.window_size
        if "gpt" in llm_name:
            token = os.getenv('openai_key')
        elif "claude" in llm_name:
            token = os.getenv('claude_key')
        else:
            raise NotImplementedError(f"llm_name {llm_name} not implemented")
        
        print("reranking")
        # for every query:
        for item in tqdm(
            rank_gpt_list, 
            desc="Ranking with rankgpt", 
            unit="query", 
            total=len(rank_gpt_list)
            ):

            new_item = sliding_windows(
                item, 
                rank_start=0, 
                rank_end=rank_end, 
                window_size=window_size,
                step=step,
                model_name=llm_name, 
                api_key=token)

            qid = new_item["hits"][0]["qid"]
            assert len(hits[qid]) == len(new_item["hits"]), f"retrieval length should be equal to reranking length. {len(hits[qid])} != {len(new_item['hits'])}"

            # sort hits[qid] to ensure the descending order
            hits[qid] = sorted(hits[qid], key=lambda x: x.score, reverse=True)

            # update doc id in the ranking list
            for i in range(len(hits[qid])):
                hits[qid][i].docid = new_item["hits"][i]["docid"]


    elif args.reranker == "rankllama":

        if args.rerank_quant == "none":
            quant_4bit = False
            quant_8bit = False
        elif args.rerank_quant == "8b":
            quant_4bit = False
            quant_8bit = True
        elif args.rerank_quant == "4b":
            quant_4bit = True
            quant_8bit = False

        print("loading rankllama model")
        tokenizer, model = load_rankllama(
            args.cache_dir,
            quant_8bit = quant_8bit,
            quant_4bit = quant_4bit
            )

        print("reranking")
        for qid, hit in tqdm(hits.items(), total=len(hits), desc="Reranking"):

            reranking_query = reranking_query_dic[qid]
            reranked_scores = rerank_rankllama(
                reranking_query,
                [json.loads(searcher.doc(doc_object.docid).raw())["contents"] for doc_object in hit[0:args.rerank_top_k]],
                tokenizer,
                model
            )

            np_reranked_scores = np.array(reranked_scores, dtype=np.float32)

            indexes = np.argsort(np_reranked_scores)[::-1]
            for rank, index in enumerate(indexes):
                hit[index].rank = rank 
            
            # change the score according to the rank
            for rank, doc_object in enumerate(hit):
                if rank < args.rerank_top_k:
                    doc_object.score = 1/(doc_object.rank + 1)
                else:
                    doc_object.score = 1/(rank + 1)
            
            # sort the hits by score
            hits[qid] = sorted(hit, key=lambda x: x.score, reverse=True)

    elif "monot5" in args.reranker:

        # get reranker_name
        if args.reranker == "monot5_base":
            reranker_name = "castorini/monot5-base-msmarco"
        elif args.reranker == "monot5_base_10k":
            reranker_name = "castorini/monot5-base-msmarco-10k"
        else:
            raise NotImplementedError(f"reranker {args.reranker} not implemented")

        # load model
        print("loading t5 model")
        tokenizer, model, decoder_stard_id, targeted_ids =\
             load_t5_DP(args.cache_dir, reranker_name)

        print("reranking")
        for qid, hit in tqdm(hits.items(), total=len(hits), desc="Reranking"):

            reranking_query = reranking_query_dic[qid]

            reranked_scores = rerank_t5_DP(
                reranking_query,
                [json.loads(searcher.doc(doc_object.docid).raw())["contents"] for doc_object in hit[0:args.rerank_top_k]],
                tokenizer,
                model,
                decoder_stard_id,
                targeted_ids,
            )

            np_reranked_scores = np.array(reranked_scores, dtype=np.float32)

            indexes = np.argsort(np_reranked_scores)[::-1]
            for rank, index in enumerate(indexes):
                hit[index].rank = rank 
            
            # change the score according to the rank
            for rank, doc_object in enumerate(hit):
                if rank < args.rerank_top_k:
                    doc_object.score = 1/(doc_object.rank + 1)
                else:
                    doc_object.score = 1/(rank + 1)
            
            # sort the hits by score
            hits[qid] = sorted(hit, key=lambda x: x.score, reverse=True)
        
    return hits

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
    query_metrics_dic = {
        "qid1" : {"ndcg_cut_10" : 0.1, "map" : 0.2},
        "qid2" : {"ndcg_cut_10" : 0.2, "map" : 0.3},
        "qid3" : {"ndcg_cut_10" : 0.3, "map" : 0.4},
    }
    '''
    metrics = {metric : [metrics[metric] for metrics in query_metrics_dic.values()] for metric in metrics_list_key_form}   

    averaged_metrics = { metric : np.average(metric_list) for metric, metric_list in metrics.items() }

    # for each query, add the number of relevant documents in query_metrics_dic
    # why not use sum(list(qrel[qid].values()))? Because relevance judgement may be graded instead of binary.
    for qid in query_metrics_dic.keys():
        query_metrics_dic[qid]["num_rel"] = sum([1 for doc in qrel[qid].values() if doc > 0])

    return query_metrics_dic, averaged_metrics


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




######################### Ranking list fusion #########################
############# adapted from fuse.py in TREC_iKAT_2024/src #############


def linear_weighted_score_fusion(
    hits_0=None, 
    hits_1=None, 
    topk=1000, 
    output=None, 
    alpha=0.1, 
    run_name='fusion'):
    """
    Performs rank list fusion by combining results from dense and sparse retrieval methods. 
    This function reads ranked lists from two input files, fuses the rankings based on the 
    provided alpha value, and writes the fused results to an output file.

    Arguments:
    - hits_0 (dict): pyserini hits object. Each element inside must can .docid and .score
    - hits_1 (dict): pyserini hits object. Each element inside must can .docid and .score
    - topk (int): Number of hits to retrieve for each query. Default is 1000.
    - output (str): Path to the output file where fused rankings will be saved.
                    The output format will be: qid Q0 docid rank score run_name.
    - alpha (float): Weighting factor for the sparse retrieval scores in the fusion process. 
                     Default is 0.1. Higher alpha gives more weight to sparse retrieval scores.
    - run_name (str): Identifier for the current run, used in the output file. Default is 'fusion'.
    
    Returns:
    - None. The function writes the output directly to the specified output file.
    """
    
    def read_rank_list(hits):
        """Reads a rank list from the a hits object and returns document IDs and scores."""

        qid_docid_list = defaultdict(list)
        qid_score_list = defaultdict(list)

        for qid, docs in hits.items():
            for doc in docs:
                qid_docid_list[qid].append(doc.docid)
                qid_score_list[qid].append(doc.score)
        return qid_docid_list, qid_score_list

    def fuse(docid_list0, docid_list1, doc_score_list0, doc_score_list1, alpha):
        """Fuses the rank lists from dense and sparse retrieval based on the alpha value."""
        score = defaultdict(float)
        score0 = defaultdict(float)
        for i, docid in enumerate(docid_list0):
            score0[docid] += doc_score_list0[i]
        min_val0 = min(doc_score_list0)
        min_val1 = min(doc_score_list1)
        for i, docid in enumerate(docid_list1):
            if score0[docid] == 0:
                score[docid] += min_val0 + doc_score_list1[i] * alpha
            else:
                score[docid] += doc_score_list1[i] * alpha
        for i, docid in enumerate(docid_list0):
            if score[docid] == 0:
                score[docid] += min_val1 * alpha
            score[docid] += doc_score_list0[i]
        score = {k: v for k, v in sorted(score.items(), key=lambda item: item[1], reverse=True)}
        return score

    # read hits
    print('Read ranked list0...')
    qid_docid_list0, qid_score_list0 = read_rank_list(hits_0)
    print('Read ranked list1...')
    qid_docid_list1, qid_score_list1 = read_rank_list(hits_1)

    # final hits list
    final_hits_dict = defaultdict(list)

    qids = qid_docid_list0.keys()
    widgets = ['Progress: ', Percentage(), ' ', Bar('#'),' ', Timer(),
               ' ', ETA(), ' ', FileTransferSpeed()]
    pbar = ProgressBar(widgets=widgets, maxval=len(qids)).start()
    start_time = time.time()
    
    for j, qid in enumerate(qids):
        rank_doc_score = fuse(qid_docid_list0[qid], qid_docid_list1[qid], qid_score_list0[qid], qid_score_list1[qid], alpha)
        for rank, doc in enumerate(rank_doc_score):
            if rank == topk:
                break
            score = rank_doc_score[doc]
            final_hits_dict[qid].append(PyScoredDoc(doc, score))
        pbar.update(j + 1)
    
    time_per_query = (time.time() - start_time) / len(qids)
    print('Fusing {} queries ({:0.3f} s/query)'.format(len(qids), time_per_query))
    print('Finished.')

    return (final_hits_dict, None)


def per_query_linear_combination(hits_list, qid_weights_dict, top_k):
    """
    perform linear combination of ranking list. Each query may have a different group
    of weights. 
    
    Arguments:
    - hits_list (List[dict]): List of pyserini hits objects. Each element inside must can .docid and .score
    - qid_weights_dict (dict): dictionary of str : List[float]. weights assignment for each query.
    - topk (int): Number of hits to retrieve for each query. Default is 1000.
    """

    def hits_2_qid_docid_score_dict(hits):
        """
        The resulting structure would be: {qid: {docid: score}}
        """

        qid_docid_score_dict = defaultdict(dict)

        for qid, docs in hits.items():
            for doc in docs:
                qid_docid_score_dict[qid][doc.docid] = doc.score

        return qid_docid_score_dict 

    def fuse_one_query(docid_score_dict_list, alpha_list):
        """
        fuse the candidate ranking lists of one query.
        docid_score_dict_list: List[dict]: List of {docid: score} dictionaries
        alpha_list: List[float]: weights for each candidate ranking list
        """

        set_of_all_possible_docids = \
            set([docid for docid_score_dict in docid_score_dict_list for docid in docid_score_dict.keys()])

        # for each candidate ranking list, fill in the missing docids with the minimum score
        for docid_score_dict in docid_score_dict_list:
            min_val = min(docid_score_dict.values())
            for docid in set_of_all_possible_docids:
                if docid not in docid_score_dict:
                    docid_score_dict[docid] = min_val 
        # fuse the ranking lists by weighted sum
        fused_docid_score_dict = defaultdict(float)
        for docid in set_of_all_possible_docids:
            for i, docid_score_dict in enumerate(docid_score_dict_list):
                fused_docid_score_dict[docid] += docid_score_dict[docid] * alpha_list[i]
        
        return fused_docid_score_dict
    
    # read hits for each candidate ranking list
    all_qid_docid_score_dict_list = [hits_2_qid_docid_score_dict(hits) for hits in hits_list]
    qids = all_qid_docid_score_dict_list[0].keys()

    # Fusion
    final_hits_dict = defaultdict(list)
    for qid in qids:
        docid_score_dict_list = [qid_docid_score_dict[qid] for qid_docid_score_dict in all_qid_docid_score_dict_list]
        alpha_list = qid_weights_dict[qid]
        fused_docid_score_dict = fuse_one_query(docid_score_dict_list, alpha_list)
        # sort the fused ranking list
        fused_docid_score_dict = {k: v for k, v in sorted(fused_docid_score_dict.items(), key=lambda item: item[1], reverse=True)}
        for rank, doc in enumerate(fused_docid_score_dict):
            if rank == top_k:
                break
            final_hits_dict[qid].append(PyScoredDoc(doc, fused_docid_score_dict[doc]))

    return final_hits_dict


def round_robin_fusion(hits_list, topk, random_seed):
    """
    Performs rank list fusion by combining results from multiple retrieval methods using round robin.
    This function reads ranked lists from multiple input, fuses the rankings based on the 
    round robin strategy, and returns the fused results.

    Arguments:
    - hits_list (List[dict]): List of pyserini hits objects. Each element inside must can .docid and .score
    - topk (int): Number of hits to retrieve for each query. Default is 1000.
    
    Returns:
    - dict: The fused ranking list.
    """
    final_hits_dict = defaultdict(list)
    qids = hits_list[0].keys()
    for qid in qids:
        for rank in range(topk):
            candidate_docs = [hits[qid][rank] for hits in hits_list]
            # shuffle the candidate docs randomly
            random.seed(random_seed)
            random.shuffle(candidate_docs)
            for doc in candidate_docs:
                # check if doc is already in the final list
                if doc.docid not in [d.docid for d in final_hits_dict[qid]]:
                    final_hits_dict[qid].append(doc)
            if len(final_hits_dict[qid]) >= topk:
                # cut to topk
                final_hits_dict[qid] = final_hits_dict[qid][:topk]
                # reassign scores according to the rank
                for i, doc in enumerate(final_hits_dict[qid]):
                    doc.score = 1/(i + 1)
                break

    return final_hits_dict