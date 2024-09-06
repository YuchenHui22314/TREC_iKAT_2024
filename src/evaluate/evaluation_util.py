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


def search(
    retrieval_query_list: List[str], 
    reranking_query_list: List[str], 
    qid_list_string: List[str], 
    args: Any
    ) -> Dict[str, List[Any]]:
    """
    Perform search and retrieval using different models and rerankers.

    Args:
        retrieval_query_list (List[str]): List of retrieval queries.
        reranking_query_list (List[str]): List of reranking queries.
        qid_list_string (List[str]): List of query IDs.
        args (Any): Additional arguments.

    Returns:
        Dict[str, List[Any]]: Pyserini hits object, where "Any" is a Anserini hit object. The List[Any] would be reranked if reranker is used. 
    """


    '''
    All required arguments are:
        - args.run_name: str
        - args.file_name_stem: str
        - args.ranking_list_path: str
        - args.save_ranking_list: bool
        - args.given_ranking_list_path: str
    # Sparse
        - args.retrieval_model: str
        - args.retrieval_top_k: int
        - args.index_dir_path: str
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

    # we have the possibility to load a custom ranking list instead of searching or reranking
    if args.retrieval_model == "none":
        assert args.given_ranking_list_path != "none", " --given_ranking_list_path should be provided when --run_from_rerank or --run_from_generate is true, because we do not do retrieval/retrieval+reranking in these cases."

        # even we do not search, we have to get access to the index (raw documents via a searcher)
        searcher = LuceneSearcher(args.index_dir_path)
        # load the ranking list
        with open(args.given_ranking_list_path, "r") as f:
            run = pytrec_eval.parse_run(f)
            hits = {qid: [PyScoredDoc(docid, score) for docid, score in docs.items()] for qid, docs in run.items()}

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
                
        hits = searcher.batch_search(retrieval_query_list, qid_list_string, k = args.retrieval_top_k, threads = 40)

    # dense search
    elif args.retrieval_model in ["ance", "dpr"]:
        print(f"{args.retrieval_model} searching...")
        searcher = FaissSearcher(
            args.index_dir_path,
            args.dense_query_encoder_path 
        )
        hits = searcher.batch_search(retrieval_query_list, qid_list_string, k = args.retrieval_top_k, threads = 40)


    ##############################
    # TODO: add splade
    ##############################


    ##############################
    # reranking
    ##############################


    if not args.reranker == "none":

        print(f"{args.reranker} reranking top {args.rerank_top_k}...")
         

        # generate a qid-reranking_query dictionary
        reranking_query_dic = {qid: reranking_query for qid, reranking_query in zip(qid_list_string, reranking_query_list)}

        # generate input format required by rankgpt
        rank_gpt_list, _ = hits_2_rankgpt_list(searcher, reranking_query_dic, hits)

    if args.reranker == "rankgpt":

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
            # hits[qid] = sorted(hit, key=lambda x: x.score, reverse=True)

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
            # hits[qid] = sorted(hit, key=lambda x: x.score, reverse=True)

    ##############################
    # save ranking list 
    ##############################


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
            for qid in qid_list_string:
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
        qrel_file_path (str): path to the trec format qrel file
        ranking_list_path (str): path to the trec format ranking list file
        metrics_list (List[str]): list of metrics to evaluate
        metrics_list_key_form (List[str]): list of metrics in key form (change . to _)
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


def extract_filename(path):
    # Extract the filename with extension
    filename_with_ext = os.path.basename(path)
    # Remove the extension
    filename, ext = os.path.splitext(filename_with_ext)
    return filename
