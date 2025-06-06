import argparse
from collections import defaultdict
from typing import  List
from itertools import product
import time
import random

import pandas as pd
import numpy as np
from progressbar import *
from tqdm import tqdm
from ranx import (
    optimize_fusion,
    Qrels,
    Run
    )
from ranx.meta import evaluate
from ranx.fusion import fusion_switch

from .utils import PyScoredDoc


def get_possible_weights(step):
    round_digits = str(step)[::-1].find(".")
    return [round(x, round_digits) for x in np.arange(0, 1 + step, step)]

def get_trial_configs(weights, n_runs):
    return [seq for seq in product(*[weights] * n_runs) if sum(seq) == 1.0]

def calculate_3_weights_from_2_consecutive_weights(weight_pair_1, weight_pair_2):

    a = weight_pair_1[0]
    b = weight_pair_1[1]
    c = weight_pair_2[0]
    d = weight_pair_2[1]
    
    list_3_final_weight = d * (a + b) / c
    return [a, b, list_3_final_weight]
    

def generate_random_numbers_sum_to_one(k: int, seed: int) -> list[float]:
    """
    Generate k random floating-point numbers that sum exactly to 1.0.

    This method works by generating k-1 random split points within the [0,1] interval.
    These points, along with 0 and 1, divide the interval into k segments. 
    The length of each segment becomes a random number. By construction (telescoping sum),
    their total sum is exactly 1.0.

    Args:
        k (int): The number of random numbers to generate. Must be a positive integer.

    Returns:
        list[float]: A list containing k random numbers that sum to 1.0.
                     All numbers in the list are non-negative.

    Raises:
        TypeError: If k is not an integer.
        ValueError: If k is not a positive integer (i.e., k <= 0).
    """
    if seed is not None:
        random.seed(seed)

    if not isinstance(k, int):
        raise TypeError("Input parameter k must be an integer.")
    if k <= 0:
        raise ValueError("Input parameter k must be a positive integer.")

    if k == 1:
        # If k is 1, the result is a list containing the single element 1.0
        return [1.0]

    # Generate k-1 random split points. random.random() generates floats in [0.0, 1.0).
    # These points will divide the [0,1] interval into k sub-intervals.
    split_points = [random.random() for _ in range(k - 1)]

    # Add 0.0 and 1.0 to the list of split points and sort all points.
    # The sorted list will contain k+1 points, from p_0=0.0 to p_k=1.0.
    # For example, if k=3, we have k-1=2 split points r1, r2.
    # all_points will be sorted([0.0, r1, r2, 1.0]).
    all_points = sorted([0.0] + split_points + [1.0])

    # Calculate the differences between consecutive sorted points.
    # The i-th random number is equal to all_points[i+1] - all_points[i].
    # There will be k such differences (i.e., lengths of k sub-intervals).
    # For example, if k=3, sorted points are p0, p1, p2, p3.
    # The random numbers are (p1-p0), (p2-p1), (p3-p2).
    # Their sum is (p1-p0) + (p2-p1) + (p3-p2) = p3 - p0 = 1.0 - 0.0 = 1.0.
    random_numbers = [all_points[i+1] - all_points[i] for i in range(k)]

    # Due to the construction method (telescoping sum), the sum is theoretically exact 1.0.
    # Actual floating-point addition might introduce tiny computational errors,
    # but this is precise enough for most applications.

    return random_numbers


def normalize_scores(hits, normalization_type):
    """
    Normalize the scores of hits objects.
    Arguments:
    - hits: (Dict[str, List[PyScoredDoc]]): Hits objects for each query.
    - normalization_type (str): Type of normalization to apply. Options are "none", 'min-max' and 'max'.
    Returns:
    - hits (Dict[str, List[PyScoredDoc]]): Normalized hits objects.
    """
    def min_max_normalization(hit):
        min_score = min([doc.score for doc in hit])
        max_score = max([doc.score for doc in hit])
        for doc in hit:
            doc.score = (doc.score - min_score) / (max_score - min_score)
        return hit

    def max_normalization(hit):
        max_score = max([doc.score for doc in hit])
        for doc in hit:
            doc.score = doc.score / max_score
        return hit

    normalization_func = {
        'min-max': min_max_normalization,
        'max': max_normalization,
        'none': lambda x: x
    }

    normalized_hits_dict = defaultdict(list)
    for qid, hit in hits.items():
        normalized_hits_dict[qid] = normalization_func[normalization_type](hit)

    return normalized_hits_dict

def customize_optimize(
    qrels: Qrels,
    runs: List[Run],
    metrics: List[str],
    step: float = 0.1,
    description: str = "Optimizing weights",
):
    weights = get_possible_weights(step)
    trials = get_trial_configs(weights, len(runs))
    fusion_method = fusion_switch("wsum")

    best_score = 0.0
    best_score_dict = {}
    best_weights = []
    optimization_report = {}

    for weights in tqdm(trials, desc=description, total=len(trials)):
        fused_run = fusion_method(runs, weights)
        metric_dic = {}
        for metric in metrics:
            score = evaluate(qrels, fused_run, metric, save_results_in_run=False)
            metric_dic[metric] = score
        optimization_report[str(weights)] = metric_dic

        score_sum = sum(metric_dic.values())
        if score_sum > best_score:
            best_score = score_sum
            best_weights = weights
            best_score_dict = metric_dic
        # transform to pandas dataframe, first column is the weights, the rest are the metrics

    pandas_optimization_report = defaultdict(list)
    for weights, metrics in optimization_report.items():
        pandas_optimization_report["weights"].append(weights)
        for metric, score in metrics.items():
            pandas_optimization_report[metric].append(score)
    
        
    df = pd.DataFrame(pandas_optimization_report)
    print("the best weights are: ", best_weights)
    print("the best scores are: ", best_score_dict)

    return best_weights, df, best_score_dict

def customize_optimize_retrieval_score(
    qrels: Qrels,
    runs: List[dict],
    metrics: List[str],
    step: float = 0.1,
    description: str = "Optimizing weights",
    top_docs: int = 100
):
    weights = get_possible_weights(step)
    trials = get_trial_configs(weights, len(runs))

    best_score = 0.0
    best_score_dict = {}
    best_weights = []
    optimization_report = {}
    best_hits = []

    # temp = []
    # for run in runs:
    #     temp.append({qid: run[qid] for qid in qrels.keys() if qid in ['15-3', '5-15', '11-9', '11-11', '7-3']})
    
    #runs = temp
    for weights in tqdm(trials, desc=description, total=len(trials)):
        print(type(weights))
        print(weights)
        qid_weights_dict = {qid: weights for qid in runs[0].keys()} 
        final_hits = per_query_linear_combination(runs, qid_weights_dict, 1000)
        score_sum = sum([doc.score for qid, docs in final_hits.items() for doc in docs[0:top_docs]])
        if score_sum > best_score:
            best_score = score_sum
            best_weights = weights
            best_hits = final_hits
        
    best_run = {qid: {doc.docid: doc.score for doc in docs}  for qid, docs in best_hits.items()}

    metric_dic = {}
    for metric in metrics:
        score = evaluate(qrels, best_run, metric, save_results_in_run=False)
        metric_dic[metric] = score
    optimization_report[str(weights)] = metric_dic
    best_score_dict = metric_dic

    # print the optimization report
     
    # transform to pandas dataframe, first column is the weights, the rest are the metrics
    pandas_optimization_report = defaultdict(list)
    for weights, metrics in optimization_report.items():
        pandas_optimization_report["weights"].append(weights)
        for metric, score in metrics.items():
            pandas_optimization_report[metric].append(score)
    
        
    df = pd.DataFrame(pandas_optimization_report)
    print("the best weights are: ", best_weights)
    print("the best scores are: ", best_score_dict)

    return best_weights, df, best_score_dict

def optimize_fusion_weights_retrieval_score(
    hits_list,
    qrels,
    target_metrics,
    step,
    top_docs
    ):

    qrels = {qid: qrel for qid, qrel in qrels.items() if qid in hits_list[0]}
    # qrels = {qid: qrels[qid] for qid in qrels.keys() if qid in ['15-3', '5-15', '11-9', '11-11', '7-3']}
    qrels = Qrels(qrels)

    # optimize weights
    weights, report_pd, best_score_dict =  \
        customize_optimize_retrieval_score(
        qrels=qrels,
        runs=hits_list,
        metrics=target_metrics,
        step = step,
        top_docs = top_docs
    )
    print("################## Optimization Report ##################")
    # set show items to infinite
    pd.set_option('display.max_rows', None)
    print(report_pd)

    return weights, report_pd, best_score_dict

def optimize_fusion_weights_n_metrics(
    hits_list,
    qrels,
    target_metrics,
    step,
    ):

    # Generate run dictionary required by ranx
    runs = [{qid: {doc.docid: doc.score for doc in docs}  for qid, docs in hits.items()}for hits in hits_list]

    # in the qrel, filter out the qids that are not in the runs
    qrels = {qid: qrel for qid, qrel in qrels.items() if qid in runs[0]}

    runs = [Run(run) for run in runs]
    qrels = Qrels(qrels)

    # optimize weights
    weights, report_pd, best_score_dict =  \
        customize_optimize(
        qrels=qrels,
        runs=runs,
        metrics=target_metrics,
        step = step
    )
    print("################## Optimization Report ##################")
    # set show items to infinite
    pd.set_option('display.max_rows', None)
    print(report_pd)

    return weights, report_pd, best_score_dict


def optimize_fusion_weights(
    hits_list, 
    qrels, 
    target_metric,
    step
    ):

    # Generate run dictionary required by ranx
    runs = [{qid: {doc.docid: doc.score for doc in docs}  for qid, docs in hits.items()}for hits in hits_list]

    # in the qrel, filter out the qids that are not in the runs
    qrels = {qid: qrel for qid, qrel in qrels.items() if qid in runs[0]}

    runs = [Run(run) for run in runs]
    qrels = Qrels(qrels)
        

    weights, report =  \
        optimize_fusion(
        qrels=qrels,
        runs=runs,
        norm=None,     
        method="wsum",      
        metric=target_metric,  
        return_optimization_report = True,
        step = step
    )
    print("Optimization report: ", report)

    return weights["weights"], 

    
def rank_list_fusion(
    rank_file0=None, 
    rank_file1=None, 
    topk=1000, 
    output=None, 
    alpha=0.1, 
    run_name='fusion'):
    """
    Performs rank list fusion by combining results from dense and sparse retrieval methods. 
    This function reads ranked lists from two input files, fuses the rankings based on the 
    provided alpha value, and writes the fused results to an output file.

    Arguments:
    - rank_file0 (str): Path to the rank file for dense retrieval. 
                        The file should have the format: qid docid rank score.
    - rank_file1 (str): Path to the rank file for sparse retrieval. 
                        The file should have the format: qid docid rank score.
    - topk (int): Number of hits to retrieve for each query. Default is 1000.
    - output (str): Path to the output file where fused rankings will be saved.
                    The output format will be: qid Q0 docid rank score run_name.
    - alpha (float): Weighting factor for the sparse retrieval scores in the fusion process. 
                     Default is 0.1. Higher alpha gives more weight to sparse retrieval scores.
    - run_name (str): Identifier for the current run, used in the output file. Default is 'fusion'.
    
    Returns:
    - None. The function writes the output directly to the specified output file.
    """
    
    def read_rank_list(file):
        """Reads a rank list from the specified file and returns document IDs and scores."""
        qid_docid_list = defaultdict(list)
        qid_score_list = defaultdict(list)
        with open(file, 'r') as f:
            for line in f:
                qid, _, docid, rank, score, _=line.strip().split(' ') 
                qid_docid_list[qid].append(docid)
                qid_score_list[qid].append(float(score))
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

    print('Read ranked list0...')
    qid_docid_list0, qid_score_list0 = read_rank_list(rank_file0)
    print('Read ranked list1...')
    qid_docid_list1, qid_score_list1 = read_rank_list(rank_file1)

    qids = qid_docid_list0.keys()
    fout = open(output, 'w')
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
            fout.write('{} Q0 {} {} {} {}\n'.format(qid, doc, rank + 1, score, run_name))
        pbar.update(j + 1)
    
    time_per_query = (time.time() - start_time) / len(qids)
    print('Fusing {} queries ({:0.3f} s/query)'.format(len(qids), time_per_query))
    fout.close()
    print('Finished.')

##############################################################################
####################### Ranking list fusion Functions #########################
##############################################################################


def from_level_to_weight_3(level, max_level, max_weight):
    """
    Convert a level to a weight.
    1) a -> 1, b ->2, c -> 3, d -> 4.
    level 1 ->  max_weight / max_level
    level 2 -> 2* max_weight / max_level
    .....
    level max_level -> max_weight
    """
    level_map = {"a": 0.4, "b": 0.5, "c": 0.6, "d": 0.6}
    return level_map[level] 

def from_level_to_weight_2(level, max_level, max_weight):
    """
    Convert a level to a weight.
    1) a -> 1, b ->2, c -> 3, d -> 4.
    level 1 ->  max_weight / max_level
    level 2 -> 2* max_weight / max_level
    .....
    level max_level -> max_weight
    """
    level_map = {"a": 1, "b": 2, "c": 3, "d": 4}
    level = level_map[level]
    unit = max_weight / max_level
    return unit * level 

def from_level_to_weight(level, max_level, max_weight):
    """
    Convert a level to a weight.
    1) a -> 1, b ->2, c -> 3, d -> 4.
    level 1 -> 0
    level 2 -> max_weight / (max_level-1)
    .....
    level max_level -> max_weight
    """
    level_map = {"a": 1, "b": 2, "c": 3, "d": 4}
    level = level_map[level]
    unit = max_weight / (max_level-1)
    return unit * (level-1) 


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


def concat(hits_list):
    final_hits_dict = defaultdict(list)

    qids = list(hits_list[0].keys())
    print("the length of qid is: ", len(qids))

    for hits in hits_list:
        qids_1 = hits.keys()
        print("the length of qid_1 is: ", len(qids_1))
        print(set(qids) - set(qids_1))
        assert  len(qids) == len(qids_1), "The number of queries in the two hits objects are different"

        for hits in hits_list:
            for qid, doc_list in hits.items():
                for doc in doc_list:
                    if doc.docid not in [d.docid for d in final_hits_dict[qid]]:
                        final_hits_dict[qid].append(doc)
        
        print([len(final_hits_dict[qid]) for qid in qids])
    
    return final_hits_dict
    


def RRF(hits_list, topk, k=60):
    """
    Perform Rank Rank Fusion (RRF) on a list of hits objects.

    Args:
        hits_list: List of pyserini hits objects. Each element inside must can .docid and .score
        topk: Number of hits to retrieve for each query. Default is 1000.
        k: Constant used in RRF score calculation. Default is 60.
    Retruns:
        The fused ranking list.
    """
    final_hits_dict = defaultdict(dict)
    qids = hits_list[0].keys()

    for qid in qids:
        doc_scores = defaultdict(float)
        for hits in hits_list:
            for rank, doc in enumerate(hits[qid]):
                doc_scores[doc.docid] += 1 / (k + rank + 1)  # +1 to adjust rank to start from 0

        # Sort documents by their RRF scores in descending order
        sorted_docs = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)

        # Convert sorted documents to pyserini's Hits format
        final_hits_dict[qid] = []
        for docid, score in sorted_docs[:topk]:
            # Create dummy hits object. We don't have access to the original score or text
            hit = type('', (), {})() 
            hit.docid = docid
            hit.score = score
            #hit.text = "" # or any dummy value for text. Pyserini may need this attribute for some functions.
            final_hits_dict[qid].append(hit)


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
        for rank in range(min(topk, min([len(hits[qid]) for hits in hits_list]))):
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

#def RRF

# Example of how to call the function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rank list fusion')
    parser.add_argument('--rank_file0', default=None, help='rank file for dense retrieval with format: qid docid rank score')
    parser.add_argument('--rank_file1', default=None, help='rank file for sparse retrieval with format: qid docid rank score')
    parser.add_argument('--topk', default=1000, type=int, help='number of hits to retrieve')
    parser.add_argument('--output', required=True)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--run_name", type=str, default='fusion')
    args = parser.parse_args()

    rank_list_fusion(args.rank_file0, args.rank_file1, args.topk, args.output, args.alpha, args.run_name)