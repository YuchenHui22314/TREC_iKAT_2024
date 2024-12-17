import argparse
from collections import defaultdict
import time
from progressbar import *
from search.utils import PyScoredDoc
import random
from typing import List
from collections import defaultdict
from ranx import (
    fuse, 
    optimize_fusion,
    Qrels,
    Run
    )

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

def optimize_fusion_weights(
    hits_list, 
    qrels, 
    target_metric
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
        norm="min-max",     
        method="wsum",      
        metric=target_metric,  
        return_optimization_report = True,
        step = 0.1
    )

    return weights["weights"], str(report)

    
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