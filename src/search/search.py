from functools import reduce
import sys
import re
import os
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from tqdm import tqdm
from pyserini.search.lucene import LuceneSearcher
from pyserini.search import FaissSearcher
import pytrec_eval

sys.path.append('..')
from search.rerank import rerank
from search.dense_search import dense_search
from search.splade_search import splade_search
from search.utils import PyScoredDoc
from search.fuse import (
    normalize_scores,
    optimize_fusion_weights,
    optimize_fusion_weights_n_metrics,
    optimize_fusion_weights_retrieval_score,
    round_robin_fusion,
    linear_weighted_score_fusion,
    per_query_linear_combination,
    calculate_3_weights_from_2_consecutive_weights,
    RRF,
    concat
    
)



def load_ranking_list_from_file(file_path: str) -> Dict[str, List[PyScoredDoc]]:
    # load the ranking list
    with open(file_path, "r") as f:
        run = pytrec_eval.parse_run(f)
        hits = {qid: [PyScoredDoc(docid, score) for docid, score in docs.items()] for qid, docs in run.items()}
    #sort the hits by score
    for qid in hits.keys():
        hits[qid] = sorted(hits[qid], key=lambda x: x.score, reverse=True)
    
    return hits

def get_run_object_and_save_ranking_list(
    hits: Dict[str, List[PyScoredDoc]], 
    args: Any
    ) -> Tuple[Dict[str, List[PyScoredDoc]], Dict[str, Dict[str, float]]]:
    '''
    Get the run object and save the ranking list if needed.
    Args:
        hits (Dict[str, List[PyScoredDoc]]): Pyserini hits object, or a "PyScoredDoc" similar to an Anserini hit object. Must include .docid and .score.
        args.save_ranking_list (bool): Whether to save the ranking list.
        args.run_name (str): The name of the run.
        args.ranking_list_path (str): The path to save the ranking list.
        args.file_name_stem (str): The file name stem (run identifier).
    Returns:
        hits (Dict[str, List[PyScoredDoc]]): Pyserini hits object, or a "PyScoredDoc" similar to an Anserini hit object. Must include .docid and .score. (same as input). The list should be sorted from highest to lowest score.
        run (Dict[qid, Dict[docid, score]]): The run object required by pytrec_eval.
    '''

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

def search(
    args: Any
    ) ->  Tuple[Dict[str, List[PyScoredDoc]], Dict[str, Dict[str, float]]]:

    """
    Perform search using different Retrieval models and rerankers.

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
        - args.file_name_stem: str e.g. S1[raw]-S2[none]-g[raw]-[BM25]-[none_4_1_none]-[s2_top50] 
        - args.file_name_stem_without_group: str e.g. S1[raw]-S2[none]-g[raw]-[BM25]-[none_4_1_none]-[s2_top50] 
        - args.ranking_list_path: str
        - args.save_ranking_list: bool
        - args.given_ranking_list_path: str
        - args.seed: int, the random seed
        ##### for loading previous runs #####
        - args.retrieval_query_type: str (optional)
    # Fusion
        - args.fusion_type: str
        - args.QRs_to_rank: List[str]
        - args.fuse_weights: List[float]
        - args.fusion_normalization: str
        - args.fusion_query_lists: List[List[str]]
        - args.per_query_weight_max_value: float
        - args.qid_personalized_level_dict: Dict[str, str]
        - args.optimize_level_weights: str ("group" or "false" or "2+1" or "no_level")
        - args.qrel_file_path: str (non-necessary if args.optimize_level_weights is "false")
        - args.target_metrics: str split by ","s.(non-necessary if args.optimize_level_weights is "false")
        - args.optimize_step: float (non-necessary if args.optimize_level_weights is "false")
        - args.top_docs
    # Retrieval
        - args.retrieval_model: str
        - args.retrieval_top_k: int, the number of topk passages to return
        # Sparse
            - args.sparse_index_dir_path: str
            - args.bm25_k1: float
            - args.bm25_b: float
            - args.qe_type: str
            - args.fb_terms: int
            - args.fb_docs: int
            - args.original_query_weight: float
        # Dense & Splade
            - args.use_pyserini_dense_search: bool
            - args.dense_query_encoder_path: str
            - args.dense_index_dir_path: str
            - args.splade_query_encoder_path: str
            - args.splade_index_dir_path: str
            - args.query_gpu_id: int, if -1, use cpu
            - args.query_encoder_batch_size : int
            # faiss specific
            - args.faiss_n_gpu: int, the number of gpus
            - args.use_gpu_for_faiss: bool, whether to use gpu for faiss
            - args.embed_dim: int, the dimension of embeddings
            - args.tempmem: int, the temporary memory for Faiss index. Set to -1 to use default value.
            - args.passage_block_num: int, the number of passage blocks
    # Reranker
        - args.reranker: str
        - args.rerank_top_k: int
        - args.rerank_batch_size: int Number of documents to rerank at a time.
        # RankGPT
            - args.step: int
            - args.window_size: int
            - args.rankgpt_llm: str
        # Rankllama
            - args.rerank_quant: str
            - args.cache_dir: str

    Returns:
        hits (Dict[str, List[PyScoredDoc]]): Pyserini hits object, or a "PyScoredDoc" similar to an Anserini hit object. Must include .docid and .score. (same as input). The list should be sorted from highest to lowest score.
        run (Dict[qid, Dict[docid, score]]): The run object required by pytrec_eval.
     
    '''

    ########################################################
    # First, try to load the ranking list from the file name
    # If found, no need to search nor rerank.
    ########################################################
    if os.path.exists(args.ranking_list_path):
        print("found a complete previous run at the begining, loading it...")
        hits = load_ranking_list_from_file(args.ranking_list_path)
        return get_run_object_and_save_ranking_list(hits, args)
    
    if args.personalization_group in ["a", "b", "c"]:

        if os.path.exists(args.ranking_list_path_without_group):
            print("found a complete previous run (no group split) at the begining, loading it...")
            hits = load_ranking_list_from_file(args.ranking_list_path_without_group)
            hits = {qid: doc_list for qid, doc_list in hits.items() if qid in args.qid_list_string}

            return get_run_object_and_save_ranking_list(hits, args)
        
    ##########################################################################################
    # we have the possibility to load a custom ranking list in stead of first stage retrieval 
    ##########################################################################################

    if args.retrieval_model == "none":
        assert args.given_ranking_list_path != "none", " --given_ranking_list_path should be provided when --run_from_rerank or --run_from_generate is true, because we do not do retrieval/retrieval+reranking in these cases."
        hits = load_ranking_list_from_file(args.given_ranking_list_path)
    
    #######################
    # First stage retrieval
    #######################
    
    assert (not args.retrieval_model == "none") or ( not args.fusion_type == "none"), "retrieval model and fusion can not be none at the same time." 

    # No fusion
    if args.fusion_type == "none" and not args.retrieval_model == "none":
        if "retrieval_query_type" in args:
            args.QR_name = args.retrieval_query_type 
        hits = Retrieval(args)

    #######################
    # fusion
    #######################

    elif not args.fusion_type == "none":
        # first search for all QR to get multiple hits
        hits_list = []
        print(f"searching for all QRs..., the number of QRs is {len(args.fusion_query_lists)}")
        for index, QR in tqdm(enumerate(args.fusion_query_lists), total = len(args.fusion_query_lists)):
            args.retrieval_query_list = QR
            args.QR_name = args.QRs_to_rank[index]
            hits_list.append(
                # score normalization is done here
                normalize_scores(
                    Retrieval(args), 
                    args.fusion_normalization
                    )
                )
        del args.QR_name
        

        # fusion 1: linear weighted score
        if args.fusion_type == "linear_weighted_score":
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
            

            # make a list of tuples, each tuple contains a query and a weight
            hits_and_weights = list(zip(hits_list, fuse_weights))

            # get the fused ranking list
            hits = reduce(linear_weighted_score_fusion_reduce_function, hits_and_weights)[0]
        
        # fusion2: lottery fusion
        if args.fusion_type == "round_robin":
            print("fusing ranking lists with round robin...")
            # round robin fusion. Random selet at each run.
            hits = round_robin_fusion(hits_list, args.retrieval_top_k,args.seed)
        
        # fusion3: linear combination.
        # difference with linear_weighted_score is that now we can specify the weight of the 1st query. More general than linear_weighted_score. 
        # I should have directly imiplemented this. 
        if args.fusion_type == "linear_combination":
            print("fusing ranking lists with linear combination...")
            
            # check weight length
            assert len(args.fuse_weights) == len(args.fusion_query_lists), "The number of weights should be equal to the number of query lists."

            # use same weights for all queries
            qid_weights_dict = {qid: args.fuse_weights for qid in args.qid_list_string}

            # linear combination fusion
            hits = per_query_linear_combination(hits_list, qid_weights_dict, args.retrieval_top_k)

        # fusion 4 RRF
        
        if args.fusion_type == "concat":
            print("fusing ranking lists with concat")
            hits = concat(hits_list)

        if args.fusion_type == "RRF":
            print("fusing ranking lists with RRF")
            hits = RRF(hits_list, args.retrieval_top_k)

            
        if args.fusion_type == "per_query_personalize_level":
            print("fusing ranking lists with per_query_personalize_level")
            qid_weights_dict = {}

            ############## Optimization of level weights ##############

            if args.optimize_level_weights == "no_level":

                with open(args.qrel_file_path, 'r') as f_qrel:
                    qrel = pytrec_eval.parse_qrel(f_qrel)

                best_weights, report_pd, best_score_dict = optimize_fusion_weights_n_metrics(
                    hits_list, 
                    qrel, 
                    args.target_metrics.split(","), 
                    args.optimize_step
                    )

                qid_weights_dict = {qid: list(best_weights) for qid in args.qid_list_string}
                args.best_weighs = best_weights
                

            elif args.optimize_level_weights == "2+1":

                with open(args.qrel_file_path, 'r') as f_qrel:
                    qrel = pytrec_eval.parse_qrel(f_qrel)
                first_2_list = hits_list[:2]
                
                # first fuse the first 2 lists and get the best weights
                stage_1_best_weights, report_pd_1, best_score_dict = \
                    optimize_fusion_weights_n_metrics(
                    first_2_list, 
                    qrel, 
                    args.target_metrics.split(","), 
                    args.optimize_step
                    )
                
                stage_1_qid_weights_dict = {qid: list(stage_1_best_weights) for qid in args.qid_list_string}

                fused_first_2_lists = per_query_linear_combination(
                    first_2_list, 
                    stage_1_qid_weights_dict,
                    args.retrieval_top_k
                    )

                hits_list = [fused_first_2_lists, hits_list[2]]

                # then fuse the fused list with the third list
                # and optimize by group.
                level_qid_dict = defaultdict(list)
                for qid, level in args.qid_personalized_level_dict.items():
                    level_qid_dict[level].append(qid)
                
                for level, qids in level_qid_dict.items():
                    print(f"level {level} has {len(qids)} queries.")
                    
                # get a "hits_list" for each level
                level_hits_list_dict = defaultdict(
                    lambda: [defaultdict(list) for _ in range(len(hits_list))]
                    )
                for i, hits in enumerate(hits_list): 
                    for qid, doc_list in hits.items():  
                        for level, qids in level_qid_dict.items():
                            if qid in qids:  
                                level_hits_list_dict[level][i][qid] = doc_list

                # for each personalization level, get the best weight
                level_weights_dict = {}
                level_best_metrics_dict = {}
                for level, sub_hits_list in level_hits_list_dict.items():
                    print(f"optimizing weights for level {level}...")
                    stage_2_best_weights, report_pd_2, best_score_dict = optimize_fusion_weights_n_metrics(
                        sub_hits_list, 
                        qrel,
                        args.target_metrics.split(","), 
                        args.optimize_step
                        )
                    level_weights_dict[level] = stage_2_best_weights
                    level_best_metrics_dict[level] = best_score_dict
                
                # get weights for each query.
                for qid, level in args.qid_personalized_level_dict.items():
                    qid_weights_dict[qid] = level_weights_dict[level]

                # now combine calculate the equivalent weights for fusing 3 lists.
                for level, stage_2_best_weights in level_weights_dict.items():
                    level_weights_dict[level] =\
                        calculate_3_weights_from_2_consecutive_weights(
                            stage_1_best_weights,
                            stage_2_best_weights
                        )
                             
                # record it in args.
                args.level_weights_dict = level_weights_dict 
                args.level_best_metrics_dict = level_best_metrics_dict
            
            elif args.optimize_level_weights == "retrieval_score":
                # load qrel, to be used for metric calculation
                with open(args.qrel_file_path, 'r') as f_qrel:
                    qrel = pytrec_eval.parse_qrel(f_qrel)
                
                # get a inverse map of level to qid
                level_qid_dict = defaultdict(list)
                for qid, level in args.qid_personalized_level_dict.items():
                    level_qid_dict[level].append(qid)
                
                for level, qids in level_qid_dict.items():
                    print(f"level {level} has {len(qids)} queries.")
                    
                # get a "hits_list" for each level
                level_hits_list_dict = defaultdict(lambda: [defaultdict(list) for _ in range(len(hits_list))])
                for i, hits in enumerate(hits_list): 
                    for qid, doc_list in hits.items():  
                        for level, qids in level_qid_dict.items():
                            if qid in qids:  
                                level_hits_list_dict[level][i][qid] = doc_list

                # for each personalization level, get the best weight
                level_weights_dict = {}
                level_best_metrics_dict = {}

                for level, sub_hits_list in level_hits_list_dict.items():
                    print(f"optimizing weights for level {level}...")
                    weights, report, best_score_dict = optimize_fusion_weights_retrieval_score(
                        sub_hits_list, 
                        qrel,
                        args.target_metrics.split(","), 
                        args.optimize_step,
                        args.top_docs
                        )
                    level_weights_dict[level] = weights
                    level_best_metrics_dict[level] = best_score_dict
                
                # record it in args.
                args.level_weights_dict = level_weights_dict
                args.level_best_metrics_dict = level_best_metrics_dict
                print(level_weights_dict)
                # get weights for each query.
                for qid, level in args.qid_personalized_level_dict.items():
                    qid_weights_dict[qid] = level_weights_dict[level]



            elif args.optimize_level_weights == "group":
                # load qrel, to be used for metric calculation
                with open(args.qrel_file_path, 'r') as f_qrel:
                    qrel = pytrec_eval.parse_qrel(f_qrel)
                
                # get a inverse map of level to qid
                level_qid_dict = defaultdict(list)
                for qid, level in args.qid_personalized_level_dict.items():
                    level_qid_dict[level].append(qid)
                
                for level, qids in level_qid_dict.items():
                    print(f"level {level} has {len(qids)} queries.")
                    
                # get a "hits_list" for each level
                level_hits_list_dict = defaultdict(lambda: [defaultdict(list) for _ in range(len(hits_list))])
                for i, hits in enumerate(hits_list): 
                    for qid, doc_list in hits.items():  
                        for level, qids in level_qid_dict.items():
                            if qid in qids:  
                                level_hits_list_dict[level][i][qid] = doc_list

                # for each personalization level, get the best weight
                level_weights_dict = {}
                level_best_metrics_dict = {}

                for level, sub_hits_list in level_hits_list_dict.items():
                    print(f"optimizing weights for level {level}...")
                    weights, report, best_score_dict = optimize_fusion_weights_n_metrics(
                        sub_hits_list, 
                        qrel,
                        args.target_metrics.split(","), 
                        args.optimize_step
                        )
                    level_weights_dict[level] = weights
                    level_best_metrics_dict[level] = best_score_dict
                
                # record it in args.
                args.level_weights_dict = level_weights_dict
                args.level_best_metrics_dict = level_best_metrics_dict
                print(level_weights_dict)
                # get weights for each query.
                for qid, level in args.qid_personalized_level_dict.items():
                    qid_weights_dict[qid] = level_weights_dict[level]

            else:
                # get weights for each query.
                # for rw fuse rwrs, always use [1,0.1].
                # for personalized query, use the level to get the weight.
                decontextualized_rwrs_weight = [1,0.1]
                for qid in args.qid_list_string:
                    level = args.qid_personalized_level_dict[qid]
                    float_level = from_level_to_weight_3(level, 4, args.per_query_weight_max_value)
                    qid_weights_dict[qid] = decontextualized_rwrs_weight + [float_level]

            # linear combination fusion
            hits = per_query_linear_combination(hits_list, qid_weights_dict, args.retrieval_top_k)



    ##############################
    # reranking
    ##############################

    if not args.reranker == "none":
        hits = rerank(hits, args)

    ##############################
    # optionally save ranking list 
    ##############################

    return get_run_object_and_save_ranking_list(hits, args)



############## First stage retrieval ###############
def Retrieval(args):

    '''
    All required arguments are:
        - args.retrieval_query_list: List[str]
        - args.qid_list_string: List[str]
        - args.retrieval_model: str
        - args.personalization_group: str
        ##### for loading previous runs #####
        - args.ranking_list_path: str (optional)
        - args.QR_name: str (optional)
    # Sparse
        - args.sparse_index_dir_path: str
        - args.retrieval_top_k: int
        - args.bm25_k1: float
        - args.bm25_b: float
        - args.qe_type: str
        - args.fb_terms: int
        - args.fb_docs: int
        - args.original_query_weight: float
    # Dense & Splade
        - args.use_pyserini_dense_search: bool
        - args.dense_query_encoder_path: str
        - args.dense_index_dir_path: str
        - args.splade_query_encoder_path: str
        - args.splade_index_dir_path: str
        - args.query_encoder_batch_size : int
        - args.faiss_n_gpu: int, the number of gpus
        - args.use_gpu: bool, whether to use gpu for faiss
        - args.embed_dim: int, the dimension of embeddings
        - args.tempmem: int, the temporary memory for Faiss index. Set to -1 to use default value.
        - args.query_gpu_id: int, Which GPU should the query encoder use. if -1, use cpu
        - args.passage_block_num: int, the number of passage blocks
    

    Returns:
        hits (Dict[str, List[Any]): Pyserini hits object, or a "PyScoredDoc" similar to an Anserini hit object. Must include .docid and .score. The lists are sorted.
    '''

    # sparse search
    if args.retrieval_model == "none":
        raise ValueError("retrieval_model should not be none when calling Retrieval function.")
    
    # before retireve, try to load the retrieved list from file
    if "ranking_list_path" in args and "QR_name" in args:

        ranking_list_dir_path = os.path.dirname(args.ranking_list_path)
        base_name = os.path.basename(args.ranking_list_path)
        if os.path.exists(ranking_list_dir_path):
            for file in os.listdir(ranking_list_dir_path):
                # find all contents in [xx], should return 6 groups
                inside_crochets = re.findall(r'\[([^\]]+)\]', file)
                inside_crochets_gold = re.findall(r'\[([^\]]+)\]', base_name)
                if (
                    inside_crochets[0] == args.QR_name and # QR type
                    inside_crochets[3] == inside_crochets_gold[3] and # retriever
                    inside_crochets[4].split("_")[0] == "none" # no reranker
                    ):
                    print("In Retrieval function, found a previous non-reranking run, loading it...")
                    hits = load_ranking_list_from_file(
                        os.path.join(ranking_list_dir_path, file)
                      )
                    return hits


    if args.retrieval_model == "BM25":
        print("BM 25 searching...")
        searcher = LuceneSearcher(args.sparse_index_dir_path)
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

        if args.use_pyserini_dense_search:
            searcher = FaissSearcher(
                args.dense_index_dir_path,
                args.dense_query_encoder_path 
            )
            hits = searcher.batch_search(args.retrieval_query_list, args.qid_list_string, k = args.retrieval_top_k, threads = 40)
        else:
            hits = dense_search(args)

    elif "splade" in args.retrieval_model:
        print(f"{args.retrieval_model} searching...")
        hits = splade_search(args)

    return hits
