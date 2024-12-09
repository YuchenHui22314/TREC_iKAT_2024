from functools import reduce
from pyserini.search.lucene import LuceneSearcher
from pyserini.search import FaissSearcher
from typing import Any, Dict, List
import sys

sys.path.append('..')
from search.rerank import rerank
from search.dense_search import dense_search
#from search.splade_search import splade_search
from search.utils import PyScoredDoc
from search.fuse import (
    round_robin_fusion,
    linear_weighted_score_fusion,
    per_query_linear_combination,
)



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
        - args.seed: int, the random seed
    # Fusion
        - args.fusion_type: str
        - args.QRs_to_rank: List[str]
        - args.fuse_weights: List[float]
        - args.fusion_query_lists: List[List[str]]
        - args.per_query_weight_max_value: float
        - args.qid_personalized_level_dict: Dict[str, str]
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
        # Dense
            - args.use_pyserini_dense_search: bool
            - args.dense_query_encoder_path: str
            - args.query_encoder_batch_size : int
            - args.dense_index_dir_path: str
            - args.faiss_n_gpu: int, the number of gpus
            - args.use_gpu_for_faiss: bool, whether to use gpu for faiss
            - args.query_gpu_id: int, if -1, use cpu
            - args.embed_dim: int, the dimension of embeddings
            - args.tempmem: int, the temporary memory for Faiss index. Set to -1 to use default value.
            - args.query_gpu_id: int, Which GPU should the query encoder use. if -1, use cpu
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

    
    '''

    ##########################################################################################
    # we have the possibility to load a custom ranking list instead of searching or reranking
    ##########################################################################################

    if args.retrieval_model == "none":
        assert args.given_ranking_list_path != "none", " --given_ranking_list_path should be provided when --run_from_rerank or --run_from_generate is true, because we do not do retrieval/retrieval+reranking in these cases."

        # even we do not search, we have to get access to the index (raw documents via a searcher)
        searcher = LuceneSearcher(args.sparse_index_dir_path)
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
        hits = round_robin_fusion(hits_list, args.retrieval_top_k,args.seed)
    
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

    elif args.fusion_type == "per_query_personalize_level":
        print("fusing ranking lists with per_query_personalize_level")
        # first search.
        hits_list = []
        for QR in args.fusion_query_lists:
            args.retrieval_query_list = QR
            hits_list.append(Retrieval(args))
        
        # get weights for each query.
        # for rw fuse rwrs, always use [1,0.1].
        # for personalized query, use the level to get the weight.
        decontextualized_rwrs_weight = [1,0.1]
        qid_weights_dict = {}
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
    # Sparse
        - args.retrieval_model: str
        - args.sparse_index_dir_path: str
        - args.retrieval_top_k: int
        - args.bm25_k1: float
        - args.bm25_b: float
        - args.qe_type: str
        - args.fb_terms: int
        - args.fb_docs: int
        - args.original_query_weight: float
    # Dense
        - args.use_pyserini_dense_search: bool
        - args.dense_query_encoder_path: str
        - args.query_encoder_batch_size : int
        - args.dense_index_dir_path: str
        - args.faiss_n_gpu: int, the number of gpus
        - args.use_gpu: bool, whether to use gpu for faiss
        - args.embed_dim: int, the dimension of embeddings
        - args.tempmem: int, the temporary memory for Faiss index. Set to -1 to use default value.
        - args.query_gpu_id: int, Which GPU should the query encoder use. if -1, use cpu
        - args.passage_block_num: int, the number of passage blocks

    Returns:
        hits (Dict[str, List[Any]): Pyserini hits object, or a "PyScoredDoc" similar to an Anserini hit object. Must include .docid and .score.
    '''

    # sparse search
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

    ##############################
    # TODO: add splade
    ##############################

    return hits
