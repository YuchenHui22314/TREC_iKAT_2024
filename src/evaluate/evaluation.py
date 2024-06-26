import pickle
import sys
import logging
import argparse
import os
import numpy as np
import json
from typing import Mapping, Tuple, List, Optional, Union
import tqdm

sys.path.append('/data/rech/huiyuche/TREC_iKAT_2024/src/')
# sys.path.append('../')
from topics import (
    Turn, 
    Result,
    Reformulation,
    save_turns_to_json, 
    load_turns_from_json,
    filter_ikat_23_evaluated_turns,
    query_type_2_query,
    query_type_rewrite
    )


from pyserini.search.lucene import LuceneSearcher
from pyserini.search import FaissSearcher
from pyserini.search import get_topics, get_qrels
import pytrec_eval





def get_args():
    parser = argparse.ArgumentParser()

    #########################
    # Some general settings
    ########################

    parser.add_argument("--collection", type=str, default="ClueWeb_ikat", 
                        help="can be [ClueWeb22B_ikat]")

    parser.add_argument("--topics", type=str, default="ikat_23_test", 
                        help="can be [ikat_23_test,ikat_24_test]")

    parser.add_argument("--input_query_path", type=str, default="/data/rech/huiyuche/TREC_iKAT_2024/data/topics/ikat_2023_test_turns.json")

    parser.add_argument("--index_dir_path", type=str, default="../../data/indexes/AP_sparse_index")

    parser.add_argument("--output_dir_path", type=str, default="../../results")

    parser.add_argument("--qrel_file_path", type=str, default="../../data/qrels/trec.nist.gov_data_ikat_2023-ptkb-qrels.txt")
    
    parser.add_argument("--retrieval_model", type=str, default="BM25",
                        help="can be [BM25, ance, dpr]")

    parser.add_argument("--reranker", type=str, default="rankllama",
                        help="can be ['None', rankllama,]")

    parser.add_argument("--dense_query_encoder_path", type=str, default="castorini/ance-msmarco-passage",
                        help="should be a huggingface face format folder/link to a model") 
    parser.add_argument("--bm25_k1", type=float, default="0.9") # 0.82
    parser.add_argument("--bm25_b", type=float, default="0.4") # 0.68
    parser.add_argument("--top_k", type=int, default="1000")
    parser.add_argument("--metrics", type=str, default="map,ndcg_cut.5,ndcg_cut.10,P.5,P.10,recall.100",
                        help= "should be a comma-separated string of metrics, such as map,ndcg_cut.5,ndcg_cut.10,P.5,P.10,recall.100,recall.1000")

    #parser.add_argument("--rel_threshold", type=int, default="1")

    parser.add_argument("--save_metrics_to_object",  action="store_true", help="if we will save metrics to turn object.")
    args = parser.parse_args()

    #########################
    # project related config
    ########################

    parser.add_argument("--rewrite_model", type=str, default="no_rewrite",
                        help="can be [no_rewrite, gpt-4-turbo]")

    parser.add_argument("--retrieval_query_type", type=str, default="current_utterance", 
                        help="""can be [
                            "current_utterance",
                            "oracle_utterance",
                            ]""")

    parser.add_argument("--reranking_query_type", type=str, default="current_utterance", 
                        help="""can be [
                            "current_utterance",
                            "oracle_utterance",
                            ]""")

    parser.add_argument("--generation_query_type", type=str, default="current_utterance", 
                        help="""can be [
                            "current_utterance",
                            "oracle_utterance",
                            ]""")

    parser.add_argument("--prompt_type", type = str, default="no_prompt", help="""could be one of 
    [no_prompt,

        ] 
    """)



    return args


    
def get_query_list(args):

    turn_list = []
    query_list = []
    qid_list_string = []
    reranking_query_list = []
    generation_query_list = []

    '''
    load turns from json file. Output format:
    retrieval_query_list: List[str] 
    reranking_query_list: List[str]
    generation_query_list: List[str]
    qid_list_string: List[str]
    turn_list: List[Turn] 
    '''


    # TODO: add ikat processing
    # apply collection specific processing
    if args.topics == "ikat_23_test":
        turn_list = load_turns_from_json(
            input_topic_path=args.input_query_path,
            range_start=0,
            range_end=-1
            )
        
        # filter out the non-evaluated turns
        evaluated_turn_list = filter_ikat_23_evaluated_turns(turn_list)

        qid_list_string = [str(turn.turn_id) for turn in evaluated_turn_list]

        # ikat 23 specific check
        assert turn_list[0].current_utterance =="Can you help me find a diet for myself?", f"The first turn first utterance is {turn_list[0].current_utterance} instead of 'Can you help me find a diet for myself?'"

        # load query/reformulated query according to query type.
        # possible to call a llm to rewrite the query at this step.
        args.retrieval_query_type = query_type_rewrite(args.retrieval_query_type) 
        retrieval_query_list = [turn.query_type_2_query(args.retrieval_query_type) for turn in evaluated_turn_list]
        args.reranking_query_type = query_type_rewrite(args.reranking_query_type)
        reranking_query_list = [turn.query_type_2_query(args.reranking_query_type) for turn in evaluated_turn_list]
        args.generation_query_type = query_type_rewrite(args.generation_query_type)
        generation_query_list = [turn.query_type_2_query(args.generation_query_type) for turn in evaluated_turn_list]
    


    assert len(retrieval_query_list) != 0, "No queries found, args.topics may be wrong"
    assert len(retrieval_query_list) == len(qid_list_string), "Number of queries and qid_list_string not match"

    
    return retrieval_query_list, reranking_query_list, generation_query_list, qid_list_string, turn_list

def get_eval_results(args):

    ###############
    # check args
    ###############

    logger.info("Checking args...")
    assert args.topics in ["ikat_23_test",], f"Invalid topics {args.topics}"
    assert args.collection in ["ClueWeb_ikat",], f"Invalid collection {args.collection}"
    assert args.retrieval_model in ["BM25", "ance", "dpr"], f"Invalid retrieval model {args.retrieval_model}"
    assert args.reranker in ["rankllama"], f"Invalid reranker {args.reranker}"
    assert args.retrieval_query_type in ["current_utterance", "oracle_utterance"], f"retrieve query type {args.retrieval_query_type} is not an invalid query_type"
    assert args.reranking_query_type in ["current_utterance", "oracle_utterance"], f"reranking query type {args.reranking_query_type} is not an invalid query_type"
    assert args.generation_query_type in ["current_utterance", "oracle_utterance"], f"generation query type {args.generation_query_type} is not an invalid query_type"

    #assert args.prompt_type in ['few_shot_narrative_prompt', 'complex_few_shot_narrative_prompt', 'real_narrative_prompt', 'complex_real_narrative_prompt', 'few_shot_pseudo_narrative_only_prompt','complex_few_shot_pseudo_narrative_only_prompt'],f"Prompt type {args.prompt_type} is not implemented."



    ###############
    # check paths
    ###############

    assert os.path.exists(args.input_query_path), "Input query file not found"
    assert os.path.exists(args.index_dir_path), "Index dir not found"
    assert os.path.exists(args.qrel_file_path), "Qrel file not found"
    assert os.path.exists(args.output_dir_path), "Output dir not found"


    ###################################################
    # get query list and qid list as well as Turn list
    ##################################################

    logger.info(f"loading quries")

    # the reason to get turn list is to add per-query 
    # search results. 
    retrieval_query_list, reranking_query_list, generation_query_list, qid_list_string, turn_list = get_query_list(args)

        
    ##############################
    # pyserini search
    ##############################

    # sparse search
    if args.retrieval_model == "BM25":
        logger.info("BM 25 searching...")
        searcher = LuceneSearcher(args.index_dir_path)
        searcher.set_bm25(args.bm25_k1, args.bm25_b)
        hits = searcher.batch_search(retrieval_query_list, qid_list_string, k = args.top_k, threads = 40)

    # dense search
    elif args.retrieval_model in ["ance", "dpr"]:
        logger.info(f"{args.retrieval_model} searching...")
        searcher = FaissSearcher(
            args.index_dir_path,
            args.dense_query_encoder_path 
        )
        hits = searcher.batch_search(retrieval_query_list, qid_list_string, k = args.top_k, threads = 40)


    ##############################
    # TODO: reranking
    ##############################
    # hits ...
    if args.reranker == "None":
        pass
    elif args.reranker == "rankllama":
        logger.info(f"{args.reranker} reranking...")
        pass


    ##############################
    # save ranking list 
    ##############################


    # save ranking list
    # format: query-id Q0 document-id rank score STANDARD
    with open(ranking_list_path, "w") as f:
        for qid in qid_list_string:
            for i, item in enumerate(hits[qid]):
                f.write("{} {} {} {} {} {}".format(
                    qid,
                    "Q0",
                    item.docid,
                    i+1,
                    item.score,
                    file_name_stem 
                    ))
                f.write('\n')

    ##############################
    # TODO: Export to ikat format
    ##############################


    ##############################
    # use pytrec_eval to evaluate
    ##############################

    # read qrels
    with open(args.qrel_file_path, 'r') as f_qrel:
        qrel = pytrec_eval.parse_qrel(f_qrel)
    # read ranking list
    with open(ranking_list_path, 'r') as f_run:
        run = pytrec_eval.parse_run(f_run)

    # then evaluate
    logger.info("trec_eval evaluating...")
    evaluator = pytrec_eval.RelevanceEvaluator(qrel, set(metrics_list))
    query_metrics_dic = evaluator.evaluate(run)

    # average metrics
    averaged_metrics = { metric : np.average(metric_list) for metric, metric_list in metrics.items() }

    # for each query, add the number of relevant documents in query_metrics_dic
    # why not use sum(list(qrel[qid].values()))? Because relevance judgement may be graded instead of binary.
    for qid in query_metrics_dic.keys():
        query_metrics_dic[qid]["num_rel"] = sum([1 for doc in qrel[qid].values() if doc > 0])

    return query_metrics_dic, averaged_metrics, turn_list

if __name__ == "__main__":

    ##########
    # get args
    ##########
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    logger.info(args)

    #########################################################
    # first generate an identifiable name for current run
    #########################################################
    file_name_stem = f"[{args.collection}]-[{args.topics}]-S1[{args.retrieval_query_type}]-S2[{args.reranking_query_type}]-g[{args.generation_query_type}][{args.retrieval_model}]-[{args.reranker}]-[top{args.top_k}]"

    # folder path where the evaluation results will be saved
    base_folder = os.path.join(args.output_dir_path, args.collection, args.topics)

    # create necessary directories 
    subdirs = ["ranking", "metrics", "per_query_metrics"]
    for subdir in subdirs:
        # create output dir if not exist
        path = os.path.join(base_folder, subdir)
        os.makedirs(path, exist_ok=True)

    # generate ranking list file name
    ranking_list_path = os.path.join(
        base_folder,
        "ranking",
        file_name_stem + ".txt")

    metrics_path = os.path.join(
        base_folder,
        "metrics",
        file_name_stem + ".json")

    # save metric dictionary
    metrics_dict_path = os.path.join(
        base_folder,
        "per_query_metrics",
        file_name_stem + "_dict.json")

    # read metrics 
    metrics_list = args.metrics.split(",")
    metrics_list_key_form = [metric.replace(".", "_") for metric in metrics_list]

    ##########################
    # evaluate
    query_metrics_dic, averaged_metrics, turn_list = get_eval_results(args)
    ##########################

    # write results to topic list and save.
    if args.save_metrics_to_object:
        for qid, result_dict in query_metrics_dic.items():
            for turn in turn_list:
                if str(turn.turn_id) == qid:
                    turn.add_result(
                        args.collection, 
                        args.retrieval_model, 
                        args.reranker,
                        args.retrieval_query_type,
                        args.reranking_query_type,
                        args.generation_query_type,
                        result_dict
                    )

        save_turns_to_json(
            turn_list,
            args.input_query_path
        )

    # save metrics
    logger.info("saving results...")


    # save metrics  
    with open(metrics_path, "w") as f:
        json.dump(averaged_metrics, f, indent=4)
    
    # save metrics dictionary
    with open(metrics_dict_path, "w") as f:
        json.dump(query_metrics_dic, f, indent=4)

    # for each metric, append a line in the corresponding file
    # for pourpose of comparison. 
    for metric_name in metrics_list_key_form:
        metric_file_path = os.path.join(
            base_folder,
            "metrics",
            f"{metric_name}.txt")
        
        # append a line in this file in the following format:
        with open(metric_file_path, "a") as f:
            f.write(file_name_stem[:-1] + f"-[{averaged_metrics[metric_name]}]\n")


    logger.info("done.")
    
        
         