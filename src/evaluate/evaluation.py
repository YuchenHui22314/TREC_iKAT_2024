import pickle
import sys
import logging
import argparse
import os
import numpy as np
import json
from typing import Mapping, Tuple, List, Optional, Union
from tqdm import tqdm
from dataclasses import asdict

sys.path.append('/data/rech/huiyuche/TREC_iKAT_2024/src/')
#sys.path.append('../')

from rank_gpt import run_retriever, sliding_windows
from topics import (
    Turn, 
    Result,
    Reformulation,
    save_turns_to_json, 
    load_turns_from_json,
    filter_ikat_23_evaluated_turns,
    )

from rerank import (
    load_rankllama, 
    rerank_rankllama,
    hits_2_rankgpt_list
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

    parser.add_argument("--input_query_path", type=str, default="../../data/topics/ikat_2023_test.json")

    parser.add_argument("--index_dir_path", type=str, default="../../data/indexes/clueweb22b_ikat23_fengran_sparse_index_2")

    parser.add_argument("--output_dir_path", type=str, default="../../results")

    parser.add_argument("--qrel_file_path", type=str, default="../../data/qrels/ikat_23_qrel.txt")
    
    parser.add_argument("--retrieval_model", type=str, default="BM25",
                        help="can be [BM25, ance, dpr, splade]")

    # hugging_face cache_dir
    parser.add_argument("--cache_dir", type=str, default="/data/rech/huiyuche/huggingface", help="cache directory for huggingface models")
    ## TODO: unify -- cache_dir and --dense_query_encoder_path

    # where is dense encoder
    parser.add_argument("--dense_query_encoder_path", type=str, default="castorini/ance-msmarco-passage",
                        help="should be a huggingface face format folder/link to a model") 


    parser.add_argument("--reranker", type=str, default="none",
                        help="can be ['none', rankllama, rankgpt]")
    # rankllama
    parser.add_argument("--rerank_quant", type=str, default="none",
                        help="can be ['none','8b','4b']")
    # rankGPT
    parser.add_argument("--rankgpt_llm", type=str, default="gpt-3.5-turbo",
                        help="can be ['gpt-3.5-turbo',]")
    parser.add_argument("--window_size", type=int, default="5") 
    parser.add_argument("--step", type=int, default="1") 

    # BM25 parameters
    parser.add_argument("--bm25_k1", type=float, default="0.9") # 0.82
    parser.add_argument("--bm25_b", type=float, default="0.4") # 0.68

    # TODO: RM3 parameters

    parser.add_argument("--top_k", type=int, default="1000")
    parser.add_argument("--metrics", type=str, default="map,ndcg_cut.1,ndcg_cut.3,ndcg_cut.5,ndcg_cut.10,P.1,P.3,P.5,P.10,recall.5,recall.100,recall.1000,recip_rank",
                        help= "should be a comma-separated string of metrics, such as map,ndcg_cut.5,ndcg_cut.10,P.5,P.10,recall.100,recall.1000")

    #parser.add_argument("--rel_threshold", type=int, default="1")

    parser.add_argument("--save_metrics_to_object",  action="store_true", help="if we will save metrics to turn object.")

    #########################
    # ikat 2024 project related config
    ########################

    parser.add_argument("--rewrite_model", type=str, default="no_rewrite",
                        help="can be [no_rewrite, gpt-4-turbo]")

    parser.add_argument("--retrieval_query_type", type=str, default="oracle_utterance", 
                        help="""can be [
                            "current_utterance",
                            "oracle_utterance",
                            "reformulation"
                            ]""")

    parser.add_argument("--reranking_query_type", type=str, default="oracle_utterance", 
                        help="""can be [
                            "current_utterance",
                            "oracle_utterance",
                            "reformulation"
                            ]""")

    parser.add_argument("--generation_query_type", type=str, default="oracle_utterance", 
                        help="""can be [
                            "current_utterance",
                            "oracle_utterance",
                            "reformulation"
                            ]""")

    parser.add_argument("--prompt_type", type = str, default="no_prompt", help="""could be one of 
    [no_prompt,

        ] 
    """)

    parser.add_argument("--just_rank_no_evaluate",  action="store_true", help="if we will use qrel to run evaluation or just yield the ranking list save metrics to turn object.")


    args = parser.parse_args()
    return args


def query_type_rewrite(
    original_query_type:str,
) -> str:
        if original_query_type == "reformulation":
        ## TODO: TBD for CIR
            query_type = f'reformulated_description_by_[{args.rewrite_model}]_using_[{args.prompt_type}]'
        else:
            query_type = original_query_type
        
        return query_type
    
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

    # apply topic specific processing
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

    print("Checking args...")
    assert args.topics in ["ikat_23_test",], f"Invalid topics {args.topics}"
    assert args.retrieval_model in ["BM25", "ance", "dpr", "splade"], f"Invalid retrieval model {args.retrieval_model}"
    assert args.reranker in ["rankllama","none", "rankgpt"], f"Invalid reranker {args.reranker}"
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

    print(f"loading quries")

    # the reason to get turn list is to add per-query 
    # search results. 
    retrieval_query_list, reranking_query_list, generation_query_list, qid_list_string, turn_list = get_query_list(args)

        
    ##############################
    # pyserini search
    ##############################

    # sparse search
    if args.retrieval_model == "BM25":
    ##############################
    # TODO: RM3
    ##############################
        print("BM 25 searching...")
        searcher = LuceneSearcher(args.index_dir_path)
        searcher.set_bm25(args.bm25_k1, args.bm25_b)
        hits = searcher.batch_search(retrieval_query_list, qid_list_string, k = args.top_k, threads = 40)

    # dense search
    elif args.retrieval_model in ["ance", "dpr"]:
        print(f"{args.retrieval_model} searching...")
        searcher = FaissSearcher(
            args.index_dir_path,
            args.dense_query_encoder_path 
        )
        hits = searcher.batch_search(retrieval_query_list, qid_list_string, k = args.top_k, threads = 40)


    ##############################
    # TODO: add splade
    ##############################


    ##############################
    # reranking
    ##############################


    if not args.reranker == "none":

        print(f"{args.reranker} reranking")
         

        # generate a qid-reranking_query dictionary
        reranking_query_dic = {qid: reranking_query for qid, reranking_query in zip(qid_list_string, reranking_query_list)}

        # generate input format required by rankgpt
        rank_gpt_list, _ = hits_2_rankgpt_list(searcher, reranking_query_dic, hits)

    if args.reranker == "rankgpt":

        # get hyperparameters
        llm_name = args.rankgpt_llm
        rank_end = args.top_k
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
                [json.loads(searcher.doc(doc_object.docid).raw())["contents"] for doc_object in hit],
                tokenizer,
                model
            )

            for i in range(len(hit)):
                hit[i].score = reranked_scores[i]
                # sort the hits by score
            hits[qid] = sorted(hit, key=lambda x: x.score, reverse=True)


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

    # read qrels
    with open(args.qrel_file_path, 'r') as f_qrel:
        qrel = pytrec_eval.parse_qrel(f_qrel)
    # read ranking list
    with open(ranking_list_path, 'r') as f_run:
        run = pytrec_eval.parse_run(f_run)

    ##############################
    #  Export to ikat format
    ##############################
    if args.just_rank_no_evaluate:
        print("generating ikat format results...")

        exit("ikat format results saved.")

    ##############################
    # TODO: enable without evaluation 
    ##############################

    ##############################
    # TODO: add run_name 
    ##############################

    ##############################
    # TODO: evaluate ptkb ranking list 
    ##############################

    ##############################
    # use pytrec_eval to evaluate
    ##############################


    # then evaluate
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

    return query_metrics_dic, averaged_metrics, turn_list

if __name__ == "__main__":

    ##########
    # get args
    ##########
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    print(args)

    #########################################################
    # first generate an identifiable name for current run
    #########################################################
    file_name_stem = f"S1[{args.retrieval_query_type}]-S2[{args.reranking_query_type}]-g[{args.generation_query_type}]-[{args.retrieval_model}]-[{args.reranker}_{args.window_size}_{args.step}_{args.rerank_quant}]-[top{args.top_k}]"

    # folder path where the evaluation results will be saved
    base_folder = os.path.join(args.output_dir_path, args.collection, args.topics)

    # create necessary directories 
    subdirs = ["ranking", "metrics", "per_query_metrics", "ikat_format_output"]
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

    # save ikat format output
    ikat_output_path = os.path.join(
        base_folder,
        "ikat_format_output",
        file_name_stem + ".json")

    ## TODO: use this path

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
    print("saving results...")


    # save metrics  
    with open(metrics_path, "w") as f:
        json.dump(averaged_metrics, f, indent=4)

    # save also the args values in the same file
    with open(metrics_path, "a") as f:
        f.write("\n")
        f.write(json.dumps(vars(args), indent=4))
    
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
            f.write(file_name_stem + f"-[{averaged_metrics[metric_name]}]\n")


    print("done.")
        