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
    )

    
from response_generation import (
    generate_responses
    ) 

from evaluation_util import (
    get_query_list,
    search,
    evaluate,
    generate_and_save_ikat_submission,
    extract_filename
)

from pyserini.search import get_topics, get_qrels
import pytrec_eval





def get_args():
    parser = argparse.ArgumentParser()

    #########################
    # Some general settings
    ########################

    parser.add_argument("--collection", type=str, default="ClueWeb_ikat", 
                        choices=["ClueWeb_ikat"])

    parser.add_argument("--topics", type=str, default="ikat_23_test",
                        choices = ["ikat_23_test", "ikat_24_test"])

    parser.add_argument("--input_query_path", type=str, default="../../data/topics/ikat_2023_test.json")

    parser.add_argument("--index_dir_path", type=str, default="../../data/indexes/clueweb22b_ikat23_fengran_sparse_index_2")

    parser.add_argument("--output_dir_path", type=str, default="../../results")

    parser.add_argument("--qrel_file_path", type=str, default="../../data/qrels/ikat_23_qrel.txt")
    
    parser.add_argument("--retrieval_model", type=str, default="BM25",
                        choices= ["none","BM25", "ance", "dpr", "splade", "repllama"])

    # hugging_face cache_dir
    parser.add_argument("--cache_dir", type=str, default="/data/rech/huiyuche/huggingface", help="cache directory for huggingface models")
    ## TODO: unify -- cache_dir and --dense_query_encoder_path

    # where is dense encoder
    parser.add_argument("--dense_query_encoder_path", type=str, default="castorini/ance-msmarco-passage",
                        help="should be a huggingface face format folder/link to a model") 


    parser.add_argument("--reranker", type=str, default="none",
                        choices=['none', 'rankllama', 'rankgpt', 'monot5_base','monot5_base_10k'])

    # rankllama
    parser.add_argument("--rerank_quant", type=str, default="none",
                        choices=['none','8b','4b'])
    # rankGPT
    parser.add_argument("--rankgpt_llm", type=str, default="gpt-3.5-turbo",
                        choices=['gpt-3.5-turbo'])
    parser.add_argument("--window_size", type=int, default="5") 
    parser.add_argument("--step", type=int, default="1") 


    # BM25 parameters
    parser.add_argument("--bm25_k1", type=float, default="0.9") # 0.82
    parser.add_argument("--bm25_b", type=float, default="0.4") # 0.68

    # Query expansion 
    parser.add_argument("--qe_type", type = str, default="rm3", choices=["rm3", "none", "llm_rm"])
    # pseudo relevance feedback parameters
    parser.add_argument("--fb_terms", type=int, default="10", help="RM3 parameter for number of expansion terms.")
    parser.add_argument("--fb_docs", type=int, default="10", help="RM3 parameter for number of expansion documents.")
    parser.add_argument("--original_query_weight", type=float, default="0.5", help="RM3 parameter for weight to assign to the original query.") 

    # response generation:
    parser.add_argument("--generation_model", type=str, default="none",
                        choices=['none','gpt-4o-2024-08-06'])
    parser.add_argument("--generation_prompt", type=str, default="none",
                        choices=['none','raw'])

    # distinguish retrieval topk and rerank topk, as well as generation top k
    parser.add_argument("--retrieval_top_k", type=int, default="1000")
    parser.add_argument("--rerank_top_k", type=int, default="50")
    parser.add_argument("--generation_top_k", type=int, default="3")

    parser.add_argument("--metrics", type=str, default="map,ndcg_cut.1,ndcg_cut.3,ndcg_cut.5,ndcg_cut.10,P.1,P.3,P.5,P.10,recall.5,recall.50,recall.100,recall.1000,recip_rank",
                        help= "should be a comma-separated string of metrics, such as map,ndcg_cut.5,ndcg_cut.10,P.5,P.10,recall.50,recall.100,recall.1000")

    parser.add_argument("--save_results_to_object",  action="store_true", help="if we will save eval results (metrics + response) to turn/topic object.")

    parser.add_argument("--save_ranking_list",  action="store_true", help="if we will save ranking list yieled by the search component.")

    parser.add_argument("--run_rag", action="store_true", help="if we will run the search + generation component (retrieval + reranking + generation, rag)")

    parser.add_argument("--given_ranking_list_path", type=str, default="none", help="when retrieval_model == 'none', we have to provide the path of a given ranking list then rerank.")


    parser.add_argument("--run_eval",  action="store_true", help="if we will run the evaluation component (+ saving)")

    #### Fusion ####
    parser.add_argument("--fusion_type", type=str, default="none",
                        choices=['none','linear_weighted_score'])

    parser.add_argument('--QRs_to_rank', type=str, nargs='+', default=["Cloud_Z", "Miyoko"], help='List of reformulation names to fuse')
    parser.add_argument('--fuse_weights', type=float, nargs='+', default = [2.71828], help='weights for linear weighted score fusion')


    ####################################
    # ikat 2024 project related config
    ###################################

    parser.add_argument("--run_name", type=str, default="none",
                        help="run name for trec ikat submission. If none, will use file name stem as run name.")

    parser.add_argument("--retrieval_query_type", type=str, default="oracle", 
                        choices=[
                            "none",
                            "raw", 
                            "oracle",
                            "rar_rwrs",
                            "rar_rw",
                            "rar_cot_rw",
                            "rar_cot_rwrs",
                            "gpt-4o_rar_rw", 
                            "gpt-4o_rar_rwrs",
                            "gpt-4o_rar_cot_rw",
                            "gpt-4o_rar_cot_rwrs",
                            # P -> personalize, D -> demo, C -> cot, Re -> rel explain
                            # O -> oracle, Rf -> rel feedback
                            "raw_llm_rm_PDCReORf",
                            "raw_llm_rm_P__Re___",
                            "raw_llm_rm____Re___",
                            "rar_ptkb_sum_cot0_rw",
                            "rar_ptkb_sum_cot0_rwrs",
                            "rar_ptkb_sum_rw",
                            "rar_ptkb_sum_rwrs",
                            'rar_personalized_cotN_rwrs',
                            'rar_personalized_cot1_rwrs',
                            'rar_personalized_cotN_rw',
                            'rar_personalized_cot1_rw',
                            'rar_rw_fuse_rar_personcot1_rw',
                            "rar_rwrs_fuse_rar_personcot1_rwrs",
                            "rar_rw_fuse_rar_rwrs",
                            "rar_rw_fuse_rar_rwrs_fuse_rar_personalized_cot1_rw",
                            "rar_rw_fuse_rar_personalized_cot1_rwrs",
                            "gpt-4o_rar_personalized_cot1_rw",
                            "gpt-4o_rar_personalized_cot1_rwrs",
                            "gpt-4o_rar_rw_fuse_rar_personalized_cot1_rw",
                            "gpt-4o_rar_rwrs_fuse_personalized_cot1_rw",
                            "gpt-4o_rar_rw_fuse_rar_rwrs",
                            "gpt-4o_rar_rw_fuse_rar_rwrs_fuse_personalized_cot1_rw",
                            "gpt-4o_rar_rw_fuse_rar_personalized_cot1_rwrs",
                            'gpt-4o_rar_non_personalized_cot1_rw',
                            "gpt-4o_rar_rw_fuse_rar_rwrs_fuse_non_personalized_cot1_rw",
                            "gpt-4o_rar_manual_depersonalized_cot1_rw",
                            "gpt-4o_rar_rw_fuse_rar_rwrs_fuse_manual_depersonalized_cot1_rw"

                            ],)

    parser.add_argument("--reranking_query_type", type=str, default="oracle_utterance", 
                        choices=[
                            "none",
                            "raw", 
                            "oracle",
                            "rar_rw",
                            "raw_llm_rm_P__Re___",
                            "raw_llm_rm____Re___",
                            "rar_ptkb_sum_cot0_rw",
                            "rar_ptkb_sum_cot0_rwrs",
                            "rar_ptkb_sum_rw",
                            "rar_ptkb_sum_rwrs",
                            "rar_personalized_cotN_rw",
                            "rar_personalized_cot1_rw",
                            "gpt-4o_rar_personalized_cot1_rw",
                            'gpt-4o_rar_non_personalized_cot1_rw',
                            ],)

    parser.add_argument("--generation_query_type", type=str, default="oracle_utterance", 
                        choices=[
                            "none",
                            "raw", 
                            "oracle",
                            "rar_rw",
                            "rar_ptkb_sum_cot0_rw",
                            "rar_ptkb_sum_rw",
                            "rar_personalized_cot1_rw",
                            "rar_personalized_cotN_rw",
                            "gpt-4o_rar_personalized_cot1_rw",
                            'gpt-4o_rar_non_personalized_cot1_rw',
                            ],)



    args = parser.parse_args()
    return args

    


if __name__ == "__main__":

    ##########
    # get args
    ##########

    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    print(args)

    ###############
    # check args
    ###############

    assert os.path.exists(args.input_query_path), "Input query file not found"
    assert os.path.exists(args.index_dir_path), "Index dir not found"
    assert os.path.exists(args.qrel_file_path), "Qrel file not found"
    assert os.path.exists(args.output_dir_path), "Output dir not found"

    

    #########################################################
    #  generate an identifiable name for current run
    #########################################################
    qe = ""
    if args.qe_type == "rm3":
        qe = f"_rm3"
    file_name_stem = f"S1[{args.retrieval_query_type}]-S2[{args.reranking_query_type}]-g[{args.generation_query_type}]-[{args.retrieval_model}{qe}]-[{args.reranker}_{args.window_size}_{args.step}_{args.rerank_quant}]-[s2_top{args.rerank_top_k}]"

    ###################################################################
    #  generate folder paths where the evaluation results will be saved
    ###################################################################

    base_folder = os.path.join(args.output_dir_path, args.collection, args.topics)

    # create necessary directories 
    subdirs = ["ranking", "metrics", "per_query_metrics", "ikat_format_output"]
    for subdir in subdirs:
        # create output dir if not exist
        path = os.path.join(base_folder, subdir)
        os.makedirs(path, exist_ok=True)

    # ranking list path
    ranking_list_path = os.path.join(
        base_folder,
        "ranking",
        file_name_stem + ".txt")
    

    # run all metrics path
    metrics_path = os.path.join(
        base_folder,
        "metrics",
        file_name_stem + ".json")

    # per query metrics dictionary path
    metrics_dict_path = os.path.join(
        base_folder,
        "per_query_metrics",
        file_name_stem + "_dict.json")

    # ikat format output path
    ikat_output_path = os.path.join(
        base_folder,
        "ikat_format_output",
        args.run_name + ".json")

    ###################################################
    # get query list and qid list as well as Turn list
    ###################################################

    print(f"loading quries")

    # get query lsit for retreival, reranking, generation, and fusion, as well as qid list.
    # the reason to get turn list is to add per-query search results. 
    (
        retrieval_query_list, 
        reranking_query_list, 
        generation_query_list, 
        fusion_query_lists, 
        qid_list_string, 
        turn_list 
        ) =         get_query_list(args)

    if args.run_rag:

        ##########################
        # Search
        ##########################
        args.ranking_list_path = ranking_list_path
        args.file_name_stem = file_name_stem

        args.retrieval_query_list = retrieval_query_list
        args.reranking_query_list = reranking_query_list
        args.fusion_query_lists = fusion_query_lists
        args.qid_list_string = qid_list_string

        hits, run = search(
            args
            )

        ##########################
        # response generation TODO
        ##########################

        response_dict = generate_responses(
            turn_list,
            hits, 
            generation_query_list,
            qid_list_string,
            args
            ) 

        ##############################
        #  Export to ikat format
        ##############################
        print("generating ikat format results...")
        generate_and_save_ikat_submission(
            ikat_output_path,
            args.run_name,
            # TODO: other mechanism for choosing the correct ptkb_provenance ...
            args.reranking_query_type,
            hits,
            turn_list,
            response_dict,
            args.generation_top_k
        )

    if args.run_eval:

        ##########################
        # evaluate ranking
        ##########################


        ##############################
        # TODO: evaluate ptkb ranking list 
        ##############################
        ####################################
        # TODO: evaluate generation quality 
        ####################################
        # process metrics 
        metrics_list = args.metrics.split(",")
        metrics_list_key_form = [metric.replace(".", "_") for metric in metrics_list]

        # evaluate
        query_metrics_dic, averaged_metrics = evaluate(
            None,
            args.qrel_file_path,
            ranking_list_path,
            metrics_list,
            metrics_list_key_form
            )

        ##########################
        # saving evaluation results
        ##########################
        # write results to topic list and save.

        if args.save_results_to_object:

            for qid, result_dict in query_metrics_dic.items():

                if args.run_rag:
                    response = response_dict[qid][0]
                else:
                    response = "rag_not_run, no response."

                for turn in turn_list:
                    if str(turn.turn_id) == qid:
                        turn.add_result(
                            args.collection, 
                            args.retrieval_model, 
                            args.reranker,
                            args.generation_model,
                            args.retrieval_query_type,
                            args.reranking_query_type,
                            args.generation_query_type,
                            result_dict,
                            response
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
        