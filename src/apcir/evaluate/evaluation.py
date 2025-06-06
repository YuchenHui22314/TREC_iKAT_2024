import logging
import argparse
import os
import json

import wandb

from apcir.functional.topics import (
    save_turns_to_json, 
    load_turns_from_json
    )
    
from apcir.search.search import search    
from apcir.functional.response_generation import (
    generate_responses
    ) 

from .evaluation_util import (
    get_query_list,
    evaluate,
    generate_and_save_ikat_submission,
    print_formatted_latex_metrics
)


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
    parser.add_argument("--output_dir_path", type=str, default="../../results")
    parser.add_argument("--qrel_file_path", type=str, default="../../data/qrels/ikat_23_qrel.txt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--machine", type=str, default="octal31",help="which lab machien to use. Optional. No effect on the program.")

    ###################
    #### retrieval ####
    ###################

    parser.add_argument("--retrieval_model", type=str, default="BM25",
                        choices= ["none","BM25", "ance", "dpr", "splade_v3", "repllama"])
    parser.add_argument("--retrieval_top_k", type=int, default="1000")
    parser.add_argument("--personalization_group", type=str, default="a", 
                        choices=["a","b","c","all"]
                        )

    
    # splade
    parser.add_argument("--splade_query_encoder_path", type=str, default="castorini/ance-msmarco-passage", help="should be a huggingface face format folder/link to a model") 
    parser.add_argument("--splade_index_dir_path", type=str, default="../../data/indexes/clueweb22b_ikat23_fengran_sparse_index_2")

    # dense & splade
    parser.add_argument("--use_pyserini_dense_search", action="store_true", help="if we will use pyserini dense search or our own dense search implementation.")
    parser.add_argument("--dense_query_encoder_path", type=str, default="castorini/ance-msmarco-passage", help="should be a huggingface face format folder/link to a model") 
    parser.add_argument("--dense_index_dir_path", type=str, default="../../data/indexes/clueweb22b_ikat23_fengran_sparse_index_2")
    parser.add_argument("--query_gpu_id", type=int, default=1)
    parser.add_argument("--query_encoder_batch_size", type=int, default=200)

    # faiss
    parser.add_argument("--faiss_n_gpu", type=int, default=4)
    parser.add_argument("--use_gpu_for_faiss", action="store_true")
    parser.add_argument("--embed_dim", type=int, default=768)
    parser.add_argument("--tempmem", type=int, default=-1)
    parser.add_argument("--passage_block_num", type=int, default=116)

    # BM25 parameters
    parser.add_argument("--sparse_index_dir_path", type=str, default="../../data/indexes/clueweb22b_ikat23_fengran_sparse_index_2")
    parser.add_argument("--bm25_k1", type=float, default="0.9") # 0.82
    parser.add_argument("--bm25_b", type=float, default="0.4") # 0.68

    # Query expansion 
    parser.add_argument("--qe_type", type = str, default="rm3", choices=["rm3", "none", "llm_rm"])
    # pseudo relevance feedback parameters
    parser.add_argument("--fb_terms", type=int, default="10", help="RM3 parameter for number of expansion terms.")
    parser.add_argument("--fb_docs", type=int, default="10", help="RM3 parameter for number of expansion documents.")
    parser.add_argument("--original_query_weight", type=float, default="0.5", help="RM3 parameter for weight to assign to the original query.") 


    ###################
    #### Fusion ####
    ###################
    parser.add_argument("--fusion_type", type=str, default="none",
                        choices=['none','linear_weighted_score','linear_combination','round_robin','per_query_personalize_level', "RRF", 'gpt-4o_judge_and_rewrite',"concat", "per_query_personalize_level",
                        "pre_calculated", "random_weights"])
    parser.add_argument('--QRs_to_rank', type=str, nargs='+', default=["Cloud_Z", "Miyoko"], help='List of reformulation names to fuse')
    parser.add_argument('--fuse_weights', type=float, nargs='+', default = [1,0.1,0.4], help='weights for linear weighted score fusion')
    parser.add_argument("--fusion_normalization", type=str, default="none",
                        choices=['none','max','min-max'])
    parser.add_argument("--level_type", type=str, default="none", help="how did we get the personalized level")
    parser.add_argument("--per_query_weight_max_value", type=float, default=0.75)
    parser.add_argument("--optimize_level_weights", type=str, default="false", choices=["false", "group", "2+1", "no_level","retrieval_score"])
    parser.add_argument("--target_metrics", type=str, default="ndcg@3,mrr", help="Please see https://amenra.github.io/ranx/ for possible metrics." )
    parser.add_argument("--optimize_step", type=float, default=0.1, help="" )
    parser.add_argument("--top_docs", type=int, default=100, help="" )


    ###################
    #### rerank    ####
    ###################
    parser.add_argument("--reranker", type=str, default="none",
                        choices=['none', 'rankllama', 'rankgpt', 'monot5_base','monot5_base_10k', 'monot5_large', 'monot5_large_10k',
                        "monot5_3b",
                        "monot5_3b_10k",
                        ])
    parser.add_argument("--rerank_top_k", type=int, default="50")
    # hugging_face cache_dir
    parser.add_argument("--cache_dir", type=str, default="/data/rech/huiyuche/huggingface", help="cache directory for huggingface models")

    # on octal31: 67 for monot5_base, 10 for rankllama 
    # on octal40: TBD
    parser.add_argument("--rerank_batch_size", type=int, default=67) 
    # rankllama
    parser.add_argument("--rerank_quant", type=str, default="none",
                        choices=['none','8b','4b'])
    # rankGPT
    parser.add_argument("--rankgpt_llm", type=str, default="gpt-3.5-turbo",
                        choices=['gpt-3.5-turbo'])
    parser.add_argument("--window_size", type=int, default="5") 
    parser.add_argument("--step", type=int, default="1") 



    # response generation:
    parser.add_argument("--generation_model", type=str, default="none",
                        choices=['none','gpt-4o-2024-08-06'])
    parser.add_argument("--generation_prompt", type=str, default="none",
                        choices=['none','raw'])
    parser.add_argument("--generation_top_k", type=int, default="3")


    # after evaluation
    parser.add_argument("--save_to_wandb", action="store_true", help="if we will save the results to wandb.")
    parser.add_argument("--metrics", type=str, default="map,ndcg_cut.1,ndcg_cut.3,ndcg_cut.5,ndcg_cut.10,P.1,P.3,P.5,P.10,recall.5,recall.50,recall.100,recall.1000,recip_rank",
                        help= "should be a comma-separated string of metrics, such as map,ndcg_cut.5,ndcg_cut.10,P.5,P.10,recall.50,recall.100,recall.1000")

    parser.add_argument("--metrics_to_print", type=str, nargs='+', default=["recip_rank", "ndcg_cut_3", "recall_10", "recall_100"])

    parser.add_argument("--save_results_to_object",  action="store_true", help="if we will save eval results (metrics + response) to turn/topic object.")

    parser.add_argument("--save_ranking_list",  action="store_true", help="if we will save ranking list yieled by the search component.")

    parser.add_argument("--run_rag", action="store_true", help="if we will run the search + generation component (retrieval + reranking + generation, rag)")

    parser.add_argument("--given_ranking_list_path", type=str, default="none", help="when retrieval_model == 'none', we have to provide the path of a given ranking list then rerank.")


    parser.add_argument("--run_eval",  action="store_true", help="if we will run the evaluation component (+ saving)")



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
                            "gpt-4o_rar_rs", 
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
                            "gpt-4o_rar_personalized_cot1_rs",
                            "gpt-4o_rar_personalized_cot1_rwrs",
                            "gpt-4o_rar_rw_fuse_rar_personalized_cot1_rw",
                            "gpt-4o_rar_rwrs_fuse_personalized_cot1_rw",
                            "gpt-4o_rar_rw_fuse_rar_rwrs",
                            "gpt-4o_rar_rw_fuse_rar_rwrs_fuse_personalized_cot1_rw",
                            "gpt-4o_rar_rw_fuse_rar_personalized_cot1_rwrs",
                            'gpt-4o_rar_non_personalized_cot1_rw',
                            "gpt-4o_rar_rw_fuse_rar_rwrs_fuse_non_personalized_cot1_rw",
                            "gpt-4o_rar_manual_depersonalized_cot1_rw",
                            "gpt-4o_rar_rw_fuse_rar_rwrs_fuse_manual_depersonalized_cot1_rw",
                            "gpt-4o_rar_rw+gpt-4o_rar_rwrs+gpt-4o_rar_personalized_cot1_rw",
                            "round_robin_gpt-4o_3_lists",
                            "personalize_level_3_lists_tune",
                            "gpt-4o_judge_and_rewrite_rw",
                            "gpt-4o_judge_and_rewrite_rs",
                            "gpt-4o_judge_and_rewrite_rwrs",
                            "gpt-4o_judge_and_rewrite_optimize_test",
                            "gpt-4o_judge_and_rewrite_optimize_mrr_test",
                            "gpt-4o_judge_and_rewrite_optimize_mrr_non_normalize_test",
                            "gpt-4o_judge_and_rewrite_optimize_mrr_non_normalize_test",
                            "original_optimize_test",
                            "2+1_test",
                            "gpt-4o_judge_and_rewrite_optimize_4_test",
                            "gpt-3.5_judge_and_rewrite_rw",
                            "gpt-4o_rar_rw_fuse_judge_and_rewrite_rw",
                            "gpt-4o_rar_rw_fuse_judge_and_rewrite_rwrs",
                            "gpt-4o_rar_rw_fuse_rar_rwrs",
                            "gpt-4o_rar_rwrs_fuse_judge_and_rewrite_rw",
                            "gpt-4o_rar_rwrs_fuse_judge_and_rewrite_rwrs",
                            "gpt-4o_judge_and_rewrite_rw_fuse_rwrs",
                            "gpt-4o_rar_rw_fuse_rar_rwrs_fuse_judge_and_rewrite_rw",
                            "gpt-4o_rar_rw_fuse_rar_rwrs_fuse_judge_and_rewrite_rwrs",
                            "gpt-4o_rar_rw_fuse_judge_and_rewrite_rwrs_fuse_judge_and_rewrite_rw",
                            "gpt-4o_rar_rwrs_fuse_judge_and_rewrite_rwrs_fuse_judge_and_rewrite_rw",
                            "alter_selon_manual_ptkb",
                            "alter_selon_manual_ptkb_rs",
                            "gpt-3.5_judge_and_rewrite_optimize_4_test",
                            "llama3.1_rar_rw",
                            "llama3.1_rar_rwrs",
                            "mistral_rar_rw",
                            "mistral_rar_rwrs",
                            "mistral_judge_and_rewrite_rw",
                            "mistral_judge_and_rewrite_rwrs",
                            "llama3.1_judge_and_rewrite_rw",
                            "llama3.1_judge_and_rewrite_rwrs",
                            "mistral_judge_and_rewrite_optimize_4_test",
                            "llama3.1_judge_and_rewrite_optimize_4_test",
                            "2024_submission",
                            "rerun_submission_from_rklist",
                            "round_robin_3_lists",
                            "RRF_3_lists",
                            "gpt-4o_rar_rw+gpt-4o_rar_rwrs+gpt-4o_judge_and_rewrite_rw",
                            "gpt-4o_judge_and_rewrite_optimize_4_test_no_normalize",
                            "wo_explicit_level",
                            "wo_finegrianed",
                            "gpt-4o_rar_rw_fuse_rar_rwrs_fuse_manual_depersonalized_cot1_rw_optimize_4",
                            "gpt-4o_judge_and_rewrite_depers_rw",
                            "gpt-4o_rar_rw_fuse_rar_rwrs_fuse_depers_jar_optimize_4",
                            "gpt-4o_MQ4CS_mq_1",
                            "gpt-4o_MQ4CS_mq_2",
                            "MQ4CS_low_resource",
                            "ours_low_resource",
                            "ours_low_resource_rerank2000",
                            "ours_low_resource_rerank1000",
                            "gpt-4o_MQ4CS_persq_rw",
                            "gpt-4o_jtr_wo_in_context_rw",
                            "gpt-4o_jtr_wo_cot_rw",
                            "wo_cot",
                            "wo_in_context",
                            "ours_low_resource_100",
                            "gpt-4o_MQ4CS_mq_3",
                            "gpt-4o_MQ4CS_mq_3_1",
                            "gpt-4o_MQ4CS_mq_3_2",
                            "gpt-4o_MQ4CS_mq_3_3",
                            "MQ4CS_low_resource_3",
                            "MQ4CS_low_resource_3_RRF",
                            "MQ4CS_low_resource_3_Round_Robin",
                            "MQ4CS_low_resource_3_grid_search",
                            "gpt-4o_GtR_mq_3_1",
                            "gpt-4o_GtR_mq_3_2",
                            "gpt-4o_GtR_mq_3_3",
                            "gpt-4o_GtR_low_resource_3_Round_Robin",
                            "gpt-4o_GtR_low_resource_3_RRF",
                            "gpt-4o_GtR_low_resource_3",
                            "gpt-4_MQ4CS_persq_rw",
                            "gpt-4o_GtR_low_resource_3_grid_search",
                            "gpt-4o_judge_and_rewrite_optimize_retrieval_score",
                            "gpt-3.5_MQ4CS_persq_rw",
                            "llama3.1_MQ4CS_persq_rw",
                            "mistral_MQ4CS_persq_rw",
                            "result_topic_entropy",
                            "DEPS",
                            "random_weights"
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
                            'gpt-4o_judge_and_rewrite_rw',
                            "gpt-4o_MQ4CS_persq_rw"
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
    print(json.dumps(vars(args), indent=4))
    # print current time:
    print("current time: ", os.popen('date').read())
    

    ###############
    # check args
    ###############
    assert os.path.exists(args.qrel_file_path), "Qrel file not found"
    assert os.path.exists(args.output_dir_path), "Output dir not found"

    #########################################################
    #  generate an identifiable name for current run
    #########################################################
    qe = ""
    if args.qe_type == "rm3":
        qe = f"_rm3"

    personalization_group = args.personalization_group
    if personalization_group == "all":
        personalization_group = ""
    else:
        personalization_group = f"_{personalization_group}"
    
    file_name_stem = f"S1[{args.retrieval_query_type}{personalization_group}]-S2[{args.reranking_query_type}]-g[{args.generation_query_type}]-[{args.retrieval_model}{qe}]-[{args.reranker}_{args.window_size}_{args.step}_{args.rerank_quant}]-[s2_top{args.rerank_top_k}]"

    file_name_stem_without_group = f"S1[{args.retrieval_query_type}]-S2[{args.reranking_query_type}]-g[{args.generation_query_type}]-[{args.retrieval_model}{qe}]-[{args.reranker}_{args.window_size}_{args.step}_{args.rerank_quant}]-[s2_top{args.rerank_top_k}]"


    print("#######################################")
    print("the file name stem is: ", file_name_stem)
    print("#######################################")

    #### WANDB initialization ####
    topic_name_map = {  
        "ikat_23_test": "TREC_iKAT_2023",
        "ikat_24_test": "TREC_iKAT_2024"  
    }
    project_name = topic_name_map[args.topics] 

    if args.save_to_wandb:
        wandb.init(
            project=project_name, 
            name=file_name_stem
            )
        

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
        qid_personalized_level_dict,
        qid_weights_dict,
        turn_list 
        ) =  get_query_list(args)

    if args.run_rag:

        ##########################
        # Search
        ##########################
        args.ranking_list_path = ranking_list_path
        args.file_name_stem = file_name_stem
        args.file_name_stem_without_group = file_name_stem_without_group

        args.retrieval_query_list = retrieval_query_list
        args.reranking_query_list = reranking_query_list
        args.fusion_query_lists = fusion_query_lists
        args.qid_list_string = qid_list_string
        args.qid_personalized_level_dict = qid_personalized_level_dict
        args.qid_weights_dict = qid_weights_dict

        ##################################
        #### Personalization Specific ####
        ##################################

        
        hits, run = search(
            args
            )
        
        # delet redundant keys in args
        del args.retrieval_query_list
        del args.reranking_query_list
        del args.fusion_query_lists
        del args.qid_list_string
        if "qid_personalized_level_dict" in args:
            del args.qid_personalized_level_dict
        if "qid_weights_dict" in args:
            del args.qid_weights_dict 


        ##########################
        # response generation 
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
            run,
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

            # first re-read the turns from the json file, because 
            # it may subject to change during the search component.
            turn_list = load_turns_from_json(
                input_topic_path=args.input_query_path,
                range_start=0,
                range_end=-1
                )

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


        # save formatted metrics
        formatted_metrics = print_formatted_latex_metrics(
            averaged_metrics,
            args.metrics_to_print
        )

        print("Print this line to your latex table:")
        print("-------------------------------------")
        print("    ", formatted_metrics)
        print("-------------------------------------")
        

        with open(metrics_path, "w") as f:
            f.write("Print this line to your latex table:\n")
            f.write("-------------------------------------\n")
            f.write("    " + formatted_metrics + "\n")
            f.write("-------------------------------------\n")

        # save metrics  
        with open(metrics_path, "a") as f:
            f.write("\n")
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

        if args.save_to_wandb:
            wandb.config.update(vars(args))
            wandb.run.summary.update(averaged_metrics)
            wandb.run.summary["formatted_metrics"] = formatted_metrics

    print("done.")
        