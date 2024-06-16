import pickle
import sys
import logging
import argparse
import os
import numpy as np
import json

sys.path.append('/data/rech/huiyuche/TREC_iKAT_2024/src/')
# sys.path.append('../')
from topics import Topic, Reformulation, Result, load_topics_from_json, save_topics_to_json

from pyserini.search.lucene import LuceneSearcher
from pyserini.search import FaissSearcher
from pyserini.search import get_topics, get_qrels
import pytrec_eval






def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", type=str, default="ClueWeb_ikat", 
                        help="can be [ClueWeb_ikat]")
    parser.add_argument("--topics", type=str, default="ikat_23_train", 
                        help="can be [ikat_23_train]")
    parser.add_argument("--input_query_path", type=str, default="../../data/topics/trec_fire_covid_data.json")
    parser.add_argument("--index_dir_path", type=str, default="../../data/indexes/AP_sparse_index")
    parser.add_argument("--output_dir_path", type=str, default="../../results")
    parser.add_argument("--qrel_file_path", type=str, default="../../data/qrels/AP_qrels.1-150.txt")
    
    parser.add_argument("--query_type", type=str, default="title", 
                        help="""can be [
                            title, 
                            description, 
                            narrative, 
                            title+description, 
                            title+narrative, 
                            description+narrative, 
                            title+description+narrative,
                            reformulation,
                            pseudo_narraitve 
                            ]""")

    parser.add_argument("--retrieval_model", type=str, default="BM25",
                        help="can be [BM25, ance, dpr]")

    parser.add_argument("--reranker", type=str, default="rankllama",
                        help="can be [rankllama,]")


    parser.add_argument("--rewrite_model", type=str, default="gpt-4-turbo",
                        help="can be [gpt-4-turbo]")

    parser.add_argument("--prompt_type", type = str, default="real_narrative_prompt", help="could be one of [few_shot_narrative_prompt, complex_few_shot_narrative_prompt, real_narrative_prompt, complex_real_narrative_prompt,'few_shot_pseudo_narrative_only_prompt','complex_few_shot_pseudo_narrative_only_prompt'],see promptor.py for more details")


    parser.add_argument("--dense_query_encoder_path", type=str, default="castorini/ance-msmarco-passage",
                        help="should be a huggingface face format folder/link to a model") 
    parser.add_argument("--bm25_k1", type=float, default="0.9") # 0.82
    parser.add_argument("--bm25_b", type=float, default="0.4") # 0.68
    parser.add_argument("--top_k", type=int, default="1000")
    parser.add_argument("--metrics", type=str, default="map,ndcg_cut.5,ndcg_cut.10,P.5,P.10,recall.100",
                        help= "should be a comma-separated string of metrics, such as map,ndcg_cut.5,ndcg_cut.10,P.5,P.10,recall.100,recall.1000")

    #parser.add_argument("--rel_threshold", type=int, default="1")

    # auxiliary fonctionalities
    parser.add_argument("--run_reformulate", action="store_true", help="if we will call a rewrite to rewrite the query when no reformulation corresponding to the query_type is found.")
    parser.add_argument("--save_metrics_to_object",  action="store_true", help="if we will save metrics to topic object.")
    args = parser.parse_args()

    return args


    
def get_query_list(args):

    topic_list = []
    query_list = []
    qid_list_string = []

    # apply collection specific processing
    if args.collection == "AP":
        topic_list = load_topics_from_json(
            input_topic_path=args.input_query_path,
            range_start=0,
            range_end=-1
            )

        # note that qid starts from 1, not 0. but list index starts from 0.
        qid_list_string = [str(i+1) for i in range(0, 150)]

        # generate id list
        qid_list = [i for i in range(0, 150)]

        qid_list_string = [str(qid + 1) for qid in qid_list]

        # AP specific check
        assert topic_list[149].topic =="U.S. Political Campaign Financing", f"Last topic is not U.S. Political Campaign Financing but {topic_list[149].topic}, which is not right."

        # load query/reformulated query according to query type.
        # possible to call a llm to rewrite the query at this step.
        args.original_query_type = args.query_type
        query_list = [topic.query_type_2_query(args) for topic in topic_list[0:150]]
    

    assert len(query_list) != 0, "No queries found, args.collection may be wrong"
    assert len(query_list) == len(qid_list_string), "Number of queries and qid_list_string not match"
    # assert len(query_list) == len(topic_list), "Number of topics and qid_list_string not match"

    
    return query_list, qid_list_string, topic_list

def get_eval_results(args):

    # check args
    logger.info("Checking args...")
    assert args.topics in ["ikat_23_train",], f"Invalid topics {args.topics}"
    assert args.collection in ["ClueWeb_ikat",], f"Invalid collection {args.collection}"
    assert args.retrieval_model in ["BM25", "ance", "dpr"], f"Invalid retrieval model {args.retrieval_model}"
    assert args.reranker in ["rankllama"], f"Invalid reranker {args.reranker}"
    assert args.query_type in ["title", "description", "narrative", "title+description", "title+narrative", "description+narrative", "title+description+narrative","reformulation","pseudo_narrative"], f"query type {args.query_type} is not an invalid query_type"

    assert args.prompt_type in ['few_shot_narrative_prompt', 'complex_few_shot_narrative_prompt', 'real_narrative_prompt', 'complex_real_narrative_prompt', 'few_shot_pseudo_narrative_only_prompt','complex_few_shot_pseudo_narrative_only_prompt'],f"Prompt type {args.prompt_type} is not implemented."


    assert os.path.exists(args.input_query_path), "Input query file not found"
    assert os.path.exists(args.index_dir_path), "Index dir not found"
    assert os.path.exists(args.qrel_file_path), "Qrel file not found"
    assert os.path.exists(args.output_dir_path), "Output dir not found"


    ###################################################
    # get query list and qid list as well as Topic list
    ##################################################
    logger.info(f"loading quries, run_reformulate is {args.run_reformulate}...")

    # the reason to get topic list is to add per-query 
    # search results. 
    query_list, qid_list_string, topic_list = get_query_list(args)

        
    ##############################
    # pyserini search
    ##############################

    # sparse search
    if args.retrieval_model == "BM25":
        logger.info("BM 25 searching...")
        searcher = LuceneSearcher(args.index_dir_path)
        searcher.set_bm25(args.bm25_k1, args.bm25_b)
        hits = searcher.batch_search(query_list, qid_list_string, k = args.top_k, threads = 40)

    # dense search
    elif args.retrieval_model == "ance" or args.retrieval_model == "dpr":
        logger.info(f"{args.retrieval_model} searching...")
        searcher = FaissSearcher(
            args.index_dir_path,
            args.dense_query_encoder_path 
        )
        hits = searcher.batch_search(query_list, qid_list_string, k = args.top_k, threads = 40)



    # generate ranking list file name
    # first generate an identifiable name for the ranking list
    file_name_stem = f"{args.collection}-{args.query_type}-{args.retrieval_model}-top{args.top_k}"
    # then add a suffix
    ranking_list_path = os.path.join(
        args.output_dir_path,
        args.collection, 
        "ranking",
        file_name_stem + ".txt")

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
    # use pytrec_eval to evaluate
    ##############################

    # read qrels
    with open(args.qrel_file_path, 'r') as f_qrel:
        qrel = pytrec_eval.parse_qrel(f_qrel)
    # read ranking list
    with open(ranking_list_path, 'r') as f_run:
        run = pytrec_eval.parse_run(f_run)

    # first read metrics
    metrics_list = args.metrics.split(",")

    # then evaluate
    logger.info("trec_eval evaluating...")
    evaluator = pytrec_eval.RelevanceEvaluator(qrel, set(metrics_list))
    query_metrics_dic = evaluator.evaluate(run)


    # for each metric, extract the result for all queries
    # then we can calculate the average of each metric.
    metrics_list_key_form = [metric.replace(".", "_") for metric in metrics_list]
    metrics = {metric : [metrics[metric] for metrics in query_metrics_dic.values()] for metric in metrics_list_key_form}

    # average metrics
    averaged_metrics = { metric : np.average(metric_list) for metric, metric_list in metrics.items() }

    # for each query, calculate the difference between the metric value and the average value
    # should be in format
    # { metric:[{qid1:diff2}, {qid2:diff2}, ...]}
    # diff_metrics = { metric : [metric_value - averaged_metrics[metric] for metric_value in metric_list] for metric, metric_list in metrics.items()}

    # for each query, add the number of relevant documents in query_metrics_dic
    # why not use sum(list(qrel[qid].values()))? Because relevance judgement may be graded instead of binary.
    for qid in query_metrics_dic.keys():
        query_metrics_dic[qid]["num_rel"] = sum([1 for doc in qrel[qid].values() if doc > 0])

    return query_metrics_dic, averaged_metrics, topic_list

if __name__ == "__main__":

    # get args
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    logger.info(args)

    # first generate an identifiable name for the ranking list
    file_name_stem = f"{args.collection}-{args.query_type}-{args.retrieval_model}-top{args.top_k}"

    # read metrics
    metrics_list = args.metrics.split(",")
    metrics_list_key_form = [metric.replace(".", "_") for metric in metrics_list]

        # create output dir if not exist
    if not os.path.exists(args.output_dir_path):
        os.makedirs(args.output_dir_path)
    if not os.path.exists(os.path.join(args.output_dir_path, args.collection)):
        os.makedirs(os.path.join(args.output_dir_path, args.collection))
    if not os.path.exists(os.path.join(args.output_dir_path, args.collection,"ranking")):
        os.makedirs(os.path.join(args.output_dir_path, args.collection,"ranking"))
    if not os.path.exists(os.path.join(args.output_dir_path, args.collection, "metrics")):
        os.makedirs(os.path.join(args.output_dir_path, args.collection, "metrics"))
    if not os.path.exists(os.path.join(args.output_dir_path, args.collection, "per_query_metrics")):
        os.makedirs(os.path.join(args.output_dir_path, args.collection, "per_query_metrics"))

    # evaluate
    query_metrics_dic, averaged_metrics, topic_list = get_eval_results(args)

    # write results to topic list and save.
    if args.save_metrics_to_object:
        for qid, result_dict in query_metrics_dic.items():
            for topic in topic_list:
                if str(topic.id) == qid:
                    reformulation = topic.find_reformulation(args.query_type)
                    assert reformulation != None, f"Reformulation {args.query_type} not found in topic {topic.id}"
                    reformulation.add_result(args.collection, args.retrieval_model, result_dict)
        save_topics_to_json(
            topic_list,
            args.input_query_path
        )

    # save metrics
    logger.info("saving results...")

    metrics_path = os.path.join(
        args.output_dir_path, 
        args.collection, 
        "metrics",
        file_name_stem + ".json")

    # save metric dictionary
    metrics_dict_path = os.path.join(
        args.output_dir_path,
        args.collection,
        "per_query_metrics",
        file_name_stem + "_dict.json")

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
            args.output_dir_path, 
            args.collection, 
            "metrics",
            f"{metric_name}.txt")
        
        # append a line in this file in the following format:
        # {collection} {query_type} {retrieval_model} {top_k} {metric_value}
        with open(metric_file_path, "a") as f:
            f.write(f"{args.collection} {args.query_type} {args.retrieval_model} {args.top_k} {averaged_metrics[metric_name]}\n")


    logger.info("done.")
    
        
        