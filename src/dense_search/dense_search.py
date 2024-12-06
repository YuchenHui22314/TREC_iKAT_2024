import csv
import argparse
import os
from os import path
from os.path import join as oj
import json
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
import faiss
import time
import copy
import pickle
import torch
import numpy as np
import pytrec_eval
from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer

from models import ANCE
from utils import check_dir_exist_or_build, pstore, pload, set_seed, get_optimizer
from data_format import padding_seq_to_same_length, Retrieval_qrecc, Retrieval_topiocqa, Search_q_Retrieval, Retrieval_trec

'''
Test process, perform dense retrieval on collection (e.g., MS MARCO):
2. establish index with Faiss on GPU for fast dense retrieval
3. load the model, build the test query dataset/dataloader, and get the query embeddings. 
4. iteratively searched on each passage block one by one to got the retrieved scores and passge ids for each query.
5. merge the results on all pasage blocks
6. output the result
'''

def build_faiss_index(args):
    '''
    Build the Faiss index for dense retrieval.
    args.n_gpu: int, the number of gpus
    args.use_gpu: bool, whether to use gpu 
    args.embed_dim: int, the dimension of embeddings
    args.tempmem: int, the temporary memory for Faiss index
    '''
    print("Building index...")
    # ngpu = faiss.get_num_gpus()
    ngpu = args.n_gpu
    gpu_resources = []
    tempmem = args.tempmem

    for i in range(ngpu):
        res = faiss.StandardGpuResources()
        if tempmem >= 0:
            res.setTempMemory(tempmem)
        gpu_resources.append(res)

    cpu_index = faiss.IndexFlatIP(args.embed_dim)  
    index = None
    if args.use_gpu:
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.usePrecomputed = False
        # gpu_vector_resources, gpu_devices_vector
        vres = faiss.GpuResourcesVector()
        vdev = faiss.Int32Vector()
        for i in range(0, ngpu):
            vdev.push_back(i)
            vres.push_back(gpu_resources[i])
        gpu_index = faiss.index_cpu_to_gpu_multiple(vres,
                                                    vdev,
                                                    cpu_index, co)
        index = gpu_index
    else:
        index = cpu_index

    return index


def search_one_by_one_with_faiss(passage_block_num, passge_embeddings_dir, index, query_embeddings, topN):
    '''
    Search on several passage embeddings blocks using a group of queries. 

    Argumets:
    passage_block_num: int, the number of passage blocks
    passge_embeddings_dir: str, the directory of passage embedding blocks
    index: faiss index
    query_embeddings: (num_query, emb_dim) the embeddings of queries
    topN: int, the number of topN results to return
    
    
    Returns:
    merged_D: (num_query, 2*topN) the scores for each query
    merged_I: (num_query, 2*topN) the indices of the topN scores
    
    '''

    merged_candidate_matrix = None

    # Search on each passage block, and merge the results on each block
    for block_id in trange(passage_block_num):

        ####################################
        # laod passage block to faiss index
        ####################################

        print("Loading passage block " + str(block_id))
        passage_embedding = None
        passage_embedding2id = None
        try:
            with open(oj(passge_embeddings_dir, "doc_emb_block.{}.pb".format(block_id)), 'rb') as handle:
                passage_embedding = pickle.load(handle)
            with open(oj(passge_embeddings_dir, "doc_embid_block.{}.pb".format(block_id)), 'rb') as handle:
                passage_embedding2id = pickle.load(handle)
                if isinstance(passage_embedding2id, list):
                    passage_embedding2id = np.array(passage_embedding2id)
        except:
            raise LoadError    

        logger.info('passage embedding shape: ' + str(passage_embedding.shape))
        logger.info("query embedding shape: " + str(query_embeddings.shape))
        index.add(passage_embedding)


        ############################
        # ann search
        ############################

        # D: (query_num, topN) the scores;
        # I: (query_num, topN) the indices of the topN scores
        tb = time.time()
        D, I = index.search(query_embeddings, topN)
        elapse = time.time() - tb

        ############################
        print({
            'time cost': elapse,
            'query num': query_embeddings.shape[0],
            'time cost per query': elapse / query_embeddings.shape[0]
        })

        # convert from index to passage id
        candidate_id_matrix = passage_embedding2id[I] 
        # list of scores for each query
        D = D.tolist()
        # list of passage ids for each query
        candidate_id_matrix = candidate_id_matrix.tolist()

        ## combine the scores and passage ids
        ''' 
        Candidate matrix:
        [
            [(score1, pid1), (score2, pid2), ...],   # query 1
            [(score1, pid1), (score2, pid2), ...],   # query 2
            ...
        ]
        '''
        candidate_matrix = []
        for score_list, passage_list in zip(D, candidate_id_matrix):
            candidate_matrix.append([])
            for score, passage in zip(score_list, passage_list):
                candidate_matrix[-1].append((score, passage))
            assert len(candidate_matrix[-1]) == len(passage_list)
        assert len(candidate_matrix) == I.shape[0]

        # empty the index and release memory
        index.reset()
        del passage_embedding
        del passage_embedding2id

        # if first block, no need to merge.
        if merged_candidate_matrix == None:
            merged_candidate_matrix = candidate_matrix
            continue
        
        # if not first bloack, merge
        ############################
        # Merge block results
        ############################
        merged_candidate_matrix_tmp = copy.deepcopy(merged_candidate_matrix)
        merged_candidate_matrix = []

        # for each query, compare the ranking list 
        # of merged block and the curreht blocks.
        for merged_list, cur_list in zip(merged_candidate_matrix_tmp,
                                         candidate_matrix):

            p1, p2 = 0, 0
            # the merged ranking list for this query
            merged_candidate_matrix.append([])
            while p1 < topN and p2 < topN:
                # [0] means the score
                if merged_list[p1][0] >= cur_list[p2][0]:
                    merged_candidate_matrix[-1].append(merged_list[p1])
                    p1 += 1
                else:
                    merged_candidate_matrix[-1].append(cur_list[p2])
                    p2 += 1
            while p1 < topN:
                merged_candidate_matrix[-1].append(merged_list[p1])
                p1 += 1
            while p2 < topN:
                merged_candidate_matrix[-1].append(cur_list[p2])
                p2 += 1

        # shape of merged_candidate_matrix: query_num * 2topN 
    
    
    # seperate again scores and passage ids.
    merged_D, merged_I = [], []
    for merged_list in merged_candidate_matrix: #
        merged_D.append([])
        merged_I.append([])
        for candidate in merged_list: # len(merged_list) = query_num * topk
            merged_D[-1].append(candidate[0])
            merged_I[-1].append(candidate[1])
    
    merged_D, merged_I = np.array(merged_D), np.array(merged_I)
    print(merged_I)
    print(merged_I.shape)
    return merged_D, merged_I


def get_test_query_embedding(args):
    '''
    Load the model, build the test query dataset/dataloader, and get the query embeddings.

    Arguments:
    args.pretrained_encoder_path: str, the path of the pretrained encoder
    args.
    '''

    set_seed(args)
    config = RobertaConfig.from_pretrained(args.pretrained_encoder_path)
    tokenizer = RobertaTokenizer.from_pretrained(args.pretrained_encoder_path, do_lower_case=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ANCE.from_pretrained(args.pretrained_encoder_path, config=config).to(device)

    # test dataset/dataloader
    args.batch_size = args.per_gpu_test_batch_size * max(1, args.n_gpu)
    logger.info("Buidling test dataset...")
    #test_dataset = Retrieval_topiocqa(args, tokenizer, args.test_file_path)
    test_dataset = Search_q_Retrieval(args, tokenizer, args.test_file_path)
    test_loader = DataLoader(test_dataset, 
                                batch_size = args.batch_size, 
                                shuffle=False, 
                                collate_fn=test_dataset.get_collate_fn(args))
    

    logger.info("Generating query embeddings for testing...")
    model.zero_grad()

    embeddings = []
    embedding2id = []

    with torch.no_grad():
        for batch in tqdm(test_loader, disable=args.disable_tqdm):
            model.eval()
            bt_sample_ids = batch["bt_sample_ids"] # question id
            # test type
            if args.test_type == "rewrite":
                input_ids = batch["bt_rewrite"].to(args.device)
                input_masks = batch["bt_rewrite_mask"].to(args.device)
            elif args.test_type == "raw":
                input_ids = batch["bt_raw_query"].to(args.device)
                input_masks = batch["bt_raw_query_mask"].to(args.device)
            elif args.test_type == "convq":
                input_ids = batch["bt_conv_q"].to(args.device)
                input_masks = batch["bt_conv_q_mask"].to(args.device)
            elif args.test_type == "convqa":
                input_ids = batch["bt_conv_qa"].to(args.device)
                input_masks = batch["bt_conv_qa_mask"].to(args.device)
            elif args.test_type == "convqp":
                input_ids = batch["bt_conv_qp"].to(args.device)
                input_masks = batch["bt_conv_qp_mask"].to(args.device)
            else:
                raise ValueError("test type:{}, has not been implemented.".format(args.test_type))
            
            query_embs = model(input_ids, input_masks)
            query_embs = query_embs.detach().cpu().numpy()
            embeddings.append(query_embs)
            embedding2id.extend(bt_sample_ids)

    embeddings = np.concatenate(embeddings, axis = 0)
    torch.cuda.empty_cache()

    return embeddings, embedding2id


def output_test_res(query_embedding2id,
                    retrieved_scores_mat, # score_mat: score matrix, test_query_num * (top_k * block_num)
                    retrieved_pid_mat, # pid_mat: corresponding passage ids
                    #offset2pid,
                    args):
    

    qids_to_ranked_candidate_passages = {}
    topN = args.top_k

    for query_idx in range(len(retrieved_pid_mat)):
        seen_pid = set()
        query_id = query_embedding2id[query_idx]

        top_ann_pid = retrieved_pid_mat[query_idx].copy()
        top_ann_score = retrieved_scores_mat[query_idx].copy()
        selected_ann_idx = top_ann_pid[:topN]
        selected_ann_score = top_ann_score[:topN].tolist()
        rank = 0

        if query_id in qids_to_ranked_candidate_passages:
            pass
        else:
            tmp = [(0, 0)] * topN
            #tmp_ori = [0] * topN
            qids_to_ranked_candidate_passages[query_id] = tmp
        
        for pred_pid, score in zip(selected_ann_idx, selected_ann_score):
            #pred_pid = offset2pid[idx]

            if not pred_pid in seen_pid:
                qids_to_ranked_candidate_passages[query_id][rank] = (pred_pid, score)
                rank += 1
                seen_pid.add(pred_pid)
        

    # for case study and more intuitive observation
    logger.info('Loading query and passages\' real text...')
    
        
    # all passages
    # all_passages = load_collection(args.passage_collection_path)

    # write to file
    logger.info('begin to write the output...')

    output_trec_file = oj(args.qrel_output_path, args.output_trec_file)
    with open(output_trec_file, "w") as g:
        for qid, passages in qids_to_ranked_candidate_passages.items():
            #query = qid2query[qid]
            rank_list = []
            for i in range(topN):
                pid, score = passages[i]
                g.write(str(qid) + " Q0 " + str(pid) + " " + str(i + 1) + " " + str(-i - 1 + 200) + ' ' + str(score) + " ance\n")

    logger.info("output file write ok at {}".format(output_trec_file))
    trec_res = print_trec_res(output_trec_file, args.trec_gold_qrel_file_path, args.rel_threshold)
    return trec_res

def print_trec_res(run_file, qrel_file, rel_threshold=1):
    with open(run_file, 'r' )as f:
        run_data = f.readlines()
    with open(qrel_file, 'r') as f:
        qrel_data = f.readlines()
    
    qrels = {}
    qrels_ndcg = {}
    runs = {}
    
    for line in qrel_data:
        line = line.split('\t')
        query = line[0]
        passage = line[2]
        rel = int(line[3])
        if query not in qrels:
            qrels[query] = {}
        if query not in qrels_ndcg:
            qrels_ndcg[query] = {}

        # for NDCG
        qrels_ndcg[query][passage] = rel
        # for MAP, MRR, Recall
        if rel >= rel_threshold:
            rel = 1
        else:
            rel = 0
        qrels[query][passage] = rel
    
    for line in run_data:
        line = line.split(' ')
        query = line[0].replace('-','_')
        #breakpoint()
        passage = line[2]
        rel = int(line[4])
        if query not in runs:
            runs[query] = {}
        runs[query][passage] = rel
    

    # pytrec_eval eval
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"map", "recip_rank", "recall.5", "recall.10", "recall.20", "recall.100"})
    res = evaluator.evaluate(runs)
    map_list = [v['map'] for v in res.values()]
    mrr_list = [v['recip_rank'] for v in res.values()]
    recall_100_list = [v['recall_100'] for v in res.values()]
    recall_20_list = [v['recall_20'] for v in res.values()]
    recall_10_list = [v['recall_10'] for v in res.values()]
    recall_5_list = [v['recall_5'] for v in res.values()]

    evaluator = pytrec_eval.RelevanceEvaluator(qrels_ndcg, {"ndcg_cut.3"})
    res = evaluator.evaluate(runs)
    ndcg_3_list = [v['ndcg_cut_3'] for v in res.values()]

    res = {
            #"MAP": np.average(map_list),
            "MRR": round(np.average(mrr_list)*100, 5),
            "NDCG@3": round(np.average(ndcg_3_list)*100, 5), 
            #"Recall@5": round(np.average(recall_5_list)*100, 5),
            "Recall@10": round(np.average(recall_10_list)*100, 5),
            #"Recall@20": round(np.average(recall_20_list)*100, 5),
            "Recall@100": round(np.average(recall_100_list)*100, 5),
        }

    
    logger.info("---------------------Evaluation results:---------------------")    
    logger.info(res)
    return res

def gen_metric_score_and_save(args, index, query_embeddings, query_embedding2id):
    # score_mat: score matrix, test_query_num * (top_n * block_num)
    # pid_mat: corresponding passage ids
    retrieved_scores_mat, retrieved_pid_mat = search_one_by_one_with_faiss(
                                                     args,
                                                     args.passage_embeddings_dir_path, 
                                                     index, 
                                                     query_embeddings, 
                                                     args.top_k) 

    #with open(args.passage_offset2pid_path, "rb") as f:
    #    offset2pid = pickle.load(f)
    
    output_test_res(query_embedding2id,
                    retrieved_scores_mat,
                    retrieved_pid_mat,
                    #offset2pid,
                    args)


def main():
    args = get_args()
    set_seed(args) 
    
    index = build_faiss_index(args)
    query_embeddings, query_embedding2id = get_test_query_embedding(args)
    gen_metric_score_and_save(args, index, query_embeddings, query_embedding2id)

    logger.info("Test finish!")
    


