from os.path import join as oj
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
import faiss
import time
import copy
import pickle
import torch
import numpy as np
import json
from transformers import RobertaConfig, RobertaTokenizer, AutoTokenizer

from apcir.models import ANCE, QwenEmbedding
from apcir.utils import set_seed
from .data_format import  Retrieval_trec, pad_and_mask




class PyScoredDoc:
    def __init__(self, docid: str, score: float):
        self.docid = docid
        self.score = score
    
    def __repr__(self):
        return f"docid: {self.docid}, score: {self.score}"

def build_faiss_index(args):
    '''
    Build the Faiss index for dense retrieval.
    args.faiss_n_gpu: int, the number of gpus
    args.use_gpu_for_faiss: bool, whether to use gpu 
    args.embed_dim: int, the dimension of embeddings
    args.tempmem: int, the temporary memory for Faiss index. Set to -1 to use default value.
    '''
    print("Building index...")
    # ngpu = faiss.get_num_gpus()
    ngpu = args.faiss_n_gpu
    gpu_resources = []
    tempmem = args.tempmem

    for i in range(ngpu):
        res = faiss.StandardGpuResources()
        if tempmem >= 0:
            res.setTempMemory(tempmem)
        gpu_resources.append(res)

    cpu_index = faiss.IndexFlatIP(args.embed_dim)  
    index = None
    if args.use_gpu_for_faiss:
        print("Using GPU for Faiss")
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

        print('passage embedding shape: ' + str(passage_embedding.shape))
        print("query embedding shape: " + str(query_embeddings.shape))
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
        
        # if not first block, merge
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


def build_ance_conv_query_tokens(tokenizer, query_json,
                                 max_query_length=64, max_response_length=64,
                                 max_concat_length=512):
    """Build the ConvDR conversational query token ids for ANCE.

    `query_json` is a JSON-serialized turn list [u1, r1, u2, r2, ..., u_current]
    (produced by topics.py `full_conversation_dense`). This reproduces
    continual_ir's training `build_conv_query_tokens` EXACTLY:
      - add the current utterance first (most recent), drop its [CLS], keep [SEP];
      - walk history backward (odd=response cap max_response_length, even=query cap
        max_query_length), drop each segment's [CLS], budget-truncate oldest-first;
      - reverse back to chronological order;
      - prepend a SINGLE [CLS].
    Result: [CLS] u1 [SEP] r1 [SEP] ... u_current [SEP]. Caps 64/64/512 match the
    ANCE training defaults (train_continually_ddp_cl.py).
    """
    turns = json.loads(query_json)
    cur_utt_text, ctx_utts_text = turns[-1], turns[:-1]
    max_context_len = max_concat_length - 1
    flat, total = [], 0

    # 1. current utterance first; drop [CLS], keep trailing [SEP]
    cur = tokenizer.encode(cur_utt_text, add_special_tokens=True,
                           max_length=max_query_length, truncation=True)[1:]
    if len(cur) > max_context_len:
        cur = cur[:max_context_len]
        cur[-1] = tokenizer.sep_token_id
    flat.append(cur)
    total += len(cur)

    # 2. history backward, oldest-first truncation
    for j in range(len(ctx_utts_text) - 1, -1, -1):
        cap = max_response_length if j % 2 == 1 else max_query_length
        tok = tokenizer.encode(ctx_utts_text[j], add_special_tokens=True,
                               max_length=cap, truncation=True)[1:]   # drop [CLS]
        remaining = max_context_len - total
        if remaining <= 0:
            break
        if len(tok) <= remaining:
            flat.append(tok)
            total += len(tok)
        else:
            tok = tok[:remaining]
            tok[-1] = tokenizer.sep_token_id   # must end with [SEP]
            flat.append(tok)
            total += len(tok)
            break

    # 3. restore chronological order; 4. flatten + prepend single [CLS]
    flat = flat[::-1]
    ids = [tokenizer.cls_token_id] + [t for seg in flat for t in seg]
    assert len(ids) <= max_concat_length
    return ids


def get_test_query_embedding(args):
    '''
    Load the model, build the test query dataset/dataloader, and get the query embeddings.

    Arguments:
    args.topics: str, the type of topics, canbe ikat-23, topiocqa, etc.
    args.dense_query_encoder_path: str, the path of the pretrained encoder
    args.retrieval_model: str, the model name 
    args.query_gpu_id: int, if None, use cpu
    args.query_encoder_batch_size : int
    args.qid_list_string: List[str], the list of query ids
    args.retrieval_query_list: List[str], the list of queries

    Returns:
    embeddings: np.array, the embeddings of queries. shape: (num_query, emb_dim)
    embedding2id: List[str], the query ids of the embeddings. shape: (num_query,)
    '''

    set_seed(args.seed, True)

    # ---- Qwen3-Embedding query encoding -------------------------------------
    # Reuse the SAME tokenizer call + QwenEmbedding (last-token pool + L2) used to
    # encode the corpus (indexing/dense: CollateClass tokenizes docs with
    # add_special_tokens=True, max_length=512, left padding; QwenEmbedding pools the
    # last token + L2). `add_special_tokens=True` appends <|endoftext|> (151643) on
    # both sides, identical to continual_ir's build_qwen_instruct_query_ids EOS, so
    # query and document embeddings are directly comparable. The instruction is
    # already baked into the query string by the `oracle_qwen_instruct` query type;
    # documents carry NO instruction (Qwen3-Embedding protocol).
    if args.retrieval_model in ("qwen3", "conv-qwen3"):
        query_device = f"cuda:{args.query_gpu_id}" if args.query_gpu_id >= 0 else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(args.dense_query_encoder_path, padding_side="left")
        model = QwenEmbedding(args.dense_query_encoder_path).to(query_device)
        model.eval()
        embeddings, embedding2id = [], []
        bs = args.query_encoder_batch_size
        with torch.no_grad():
            for i in trange(0, len(args.retrieval_query_list), bs,
                            desc="generating qwen3 query embeddings"):
                batch_q  = args.retrieval_query_list[i:i + bs]
                batch_id = args.qid_list_string[i:i + bs]
                # Qwen3-Embedding has a long context (32k). Do NOT cap conversational
                # queries to the 512-token doc length — that would chop conversation/PTKB
                # context. We pad to the batch's longest (padding=True), not to max_length,
                # so memory stays proportional to the actual query length.
                enc = tokenizer(batch_q, add_special_tokens=True, max_length=32768,
                                padding=True, truncation=True, return_tensors="pt")
                embs = model(enc["input_ids"].to(query_device),
                             enc["attention_mask"].to(query_device))
                embeddings.append(embs.detach().cpu().float().numpy())
                embedding2id.extend(batch_id)
        embeddings = np.concatenate(embeddings, axis=0)
        torch.cuda.empty_cache()
        return embeddings, embedding2id

    # ---- ANCE / conv-ANCE query encoding ------------------------------------
    # Symmetric with the qwen branch above: load encoder, then inline batched encoding
    # (no Dataset/DataLoader). For full_conversation_dense the query is a JSON turn list
    # -> build_ance_conv_query_tokens (ConvDR build, matches continual_ir training); every
    # other ANCE query type is a single query -> plain tokenizer.encode. Works for both
    # iKAT and topiocqa topics (the distinction is the query type, not the corpus).
    if args.retrieval_model in ("ance", "conv-ance"):
        query_device = f"cuda:{args.query_gpu_id}" if args.query_gpu_id >= 0 else "cpu"
        config = RobertaConfig.from_pretrained(args.dense_query_encoder_path)
        tokenizer = RobertaTokenizer.from_pretrained(args.dense_query_encoder_path, do_lower_case=True)
        model = ANCE.from_pretrained(args.dense_query_encoder_path, config=config).to(query_device)
        model.eval()

        is_conv = (args.retrieval_query_type == "full_conversation_dense")
        embeddings, embedding2id = [], []
        bs = args.query_encoder_batch_size
        with torch.no_grad():
            for i in trange(0, len(args.retrieval_query_list), bs,
                            desc="generating ance query embeddings"):
                batch_q  = args.retrieval_query_list[i:i + bs]
                batch_id = args.qid_list_string[i:i + bs]
                if is_conv:
                    seqs = [build_ance_conv_query_tokens(tokenizer, q) for q in batch_q]
                else:
                    seqs = [tokenizer.encode(q, add_special_tokens=True,
                                             max_length=512, truncation=True) for q in batch_q]
                input_ids, input_masks = pad_and_mask(seqs, tokenizer.pad_token_id)
                query_embs = model(input_ids.to(query_device), input_masks.to(query_device))
                embeddings.append(query_embs.detach().cpu().numpy())
                embedding2id.extend(batch_id)
        embeddings = np.concatenate(embeddings, axis=0)
        torch.cuda.empty_cache()
        return embeddings, embedding2id

    raise NotImplementedError(
        f"query encoding not implemented for retrieval_model={args.retrieval_model}")


def get_dense_ranking_list(
    query_embedding2id,
    retrieved_scores_mat, 
    retrieved_pid_mat, 
    #offset2pid,
    retrieval_top_k
    ):

    '''
    Arguments:
    query_embedding2id: List[str], the query ids of the embeddings. shape: (num_query,)
    retrieved_scores_mat: np.array, the scores of retrieved passages. shape: (num_query, topk*2)
    retrieved_pid_mat: np.array, the passage ids of the retrieved passages. shape: (num_query, topk*2)
    retrieval_top_k: int, the number of topk passages to return

    Returns:
    PyScoredDoc_list: Dict[qid, list_of(PyScoredDoc)], the ranking list of passages for each query

    ''' 
    qids_to_ranked_candidate_passages = {}
    topN = retrieval_top_k


    # for each query
    for query_idx in range(len(retrieved_pid_mat)):
        seen_pid = set()
        query_id = query_embedding2id[query_idx]

        top_ann_pid = retrieved_pid_mat[query_idx].copy()
        top_ann_score = retrieved_scores_mat[query_idx].copy()
        selected_ann_idx = top_ann_pid[:topN]
        selected_ann_score = top_ann_score[:topN].tolist()

        if query_id in qids_to_ranked_candidate_passages:
            pass
        else:
            ranking_list = []
            #tmp_ori = [0] * topN
            qids_to_ranked_candidate_passages[query_id] = ranking_list
        
        for pred_pid, score in zip(selected_ann_idx, selected_ann_score):
            #pred_pid = offset2pid[idx]
            scored_doc = PyScoredDoc(str(pred_pid), float(score))
            if not pred_pid in seen_pid:
                qids_to_ranked_candidate_passages[query_id].append(scored_doc)
                seen_pid.add(pred_pid)
        
        
    
    return qids_to_ranked_candidate_passages 



def calculate_score_and_get_rakning_list(args, index, query_embeddings, query_embedding2id):
    # score_mat: score matrix, test_query_num * (top_n * block_num)
    # pid_mat: corresponding passage ids
    '''
    Arguments:
    args.passage_block_num: int, the number of passage blocks
    args.dense_index_dir_path: str, the directory of passage embedding blocks
    args.retrieval_top_k: int, the number of topk passages to return
    index: faiss index
    query_embeddings: (num_query, emb_dim) the embeddings of queries
    query_embedding2id: List[str], the query ids of the embeddings. shape: (num_query,)
    
    Returns:
    PyScoredDoc_dict: Dict[qid, list_of(PyScoredDoc)], the ranking list of passages for each query
    '''

    retrieved_scores_mat, retrieved_pid_mat = search_one_by_one_with_faiss(
        args.passage_block_num,
        args.dense_index_dir_path, 
        index, 
        query_embeddings, 
        args.retrieval_top_k
        ) 

    #with open(args.passage_offset2pid_path, "rb") as f:
    #    offset2pid = pickle.load(f)
    
    PyScoredDoc_dict = get_dense_ranking_list(
        query_embedding2id,
        retrieved_scores_mat,
        retrieved_pid_mat,
        #offset2pid,
        args.retrieval_top_k)
    
    return PyScoredDoc_dict


def dense_search(args):

    '''
    Perform dense retrieval on collection (e.g., MS MARCO):
    2. establish index with Faiss on GPU for fast dense retrieval
    3. load the model, build the test query dataset/dataloader, and get the query embeddings. 
    4. iteratively searched on each passage block one by one to got the retrieved scores and passge ids for each query.
    5. merge the results on all pasage blocks
    6. output the result

    Arguments:
    args.seed: int, the random seed
    args.faiss_n_gpu: int, the number of gpus
    args.use_gpu_for_faiss: bool, whether to use gpu 
    args.embed_dim: int, the dimension of embeddings
    args.tempmem: int, the temporary memory for Faiss index. Set to -1 to use default value.

    args.topics: str, the type of topics, canbe ikat-23, topiocqa, etc.
    args.dense_query_encoder_path: str, the path of the pretrained encoder
    args.retrieval_model: str, the model name 
    args.query_gpu_id: int, if -1, use cpu
    args.query_encoder_batch_size : int
    args.qid_list_string: List[str], the list of query ids
    args.retrieval_query_list: List[str], the list of queries

    args.passage_block_num: int, the number of passage blocks
    args.dense_index_dir_path: str, the directory of passage embedding blocks
    args.retrieval_top_k: int, the number of topk passages to return

    Returns:
    PyScoredDoc_dict: Dict[qid, list_of(PyScoredDoc)], the ranking list of passages for each query
    '''

    set_seed(args.seed, args.faiss_n_gpu >= 0) 
    index = build_faiss_index(args)
    query_embeddings, query_embedding2id = get_test_query_embedding(args)
    PyScoredDoc_dict = calculate_score_and_get_rakning_list(args, index, query_embeddings, query_embedding2id)

    return PyScoredDoc_dict

    


