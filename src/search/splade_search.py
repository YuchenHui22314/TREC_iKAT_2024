import os
import sys
sys.path.append('..')
sys.path.append('.')
import time
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import  AutoTokenizer

from search.utils import set_seed 

from search.libs import SparseRetrieval
from search.models import Splade
from search.data_format import Retrieval_trec

def splade_search(args):
    '''
    Perform Splade Sparse Retrieval.
    Args:
        args.seed
        args.splade_query_encoder_path: str
        args.splade_index_dir_path: str
        args.query_gpu_id: int, if -1, use cpu
        args.query_encoder_batch_size: int
        args.qid_list_string: List[str], the list of query ids
        args.retrieval_query_list: List[str], the list of queries
        args.retrieval_top_k: int, the number of retrieved documents
    '''

    device = torch.device(f"cuda:{args.query_gpu_id}" if args.query_gpu_id >= 0 else "cpu")
    set_seed(args.seed, True)

    model = Splade(
        args.splade_query_encoder_path, 
        agg = "max"
        )
    model.to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(args.splade_query_encoder_path) 
    
    # test dataset/dataloader
    print("Buidling test dataset...")
    test_dataset = Retrieval_trec(
        tokenizer = tokenizer,
        retrieval_query_list = args.retrieval_query_list,
        qid_list_string = args.qid_list_string,
        max_length = 256,
        )
    test_loader = DataLoader(
        test_dataset, 
        batch_size = args.query_encoder_batch_size, 
        shuffle=False, 
        collate_fn=test_dataset.get_collate_fn()
        )
    
    # get query embeddings
    qid2emb = {}
    
    with torch.no_grad():
        model.eval()
        for batch in tqdm(test_loader, desc="generating query embeddings"):
            inputs = {}
            inputs["input_ids"] = batch["input_ids"].to(device)
            inputs["attention_mask"] = batch["attention_mask"].to(device)


            batch_query_embs = model(q_kwargs = inputs)["q_rep"]
            qids = batch["qid"] 
            for i, qid in enumerate(qids):
                qid2emb[qid] = batch_query_embs[i]
    
    # retrieve
    dim_voc = None
    for qid, emb in qid2emb.items():
        dim_voc = emb.shape[0]
        break
    assert dim_voc is not None, "dim_voc is None"

    retriever = SparseRetrieval(
        args.splade_index_dir_path, 
        "None",  # this the output path to save the retrieval results. Useful in Kelong's code, but not here. 
        dim_voc, 
        args.retrieval_top_k
    )
    result,hits = retriever.retrieve(qid2emb)

    return hits
    

