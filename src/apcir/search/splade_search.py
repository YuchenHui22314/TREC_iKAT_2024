from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import  AutoTokenizer

from apcir.utils import set_seed 
from apcir.splade_index import SparseRetrieval
from apcir.models import Splade
from .data_format import Retrieval_trec

import threading
from transformers import AutoModelForMaskedLM

# Cache the SPLADE query encoder (BERT-base MLM) by (path, device) — load once, reuse across turns
# in the resident interactive server. Encodes via the HF MLM model directly (the standard SPLADE-max
# aggregation), which avoids the `Splade` class's CPU autocast `NoneType` bug.
_SPLADE_ENCODER_CACHE = {}
_SPLADE_ENCODER_LOCK = threading.Lock()


def _get_splade_encoder(encoder_path, device):
    key = (encoder_path, str(device))
    cached = _SPLADE_ENCODER_CACHE.get(key)
    if cached is None:                                    # fast path, no lock
        with _SPLADE_ENCODER_LOCK:
            cached = _SPLADE_ENCODER_CACHE.get(key)
            if cached is None:
                tok = AutoTokenizer.from_pretrained(encoder_path)
                model = AutoModelForMaskedLM.from_pretrained(encoder_path).to(device).eval()
                _SPLADE_ENCODER_CACHE[key] = cached = (tok, model)
    return cached


def splade_encode_query(query, encoder_path, device):
    """Encode a query string into a SPLADE sparse vocab vector (1-D tensor on `device`):
    SPLADE-max = max over tokens of log(1+relu(MLM logits)) * attention_mask. Cached encoder."""
    tok, model = _get_splade_encoder(encoder_path, device)
    enc = tok(query, return_tensors="pt", truncation=True, max_length=256).to(device)
    with torch.no_grad():
        logits = model(**enc).logits                                          # (1, seq, vocab)
        rep = torch.max(torch.log1p(torch.relu(logits))
                        * enc["attention_mask"].unsqueeze(-1), dim=1).values[0]   # (vocab,)
    return rep


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
            print(batch["input_ids"].shape)
            print(batch["input_ids"])
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
    

