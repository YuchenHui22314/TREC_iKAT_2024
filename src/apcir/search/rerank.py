from multiprocessing import Manager
import json
import os

from tqdm import tqdm
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import DataParallel
import torch.nn as nn

from peft import PeftModel, PeftConfig
from typing import List, Tuple, Any, Dict
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    T5ForConditionalGeneration,
    PreTrainedTokenizer,
    )

from pyserini.search.lucene import LuceneSearcher

from apcir.functional.llm import monoT5
from .rank_gpt import  sliding_windows


# ----------------------------------------------------------------------------- #
# Qwen3-Reranker instructions (the model card recommends customizing `instruct`
# per task, +1-5%). Routing: reranking_query_type == "qwen_3_rerank_instruct_full"
# -> the conversational instruction (query text = profile-first + conversation,
# built by topics.query_type_2_query); any other reranking_query_type (e.g. a
# personalized rewrite like gpt-4o_rar_personalized_cot1_rw) -> the NATIVE default
# instruction from the model card.
# ----------------------------------------------------------------------------- #
QWEN3_RERANK_DEFAULT_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query")
QWEN3_RERANK_CONV_INSTRUCTION = (
    "Given a conversation between a user and an AI assistant and the user's profile, "
    "judge whether the document helps answer the user's last question in a way "
    "consistent with the user's profile.")


def _fetch_contents_cached(searcher, docid, _cache):
    """docid -> passage contents with a per-rerank-call dict cache. The same doc often
    appears in many queries' top-k; the lucene fetch is mmap-cheap but the raw()+json
    parse is not — cache it once per rerank() call."""
    c = _cache.get(docid)
    if c is None:
        c = json.loads(searcher.doc(docid).raw())["contents"]
        _cache[docid] = c
    return c


class QwenReranker:
    """Qwen3-Reranker (0.6B/4B/8B): a CausalLM scored by P("yes") at the last position.

    Input format is VERBATIM from the model card (Qwen/Qwen3-Reranker-4B):
        system: Judge whether the Document meets the requirements based on the Query
                and the Instruct provided. Note that the answer can only be "yes" or "no".
        user:   <Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}
        assistant prefix: <think>\n\n</think>\n\n   (no thinking content)
    Score = softmax over the ("yes","no") token logits at the LAST position -> P(yes).
    LEFT padding so position -1 is the real last token for every row in the batch.

    Shared by evaluation.py (offline batch) and the interactive server (co-hosted on
    the gen-LLM GPU).

    Quantization (computed VRAM, NOT measured; quality NOT yet validated on iKAT):
      none -> bf16, ~8G weights (~9-9.5G with activations) — DEFAULT, safest.
      8b   -> bitsandbytes int8, ~4G weights (~4.5-5G). Near-lossless for Qwen3 in
              general, BUT a reranker scores the yes/no LOGIT DIFFERENCE, which is more
              quantization-sensitive than generation — the field consensus (IntelLabs
              fastRAG, sentence-transformers) is to VALIDATE NDCG/MRR before trusting it.
      4b   -> bitsandbytes nf4, ~2G. EXPERIMENTAL for reranking — int4 on cross-encoder
              rerankers is under-documented; not recommended without an ablation.
    Given the remote-reranker option (octal31) and the 0.72 co-host headroom, bf16 fits
    everywhere — quantization is a convenience, not a necessity.
    """

    PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements "
              "based on the Query and the Instruct provided. Note that the answer can "
              "only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
    SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def __init__(self, model_path="Qwen/Qwen3-Reranker-4B", cache_dir=None,
                 quant="none", device="cuda", max_length=8192):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, cache_dir=cache_dir, padding_side="left")
        model_kwargs = {"cache_dir": cache_dir, "torch_dtype": torch.bfloat16}
        if quant in ("8b", "4b"):
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=(quant == "8b"), load_in_4bit=(quant == "4b"))
            model_kwargs["device_map"] = {"": device}
        else:
            model_kwargs["device_map"] = {"": device}
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        self.model.eval()
        self.device = device
        self.token_yes = self.tokenizer.convert_tokens_to_ids("yes")
        self.token_no = self.tokenizer.convert_tokens_to_ids("no")
        self.prefix_tokens = self.tokenizer.encode(self.PREFIX, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.SUFFIX, add_special_tokens=False)

    @staticmethod
    def format_pair(instruction, query, doc):
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

    def _score_batch(self, pairs):
        """pairs: list[str] (already format_pair'ed). Returns list[float] P(yes)."""
        body_max = self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        enc = self.tokenizer(pairs, padding=False, truncation="longest_first",
                             max_length=body_max, add_special_tokens=False)
        input_ids = [self.prefix_tokens + ids + self.suffix_tokens
                     for ids in enc["input_ids"]]
        batch = self.tokenizer.pad({"input_ids": input_ids}, padding=True,
                                   return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**batch).logits[:, -1, :]
            pair_logits = torch.stack(
                [logits[:, self.token_no], logits[:, self.token_yes]], dim=1)
            scores = torch.nn.functional.log_softmax(pair_logits, dim=1)[:, 1].exp()
        return scores.float().cpu().tolist()

    def score(self, instruction, query, docs, batch_size, max_batch_tokens=24000):
        """One query vs many docs -> list[float].

        TOKEN-BUDGET batching (the fixed `batch_size` is only a hard cap): tokenize each
        pair once, sort by length, and greedily pack each forward batch up to
        ~max_batch_tokens PADDED tokens (rows_in_batch * longest_in_batch). Short pairs
        (oracle mode, ~280 tok) pack ~50/batch -> 1 forward instead of 7; long pairs
        (instruct_full, full conversation per pair) pack few/batch -> no OOM. Sorting by
        length also minimises padding waste. Original order is restored for the scores.
        """
        pairs = [self.format_pair(instruction, query, d) for d in docs]
        body_max = self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        enc = self.tokenizer(pairs, padding=False, truncation="longest_first",
                             max_length=body_max, add_special_tokens=False)["input_ids"]
        n_extra = len(self.prefix_tokens) + len(self.suffix_tokens)
        lens = [len(ids) + n_extra for ids in enc]
        order = sorted(range(len(pairs)), key=lambda i: lens[i])   # short -> long

        scores = [0.0] * len(pairs)
        i = 0
        while i < len(order):
            # grow a batch while it fits the token budget and the hard count cap
            j = i
            longest = 0
            while j < len(order):
                cand = max(longest, lens[order[j]])
                if j > i and (cand * (j - i + 1) > max_batch_tokens or (j - i) >= max(batch_size, 1) * 8):
                    break
                longest = cand
                j += 1
            idxs = order[i:j]
            batch_scores = self._score_batch([pairs[k] for k in idxs])
            for k, s in zip(idxs, batch_scores):
                scores[k] = s
            i = j
        return scores


class RemoteReranker:
    """Same .score() interface as QwenReranker, but the model lives on ANOTHER machine
    (a rerank_server.py instance, e.g. on octal31). Use when the local GPUs are full —
    the LAN round-trip is negligible (~100KB per 50-doc request, 0.37ms RTT measured
    octal40<->octal31)."""

    def __init__(self, base_url: str, timeout: float = 600.0):
        import requests as _requests
        self._requests = _requests
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        # fail fast with a clear message if the remote server isn't up
        r = self._requests.get(f"{self.base_url}/health", timeout=10)
        r.raise_for_status()
        info = r.json()
        print(f"[RemoteReranker] connected: {info}")

    def score(self, instruction, query, docs, batch_size):
        r = self._requests.post(
            f"{self.base_url}/rerank",
            json={"instruction": instruction, "query": query,
                  "docs": list(docs), "batch_size": batch_size},
            timeout=self.timeout)
        r.raise_for_status()
        return r.json()["scores"]


class RemoteRerankerPool:
    """N resident rerank servers (one per GPU) + a thread pool that fires queries
    CONCURRENTLY round-robin → data parallelism across N GPUs, realized as persistent
    servers instead of an accelerate one-shot launch. Each server loads the model ONCE
    and stays resident (warm across configs AND sessions). The interactive single-server
    is just N=1 of this. `score()` = one query (server 0); `score_many()` = a list of
    queries reranked concurrently (the multi-GPU win for a config's many queries)."""

    def __init__(self, urls, timeout: float = 600.0):
        import requests as _requests
        self._requests = _requests
        self.urls = [u.rstrip("/") for u in urls]
        self.timeout = timeout
        for u in self.urls:
            self._requests.get(f"{u}/health", timeout=10).raise_for_status()
        print(f"[RemoteRerankerPool] {len(self.urls)} servers: {self.urls}")

    def _one(self, url, instruction, query, docs, batch_size):
        r = self._requests.post(
            f"{url}/rerank",
            json={"instruction": instruction, "query": query,
                  "docs": list(docs), "batch_size": batch_size},
            timeout=self.timeout)
        r.raise_for_status()
        return r.json()["scores"]

    def score(self, instruction, query, docs, batch_size):
        return self._one(self.urls[0], instruction, query, docs, batch_size)

    def score_many(self, jobs, batch_size):
        """jobs: list of (instruction, query, docs) -> list of score-lists IN ORDER,
        scored concurrently across the N servers (round-robin by index)."""
        from concurrent.futures import ThreadPoolExecutor
        results = [None] * len(jobs)

        def run(i):
            inst, q, docs = jobs[i]
            return i, self._one(self.urls[i % len(self.urls)], inst, q, docs, batch_size)

        with ThreadPoolExecutor(max_workers=len(self.urls)) as ex:
            for i, sc in ex.map(run, range(len(jobs))):
                results[i] = sc
        return results


def get_model(
    peft_model_name, 
    cache_dir,
    quant_8bit = True,
    quant_4bit = False,
    ):
    config = PeftConfig.from_pretrained(peft_model_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model_name_or_path, 
        cache_dir=cache_dir, 
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
        load_in_8bit = quant_8bit,
        load_in_4bit = quant_4bit,
        num_labels=1)

    model = PeftModel.from_pretrained(base_model, peft_model_name)
    model = model.merge_and_unload()
    model.eval()

    return model




def load_rankllama(
    cache_dir: str,
    quant_8bit: bool = True,
    quant_4bit: bool = False
    ) -> Tuple[Any,Any]:

    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')

    model = get_model(
        'castorini/rankllama-v1-7b-lora-passage', 
        cache_dir,
        quant_8bit,
        quant_4bit
        )

    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "right"
    model.config.pad_token_id = 0

    return tokenizer, model



def load_t5_DP(
    cache_dir: str,
    model_name: str = 'castorini/monot5-base-msmarco',
    ) -> Tuple[PreTrainedTokenizer,T5ForConditionalGeneration,Any,Any]:
    

    # load model
    model = monoT5.from_pretrained(model_name, cache_dir=cache_dir)
    model.set_tokenizer()
    model.set_targets(['true', 'false'])
    tokenizer = model.tokenizer
    decoder_stard_id = model.config.decoder_start_token_id
    targeted_ids = model.targeted_ids


    parallel = True
    # data parallel
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if parallel:
        model = DataParallel(model).to(device)
    else:
        model = model.to(device)

    model.eval()

    return tokenizer, model, decoder_stard_id, targeted_ids


def load_t5_DDP(
    cache_dir: str,
    model_name: str = 'castorini/monot5-base-msmarco',
    ) -> Tuple[PreTrainedTokenizer,T5ForConditionalGeneration,Any,Any]:
    

    # load model
    model = monoT5.from_pretrained(model_name, cache_dir=cache_dir)
    model.set_tokenizer()
    model.set_targets(['true', 'false'])
    tokenizer = model.tokenizer
    decoder_stard_id = model.config.decoder_start_token_id
    targeted_ids = model.targeted_ids

    model.eval()

    return tokenizer, model, decoder_stard_id, targeted_ids

def get_split_num(length, batch_size):

    if length < batch_size:
        num_to_split = 1
    elif length % batch_size == 0:
        num_to_split = length // batch_size
    else:
        num_to_split = length // batch_size + 1
    return num_to_split

def rerank_rankllama(
    query: str,
    passages: List[str],
    tokenizer: Any,
    model: Any,
    rerank_batch_size: int 
) -> List[float]: 

    # Split passages into groups of 10 passages (on octal31)
    # due to GPU resources limitation.


    num_to_split = get_split_num(len(passages), rerank_batch_size)
    passages_parts = np.array_split(passages, num_to_split)
    scores = []

    for passages_part in passages_parts:
        inputs = tokenizer(
            [f'query: {query}'] * len(passages_part), [f'document: {passage}' for passage in passages_part], 
            return_tensors='pt',
            padding = True,
            max_length = 2048,
            truncation = True
            )

        # Run the model forward
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            part_scores = logits[:,0]
            # one batched GPU->CPU transfer (the old per-element float(tensor) forced a
            # cuda sync per score, and re-converted the whole accumulated list each batch)
            scores.extend(part_scores.float().cpu().tolist())

    return scores

def rerank_t5_DP(
    query: str,
    passages: List[str],
    tokenizer: Any,
    model: Any,
    decoder_input_ids: Any,
    targeted_ids: Any,
    rerank_batch_size: int 
    ) -> List:

    # Split passages into groups of 67 passages
    # due to GPU resources limitation.
    # 15 on octal31, 6 on octal40 when reranking top 1000

    num_to_split = get_split_num(len(passages), rerank_batch_size)
    passages_parts = np.array_split(passages, num_to_split)
    scores = []

    for passages_part in passages_parts:
        inputs = tokenizer(
            [f"Query: {query} Document: {passage} Relevant:" for passage in passages_part],
            max_length = 512,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        # predict
        with torch.no_grad():

            softmax = nn.Softmax(dim=1)
            for k in inputs:
                inputs[k] = inputs[k].to("cuda")

            dummy_labels = torch.full(
                inputs.input_ids.size(), 
                decoder_input_ids
            ).to("cuda")
            
            batch_logits = model(**inputs, labels=dummy_labels).logits
            true_false = softmax(batch_logits[:, 0, targeted_ids]).detach().cpu().numpy() # B 2
            true_prob = true_false[:,0]
            scores.extend(true_prob.tolist())
        

    return scores


def rerank_t5_DDP(
    rank: int,
    world_size: int,
    query: str,
    passages: List[str],
    tokenizer: Any,
    decoder_input_ids: Any,
    targeted_ids: Any,
    outputs ,#Manager().list(),
    rerank_batch_size: int 
    ) -> None:

    dist.init_process_group("nccl", rank=rank, world_size=world_size)


    # construct DDP model
    tokenizer, model, decoder_stard_id, targeted_ids= load_t5_DDP(
        cache_dir = cache_dir,
        model_name = 'castorini/monot5-base-msmarco'
        )
    model = model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    # Split passages into groups of 67 passages
    # due to GPU resources limitation.
    num_to_split = get_split_num(len(passages), rerank_batch_size)
    passages_parts = np.array_split(passages, num_to_split)
    scores = []

    for passages_part in passages_parts:
        inputs = tokenizer(
            [f"Query: {query} Document: {passage} Relevant:" for passage in passages_part],
            max_length = 512,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        # predict
        with torch.no_grad():

            softmax = nn.Softmax(dim=1)
            for k in inputs:
                inputs[k] = inputs[k].to(rank)

            dummy_labels = torch.full(
                inputs.input_ids.size(), 
                decoder_input_ids
            ).to(rank)
            
            batch_logits = ddp_model(**inputs, labels=dummy_labels).logits
            true_false = softmax(batch_logits[:, 0, targeted_ids]).detach().cpu().numpy() # B 2
            true_prob = true_false[:,0]
            scores.extend(true_prob.tolist())

    if rank == 0:
        outputs.append(scores)

    # Clean up
    dist.destroy_process_group()


def hits_2_rankgpt_list(
    searcher: Any,
    query_dict: Dict[str, str],
    hits_dict: Dict[str, List[Any]],
    rank_end: int = None,
    ) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, List[Dict[str, Any]]]]:

    rankgpt_list = []
    new_hits_dict = {}
    doc_cache = {}

    for qid, query in query_dict.items():
        hits = hits_dict[qid]
        new_hits_dict[qid] = []
        rankgpt_list.append({'query': query, 'hits': []})

        # assuming that hits are sorted by rank
        for rank, hit in enumerate(hits):
            # get passage text — ONLY for ranks the sliding window will actually look at
            # (rank < rank_end == rerank_top_k). Beyond that, sliding_windows never reads
            # `content` and the final reassignment uses docid only, so skip the fetch+parse
            # (the old code fetched all 1000 hits to rerank 50).
            if rank_end is None or rank < rank_end:
                content = _fetch_contents_cached(searcher, hit.docid, doc_cache)
                content = ' '.join(content.split())
            else:
                content = ''

            document_hit_dict = {
                'content': content,
                'qid': qid,
                'docid': hit.docid,
                'rank': rank,
                'score': hit.score}

            rankgpt_list[-1]['hits'].append(document_hit_dict)

            new_hits_dict[qid].append(document_hit_dict)

    return rankgpt_list, new_hits_dict



# ----------------------------------------------------------------------------- #
# Model-agnostic local scorers: each exposes .score(instruction, query, docs, bs)
# so rerank_server.py can serve ANY HF reranker and the RemoteReranker(Pool) client
# is reranker-agnostic. (monot5/rankllama ignore `instruction`; qwen3 uses it.)
# ----------------------------------------------------------------------------- #
MONOT5_NAMES = {
    "monot5_base": "castorini/monot5-base-msmarco",
    "monot5_base_10k": "castorini/monot5-base-msmarco-10k",
    "monot5_large": "castorini/monot5-large-msmarco",
    "monot5_large_10k": "castorini/monot5-large-msmarco-10k",
    "monot5_3b": "castorini/monot5-3b-msmarco",
    "monot5_3b_10k": "castorini/monot5-3b-msmarco-10k",
}


class MonoT5Scorer:
    """Single-GPU monoT5 reranker with the unified .score() signature. Under the
    per-GPU server (CUDA_VISIBLE_DEVICES=<g>), the one visible GPU is 'cuda'."""

    def __init__(self, model_name, cache_dir, device="cuda"):
        self.model = monoT5.from_pretrained(model_name, cache_dir=cache_dir)
        self.model.set_tokenizer()
        self.model.set_targets(["true", "false"])
        self.tokenizer = self.model.tokenizer
        self.decoder_start_id = self.model.config.decoder_start_token_id
        self.targeted_ids = self.model.targeted_ids
        self.model = self.model.to(device)
        self.model.eval()

    def score(self, instruction, query, docs, batch_size):   # instruction ignored
        return rerank_t5_DP(query, list(docs), self.tokenizer, self.model,
                            self.decoder_start_id, self.targeted_ids, batch_size)


class RankLlamaScorer:
    """RankLLaMA reranker with the unified .score() signature (ignores instruction)."""

    def __init__(self, cache_dir, quant="none"):
        self.tokenizer, self.model = load_rankllama(
            cache_dir, quant_8bit=(quant == "8b"), quant_4bit=(quant == "4b"))

    def score(self, instruction, query, docs, batch_size):
        return rerank_rankllama(query, list(docs), self.tokenizer, self.model, batch_size)


def build_local_reranker(reranker_type, cache_dir, device="cuda", quant="none",
                         qwen3_path="Qwen/Qwen3-Reranker-4B"):
    """reranker_type -> a scorer object with .score(instruction, query, docs, batch_size).
    Used by rerank_server.py so ONE server serves any HF reranker."""
    if reranker_type == "qwen3_reranker":
        return QwenReranker(model_path=qwen3_path, cache_dir=cache_dir, quant=quant, device=device)
    if reranker_type in MONOT5_NAMES:
        return MonoT5Scorer(MONOT5_NAMES[reranker_type], cache_dir, device="cuda")
    if reranker_type == "rankllama":
        return RankLlamaScorer(cache_dir, quant=quant)
    raise NotImplementedError(f"reranker_type {reranker_type} not supported by the server")


def _qwen3_instruction(args):
    return (QWEN3_RERANK_CONV_INSTRUCTION
            if args.reranking_query_type == "qwen_3_rerank_instruct_full"
            else QWEN3_RERANK_DEFAULT_INSTRUCTION)


def _score_and_writeback(hits, args, reranking_query_dic, searcher, doc_cache, scorer, instruction):
    """Shared: build per-query (instruction, query, docs) jobs -> score (pool = concurrent
    across N GPUs; single scorer = sequential) -> writeback rank->1/(rank+1). Reranker-agnostic."""
    qid_order = list(hits.keys())
    jobs = []
    for qid in qid_order:
        new_hits = hits[qid][0:args.rerank_top_k]
        docs = [_fetch_contents_cached(searcher, d.docid, doc_cache) for d in new_hits]
        jobs.append((instruction, reranking_query_dic[qid], docs))
    if isinstance(scorer, RemoteRerankerPool):
        all_scores = scorer.score_many(jobs, args.rerank_batch_size)
    else:
        all_scores = [scorer.score(inst, q, docs, args.rerank_batch_size)
                      for inst, q, docs in tqdm(jobs, desc="Reranking")]
    for qid, scores in zip(qid_order, all_scores):
        hit = hits[qid]
        indexes = np.argsort(np.array(scores, dtype=np.float32))[::-1]
        for rank, index in enumerate(indexes):
            hit[index].rank = rank
        for rank, doc_object in enumerate(hit):
            doc_object.score = (1/(doc_object.rank + 1) if rank < args.rerank_top_k
                                else 1/(rank + 1))
        hits[qid] = sorted(hit, key=lambda x: x.score, reverse=True)


#######################################################
###################### Reranking ######################
#######################################################
def rerank(hits, args):

    """
    Perform reranking on hits objects.

    Args:
        hits (Dict[str, List[Any]): Pyserini hits object, or a "PyScoredDoc" similar to an Anserini hit object. Must include .docid and .score.
        args (Any): Additional arguments.

    Returns:
        hits (Dict[str, List[Any]): Pyserini hits object, or a "PyScoredDoc" similar to an Anserini hit object. Must include .docid and .score. The result list should be sorted 
    """

    '''
    All required arguments are:
        - args.reranking_query_list: List[str]: List of reranking queries.
        - args.reranker: str
        - args.rerank_top_k: int
        - args.qid_list_string: List[str]: List of query IDs.
        - args.sparse_index_dir_path: str path to the pyserini index.
        - args.rerank_batch_size: int batch size for reranking.
        # RankGPT
            - args.step: int
            - args.window_size: int
            - args.rankgpt_llm: str
        # Rankllama
            - args.rerank_quant: str
            - args.cache_dir: str  (also applied for T5)
    '''

    print(f"{args.reranker} reranking top {args.rerank_top_k}...")

    # generate a qid-reranking_query dictionary
    reranking_query_dic = {qid: reranking_query for qid, reranking_query in zip(args.qid_list_string, args.reranking_query_list)}

    searcher = LuceneSearcher(args.sparse_index_dir_path)

    # per-call docid->contents cache shared by all queries (same doc appears in many
    # queries' top-k; saves the repeated raw()+json.loads)
    doc_cache = {}

    # ---- UNIFIED REMOTE PATH: any reranker served by resident server(s). One URL ->
    # RemoteReranker; comma-list -> RemoteRerankerPool (data-parallel across N GPUs). The
    # server is configured with --reranker_type, so this client is reranker-agnostic
    # (qwen3 sends its instruction; monot5/rankllama send "" which the server ignores).
    # rankgpt is API-based (no model server) -> falls through to its own branch.
    remote_url = getattr(args, "rerank_remote_url", "none")
    urls = [u.strip() for u in str(remote_url).split(",") if u.strip() and u.strip() != "none"]
    if urls and args.reranker != "rankgpt":
        instruction = _qwen3_instruction(args) if args.reranker == "qwen3_reranker" else ""
        scorer = RemoteRerankerPool(urls) if len(urls) > 1 else RemoteReranker(urls[0])
        print(f"[rerank] remote {args.reranker} via {len(urls)} server(s)")
        _score_and_writeback(hits, args, reranking_query_dic, searcher, doc_cache, scorer, instruction)
        return hits

    if args.reranker == "rankgpt":

        # generate input format required by rankgpt
        rank_gpt_list, _ = hits_2_rankgpt_list(searcher, reranking_query_dic, hits,
                                               rank_end=args.rerank_top_k)

        # get hyperparameters
        llm_name = args.rankgpt_llm
        rank_end = args.rerank_top_k
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
                [_fetch_contents_cached(searcher, doc_object.docid, doc_cache)
                 for doc_object in hit[0:args.rerank_top_k]],
                tokenizer,
                model,
                args.rerank_batch_size
            )

            np_reranked_scores = np.array(reranked_scores, dtype=np.float32)

            indexes = np.argsort(np_reranked_scores)[::-1]
            for rank, index in enumerate(indexes):
                hit[index].rank = rank 
            
            # change the score according to the rank
            for rank, doc_object in enumerate(hit):
                if rank < args.rerank_top_k:
                    doc_object.score = 1/(doc_object.rank + 1)
                else:
                    doc_object.score = 1/(rank + 1)
            
            # sort the hits by score
            hits[qid] = sorted(hit, key=lambda x: x.score, reverse=True)

    elif "monot5" in args.reranker:

        # get reranker_name
        if args.reranker == "monot5_base":
            reranker_name = "castorini/monot5-base-msmarco"
        elif args.reranker == "monot5_base_10k":
            reranker_name = "castorini/monot5-base-msmarco-10k"
        elif args.reranker == "monot5_large":
            reranker_name = "castorini/monot5-large-msmarco"
        elif args.reranker == "monot5_large_10k":
            reranker_name = "castorini/monot5-large-msmarco-10k"
        elif args.reranker == "monot5_3b":
            reranker_name = "castorini/monot5-3b-msmarco"
        elif args.reranker == "monot5_3b_10k":
            reranker_name = "castorini/monot5-3b-msmarco-10k"
        else:
            raise NotImplementedError(f"reranker {args.reranker} not implemented")

        # load model
        print("loading t5 model")
        tokenizer, model, decoder_stard_id, targeted_ids =\
             load_t5_DP(args.cache_dir, reranker_name)

        print("reranking")
        for qid, hit in tqdm(hits.items(), total=len(hits), desc="Reranking"):

            reranking_query = reranking_query_dic[qid]

            new_hits = hit[0:args.rerank_top_k]
            doc_contents = [_fetch_contents_cached(searcher, doc_object.docid, doc_cache)
                            for doc_object in new_hits]

            reranked_scores = rerank_t5_DP(
                reranking_query,
                doc_contents,
                tokenizer,
                model,
                decoder_stard_id,
                targeted_ids,
                args.rerank_batch_size
            )

            np_reranked_scores = np.array(reranked_scores, dtype=np.float32)

            indexes = np.argsort(np_reranked_scores)[::-1]
            for rank, index in enumerate(indexes):
                hit[index].rank = rank 
            
            # change the score according to the rank
            for rank, doc_object in enumerate(hit):
                if rank < args.rerank_top_k:
                    doc_object.score = 1/(doc_object.rank + 1)
                else:
                    doc_object.score = 1/(rank + 1)

            # sort the hits by score
            hits[qid] = sorted(hit, key=lambda x: x.score, reverse=True)

    elif args.reranker == "qwen3_reranker":
        # in-process (no remote): the unified remote path above already handled the
        # server/pool case. Load QwenReranker locally + score via the shared helper.
        instruction = _qwen3_instruction(args)
        print(f"loading qwen3 reranker; instruction: {instruction!r}")
        reranker = QwenReranker(
            model_path=getattr(args, "qwen3_reranker_path", "Qwen/Qwen3-Reranker-4B"),
            cache_dir=args.cache_dir, quant=args.rerank_quant,
            device=f"cuda:{getattr(args, 'rerank_gpu_id', 0)}" if torch.cuda.is_available() else "cpu",
        )
        _score_and_writeback(hits, args, reranking_query_dic, searcher, doc_cache, reranker, instruction)

    return hits