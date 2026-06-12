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
    the gen-LLM GPU). Quantization: none -> bf16 (~9G for 4B); 8b/4b -> bitsandbytes
    (same interface as rankllama's rerank_quant).
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

    def score(self, instruction, query, docs, batch_size):
        """One query vs many docs -> list[float], batched by rerank_batch_size."""
        pairs = [self.format_pair(instruction, query, d) for d in docs]
        num_to_split = get_split_num(len(pairs), batch_size)
        scores = []
        for part in np.array_split(np.array(pairs, dtype=object), num_to_split):
            scores.extend(self._score_batch(list(part)))
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

        # instruction routing: the conversational query field carries profile+conversation
        # (judge if the doc helps answer the last question); any other reranking_query_type
        # (e.g. a personalized rewrite) uses the model's NATIVE default instruction.
        if args.reranking_query_type == "qwen_3_rerank_instruct_full":
            instruction = QWEN3_RERANK_CONV_INSTRUCTION
        else:
            instruction = QWEN3_RERANK_DEFAULT_INSTRUCTION
        print(f"qwen3 reranker instruction: {instruction!r}")

        remote_url = getattr(args, "rerank_remote_url", "none")
        if remote_url and remote_url != "none":
            # model hosted on another machine (rerank_server.py, e.g. octal31) —
            # no local VRAM used at all
            print(f"using REMOTE qwen3 reranker at {remote_url}")
            reranker = RemoteReranker(remote_url)
        else:
            print("loading qwen3 reranker")
            reranker = QwenReranker(
                model_path=getattr(args, "qwen3_reranker_path", "Qwen/Qwen3-Reranker-4B"),
                cache_dir=args.cache_dir,
                quant=args.rerank_quant,
                device=f"cuda:{getattr(args, 'rerank_gpu_id', 0)}" if torch.cuda.is_available() else "cpu",
            )

        print("reranking")
        for qid, hit in tqdm(hits.items(), total=len(hits), desc="Reranking"):

            reranking_query = reranking_query_dic[qid]

            new_hits = hit[0:args.rerank_top_k]
            doc_contents = [_fetch_contents_cached(searcher, doc_object.docid, doc_cache)
                            for doc_object in new_hits]

            reranked_scores = reranker.score(
                instruction, reranking_query, doc_contents, args.rerank_batch_size)

            np_reranked_scores = np.array(reranked_scores, dtype=np.float32)

            indexes = np.argsort(np_reranked_scores)[::-1]
            for rank, index in enumerate(indexes):
                hit[index].rank = rank

            # change the score according to the rank (same convention as the others)
            for rank, doc_object in enumerate(hit):
                if rank < args.rerank_top_k:
                    doc_object.score = 1/(doc_object.rank + 1)
                else:
                    doc_object.score = 1/(rank + 1)

            # sort the hits by score
            hits[qid] = sorted(hit, key=lambda x: x.score, reverse=True)

    return hits