import os
import numpy as np
import json
import re
from typing import List, Any, Dict
import pytrec_eval

from apcir.functional.topics import (
    Turn, 
    load_turns_from_json,
    filter_ikat_23_evaluated_turns,
    filter_ikat_24_evaluated_turns,
    filter_ikat_25_evaluated_turns,
    get_turn_by_qid
    )

from apcir.functional.constants import IKAT_AUTOMATIC_RUN_TEMPLATE_DICT



# Get the filename without extension nor parent directory
def extract_filename(path):
    # Extract the filename with extension
    filename_with_ext = os.path.basename(path)
    # Remove the extension
    filename, ext = os.path.splitext(filename_with_ext)
    return filename

# get query list & qid list for pyserirni batch search.
def get_query_list(args):

    turn_list = []
    query_list = []
    qid_list_string = []
    reranking_query_list = []
    generation_query_list = []

    '''
    Arguments:
    args.topics: str
    args.input_query_path: str
    args.retrieval_model: str
    args.retrieval_query_type: str
    args.reranking_query_type: str
    args.generation_query_type: str
    args.fb_terms: int
    args.original_query_weight: float
    args.fusion_type: str
    args.QRs_to_rank: List[str]
    args.fuse_weights: List[float]
    args.fusion_query_lists: List[List[str]]
    args.personalization_group: str
    args.qid_personalized_level_dict: Dict[str, str]
    args.level_type: str

    Returns:
    retrieval_query_list: List[str]
    reranking_query_list: List[str]
    generation_query_list: List[str]
    fusion_query_lists: List[List[str]]
    qid_list_string: List[str]
    turn_list: List[Turn]

    '''

    # Route the logical "full_conversation" QR to a model-appropriate variant:
    #   BM25 (lexical retrieval)              -> full_conversation_sparse (plain-text concat)
    #   any dense/neural model (ance, qwen3,  -> full_conversation_dense  (placeholder "[SEP]"
    #     splade, dpr, ...)                       + conversational token build, ANCE-only today)
    # The test is on BM25, so EVERYTHING non-BM25 falls through to dense. The outer guard is
    # REQUIRED: without it the `else` clobbers every non-BM25 run's query type (that was a bug).
    if args.retrieval_query_type == "full_conversation":
        if args.retrieval_model == "BM25":
            args.retrieval_query_type = "full_conversation_sparse"
        else:
            args.retrieval_query_type = "full_conversation_dense"
    
    # TODO: for reranking and generation, conceptually we use LLM readable format full conversation, right?

    # apply topic specific processing
    if ("ikat" in args.topics or "topiocqa" in args.topics
            or args.topics.startswith("perso_dense") or args.topics.startswith("cast_")):
        turn_list = load_turns_from_json(
            input_topic_path=args.input_query_path,
            range_start=0,
            range_end=-1
            )

        # full_conversation (iKAT/ANCE): attach each turn's interleaved (user, system)
        # history so query_type_2_query can build the conversational query. iKAT stores
        # only prior USER utterances on `context_utterances`, so we reconstruct the
        # interleaved [u1, r1, u2, r2, ...] from the full turn list (grouped by
        # conversation, ordered by turn). Each turn.fullconv_ctx = everything BEFORE it.
        if "ikat" in args.topics:
            from collections import defaultdict
            conv_groups = defaultdict(list)
            for t in turn_list:
                conv_groups[t.conversation_id].append(t)
            for conv_turns in conv_groups.values():
                conv_turns.sort(key=lambda x: x.get_turn_order())
                hist = []
                for t in conv_turns:
                    t.fullconv_ctx = list(hist)
                    hist.append(t.current_utterance)
                    hist.append(t.current_response)

            # previous-conversation context (qwen_conversation_ptkb_previous_conv_as_ptkb).
            # iKAT-2025 ONLY: only 2025 has a persona X with multiple conversations X-1, X-2
            # (conversation_id like "1-2"); 23 uses "9-1" (single conv/persona) and 24 uses a
            # bare int (0,1,...) — neither has a "previous conversation", and 24's int id would
            # break .split. The "previous conversation" of a turn = the FULL interleaved
            # [u, r, ...] of that persona's EARLIER conversation(s). str(cid) is defensive.
            if args.topics == "ikat_25_test":
                full_conv = {}          # conversation_id -> [u1, r1, u2, r2, ...] (whole conv)
                persona_to_cids = defaultdict(list)
                for cid, conv_turns in conv_groups.items():
                    flat = []
                    for t in conv_turns:
                        flat.append(t.current_utterance)
                        flat.append(t.current_response)
                    full_conv[cid] = flat
                    persona_to_cids[str(cid).split('-')[0]].append(cid)
                for cid, conv_turns in conv_groups.items():
                    siblings = sorted(persona_to_cids[str(cid).split('-')[0]],
                                      key=lambda x: [int(p) for p in str(x).split('-') if p.isdigit()])
                    prev_flat = []
                    for prev_cid in siblings[:siblings.index(cid)]:
                        prev_flat += full_conv[prev_cid]
                    for t in conv_turns:
                        t.prev_conv_ctx = list(prev_flat)

                # new_ptkb (for qwen_conversation_rel_new_ptkb): organizer-oracle carried-over
                # facts in ptkb-update.json (sibling of the topics file), keyed by conversation
                # `number`, each with a turn_dependence list of turn numbers. Attach the statements
                # that apply to each turn (its turn order is in turn_dependence).
                import os as _os, json as _json
                _upd = _os.path.join(_os.path.dirname(args.input_query_path), "ptkb-update.json")
                _new_map = {}
                if _os.path.exists(_upd):
                    for _e in _json.load(open(_upd)):
                        _new_map[str(_e["number"])] = _e.get("new_ptkb", [])
                for _cid, _conv_turns in conv_groups.items():
                    _entries = _new_map.get(str(_cid), [])
                    for _t in _conv_turns:
                        _tno = _t.get_turn_order()
                        _t.applicable_new_ptkb = [n["statement"].strip() for n in _entries
                                                  if _tno in (n.get("turn_dependence") or [])]

        # filter out the non-evaluated turns for ikat 23
        if args.topics == "ikat_23_test":
            evaluated_turn_list = filter_ikat_23_evaluated_turns(turn_list)
        elif args.topics == "ikat_24_test":
            evaluated_turn_list = filter_ikat_24_evaluated_turns(turn_list)
        elif args.topics == "ikat_25_test":
            evaluated_turn_list = filter_ikat_25_evaluated_turns(turn_list)
        elif "topiocqa" in args.topics:
            evaluated_turn_list = turn_list
        elif args.topics.startswith("cast_"):
            # CAsT has NO user profile (build_cast_topics writes ptkb={}). A profile- or
            # conversation-template query type would therefore silently build a query with an
            # empty "User Profile:" section and still produce a plausible-looking run named
            # after the profile method. Fail loudly instead.
            _profile_qts = ("ptkb", "perso", "personalized", "qwen_conversation")
            if any(k in args.retrieval_query_type for k in _profile_qts):
                raise ValueError(
                    f"--retrieval_query_type={args.retrieval_query_type!r} needs a user profile, "
                    f"but CAsT topics ({args.topics}) have none. Use raw / oracle / "
                    f"cast_automatic_rewrite, or a non-profile reformulation.")
            # CAsT topics are converted by apcir/preprocess/build_cast_topics.py and already
            # contain exactly the turns of the official topic file; turn_id == the qrel qid
            # ("{topic}_{turn}"), so every turn is an evaluated turn. Turns absent from the
            # qrel are simply not scored by pytrec_eval.
            evaluated_turn_list = turn_list
        elif args.topics.startswith("perso_dense"):
            # The perso_dense_{val,train} topics file already contains ONLY the held-out / train-split
            # turns of the personalized-dense-retriever experiment. Their query is a PRE-BUILT
            # reformulation (perso_dense_val_ptkb / perso_dense_val_rel_ptkb), so the per-year
            # context attachment + builders above are skipped (the `if "ikat"` block did not run),
            # and every turn here is "evaluated".
            evaluated_turn_list = turn_list
        
        qid_list_string = [str(turn.turn_id) for turn in evaluated_turn_list]

        # Get the personalization level for each turn (if applicable)
        if (
            args.fusion_type == 'per_query_personalize_level' or
            args.personalization_group != "all" 
        ):
            print("get personalized level for each query....")
            qid_personalized_level_dict = \
            {
                turn.turn_id: turn.get_personalization_level(args.level_type) for turn in evaluated_turn_list
            }

            if args.personalization_group != "all":
                evaluated_turn_list = [turn for turn in evaluated_turn_list if qid_personalized_level_dict[turn.turn_id] == args.personalization_group] 

                qid_list_string = [qid for qid in qid_list_string if qid_personalized_level_dict[qid] == args.personalization_group]
        else:
            qid_personalized_level_dict = None
        

        # get different query representations for fusion
        if args.fusion_type != "none":
            fusion_query_lists = []
            for QR_name in args.QRs_to_rank:
                fusion_query_lists.append([turn.query_type_2_query(QR_name, args.fb_terms, args.original_query_weight,args) for turn in evaluated_turn_list])
        else:
            fusion_query_lists = None

        # load query according to query type.
        retrieval_query_list = [turn.query_type_2_query(args.retrieval_query_type, args.fb_terms, args.original_query_weight,args) for turn in evaluated_turn_list]
        reranking_query_list = [turn.query_type_2_query(args.reranking_query_type , args.fb_terms, args.original_query_weight,args) for turn in evaluated_turn_list]
        generation_query_list = [turn.query_type_2_query(args.generation_query_type, args.fb_terms, args.original_query_weight,args) for turn in evaluated_turn_list]
    

        # Load the fusion weights from turn object, if applicable
        if args.fusion_type == "pre_calculated":
            qid_weights_dict = {turn.turn_id : turn.query_type_2_query(args.retrieval_query_type, 0, 0,args) for turn in evaluated_turn_list}
            assert len(qid_weights_dict[evaluated_turn_list[0].turn_id]) == len(args.QRs_to_rank), "The number of weights does not match the number of queries."
        else:
            qid_weights_dict = None
        

    assert len(retrieval_query_list) != 0, "No queries found, args.topics may be wrong"
    assert len(retrieval_query_list) == len(qid_list_string), "Number of queries and qid_list_string not match"

    
    return retrieval_query_list, reranking_query_list, generation_query_list, fusion_query_lists, qid_list_string, qid_personalized_level_dict, qid_weights_dict, turn_list



def collapse_passages_to_docs(run: dict, sep: str = "_"):
    """Collapse passage ids to document ids, keeping each document's best passage score
    (the "max passage" rule) and dropping duplicate documents.

    WHY. TREC CAsT 2021 is judged at DOCUMENT level. Per the track overview this was not
    the plan: NIST found that participants running different spaCy versions produced
    different passage segmentations, so some submitted passage ids did not exist. NIST
    therefore "truncated passage identifiers, used a max passage algorithm to convert
    passage runs to document runs, and removed duplicate retrieved documents". There is no
    passage-level 2021 qrel, so a passage-level run must be mapped down before scoring or
    every metric comes out 0.

    VERIFIED. Applying this function to the official baseline passage run
    (`org_automatic_results_1000.v1.0.run`) and scoring at cutoff 500 with
    relevance_level=2 reproduces the overview's `org_auto_bm25_t5` row EXACTLY:
    Recall .636, MAP .291, MRR .607, NDCG .504, NDCG@3 .436.

    Note the pre-converted `document_runs/` files shipped in the treccastweb repo do NOT
    reproduce those numbers (.623/.282/.597/.493/.424): they cut Washington Post ids at the
    first hyphen, so `WAPO_50658292-34ef-11e2-92f0-496af208bf23-0` becomes `WAPO_50658292`,
    which matches no qrel entry. The qrels use the FULL uuid. Keep the uuid intact.

    SEPARATOR. The passage index differs by source: the released 2021 collection uses
    `_<n>` (`MARCO_D1167206_1`) while the official 2021 run files use `-<n>`
    (`MARCO_D1599536-11`). Pass `sep` accordingly. Do NOT accept both at once: WaPo uuids
    are hyphen-separated, so a `-\\d+$` rule would corrupt any uuid whose final group is
    all digits.

    GUARD. Stripping is skipped when it would leave only the corpus prefix, so a run that
    already carries document-level ids (e.g. `KILT_1001165`, where the id itself ends in
    `_<digits>`) passes through unchanged instead of collapsing to `KILT`.
    """
    pat = re.compile(re.escape(sep) + r"\d+$")
    prefixes = {"MARCO", "WAPO", "KILT", "CAR"}
    collapsed = {}
    for qid, docs in run.items():
        best = {}
        for pid, score in docs.items():
            did = pat.sub("", pid)
            if did in prefixes:          # would have destroyed the id
                did = pid
            if did not in best or score > best[did]:
                best[did] = score
        collapsed[qid] = best
    return collapsed


def evaluate(
    run: dict,
    qrel_file_path: str,
    ranking_list_path: str,
    metrics_list: List[str],
    metrics_list_key_form: List[str],
    passage_to_doc: bool = False
):

    '''
    Evaluate the ranking list using pytrec_eval.
    Args:
        run (dict): ranking list in dictionary format required by pytrec_eval. If None, the ranking list will be read from ranking_list_path.
            - example:     
                     run = {qid: {doc.docid: doc.score for doc in docs} for qid, docs in hits.items()}
        qrel_file_path (str): path to the trec format qrel file
        ranking_list_path (str): path to the trec format ranking list file
        metrics_list (List[str]): list of metrics to evaluate
            - example: ["map", "ndcg_cut.10", "P.5"]
        metrics_list_key_form (List[str]): list of metrics in key form (change . to _)
            - example: ["map", "ndcg_cut_10", "P_5"]
    
    Returns:
        query_metrics_dic (dict): metrics for each query
            - example:          
                query_metrics_dic = {       
                    "qid1" : {"ndcg_cut_10" : 0.1, "map" : 0.2},
                    "qid2" : {"ndcg_cut_10" : 0.2, "map" : 0.3},
                    "qid3" : {"ndcg_cut_10" : 0.3, "map" : 0.4},
                }
        averaged_metrics (dict): averaged metrics
            - example:
                averaged_metrics = {
                    "ndcg_cut_10" : 0.1,
                    "map" : 0.2,
                }
    '''

    # read qrels
    with open(qrel_file_path, 'r') as f_qrel:
        qrel = pytrec_eval.parse_qrel(f_qrel)
    # read ranking list
    if run is None:
        with open(ranking_list_path, 'r') as f_run:
            run = pytrec_eval.parse_run(f_run)

    # CAsT 2021: passage-level run -> document-level qrels (see collapse_passages_to_docs)
    if passage_to_doc:
        n_before = sum(len(v) for v in run.values())
        run = collapse_passages_to_docs(run)
        n_after = sum(len(v) for v in run.values())
        print(f"passage->doc collapse: {n_before} passages -> {n_after} documents")

    # Filter the qrels down to the qids the run actually covers.
    # NOTE pytrec_eval.parse_qrel returns a defaultdict, so a qid absent from the qrels
    # yields {} rather than raising; pytrec_eval then SKIPS such queries instead of
    # scoring them 0. That is what we want for CAsT, where only a subset of turns is
    # judged (2019: 173/479, 2021: 158/239, 2022: 163/205). But it also means a judged
    # qid MISSING FROM THE RUN is silently dropped instead of counting as 0 the way
    # trec_eval would, which would inflate the average -- so warn about it.
    n_judged = sum(1 for v in qrel.values() if v)
    covered = sum(1 for qid in run.keys() if qrel[qid])
    if covered < n_judged:
        print(f"WARNING: run covers only {covered}/{n_judged} judged queries; "
              f"{n_judged - covered} judged queries are missing from the run and will "
              f"NOT be counted as 0 (this inflates the averages)")
    qrel = {qid: qrel[qid] for qid in run.keys()}

    #  evaluate
    print("trec_eval evaluating...")
    evaluator = pytrec_eval.RelevanceEvaluator(qrel, set(metrics_list))
    query_metrics_dic = evaluator.evaluate(run)

    # average metrics
    '''
    example:
    metrics = { 
        "ndcg_cut_10" : [0.1, 0.2, 0.3, 0.4, 0.5], 
        "map" : [0.1, 0.2, 0.3, 0.4, 0.5],
            }
    '''
    metrics = {metric : [metrics[metric] for metrics in query_metrics_dic.values()] for metric in metrics_list_key_form}   

    averaged_metrics = { metric : np.average(metric_list) for metric, metric_list in metrics.items() }

    # for each query, add the number of relevant documents in query_metrics_dic
    # why not use sum(list(qrel[qid].values()))? Because relevance judgement may be graded instead of binary.
    for qid in query_metrics_dic.keys():
        query_metrics_dic[qid]["num_rel"] = sum([1 for doc in qrel[qid].values() if doc > 0])

    print("################# Retrieval eval Results #################")
    print(json.dumps(averaged_metrics, indent=4))
    print("##########################################################")

    return query_metrics_dic, averaged_metrics


def print_formatted_latex_metrics(metrics_dict, metrics_list):
    '''
    usage example:
    print_formateed_latex_metrics(
        {"ndcg_cut_10": 0.1, "map": 0.2}, 
        ["ndcg_cut_10", "map"]
        )
    '''

    # Calculate the metrics and format them
    result = []
    for metric in metrics_list:
        value = metrics_dict.get(metric, 0) * 100
        result.append(f"{value:.1f}")
    
    # Print the result separated by tab
    return " & ".join(result)


def generate_and_save_ikat_submission(
    ikat_output_path: str,
    run_name: str,
    reformulation_name: str,
    hits: Dict[str, List[Any]],
    turn_list : List[Turn],
    response_dict: Dict[str, List[str]],
    top_k: int
    ) -> None:

    # resulting dictionary
    result_dict = IKAT_AUTOMATIC_RUN_TEMPLATE_DICT
    result_dict["run_name"] = run_name

    for qid, ordered_doc_object_list in hits.items():

        three_chiffres = qid.split("-")
        # adapt turn_id format. work for both ikat23 and ikat24
        conversation_id = "-".join(three_chiffres[:-1])
        turn_id = f"{conversation_id}_{three_chiffres[-1]}"
        
        # get ptkb_provenance, which should be from the reformulation.
        turn_object = get_turn_by_qid(qid,turn_list)

        #ptkb_provenance = turn_object.get_ptkb_provenance(reformulation_name)
        # TODO: add ptkb_provenance
        ptkb_provenance = []

        responses = []
        for rank, response in enumerate(response_dict[qid]):
            real_rank = rank + 1
            responses.append(
                {
                    "rank": real_rank,
                    "text": response,
                    "ptkb_provenance": ptkb_provenance,
                    "passage_provenance": [
                        {
                            "id": doc_object.docid,
                            "score": doc_object.score,
                            "used": False if i >= top_k else True
                        } for i, doc_object in enumerate(ordered_doc_object_list)
                    ]
                }
            )
        

        result_dict["turns"].append(
            {
                "turn_id": turn_id,
                "responses": responses
            }
            )
        
    with open(ikat_output_path, "w") as f:
        json.dump(result_dict, f, indent=4)



