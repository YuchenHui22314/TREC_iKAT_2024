"""SPLADE inverted-index structure + sparse retrieval (numba-isolated).

DRY refactor: consolidated from the former {search,indexing/dense,indexing/splade}/libs.py.
Only the genuinely splade-specific, numba-heavy pieces live here so the numba import does not
leak into the widely-imported apcir.utils. Kept the search-side (live retrieval) versions:
  - IndexDictOfArray (parallel p_map load; indexing builds with force_new=True so never loads);
  - SparseRetrieval (the only live copy; indexing copies were dead) + its numba_score_float;
  - load_key (helper for the parallel load).
MixedPrecisionManager was dropped (0 references). NullContextManager stays with the splade model.
"""
from collections import defaultdict
import os
import h5py
import json
import numba
import array
import pickle
import numpy as np
from tqdm import tqdm
from p_tqdm import p_map

import torch
from torch.utils.data import IterableDataset

from apcir.utils import tensor_to_list, PyScoredDoc



def load_key(key,file_name):
    try:
        file = h5py.File(file_name, "r")
        doc_id = np.array(file[f"index_doc_id_{key}"], dtype=np.int32)
        doc_value = np.array(file[f"index_doc_value_{key}"], dtype=np.float32)
        file.close()
        return key, doc_id, doc_value
    except:
        file.close()
        return key, np.array([], dtype=np.int32), np.array([], dtype=np.float32)


class IndexDictOfArray:
    def __init__(self, index_path=None, force_new=False, filename="array_index.h5py", dim_voc=None):
        # index_path = None # for debug
        if index_path is not None:
            self.index_path = index_path
            if not os.path.exists(index_path):
                os.makedirs(index_path)
            self.filename = os.path.join(self.index_path, filename)
            if os.path.exists(self.filename) and not force_new:
                print("index already exists, loading...")
    
                self.file = h5py.File(self.filename, "r")
                if dim_voc is not None:
                    dim = dim_voc
                else:
                    dim = self.file["dim"][()]
                self.index_doc_id = dict()
                self.index_doc_value = dict()

                # A parallel version of the commented loop
                results = p_map(load_key, range(dim), [self.filename]*dim, num_cpus=50)
                self.index_doc_id = {key: doc_id for key, doc_id, _ in results}
                self.index_doc_value = {key: doc_value for key, _, doc_value in results}

                # for key in tqdm(range(dim)):
                #     try:
                #         self.index_doc_id[key] = np.array(self.file["index_doc_id_{}".format(key)],
                #                                           dtype=np.int32)
                #         # ideally we would not convert to np.array() but we cannot give pool an object with hdf5
                #         self.index_doc_value[key] = np.array(self.file["index_doc_value_{}".format(key)],
                #                                              dtype=np.float32)
                #     except:
                #         self.index_doc_id[key] = np.array([], dtype=np.int32)
                #         self.index_doc_value[key] = np.array([], dtype=np.float32)
                self.file.close()
                del self.file
                print("done loading index...")
                doc_ids = pickle.load(open(os.path.join(self.index_path, "doc_ids.pkl"), "rb"))
                self.n = len(doc_ids)
            else:
                self.n = 0
                print("initializing new index...")
                # Empty dictionary of arrays of (1) unsigned int and (2) floats
                self.index_doc_id = defaultdict(lambda: array.array("I"))
                self.index_doc_value = defaultdict(lambda: array.array("f"))
        else:
            self.n = 0
            print("initializing new index...")
            # This is an inverted index. It has the following structure:
            # (1) For each token in vocab, a list of document id of which 
            # the document embedding vector has non-zero value for that token.
            # It is just like the inverted index in the search engine.
            # (2) save the float value for each element in dictionary (1)
            self.index_doc_id = defaultdict(lambda: array.array("I"))
            self.index_doc_value = defaultdict(lambda: array.array("f"))

    def add_batch_document(self, row, col, data, n_docs=-1):
        """add a batch of documents to the index
            Example:
            if batch_documents = [[0, 0, 0.3], [0, 0.4, 0], [0.5, 0, 0]]
            then row = [0, 1, 2], col = [2, 1, 0], data = [0.3, 0.4, 0.5]
        """
        if n_docs < 0:
            self.n += len(set(row))
        else:
            self.n += n_docs
        for doc_id, dim_id, value in zip(row, col, data):
            self.index_doc_id[dim_id].append(doc_id)
            self.index_doc_value[dim_id].append(value)

    def __len__(self):
        return len(self.index_doc_id)

    def nb_docs(self):
        return self.n

    def save(self, dim=None):
        print("converting to numpy")
        for key in tqdm(list(self.index_doc_id.keys())):
            self.index_doc_id[key] = np.array(self.index_doc_id[key], dtype=np.int32)
            self.index_doc_value[key] = np.array(self.index_doc_value[key], dtype=np.float32)
        print("save to disk")
        with h5py.File(self.filename, "w") as f:
            if dim:
                f.create_dataset("dim", data=int(dim))
            else:
                f.create_dataset("dim", data=len(self.index_doc_id.keys()))
            for key in tqdm(self.index_doc_id.keys()):
                f.create_dataset("index_doc_id_{}".format(key), data=self.index_doc_id[key])
                f.create_dataset("index_doc_value_{}".format(key), data=self.index_doc_value[key])
            f.close()
        print("saving index distribution...")  
        # => size of each posting list in a dict
        index_dist = {}
        for k, v in self.index_doc_id.items():
            index_dist[int(k)] = len(v)
        json.dump(index_dist, open(os.path.join(self.index_path, "index_dist.json"), "w"))


class SparseRetrieval:
    """
    retrieval from Splade SparseIndexing
    """

    @staticmethod
    def select_topk(filtered_indexes, scores, k):
        if len(filtered_indexes) > k:
            # take k smallist values indexes
            # NOTE: the topk may not be sorted. argpartition
            # just put the k smallest values in the first k indexes,
            # but the internal order is not guaranteed. 
            sorted_ = np.argpartition(scores, k)[:k]
            filtered_indexes, scores = filtered_indexes[sorted_], -scores[sorted_]
        else:
            scores = -scores
        return filtered_indexes, scores

    @staticmethod
    @numba.njit(nogil=True, parallel=True, cache=True)
    def numba_score_float(inverted_index_ids: numba.typed.Dict,
                          inverted_index_floats: numba.typed.Dict,
                          indexes_to_retrieve: np.ndarray,
                          query_values: np.ndarray,
                          threshold: float,
                          size_collection: int):
        '''
        Get all document scores for a given query
        Args:
            inverted_index_ids: dict[vocab, [doc1, doc2, ...]]
            inverted_index_floats: dict[vocab, [value1, value2, ...]]
            indexes_to_retrieve: [235,22314,114,514], vocab id of non-zero tokens in the query 
            query_values: [value1, value2, ...], of length non-zero tokens in the query
            threshold: 0
            size_collection: literally.
        Returns:
            filtered_indexes: [3,5,8,...,] (all the indexes where the score > threshold)
            -scores[filtered_indexes]: [0.1, 0.2, 0.3, ...] (the scores of the filtered indexes)
            
        '''
        # initialize array with size = size of collection
        # like: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        scores = np.zeros(size_collection, dtype=np.float32)  
        n = len(indexes_to_retrieve)
        # for every non-zero tokens in the query
        for _idx in range(n):
            # get the position of the token in vocabulary.abs
            # i.e. which posting list to search
            local_idx = indexes_to_retrieve[_idx]  

            # what is the value of the token in the query
            query_float = query_values[_idx]  

            # get the document id list containing the token
            retrieved_indexes = inverted_index_ids[local_idx]  

            # get the value of the token in each documents
            retrieved_floats = inverted_index_floats[local_idx]  

            for j in numba.prange(len(retrieved_indexes)):
                # for each document containing the token, calculate the score
                # which is the product of the value of the token in the query and the value of the token in the document 
                scores[retrieved_indexes[j]] += query_float * retrieved_floats[j]
        
        # filter the documents with score > threshold
        # filtered_indexes = [3,5,8,...,] (all the indexes where the score > threshold)
        filtered_indexes = np.argwhere(scores > threshold)[:, 0]  
        # unused documents => this should be tuned, currently it is set to 0
        return filtered_indexes, -scores[filtered_indexes]

    def __init__(self, index_dir_path, retrieval_output_path, dim_voc, top_k):
        self.sparse_index = IndexDictOfArray(index_dir_path, dim_voc=dim_voc)
        self.doc_ids = pickle.load(open(os.path.join(index_dir_path, "doc_ids.pkl"), "rb"))
        self.top_k = top_k
        self.retrieval_output_path = retrieval_output_path

        # Convert the python inverted index (~235 GB) to numba typed dicts. MEMORY: build
        # by POPPING each posting list out of the python dict as it is moved into the numba
        # dict, so we never hold two full copies at once. The old code iterated .items()
        # leaving the python dict fully alive alongside the numba copy -> ~470 GB peak (it
        # OOMs near the 503 GB box). Popping caps the peak at ~235 GB. The numba dict
        # CONTENTS are identical (same keys -> same arrays), so retrieval is byte-identical.
        self.numba_index_doc_ids = numba.typed.Dict()
        self.numba_index_doc_values = numba.typed.Dict()
        _ids = self.sparse_index.index_doc_id
        for key in list(_ids.keys()):
            self.numba_index_doc_ids[key] = _ids.pop(key)
        _vals = self.sparse_index.index_doc_value
        for key in list(_vals.keys()):
            self.numba_index_doc_values[key] = _vals.pop(key)
        # the python dicts are now empty; drop the references
        self.sparse_index.index_doc_id = None
        self.sparse_index.index_doc_value = None
        
    
    def retrieve(self, qid2emb):
        '''
        Given a group of queries, retrieve the top k documents for each query
        Args:
            qid2emb: {qid: query_embedding}
        Returns:
            res: {qid: {doc_id: score}}
            hits: {qid: [PyScoredDoc]}, PyScoredDoc is a namedtuple(docid, score)
        '''
            
        res = defaultdict(dict)
        hits = defaultdict(list)
        for qid in tqdm(qid2emb):
            query_emb = qid2emb[qid]
            query_emb = query_emb.view(1, -1)
            row, col = torch.nonzero(query_emb, as_tuple=True)
            values = query_emb[tensor_to_list(row), tensor_to_list(col)]
            threshold = 0
            filtered_indexes, scores = self.numba_score_float(
                self.numba_index_doc_ids,
                self.numba_index_doc_values,
                col.cpu().numpy(),
                values.cpu().numpy().astype(np.float32),
                threshold=threshold,
                size_collection=self.sparse_index.nb_docs()
            )
            # threshold set to 0 by default, could be better
            filtered_indexes, scores = self.select_topk(filtered_indexes, scores, k=self.top_k)
            for id_, sc in zip(filtered_indexes, scores):
                res[str(qid)][str(self.doc_ids[id_])] = float(sc)
                hits[str(qid)].append(PyScoredDoc(docid = str(self.doc_ids[id_]), score = float(sc)))
            # sort the hits by score
            hits[str(qid)] = sorted(hits[str(qid)], key=lambda x: x.score, reverse=True)

        print("Splade Sparse Retrieval Done")

        return res, hits
        





PyTorch_over_1_6 = float((torch.__version__.split('.')[1])) >= 6 and float((torch.__version__.split('.')[0])) >= 1

# replace this with  contextlib.nullcontext if python >3.7
# see https://stackoverflow.com/a/45187287
