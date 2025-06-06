# compute personalized weight according to PTKB - topic_entropy
import json
from pyserini.search.lucene import LuceneSearcher
searcher = LuceneSearcher("/part/01/Tmp/yuchen/indexes/clueweb22b_ikat23_fengran_sparse_index_2/")

def load_document_by_id(doc_id):

    doc_content = json.loads(searcher.doc(doc_id).raw())

    return doc_content["contents"]

from transformers import RobertaTokenizer, RobertaModel
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import math


from apcir.search.search import load_ranking_list_from_file

# 可能要按需稍微改一下输入函数的参数
def calculate_topic_entrop(
    encoder_model, 
    tokenizer, 
    turn_list,
    result_file = "None", 
    top_k = 100
    ):


    @torch.no_grad()
    def encode_texts(texts, batch_size=128):
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=128).to(model.device)
            outputs = model(**inputs)

        # normalize the embeddings
        return outputs/outputs.norm(dim=-1, keepdim=True)

    def calculate_entropy(probabilities):
        return -sum(p * math.log(p) for p in probabilities if p > 0)

    def personalization_need(probabilities):
        entropy = calculate_entropy(probabilities)
        max_entropy = math.log(len(probabilities))
        return entropy / max_entropy if max_entropy > 0 else 0

    def compute_personalization_for_one_query(docs):

        # 2. encode topic句子
        embeddings = encode_texts(docs)
        # sentence transformer
        # embeddings = encoder_model.encode(docs)

        # 3. 计算主题之间的相似度矩阵
        similarity_matrix = torch.matmul(embeddings, embeddings.T).cpu().numpy()
        #sentence transformer
        # similarity_matrix = encoder_model.similarity(embeddings, embeddings).cpu().numpy()

        # 4. 每个主题平均相似度
        avg_similarities = np.mean(similarity_matrix, axis=1)

        # 5. 差异度（1-相似度）
        dissimilarities = 1 - avg_similarities
        dissimilarities = np.clip(dissimilarities, 0, None)

        # 6. 归一化成概率分布
        total = np.sum(dissimilarities)
        probabilities = dissimilarities / total if total > 0 else np.ones(len(dissimilarities)) / len(dissimilarities)

        # 7. 计算个性化需求
        score = personalization_need(probabilities)
        #score = max(0.0, min(score, 1.0))

        return score



    # hits = load_ranking_list_from_file(result_file) 
    # ranking_list = hits[qid][:top_k] 
    # ranked_doc_contents = [load_document_by_id(hit.docid) for hit in ranking_list] # load the document by doc_id

    qid_personalized_weight_dict = {}

    for turn in turn_list:
        qid = turn.turn_id
        ptkb_sentences = list(turn.ptkb.values()) #"ptkb: {"1": s1, "2": s2, ...}"

        model = encoder_model
        model.eval()
        w3_score = compute_personalization_for_one_query(ptkb_sentences)

        qid_personalized_weight_dict[qid] = w3_score
   
    # weight normalization
    max_weight = max(qid_personalized_weight_dict.values())
    min_weight = min(qid_personalized_weight_dict.values())

    for qid, weight in qid_personalized_weight_dict.items():
        if max_weight - min_weight != 0:
            qid_personalized_weight_dict[qid] = (weight - min_weight) / (max_weight - min_weight)
        else:
            qid_personalized_weight_dict[qid] = 0.5
    
    # calculate w1 and w2 according to w3
    for qid, weight in qid_personalized_weight_dict.items():
        non_p_w = abs(1 - weight) / 2
        qid_personalized_weight_dict[qid] = [non_p_w, non_p_w, weight]
        
       
    return qid_personalized_weight_dict  

