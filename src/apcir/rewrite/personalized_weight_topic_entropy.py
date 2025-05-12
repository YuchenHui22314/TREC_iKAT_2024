# compute personalized weight according to PTKB - topic_entropy
from transformers import RobertaTokenizer, RobertaModel
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import math

# 可能要按需稍微改一下输入函数的参数
def calculate_topic_entrop(encoder_model, tokenizer, turn):

    model = encoder_model
    model.eval()

    @torch.no_grad()
    def encode_texts(texts, batch_size=32):
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=128)
            outputs = model(**inputs)

        # normalize the embeddings
        return outputs/outputs.norm(dim=-1, keepdim=True)

    def calculate_entropy(probabilities):
        return -sum(p * math.log(p) for p in probabilities if p > 0)

    def personalization_need(probabilities):
        entropy = calculate_entropy(probabilities)
        max_entropy = math.log(len(probabilities))
        return entropy / max_entropy if max_entropy > 0 else 0

    def compute_personalization_for_one_query(ptkb_sentences):

        # 2. encode topic句子
        embeddings = encode_texts(ptkb_sentences)
        # sentence transformer
        # embeddings = encoder_model.encode(ptkb_sentences)

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
        score = max(0.0, min(score, 1.0))

        return score


    ptkb_sentences = turn.ptkb # 获取ptkb, "ptkb: {"1": s1, "2": s2, ...}"

    w3_score = compute_personalization_for_one_query(list(ptkb_sentences.values()))

    w1 = abs(1 - w3_score) / 2
    w2 = abs(1 - w3_score) / 2
    
    return [w1, w2, w3_score] # w1, w2, w3_score

