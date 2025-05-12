# compute personalized weight for DEPS method - heuristic-based
# input是根据非personalized reformulated query检索得到的top-k output file
def variance_of_first_k_elements(data, k):
    result = {}
    for key, value_list in data.items():
        if value_list:  # 检查列表是否为空
            sub_list = value_list[:k]  # 取前k个元素
            if sub_list:
                mean_value = sum(sub_list) / len(sub_list)
                variance = sum((x - mean_value) ** 2 for x in sub_list) / len(sub_list)
            else:
                variance = None
        else:
            variance = None
        result[key] = variance
    return result

def calculate_fusion_weight(data):
    result = {}
    for key, value in data.items():
        result[key] = []
        non_p_w = abs(1 - value) / 2
        result[key].append(non_p_w)
        result[key].append(non_p_w)
        result[key].append(value)
    return result

def calculate_std_top_k_list(output_file, k=100):
    with open(output_file, 'r') as f:
        top_k_result = f.readlines()
    
    qid2score_list = {} # {qid: [p1_score, p2_score, ...]}
    for line in top_k_result:
        qid, _, pid, rank, score, __ = line.strip().split()
        score = float(score)
        if qid not in qid2score_list:
            qid2score_list[qid] = []
            qid2score_list[qid].append(score)
        elif qid in qid2score_list:
            qid2score_list[qid].append(score)
    
    # indicate the personalized weight w3 for each query, which should apply on q_n^u+r_n^u
    qid2std = variance_of_first_k_elements(qid2score_list, k) # {qid: std}
    
    # then the weight w1 and w2 for non-personalized query q_n+r_n and q_n could be equal to |(1 - w3)| / 2
    qid2personalized_weight = calculate_fusion_weight(qid2std)
    #for key, value in qid2std.items():
    #    print(key, value)
    
    return qid2personalized_weight
    
# input是根据非personalized reformulated query检索得到的top-k output file
#calculate_std_top_k_list("test_case.txt")