import regex as re
import string
from collections import Counter

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def bleu4(pred_text, gold_text):
    pred_tokens = pred_text.split()
    gold_tokens = gold_text.split()
    
    precision_scores = []
    for n in range(1, 5):  #n-grams
        pred_ngrams = Counter([' '.join(pred_tokens[i:i+n]) for i in range(len(pred_tokens)-n+1)])
        gold_ngrams = Counter([' '.join(gold_tokens[i:i+n]) for i in range(len(gold_tokens)-n+1)])
        
        common_ngrams = pred_ngrams & gold_ngrams
        num_common = sum(common_ngrams.values())
        
        if num_common == 0:
            precision_scores.append(0)
        else:
            precision_scores.append(num_common / sum(pred_ngrams.values()))
    
    precision = sum(precision_scores) / 4  #average
    
    if len(pred_tokens) < len(gold_tokens):
        brevity_penalty = len(pred_tokens) / len(gold_tokens)
    else:
        brevity_penalty = 1.0
    
    bleu4_score = brevity_penalty * precision
    return bleu4_score


def rouge(pred_text, gold_text):
    pred_tokens = pred_text.split()
    gold_tokens = gold_text.split()
    
    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)
    
    common_tokens = pred_counter & gold_counter
    num_common = sum(common_tokens.values())
    
    if num_common == 0:
        return 0
    
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    
    rouge_l_score = 2 * precision * recall / (precision + recall)
    return rouge_l_score
