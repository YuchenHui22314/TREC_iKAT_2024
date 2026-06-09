"""Dataset / padding helpers used by the dense & splade search query encoders.

DRY refactor: the old training-dataset classes (T5FT_context, T5RewriterIRDataset_*,
Retrieval_qrecc, ConvGQR_Retrieval, Search_q_Retrieval, keywords_Retrieval, QR_qrecc,
QR_cast [defined twice], QR_topiocqa, llama_prompt_topiocqa) had 0 references anywhere
(incl. notebooks) and were removed. Only the three symbols actually imported by
dense_search.py / splade_search.py are kept.
"""
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

def padding_seq_to_same_length(input_ids, max_pad_length, pad_token = 0):
    padding_length = max_pad_length - len(input_ids)
    padding_ids = [pad_token] * padding_length
    attention_mask = []

    if padding_length <= 0:
        attention_mask = [1] * max_pad_length
        input_ids = input_ids[:max_pad_length]
    else:
        attention_mask = [1] * len(input_ids) + [0] * padding_length
        input_ids = input_ids + padding_ids
            
    assert len(input_ids) == max_pad_length
    assert len(attention_mask) == max_pad_length
  
    return input_ids, attention_mask

def pad_and_mask(seqs, pad_token_id=0):
    """
    Shared helper to pad a list of variable-length sequences to the maximum length
    in the batch and build attention masks.
    """
    # create a tenror of length max_length, add it to the seqs, do the pad, then remove this extra row
    # # this is a temporary fix to the problem of experience replay batch size mismatch. TODO: find a remedy for this.
    # seqs.append([pad_token_id] * max_length)
    # Convert to tensors

    tensors = [torch.tensor(s, dtype=torch.long) for s in seqs]

    # Pad to batch-longest
    padded = pad_sequence(
        tensors,
        batch_first=True,
        padding_value=pad_token_id
    )

    # padded = padded[:-1, :]  # remove the extra row

    # Attention mask
    mask = (padded != pad_token_id).long()
    return padded, mask


class Retrieval_trec(Dataset):
    def __init__(
        self, 
        tokenizer, 
        retrieval_query_list, 
        qid_list_string,
        max_length = 512
        ):

        self.queries = retrieval_query_list
        self.qids = qid_list_string 
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, index):
        qid = self.qids[index]
        query = self.queries[index]
        
        encoded_query = self.tokenizer(
            query, 
            add_special_tokens = True, 
            padding = "max_length",
            truncation = True,
            return_tensors = "pt",
            max_length = self.max_length)
        
        return {
            "qid": qid,
            "query": query,
            "input_ids": encoded_query["input_ids"].squeeze(),
            "attention_mask": encoded_query["attention_mask"].squeeze(),
        }
            
    
    @staticmethod
    def get_collate_fn(pad_token_id = 0 ):
        
        def collate_fn(batch: list):
            # padding
            input_ids = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(item["input_ids"]) for item in batch],
                batch_first = True,
                padding_value = pad_token_id
            )
            attention_mask = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(item["attention_mask"]) for item in batch],
                batch_first = True,
                padding_value = pad_token_id
            )
            return {
                "qid": [item["qid"] for item in batch],
                "query": [item["query"] for item in batch],
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }

        return collate_fn
