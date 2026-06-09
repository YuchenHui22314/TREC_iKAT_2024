"""Dense encoder architectures (consolidated from search/models.py + indexing/dense/models.py).

ANCE + QwenEmbedding are the search-side (live retrieval) versions; behaviorally identical to
the indexing copies. TCTColBERT comes from the indexing side. All three were duplicated; merged here.
"""
import torch
import torch.nn.functional as F
from torch import nn
from transformers import RobertaConfig, RobertaForSequenceClassification, AutoModel, BertModel


class ANCE(RobertaForSequenceClassification):
    # class Pooler:   # adapt to DPR
    #     def __init__(self, pooler_output):
    #         self.pooler_output = pooler_output

    def __init__(self, config):
        RobertaForSequenceClassification.__init__(self, config)
        self.embeddingHead = nn.Linear(config.hidden_size, 768)
        self.norm = nn.LayerNorm(768)
        self.apply(self._init_weights)
        self.use_mean = False
    
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    def query_emb(self, input_ids, attention_mask):
        outputs1 = self.roberta(input_ids=input_ids,
                                attention_mask=attention_mask)
        outputs1 = outputs1.last_hidden_state
        full_emb = self.masked_mean_or_first(outputs1, attention_mask)
        query1 = self.norm(self.embeddingHead(full_emb))
        return query1


    def doc_emb(self, input_ids, attention_mask):
        return self.query_emb(input_ids, attention_mask)
    

    def masked_mean_or_first(self, emb_all, mask):
        if self.use_mean:
            return self.masked_mean(emb_all, mask)
        else:
            return emb_all[:, 0]
    
    def masked_mean(self, t, mask):
        s = torch.sum(t * mask.unsqueeze(-1).float(), axis=1)
        d = mask.sum(axis=1, keepdim=True).float()
        return s / d
    
    def forward(self, input_ids, attention_mask, wrap_pooler=False):
        return self.query_emb(input_ids, attention_mask)


class QwenEmbedding(nn.Module):
    def __init__(self, model_path):
        super(QwenEmbedding, self).__init__()
        self.model = AutoModel.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")

    @staticmethod
    def last_token_pool(last_hidden_states, attention_mask):
        """Pool the last non-padding token. Works for left- and right-padding."""
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths
        ]

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = self.last_token_pool(outputs.last_hidden_state, attention_mask)
        # L2 normalize for cosine similarity retrieval (FAISS IndexFlatIP).
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

#############################################################
#############################################################
#############################################################
#############################################################
#############################################################
#                                                           #
#                      SPLADE                               #
#                                                           #
#############################################################
#############################################################
#############################################################
#############################################################
#############################################################
from abc import ABC
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel

"""
we provide abstraction classes from which we can easily derive representation-based models with transformers like SPLADE
with various options (one or two encoders, freezing one encoder etc.) 
"""


class TCTColBERT(nn.Module):
    def __init__(self, model_path) -> None:
        super(TCTColBERT, self).__init__()
        self.model = BertModel.from_pretrained(model_path)

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        if "cur_utt_end_position" in kwargs:
            device = outputs.device
            cur_utt_end_positions = kwargs["cur_utt_end_positions"]
            output_mask = torch.zeros(attention_mask.size()).to(device)
            mask_row = []
            mask_col = []
            for i in range(len(cur_utt_end_positions)):
                mask_row += [i] * (cur_utt_end_positions[i] - 3)
                mask_col += list(range(4, cur_utt_end_positions[i] + 1))
                
            mask_index = (
                    torch.tensor(mask_row).long().to(device),
                    torch.tensor(mask_col).long().to(device)
                )
            values = torch.ones(len(mask_row)).to(device)
            output_mask = output_mask.index_put(mask_index, values)
        else:
            output_mask = attention_mask
            output_mask[:, :4] = 0 # filter the first 4 tokens: [CLS] "[" "Q/D" "]"
            
        # sum / length
        sum_outputs = torch.sum(outputs * output_mask.unsqueeze(-1), dim = -2) 
        real_seq_length = torch.sum(output_mask, dim = 1).view(-1, 1)

        return sum_outputs / real_seq_length




# Qwen3-Embedding model
