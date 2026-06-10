"""Model factory (the indexing-side richer load_model; the search-side ANCE-only one was dead)."""
from transformers import (RobertaConfig, RobertaTokenizer, AutoTokenizer,
                          DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer,
                          DPRContextEncoder, DPRQuestionEncoder)
from apcir.models.architectures import ANCE, TCTColBERT, QwenEmbedding


def load_model(model_type, query_or_doc, model_path):
    assert query_or_doc in ("query", "doc")
    if model_type.lower() == "ance":
        config = RobertaConfig.from_pretrained(
            model_path,
            finetuning_task="MSMarco",
        )
        tokenizer = RobertaTokenizer.from_pretrained(
            model_path,
            do_lower_case=True
        )
        model = ANCE.from_pretrained(model_path, config=config)
    elif model_type.lower() == "dpr-nq":
        if query_or_doc == "query":
            tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(model_path)
            model = DPRQuestionEncoder.from_pretrained(model_path)
        else:
            tokenizer = DPRContextEncoderTokenizer.from_pretrained(model_path)
            model = DPRContextEncoder.from_pretrained(model_path)
    elif model_type.lower() == "tctcolbert":
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = TCTColBERT(model_path)
    elif model_type.lower() == "qwen-embedding":
        # padding_side='left' is required for correct last-token pooling
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
        model = QwenEmbedding(model_path)
    else:
        raise ValueError
    
    # tokenizer.add_tokens(["<CUR_Q>", "<CTX>", "<CTX_R>", "<CTX_Q>"])
    # model.resize_token_embeddings(len(tokenizer))

    return tokenizer, model
