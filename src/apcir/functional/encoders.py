"""BEIR-evaluation dense encoders, split out of functional/llm.py (which mixed generative
LLM clients with these retrieval encoders). Each class keeps its own encode loop verbatim
(embedding byte-identity preserved); only the home changed.
"""
from typing import List, Dict, Optional, Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, RobertaConfig, RobertaTokenizer

from apcir.models import ANCE


class BeirConvdrEncoder:
    def __init__(
        self,
        model=None,
        tokenizer=None,
        device=None,
        max_length_query=512,
        max_length_doc=512
    ):
        """
        
        Args:
            model: convdr BiEncoder instance
            tokenizer: corresponding tokenizer
            device: torch.device
            max_length_query: maximum length for queries
            max_length_doc: maximum length for documents
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device("cpu")
        self.max_length_query = max_length_query
        self.max_length_doc = max_length_doc
        
        if self.model is not None:
            self.model.eval()
            self.model.to(self.device)

    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:
        embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(queries), batch_size), desc="Encoding Queries"):
                batch = queries[i:i + batch_size]
                encoded = self.tokenizer(
                    batch,
                    max_length=self.max_length_query,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)
                
                q_embs = self.model.query_emb(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"]
                )
                embeddings.append(q_embs.cpu().numpy())
        
        return np.concatenate(embeddings, axis=0)

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs) -> np.ndarray:
        texts = []
        for doc in corpus:
            title = doc.get("title", "").strip()
            text = doc.get("text", "").strip()
            if title and text:
                texts.append(f"{title} {text}")
            elif title:
                texts.append(title)
            else:
                texts.append(text)
        
        embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Encoding Corpus"):
                batch = texts[i:i + batch_size]
                encoded = self.tokenizer(
                    batch,
                    max_length=self.max_length_doc,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)
                
                d_embs = self.model.body_emb(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"]
                )
                embeddings.append(d_embs.cpu().numpy())
        
        return np.concatenate(embeddings, axis=0)


class BeirCLSEncoder:
    def __init__(
        self,
        model_path: str,
        device: Optional[torch.device] = None,
        max_length_query: int = 512,
        max_length_doc: int = 512,
    ):
        """
        Encoder using Bert-based encoder with [CLS] token embedding (not mean pooling).

        Args:
            model_path (str): Hugging Face model repo name or local path (e.g., "roberta-base")
            device (torch.device, optional): Device to run the model on. Defaults to CPU if not provided.
            max_length_query (int): Max length for query encoding.
            max_length_doc (int): Max length for document encoding.
        """
        self.device = device or torch.device("cpu")
        self.max_length_query = max_length_query
        self.max_length_doc = max_length_doc

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.model = AutoModel.from_pretrained(model_path)

        # Move to device and set eval mode
        self.model.eval()
        self.model.to(self.device)

    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:
        embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(queries), batch_size), desc="Encoding Queries"):
                batch = queries[i:i + batch_size]
                encoded = self.tokenizer(
                    batch,
                    max_length=self.max_length_query,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)

                # Forward pass
                outputs = self.model(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"]
                )
                #  Use [CLS] token embedding (first token)
                cls_embeddings = outputs.last_hidden_state[:, 0, :]  # (batch_size, hidden_dim)
                # L2 normalize
                cls_embeddings = F.normalize(cls_embeddings, p=2, dim=-1)
                embeddings.append(cls_embeddings.cpu().numpy())

        return np.concatenate(embeddings, axis=0)

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs) -> np.ndarray:
        # Preprocess: combine title and text like BEIR expects
        texts = []
        for doc in corpus:
            title = doc.get("title", "").strip()
            text = doc.get("text", "").strip()
            if title and text:
                texts.append(f"{title} {text}")
            elif title:
                texts.append(title)
            else:
                texts.append(text)

        embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Encoding Corpus"):
                batch = texts[i:i + batch_size]
                encoded = self.tokenizer(
                    batch,
                    max_length=self.max_length_doc,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)

                outputs = self.model(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"]
                )
                #  Use [CLS] token embedding
                cls_embeddings = outputs.last_hidden_state[:, 0, :]
                # L2 normalize
                cls_embeddings = F.normalize(cls_embeddings, p=2, dim=-1)
                embeddings.append(cls_embeddings.cpu().numpy())

        return np.concatenate(embeddings, axis=0)


class BeirMPoolingEncoder:
    def __init__(
        self,
        model_path: str,
        device: Optional[torch.device] = None,
        max_length_query: int = 512,
        max_length_doc: int = 512,
    ):
        """
        Encoder using Bert-based encoder with mean pooling (not [CLS]).

        Args:
            model_path (str): Hugging Face model repo name or local path (e.g., "roberta-base")
            device (torch.device, optional): Device to run the model on. Defaults to CPU if not provided.
            max_length_query (int): Max length for query encoding.
            max_length_doc (int): Max length for document encoding.
        """
        self.device = device or torch.device("cpu")
        self.max_length_query = max_length_query
        self.max_length_doc = max_length_doc

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.model = AutoModel.from_pretrained(model_path)

        # Move to device and set eval mode
        self.model.eval()
        self.model.to(self.device)

    def _mean_pooling(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply mean pooling with attention mask.
        
        Args:
            token_embeddings: (batch_size, seq_len, hidden_dim)
            attention_mask: (batch_size, seq_len)
            
        Returns:
            embeddings: (batch_size, hidden_dim)
        """
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:
        embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(queries), batch_size), desc="Encoding Queries"):
                batch = queries[i:i + batch_size]
                encoded = self.tokenizer(
                    batch,
                    max_length=self.max_length_query,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)

                outputs = self.model(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"]
                )
                # Mean pooling over non-padding tokens
                mean_embeddings = self._mean_pooling(outputs.last_hidden_state, encoded["attention_mask"])
                # L2 normalize (required by BEIR for dot-product retrieval)
                mean_embeddings = F.normalize(mean_embeddings, p=2, dim=-1)
                embeddings.append(mean_embeddings.cpu().numpy())

        return np.concatenate(embeddings, axis=0)

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs) -> np.ndarray:
        # Preprocess: combine title and text like BEIR expects
        texts = []
        for doc in corpus:
            title = doc.get("title", "").strip()
            text = doc.get("text", "").strip()
            if title and text:
                texts.append(f"{title} {text}")
            elif title:
                texts.append(title)
            else:
                texts.append(text)

        embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Encoding Corpus"):
                batch = texts[i:i + batch_size]
                encoded = self.tokenizer(
                    batch,
                    max_length=self.max_length_doc,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)

                outputs = self.model(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"]
                )
                # Mean pooling
                mean_embeddings = self._mean_pooling(outputs.last_hidden_state, encoded["attention_mask"])
                # L2 normalize
                mean_embeddings = F.normalize(mean_embeddings, p=2, dim=-1)
                embeddings.append(mean_embeddings.cpu().numpy())

        return np.concatenate(embeddings, axis=0)


class BEIRQwenEncoder:
    def __init__(
        self,
        model_path: str,
        device: Optional[torch.device] = None,
        cache_dir: Optional[str] = None,
        max_length_query: int = 512,
        max_length_doc: int = 512,
        query_instruction: str = "Given a web search query, retrieve relevant passages that answer the query",
        attn_implementation: Optional[str] = None,
    ):
        self.device = device or torch.device("cpu")
        self.cache_dir = cache_dir
        self.max_length_query = max_length_query
        self.max_length_doc = max_length_doc
        self.query_instruction = query_instruction

        tokenizer_kwargs = {
            "padding_side": "left",
            "trust_remote_code": True,
        }
        if cache_dir is not None:
            tokenizer_kwargs["cache_dir"] = cache_dir

        model_kwargs = {
            "trust_remote_code": True,
        }
        if cache_dir is not None:
            model_kwargs["cache_dir"] = cache_dir
        if self.device.type == "cuda":
            model_kwargs["torch_dtype"] = torch.bfloat16
            if attn_implementation is not None:
                model_kwargs["attn_implementation"] = attn_implementation

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModel.from_pretrained(model_path, **model_kwargs)
        self.model.eval()
        self.model.to(self.device)

    @staticmethod
    def _last_token_pool(
        last_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        left_padding = bool(
            torch.all(attention_mask[:, -1] == 1).item()
        )
        if left_padding:
            return last_hidden_states[:, -1]

        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths,
        ]

    def _format_query(self, query: str) -> str:
        return f"Instruct: {self.query_instruction}\nQuery:{query}"

    def _encode_texts(
        self,
        texts: List[str],
        batch_size: int,
        max_length: int,
        desc: str,
    ) -> np.ndarray:
        embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc=desc):
                batch = texts[i : i + batch_size]
                encoded = self.tokenizer(
                    batch,
                    max_length=max_length,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.device)

                outputs = self.model(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                )
                pooled = self._last_token_pool(
                    outputs.last_hidden_state,
                    encoded["attention_mask"],
                )
                pooled = F.normalize(pooled, p=2, dim=-1)
                embeddings.append(pooled.float().cpu().numpy())
                del outputs
                del pooled
                del encoded

        return np.concatenate(embeddings, axis=0)

    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:
        formatted_queries = [self._format_query(query) for query in queries]
        return self._encode_texts(
            texts=formatted_queries,
            batch_size=batch_size,
            max_length=self.max_length_query,
            desc="Encoding Queries (Qwen3)",
        )

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs) -> np.ndarray:
        texts = []
        for doc in corpus:
            title = doc.get("title", "").strip()
            text = doc.get("text", "").strip()
            if title and text:
                texts.append(f"{title} {text}")
            elif title:
                texts.append(title)
            else:
                texts.append(text)

        return self._encode_texts(
            texts=texts,
            batch_size=batch_size,
            max_length=self.max_length_doc,
            desc="Encoding Corpus (Qwen3)",
        )


# ---- assume your ANCE class is already defined exactly as you pasted ----
# class ANCE(RobertaForSequenceClassification): ...
# def load_model(...): ...

class BeirANCEEncoder:
    def __init__(
        self,
        model_path: str,
        device: Optional[torch.device] = None,
        max_length_query: int = 512,
        max_length_doc: int = 512,
    ):
        """
        Encoder using ANCE-style encoder producing 768-d dense embeddings
        (CLS token by default, or mean pooling if ANCE.use_mean=True).

        Args:
            model_path (str): Hugging Face model repo name or local path (ANCE checkpoint dir)
            device (torch.device, optional): Device to run the model on. Defaults to CPU if not provided.
            max_length_query (int): Max length for query encoding.
            max_length_doc (int): Max length for document encoding.
        """
        self.device = device or torch.device("cpu")
        self.max_length_query = max_length_query
        self.max_length_doc = max_length_doc

        # Load ANCE tokenizer + model (same as your load_model behavior)
        self.tokenizer = RobertaTokenizer.from_pretrained(model_path, do_lower_case=True)

        config = RobertaConfig.from_pretrained(
            model_path,
            finetuning_task="MSMarco",
        )
        self.model = ANCE.from_pretrained(model_path, config=config)

        # Move to device and set eval mode
        self.model.eval()
        self.model.to(self.device)

    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:
        embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(queries), batch_size), desc="Encoding Queries"):
                batch = queries[i : i + batch_size]
                encoded = self.tokenizer(
                    batch,
                    max_length=self.max_length_query,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}

                # ANCE forward returns (batch_size, 768)
                dense = self.model(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                )

                # L2 normalize
                dense = F.normalize(dense, p=2, dim=-1)
                embeddings.append(dense.cpu().numpy())

        return np.concatenate(embeddings, axis=0)

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs) -> np.ndarray:
        # Preprocess: combine title and text like BEIR expects
        texts = []
        for doc in corpus:
            title = doc.get("title", "").strip()
            text = doc.get("text", "").strip()
            if title and text:
                texts.append(f"{title} {text}")
            elif title:
                texts.append(title)
            else:
                texts.append(text)

        embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Encoding Corpus"):
                batch = texts[i : i + batch_size]
                encoded = self.tokenizer(
                    batch,
                    max_length=self.max_length_doc,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}

                dense = self.model(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                )

                dense = F.normalize(dense, p=2, dim=-1)
                embeddings.append(dense.cpu().numpy())

        return np.concatenate(embeddings, axis=0)
        


class BeirAsymmetricANCEEncoder:
    def __init__(
        self,
        query_encoder_path: str,
        passage_encoder_path: str,
        device: Optional[torch.device] = None,
        max_length_query: int = 512,
        max_length_doc: int = 512,
    ):
        """
        Asymmetric ANCE encoder:
        - one encoder for queries
        - one encoder for passages (documents)

        Args:
            query_encoder_path (str): HF repo or local path for query encoder
            passage_encoder_path (str): HF repo or local path for passage encoder
            device (torch.device, optional): torch device
            max_length_query (int): max length for query encoding
            max_length_doc (int): max length for document encoding
        """
        self.device = device or torch.device("cpu")
        self.max_length_query = max_length_query
        self.max_length_doc = max_length_doc

        # -------------------------
        # Query encoder
        # -------------------------
        self.query_tokenizer = RobertaTokenizer.from_pretrained(
            query_encoder_path, do_lower_case=True
        )
        query_config = RobertaConfig.from_pretrained(
            query_encoder_path,
            finetuning_task="MSMarco",
        )
        self.query_encoder = ANCE.from_pretrained(
            query_encoder_path, config=query_config
        )
        self.query_encoder.eval()
        self.query_encoder.to(self.device)

        # -------------------------
        # Passage encoder
        # -------------------------
        self.passage_tokenizer = RobertaTokenizer.from_pretrained(
            passage_encoder_path, do_lower_case=True
        )
        passage_config = RobertaConfig.from_pretrained(
            passage_encoder_path,
            finetuning_task="MSMarco",
        )
        self.passage_encoder = ANCE.from_pretrained(
            passage_encoder_path, config=passage_config
        )
        self.passage_encoder.eval()
        self.passage_encoder.to(self.device)

    def encode_queries(
        self,
        queries: List[str],
        batch_size: int,
        **kwargs,
    ) -> np.ndarray:
        """
        Encode queries using the query encoder.
        """
        embeddings = []
        with torch.no_grad():
            for i in tqdm(
                range(0, len(queries), batch_size),
                desc="Encoding Queries (Asymmetric)",
            ):
                batch = queries[i : i + batch_size]
                encoded = self.query_tokenizer(
                    batch,
                    max_length=self.max_length_query,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}

                dense = self.query_encoder(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                )

                dense = F.normalize(dense, p=2, dim=-1)
                embeddings.append(dense.cpu().numpy())

        return np.concatenate(embeddings, axis=0)

    def encode_corpus(
        self,
        corpus: List[Dict[str, str]],
        batch_size: int,
        **kwargs,
    ) -> np.ndarray:
        """
        Encode documents using the passage encoder.
        """
        # BEIR-style preprocessing: title + text
        texts = []
        for doc in corpus:
            title = doc.get("title", "").strip()
            text = doc.get("text", "").strip()
            if title and text:
                texts.append(f"{title} {text}")
            elif title:
                texts.append(title)
            else:
                texts.append(text)

        embeddings = []
        with torch.no_grad():
            for i in tqdm(
                range(0, len(texts), batch_size),
                desc="Encoding Corpus (Asymmetric)",
            ):
                batch = texts[i : i + batch_size]
                encoded = self.passage_tokenizer(
                    batch,
                    max_length=self.max_length_doc,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}

                dense = self.passage_encoder(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                )

                dense = F.normalize(dense, p=2, dim=-1)
                embeddings.append(dense.cpu().numpy())

        return np.concatenate(embeddings, axis=0)
