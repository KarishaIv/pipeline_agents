"""
Эмбеддинги E5 (intfloat/multilingual-e5-*) через sentence-transformers
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "intfloat/multilingual-e5-base"
E5_QUERY_PREFIX = "query: "
E5_DOC_PREFIX = "passage: "

_model: Optional[SentenceTransformer] = None
_vector_size: Optional[int] = None


def _model_name() -> str:
    return os.getenv("RAG_EMBEDDING_MODEL", DEFAULT_MODEL)


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_model_name())
    return _model


def get_vector_size() -> int:
    global _vector_size
    if _vector_size is None:
        _vector_size = int(_get_model().get_sentence_embedding_dimension())
    return _vector_size


def get_embeddings(texts: List[str], *, query: bool) -> List[List[float]]:
    model = _get_model()
    prefix = E5_QUERY_PREFIX if query else E5_DOC_PREFIX
    prefixed = [prefix + t for t in texts]
    emb = model.encode(
        prefixed,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return np.asarray(emb, dtype=np.float32).tolist()


def get_embedding(text: str, *, query: bool = True) -> List[float]:
    return get_embeddings([text], query=query)[0]


def get_document_embeddings(texts: List[str]) -> List[List[float]]:
    return get_embeddings(texts, query=False)


def get_query_embedding(text: str) -> List[float]:
    return get_embedding(text, query=True)
