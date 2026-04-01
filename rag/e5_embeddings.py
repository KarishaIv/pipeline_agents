"""
Эмбеддинги E5 (intfloat/multilingual-e5-*): ONNX Runtime (по умолчанию, если есть экспорт)
или sentence-transformers (fallback)

Экспорт ONNX в кэш:
  python -m rag.onnx_quant_bench_e5 --model intfloat/multilingual-e5-large --iters 1 --batches 1

Переменные окружения:
  RAG_EMBEDDING_MODEL — HF id модели (по умолчанию intfloat/multilingual-e5-large)
  RAG_EMBEDDING_BACKEND — auto | onnx | sentence_transformers
      auto: ONNX если в каталоге есть model.onnx / model.int8.onnx, иначе ST
  RAG_ONNX_MODEL_DIR — каталог с model.onnx, model.int8.onnx и tokenizer (иначе .cache/onnx_bench/...)
  RAG_ONNX_USE_INT8 — 1/0, брать model.int8.onnx если есть (по умолчанию 1)
  RAG_EMBEDDING_MAX_LENGTH — max_length токенизации (по умолчанию 512)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import numpy as np

DEFAULT_MODEL = "intfloat/multilingual-e5-large"
E5_QUERY_PREFIX = "query: "
E5_DOC_PREFIX = "passage: "

_model = None 
_ort_session = None
_tokenizer = None
_vector_size: Optional[int] = None
_backend: Optional[str] = None
_model_name_bound: Optional[str] = None


def _model_name() -> str:
    return os.getenv("RAG_EMBEDDING_MODEL", DEFAULT_MODEL)


def _onnx_cache_dir(model_name: str) -> Path:
    root = Path(__file__).resolve().parent.parent
    return root / ".cache" / "onnx_bench" / model_name.replace("/", "__")


def _onnx_model_dir() -> Path:
    override = os.getenv("RAG_ONNX_MODEL_DIR")
    if override:
        return Path(override)
    return _onnx_cache_dir(_model_name())


def _pick_onnx_file(model_dir: Path) -> Optional[Path]:
    use_int8 = os.getenv("RAG_ONNX_USE_INT8", "1") != "0"
    int8 = model_dir / "model.int8.onnx"
    fp32 = model_dir / "model.onnx"
    if use_int8 and int8.is_file():
        return int8
    if fp32.is_file():
        return fp32
    return None


def _embedding_backend_choice() -> str:
    raw = os.getenv("RAG_EMBEDDING_BACKEND", "auto").strip().lower()
    if raw in ("", "auto"):
        return "auto"
    if raw in ("onnx", "ort"):
        return "onnx"
    if raw in ("sentence_transformers", "st", "torch", "pytorch"):
        return "sentence_transformers"
    return raw


def _mean_pool(last_hidden: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    mask = attention_mask.astype(np.float32)
    mask = np.expand_dims(mask, axis=-1)
    summed = np.sum(last_hidden * mask, axis=1)
    counts = np.clip(np.sum(mask, axis=1), 1e-9, None)
    return summed / counts


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm = np.clip(norm, 1e-12, None)
    return x / norm


def _max_length() -> int:
    return int(os.getenv("RAG_EMBEDDING_MAX_LENGTH", "512"))


def _ensure_sentence_transformer():
    global _model, _vector_size, _backend, _model_name_bound
    from sentence_transformers import SentenceTransformer

    mn = _model_name()
    if _model is not None and _model_name_bound == mn:
        return
    _model = SentenceTransformer(mn)
    _vector_size = int(_model.get_sentence_embedding_dimension())
    _backend = "sentence_transformers"
    _model_name_bound = mn


def _ensure_onnx():
    global _ort_session, _tokenizer, _vector_size, _backend, _model_name_bound
    import onnxruntime as ort
    from transformers import AutoTokenizer

    mn = _model_name()
    if _ort_session is not None and _model_name_bound == mn:
        return

    model_dir = _onnx_model_dir()
    onnx_path = _pick_onnx_file(model_dir)
    if onnx_path is None:
        raise RuntimeError(
            f"ONNX не найден в {model_dir}. Экспорт: "
            f"python -m rag.onnx_quant_bench_e5 --model {mn} --iters 1 --batches 1"
        )

    _tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    _ort_session = ort.InferenceSession(
        str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"]
    )
    enc = _tokenizer(
        [E5_QUERY_PREFIX + "x"],
        padding=True,
        truncation=True,
        max_length=16,
        return_tensors="np",
    )
    ort_inputs = {
        "input_ids": enc["input_ids"].astype(np.int64),
        "attention_mask": enc["attention_mask"].astype(np.int64),
    }
    if "token_type_ids" in enc:
        ort_inputs["token_type_ids"] = enc["token_type_ids"].astype(np.int64)
    outputs = _ort_session.run(None, ort_inputs)
    last_hidden = outputs[0]
    _vector_size = int(last_hidden.shape[-1])
    _backend = "onnx"
    _model_name_bound = mn


def _ensure_model():
    global _backend
    choice = _embedding_backend_choice()
    if choice == "sentence_transformers":
        _ensure_sentence_transformer()
        return
    if choice == "onnx":
        _ensure_onnx()
        return
    # auto
    if _pick_onnx_file(_onnx_model_dir()) is not None:
        _ensure_onnx()
    else:
        _ensure_sentence_transformer()


def get_vector_size() -> int:
    global _vector_size
    if _vector_size is None:
        _ensure_model()
    assert _vector_size is not None
    return _vector_size


def _encode_onnx(texts: List[str], *, query: bool) -> np.ndarray:
    assert _ort_session is not None and _tokenizer is not None
    prefix = E5_QUERY_PREFIX if query else E5_DOC_PREFIX
    prefixed = [prefix + t for t in texts]
    ml = _max_length()
    enc = _tokenizer(
        prefixed,
        padding=True,
        truncation=True,
        max_length=ml,
        return_tensors="np",
    )
    ort_inputs = {
        "input_ids": enc["input_ids"].astype(np.int64),
        "attention_mask": enc["attention_mask"].astype(np.int64),
    }
    if "token_type_ids" in enc:
        ort_inputs["token_type_ids"] = enc["token_type_ids"].astype(np.int64)
    outputs = _ort_session.run(None, ort_inputs)
    last_hidden = outputs[0]
    pooled = _mean_pool(last_hidden, enc["attention_mask"])
    return _l2_normalize(pooled)


def get_embeddings(texts: List[str], *, query: bool) -> List[List[float]]:
    _ensure_model()
    if _backend == "onnx":
        emb = _encode_onnx(texts, query=query)
        return np.asarray(emb, dtype=np.float32).tolist()
    assert _model is not None
    prefix = E5_QUERY_PREFIX if query else E5_DOC_PREFIX
    prefixed = [prefix + t for t in texts]
    emb = _model.encode(
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
