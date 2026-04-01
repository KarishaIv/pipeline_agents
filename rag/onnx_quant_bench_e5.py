from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Iterable

import numpy as np


E5_QUERY_PREFIX = "query: "


def _cache_dir(model_name: str) -> Path:
    root = Path(__file__).resolve().parent.parent
    return root / ".cache" / "onnx_bench" / model_name.replace("/", "__")


def _export_onnx(model_name: str, out_dir: Path) -> Path:
    from optimum.onnxruntime import ORTModelForFeatureExtraction
    from transformers import AutoTokenizer

    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "model.onnx"
    if onnx_path.exists():
        return onnx_path

    print(f"[export] {model_name} -> {onnx_path}")
    model = ORTModelForFeatureExtraction.from_pretrained(model_name, export=True)
    model.save_pretrained(str(out_dir))
    AutoTokenizer.from_pretrained(model_name).save_pretrained(str(out_dir))
    return onnx_path


def _quantize_int8(onnx_fp32: Path, onnx_int8: Path) -> Path:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    if onnx_int8.exists():
        return onnx_int8

    print(f"[quant] dynamic INT8 -> {onnx_int8.name}")
    quantize_dynamic(
        model_input=str(onnx_fp32),
        model_output=str(onnx_int8),
        weight_type=QuantType.QInt8,
    )
    return onnx_int8


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


def _make_texts(n: int) -> list[str]:
    base = [
        "ключевая ставка инфляция прогноз",
        "льготная ипотека ставки новостройки",
        "курс доллара евро рубля прогноз",
        "дивиденды сбер и другие банки",
        "бюджет расходы доходы нефть газ",
        "индекс мосбиржи падение рост причины",
    ]
    out = []
    i = 0
    while len(out) < n:
        out.append(base[i % len(base)])
        i += 1
    return out


def _bench_torch(model_name: str, batch: int, iters: int, warmup: int, max_length: int) -> float:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    texts = [E5_QUERY_PREFIX + t for t in _make_texts(batch)]
    enc = tok(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")

    def run_once():
        with torch.inference_mode():
            out = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            torch.nn.functional.normalize(pooled, p=2, dim=1, out=pooled)

    for _ in range(warmup):
        run_once()

    t0 = time.perf_counter()
    for _ in range(iters):
        run_once()
    return (time.perf_counter() - t0) * 1000.0


def _bench_ort(onnx_path: Path, model_name: str, batch: int, iters: int, warmup: int, max_length: int) -> float:
    import onnxruntime as ort
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    texts = [E5_QUERY_PREFIX + t for t in _make_texts(batch)]
    enc = tok(texts, padding=True, truncation=True, max_length=max_length, return_tensors="np")
    ort_inputs = {
        "input_ids": enc["input_ids"].astype(np.int64),
        "attention_mask": enc["attention_mask"].astype(np.int64),
    }
    if "token_type_ids" in enc:
        ort_inputs["token_type_ids"] = enc["token_type_ids"].astype(np.int64)

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"])

    def run_once():
        outputs = sess.run(None, ort_inputs)
        last_hidden = outputs[0]
        pooled = _mean_pool(last_hidden, enc["attention_mask"])
        _ = _l2_normalize(pooled)

    for _ in range(warmup):
        run_once()

    t0 = time.perf_counter()
    for _ in range(iters):
        run_once()
    return (time.perf_counter() - t0) * 1000.0


def _fmt(ms: float, batch: int, iters: int) -> str:
    per_call = ms / iters
    per_item = per_call / batch
    return f"{ms:9.1f} ms total | {per_call:7.1f} ms/call | {per_item:6.2f} ms/item"


def main(argv: Iterable[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="PyTorch vs ONNX vs ONNX INT8 benchmark for multilingual E5")
    ap.add_argument("--model", default="intfloat/multilingual-e5-large")
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32, 64])
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args(argv)

    model_name = args.model
    cache = _cache_dir(model_name)
    onnx_fp32 = _export_onnx(model_name, cache)
    onnx_int8 = _quantize_int8(onnx_fp32, cache / "model.int8.onnx")

    print("\n== Bench settings ==")
    print(f"model      : {model_name}")
    print(f"max_length : {args.max_length}")
    print(f"iters      : {args.iters} (warmup {args.warmup})")
    print(f"batches    : {args.batches}")
    print("")

    for b in args.batches:
        print(f"--- batch={b} ---")
        ms_torch = _bench_torch(model_name, b, args.iters, args.warmup, args.max_length)
        ms_ort = _bench_ort(onnx_fp32, model_name, b, args.iters, args.warmup, args.max_length)
        ms_int8 = _bench_ort(onnx_int8, model_name, b, args.iters, args.warmup, args.max_length)
        print(f"PyTorch     : {_fmt(ms_torch, b, args.iters)}")
        print(f"ONNX FP32   : {_fmt(ms_ort, b, args.iters)}")
        print(f"ONNX INT8   : {_fmt(ms_int8, b, args.iters)}")
        print(f"speedup FP32 vs torch: {ms_torch/ms_ort:.2f}x | INT8 vs torch: {ms_torch/ms_int8:.2f}x")
        print("")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()

