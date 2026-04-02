# Compare embeddings FP32 vs INT8 

- model: `intfloat/multilingual-e5-large`
- n: 100, max_length: 256
- fp32 time: 185414.5 ms total
- int8 time: 72382.0 ms total

## Cosine similarity (FP32 vs INT8)
- mean=0.98788, p50=0.98798, p05=0.98492, p01=0.98245, min=0.98123

JSON: `outputs/compare_embeddings_fp32_int8.json`
