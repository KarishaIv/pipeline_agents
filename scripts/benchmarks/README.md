# Benchmark Suite (MVP, protocol-compatible)

## 1) Prepare external datasets

```bash
/Users/timofeysukhoparov/Documents/cursach/.venv/bin/python scripts/benchmarks/prepare_external_data.py
```

If HF datasets-server returns 401/403, use datasets-library backend:
```bash
/Users/timofeysukhoparov/Documents/cursach/.venv/bin/pip install datasets
/Users/timofeysukhoparov/Documents/cursach/.venv/bin/python scripts/benchmarks/prepare_external_data.py \
  --fetch-backend datasets-lib \
  --trait-dataset mirlab/TRAIT \
  --trait-split all \
  --emobench-config emotional_understanding \
  --emobench-split train
```

Optional (for gated datasets):
```bash
export HF_TOKEN='hf_...'
```

This downloads and normalizes:
- TRAIT -> `data/benchmark_external/trait/trait_train.jsonl`
- EmoBench -> `data/benchmark_external/emobench/emobench_test.jsonl`

## 2) Run full suite

```bash
/Users/timofeysukhoparov/Documents/cursach/.venv/bin/python scripts/benchmarks/run_suite.py \
  --locale both \
  --benchmarks trait,personallm,personagym,emobench \
  --persona-sample 15 \
  --concurrency 3 \
  --max-calls 2500 \
  --max-runtime-min 180
```

Outputs go to:
- `outputs/benchmarks/suite/<timestamp>/run_config.json`
- `outputs/benchmarks/suite/<timestamp>/suite_summary.json`
- `outputs/benchmarks/suite/<timestamp>/suite_report.md`

## 3) Run single benchmark

### TRAIT
```bash
/Users/timofeysukhoparov/Documents/cursach/.venv/bin/python scripts/benchmarks/benchmark_trait.py \
  --locale en \
  --persona-sample 15 \
  --items-per-persona 20
```

### PersonaLLM protocol
```bash
/Users/timofeysukhoparov/Documents/cursach/.venv/bin/python scripts/benchmarks/benchmark_personallm.py \
  --locale en \
  --persona-sample 15
```

### PersonaGym protocol-compatible
```bash
/Users/timofeysukhoparov/Documents/cursach/.venv/bin/python scripts/benchmarks/benchmark_personagym.py \
  --locale en \
  --persona-sample 15 \
  --scenarios-per-persona 8
```

### EmoBench
```bash
/Users/timofeysukhoparov/Documents/cursach/.venv/bin/python scripts/benchmarks/benchmark_emobench.py \
  --locale en \
  --persona-sample 15 \
  --items-per-persona 15 \
  --prompt-variant appraisal \
  --balance-col auto \
  --parse-retries 1
```

### EmoBench prompt A/B on identical items
```bash
/Users/timofeysukhoparov/Documents/cursach/.venv/bin/python scripts/benchmarks/run_emobench_ablation.py \
  --locale en \
  --persona-sample 12 \
  --items-per-persona 15 \
  --seed 42 \
  --variants baseline,facts_first,appraisal \
  --concurrency 1 \
  --max-calls 1000 \
  --max-runtime-min 90
```

This script:
- samples one fixed item set once;
- runs all prompt variants on the same items;
- saves `summary.json` / `summary.csv` with comparable metrics.

### EmoBench error analysis after runs
```bash
/Users/timofeysukhoparov/Documents/cursach/.venv/bin/python scripts/benchmarks/analyze_emobench_errors.py \
  --predictions \
  outputs/benchmarks/budget2k/emobench_v2_seed42/predictions.csv \
  outputs/benchmarks/budget2k/emobench_v2_seed43/predictions.csv
```

## Notes
- PersonaGym and PersonaLLM scripts are protocol-compatible adapters, not official leaderboard runners.
- RU mode caches translated items in:
  - `data/benchmark_external/trait/cache_ru_items.json`
  - `data/benchmark_external/emobench/cache_ru_items.json`
- For controlled EmoBench comparisons use `--fixed-items-path` (or run via `run_emobench_ablation.py`).
