# Credit reasoning research bundle

This folder preserves the experimental credit-mode work as an isolated research package. It is intentionally not wired into the current main survey pipeline.

## What this branch studied

The credit branch evaluated the last decision layer of the original credit-mode runtime. The main research question was whether a direct final decision can be replaced by a compact internal debate while keeping outputs stable and making the final explanation more interpretable.

The research branch contains three ideas:

- direct baseline: one final LLM decision call;
- compact debate: emotional and rational voices followed by code-level aggregation and synthesis;
- current-news variant: the same compact debate with external news signals used as contextual enrichment.

## Frozen decision packets

The benchmark uses frozen decision packets instead of rerunning the whole upstream simulation. This makes the comparison fair because every mode receives the same profile, goal, session history, push information, reaction, and baseline context.

Packet artifacts:

- `decision_packets/decision_packets.jsonl`
- `decision_packets/decision_packets_summary.json`

The preserved sample contains 10 benchmark packets sampled from 1000 valid scanned runs:

- 4 fathers;
- 6 mothers;
- mostly informational or exploratory goals rather than explicit credit-application goals.

This explains why the compact modes usually keep decision rate at zero: the branch is intentionally conservative when the user is browsing or exploring rather than applying for credit.

## Preserved results

Canonical results are stored under `results/`. Each run contains:

- `metrics.json`;
- `predictions.csv`;
- `judge_results.csv` when judge evaluation was run.

The compact summary tables are:

- `metrics_summary.csv`;
- `metrics_summary.json`.

## Reconstructed code

The `code/` folder contains a runnable reconstruction of the credit research implementation:

- `code/credit_schemas.py` defines the packet, voice, decision, and news-signal contracts;
- `code/credit_reasoning_agent.py` restores direct and compact-debate decision modes;
- `code/credit_news_adapter.py` maps general news snapshots into credit-specific signals;
- `code/benchmark_credit_reasoning.py` reruns the restored benchmark on frozen packets;
- `code/build_credit_decision_packets.py` rebuilds frozen packets from old full-run outputs when those outputs are available.

This code is kept as research-only archival code and is not imported by the current main survey pipeline. The preserved historical metrics under `results/` remain the canonical reported numbers.

Minimal local smoke run:

```bash
python research/credit_reasoning/code/benchmark_credit_reasoning.py \
  --decision-packets research/credit_reasoning/decision_packets/decision_packets.jsonl \
  --decision-mode compact_debate \
  --packet-sample 2 \
  --repeats 1 \
  --narrative-mode heuristic \
  --out-dir /tmp/credit_reasoning_smoke
```

## Main metric summary

| Run | Mode | News | Decision rate | Stability | Calls | Latency, s | Persona alignment | Emotional nuance | Coherence |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `v2_direct` | direct | no | 0.100 | 1.000 | 1.0 | 2.425 | 4.160 | 3.832 | 5.000 |
| `v2_compact_debate` | compact debate | no | 0.000 | 1.000 | 3.0 | 6.716 | 4.020 | 4.720 | 5.000 |
| `v3_compact` | compact debate | no | 0.000 | 1.000 | 3.0 | 14.182 | 3.880 | 4.440 | 5.000 |
| `v3b_compact` | compact debate | no | 0.000 | 1.000 | 3.0 | 8.440 | 4.160 | 4.720 | 5.000 |
| `v3c_compact` | compact debate | no | 0.000 | 1.000 | 3.0 | 8.749 | 4.020 | 4.720 | 5.000 |
| `v3_no_news_current` | compact debate | no | 0.000 | 1.000 | 3.0 | 9.222 | 4.020 | 4.252 | 5.000 |
| `v3_with_news_current` | compact debate | yes | 0.000 | 1.000 | 3.0 | 9.265 | 4.160 | 4.860 | 5.000 |

## Interpretation

The direct baseline is cheaper and faster because it uses one LLM call. Its main weakness is lower emotional nuance.

The compact-debate branch is more expensive, but it exposes an interpretable internal structure. Its most consistent semantic gain is emotional nuance: `3.832` in direct mode versus `4.720` in the mature compact variants.

The current-news comparison should be interpreted narrowly. It does not change binary decisions on this frozen packet set, but it enriches explanations and improves the judged semantic profile:

- persona alignment: `4.020 -> 4.160`;
- emotional nuance: `4.252 -> 4.860`;
- decision coherence remains `5.000`.

This branch should be treated as research evidence for improving the final decision layer, not as the primary production result of the project. The primary final engineering result remains the structured survey mode.
