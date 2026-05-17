# Reconstructed credit reasoning code

This folder contains a reconstruction of the credit-mode research implementation from the preserved benchmark artifacts and chat history.

It is intentionally isolated from the current production survey pipeline. Treat the historical artifacts in `../results/` as the canonical source of reported metrics; this code is a readable, runnable reconstruction of the experiment protocol.

## Main files

- `credit_schemas.py` defines the decision, voice, packet, and news-signal contracts.
- `credit_news_adapter.py` converts a news context snapshot into compact credit-specific signals.
- `credit_reasoning_agent.py` implements direct and compact-debate credit decisions.
- `benchmark_credit_reasoning.py` runs the restored benchmark over frozen decision packets.
- `build_credit_decision_packets.py` rebuilds frozen packets from old full simulation runs when those runs are available.

## Smoke run

From the repository root:

```bash
python research/credit_reasoning/code/benchmark_credit_reasoning.py \
  --decision-packets research/credit_reasoning/decision_packets/decision_packets.jsonl \
  --decision-mode compact_debate \
  --packet-sample 2 \
  --repeats 1 \
  --narrative-mode heuristic \
  --out-dir /tmp/credit_reasoning_smoke
```

Use `--narrative-mode llm` only when the project LLM configuration is available.

