#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests


HF_ROWS_URL = "https://datasets-server.huggingface.co/rows"


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(rows: Iterable[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def _pick_first(row: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return default


def _normalize_space(text: Any) -> str:
    if text is None:
        return ""
    return " ".join(str(text).split())


def _extract_four_options(row: Dict[str, Any]) -> List[str]:
    # TRAIT (mirlab/TRAIT) schema: response_high1/high2/low1/low2
    trait_direct = [
        row.get("response_high1"),
        row.get("response_high2"),
        row.get("response_low1"),
        row.get("response_low2"),
    ]
    if all(v is not None for v in trait_direct):
        return [_normalize_space(v) for v in trait_direct]

    direct = [
        row.get("option_1"),
        row.get("option_2"),
        row.get("option_3"),
        row.get("option_4"),
    ]
    if all(v is not None for v in direct):
        return [_normalize_space(v) for v in direct]

    choices = _pick_first(row, ["choices", "options", "candidate_answers"], default=None)
    if isinstance(choices, dict):
        for keyset in (["A", "B", "C", "D"], ["a", "b", "c", "d"], ["1", "2", "3", "4"]):
            if all(k in choices for k in keyset):
                return [_normalize_space(choices[k]) for k in keyset]
    if isinstance(choices, list) and len(choices) >= 4:
        return [_normalize_space(x) for x in choices[:4]]

    # Last chance: specific letter fields.
    letter_fields = [row.get("option_a"), row.get("option_b"), row.get("option_c"), row.get("option_d")]
    if all(v is not None for v in letter_fields):
        return [_normalize_space(v) for v in letter_fields]

    return []


def _answer_to_index(answer: Any, options: List[str]) -> Optional[str]:
    if answer is None:
        return None
    if isinstance(answer, (int, float)):
        try:
            value = int(answer)
        except Exception:
            value = None
        if value is not None:
            # Common dataset encodings: 0..3 or 1..4
            if 0 <= value < len(options):
                return str(value + 1)
            if 1 <= value <= len(options):
                return str(value)
    text = str(answer).strip()
    if text in {"1", "2", "3", "4"}:
        return text
    if text == "0":
        return "1"
    if text.upper() in {"A", "B", "C", "D"}:
        return str(ord(text.upper()) - ord("A") + 1)

    text_norm = _normalize_space(text).lower()
    for idx, option in enumerate(options, start=1):
        if text_norm == _normalize_space(option).lower():
            return str(idx)
    return None


def _answer_to_letter(answer: Any, options: List[str]) -> Optional[str]:
    idx = _answer_to_index(answer, options)
    if idx is None:
        return None
    val = int(idx)
    return chr(ord("A") + val - 1)


def fetch_rows(
    dataset: str,
    config: str,
    split: str,
    max_rows: int,
    chunk_size: int = 100,
    timeout: int = 60,
) -> List[Dict[str, Any]]:
    all_rows: List[Dict[str, Any]] = []
    offset = 0
    headers = {}
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    while len(all_rows) < max_rows:
        length = min(chunk_size, max_rows - len(all_rows))
        params = {"dataset": dataset, "config": config, "split": split, "offset": offset, "length": length}
        resp = requests.get(HF_ROWS_URL, params=params, headers=headers, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
        rows = payload.get("rows", [])
        if not rows:
            break
        for item in rows:
            row = item.get("row", item)
            if isinstance(row, dict):
                all_rows.append(row)
        offset += len(rows)
    return all_rows


def fetch_rows_via_datasets_lib(
    dataset: str,
    split: str,
    max_rows: int,
    config: Optional[str] = None,
) -> List[Dict[str, Any]]:
    try:
        from datasets import get_dataset_config_names, load_dataset
    except Exception as exc:
        raise RuntimeError(
            "datasets package is required for fallback loading. "
            "Install with: pip install datasets"
        ) from exc

    hf_token = os.getenv("HF_TOKEN")
    kwargs: Dict[str, Any] = {}
    config_name = config
    if not config_name and dataset.lower() == "sahandsab/emobench":
        # EmoBench requires explicit config.
        config_name = "emotional_understanding"
    if config_name:
        kwargs["name"] = config_name
    if hf_token:
        kwargs["token"] = hf_token

    split_norm = (split or "").strip()
    split_norm_lower = split_norm.lower()

    if split_norm_lower == "all":
        try:
            ds_dict = load_dataset(dataset, **kwargs)
        except Exception as exc:
            if "Config name is missing" in str(exc):
                cfg_names = get_dataset_config_names(dataset, token=hf_token) if hf_token else get_dataset_config_names(dataset)
                if not cfg_names:
                    raise
                preferred = "emotional_understanding" if "emotional_understanding" in cfg_names else cfg_names[0]
                kwargs["name"] = preferred
                ds_dict = load_dataset(dataset, **kwargs)
            else:
                raise
        if not hasattr(ds_dict, "keys"):
            # Fallback for non-dict result.
            rows = [dict(row) for row in ds_dict]
            return rows[:max_rows]

        split_names = list(ds_dict.keys())
        if not split_names:
            return []

        rows: List[Dict[str, Any]] = []
        per_split = max(1, max_rows // len(split_names))
        for split_name in split_names:
            ds_split = ds_dict[split_name]
            take = min(len(ds_split), per_split)
            for row in ds_split.select(range(take)):
                item = dict(row)
                item.setdefault("subset", split_name)
                item["__split_name"] = split_name
                rows.append(item)

        # If any budget remains, fill from splits in order.
        if len(rows) < max_rows:
            needed = max_rows - len(rows)
            for split_name in split_names:
                if needed <= 0:
                    break
                ds_split = ds_dict[split_name]
                already = sum(1 for r in rows if r.get("__split_name") == split_name)
                if already >= len(ds_split):
                    continue
                extra_take = min(len(ds_split) - already, needed)
                for row in ds_split.select(range(already, already + extra_take)):
                    item = dict(row)
                    item.setdefault("subset", split_name)
                    item["__split_name"] = split_name
                    rows.append(item)
                    needed -= 1
                    if needed <= 0:
                        break
        return rows[:max_rows]

    try:
        ds = load_dataset(dataset, split=split_norm, **kwargs)
    except Exception as exc:
        if "Config name is missing" in str(exc):
            cfg_names = get_dataset_config_names(dataset, token=hf_token) if hf_token else get_dataset_config_names(dataset)
            if not cfg_names:
                raise
            preferred = "emotional_understanding" if "emotional_understanding" in cfg_names else cfg_names[0]
            kwargs["name"] = preferred
            ds = load_dataset(dataset, split=split_norm, **kwargs)
            total = len(ds)
            if max_rows < total:
                ds = ds.select(range(max_rows))
            rows = [dict(row) for row in ds]
            for row in rows:
                row.setdefault("__split_name", split_norm)
                row.setdefault("subset", split_norm)
            return rows
        # Improve readability of split-related errors.
        if "Unknown split" in str(exc):
            ds_dict = load_dataset(dataset, **kwargs)
            available = list(ds_dict.keys()) if hasattr(ds_dict, "keys") else []
            if len(available) == 1:
                fallback_split = available[0]
                print(
                    f"Unknown split '{split_norm}' for dataset '{dataset}'. "
                    f"Auto-fallback to split '{fallback_split}'."
                )
                ds = load_dataset(dataset, split=fallback_split, **kwargs)
                total = len(ds)
                if max_rows < total:
                    ds = ds.select(range(max_rows))
                rows = [dict(row) for row in ds]
                for row in rows:
                    row.setdefault("__split_name", fallback_split)
                    row.setdefault("subset", fallback_split)
                return rows
            raise RuntimeError(
                f"Unknown split '{split_norm}' for dataset '{dataset}'. "
                f"Available splits: {available}. Use --trait-split all or --emobench-split all to combine splits."
            ) from exc
        raise

    total = len(ds)
    if max_rows < total:
        ds = ds.select(range(max_rows))
    rows = [dict(row) for row in ds]
    for row in rows:
        row.setdefault("__split_name", split_norm)
        row.setdefault("subset", split_norm)
    return rows


def normalize_trait_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        prompt = _pick_first(row, ["prompt", "question", "query", "text", "instruction"], default="")
        options = _extract_four_options(row)
        if not prompt or len(options) != 4:
            continue
        answer_raw = _pick_first(row, ["answer", "label", "gold", "target", "correct_answer"])
        answer = _answer_to_index(answer_raw, options)
        subset = _normalize_space(
            _pick_first(
                row,
                ["subset", "category", "task", "type", "personality", "__split_name"],
                default="default",
            )
        )
        item_id = _normalize_space(_pick_first(row, ["id", "item_id", "uid"], default=f"trait_{idx}"))
        out.append(
            {
                "item_id": item_id,
                "prompt": _normalize_space(prompt),
                "option_1": options[0],
                "option_2": options[1],
                "option_3": options[2],
                "option_4": options[3],
                "answer": answer,
                "subset": subset or "default",
            }
        )
    return out


def normalize_emobench_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        base_prompt = _pick_first(
            row,
            ["prompt", "question", "text", "query", "instruction", "scenario", "context"],
            default="",
        )
        if not base_prompt:
            continue
        domain = _normalize_space(
            _pick_first(row, ["domain", "coarse_category", "category", "task", "subset"], default="default")
        )
        taxonomy_base = _normalize_space(
            _pick_first(
                row,
                ["taxonomy", "finegrained_category", "emotion_type", "emotion_category", "question type", "question_type"],
                default="default",
            )
        )
        item_id = _normalize_space(_pick_first(row, ["id", "item_id", "uid", "qid"], default=f"emobench_{idx}"))

        task_specs = [
            ("emotion", row.get("emotion_choices"), row.get("emotion_label")),
            ("cause", row.get("cause_choices"), row.get("cause_label")),
        ]

        added = False
        for task_name, raw_choices, raw_label in task_specs:
            if raw_choices is None:
                continue
            if isinstance(raw_choices, dict):
                choices = [raw_choices.get(k) for k in ["A", "B", "C", "D"] if k in raw_choices]
                if len(choices) != 4:
                    # Fall back to deterministic key order.
                    choices = [raw_choices[k] for k in sorted(raw_choices.keys())[:4]]
            elif isinstance(raw_choices, list):
                choices = raw_choices[:4]
            else:
                choices = []
            choices = [_normalize_space(x) for x in choices if x is not None]
            if len(choices) != 4:
                continue

            label = _answer_to_letter(raw_label, choices)
            task_prompt_suffix = (
                "What is the most likely emotion?" if task_name == "emotion" else "What is the most likely cause?"
            )
            prompt = f"{_normalize_space(base_prompt)}\n{task_prompt_suffix}"
            out.append(
                {
                    "item_id": f"{item_id}_{task_name}",
                    "prompt": prompt,
                    "option_a": choices[0],
                    "option_b": choices[1],
                    "option_c": choices[2],
                    "option_d": choices[3],
                    "label": label,
                    "domain": domain or "default",
                    "taxonomy": f"{taxonomy_base}:{task_name}",
                }
            )
            added = True

        if added:
            continue

        # Fallback for older schema.
        options = _extract_four_options(row)
        if len(options) == 4:
            label_raw = _pick_first(
                row,
                ["label", "answer", "gold", "target", "correct_answer", "emotion_label", "cause_label"],
            )
            label = _answer_to_letter(label_raw, options)
            out.append(
                {
                    "item_id": item_id,
                    "prompt": _normalize_space(base_prompt),
                    "option_a": options[0],
                    "option_b": options[1],
                    "option_c": options[2],
                    "option_d": options[3],
                    "label": label,
                    "domain": domain or "default",
                    "taxonomy": taxonomy_base or "default",
                }
            )
    return out


def _save_metadata(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and normalize external benchmark datasets.")
    parser.add_argument("--output-root", default="data/benchmark_external", help="Output root directory.")
    parser.add_argument("--trait-dataset", default="mirlab/TRAIT", help="HF dataset id for TRAIT.")
    parser.add_argument("--trait-config", default="default", help="HF config for TRAIT.")
    parser.add_argument(
        "--trait-split",
        default="all",
        help="HF split for TRAIT (use 'all' to combine all dataset splits).",
    )
    parser.add_argument("--emobench-dataset", default="SahandSab/EmoBench", help="HF dataset id for EmoBench.")
    parser.add_argument(
        "--emobench-config",
        default="emotional_understanding",
        help="HF config for EmoBench (e.g. emotional_understanding, emotional_application).",
    )
    parser.add_argument("--emobench-split", default="train", help="HF split for EmoBench.")
    parser.add_argument(
        "--fetch-backend",
        choices=["auto", "datasets-server", "datasets-lib"],
        default="auto",
        help="How to fetch rows from HF.",
    )
    parser.add_argument("--trait-max-rows", type=int, default=5000, help="Max rows to fetch for TRAIT.")
    parser.add_argument("--emobench-max-rows", type=int, default=5000, help="Max rows to fetch for EmoBench.")
    parser.add_argument("--force", action="store_true", help="Re-download even if cached files exist.")
    parser.add_argument("--timeout", type=int, default=60, help="HTTP timeout in seconds.")
    args = parser.parse_args()

    # Allow override from CLI while preserving function defaults.
    requests.adapters.DEFAULT_RETRIES = 3

    output_root = Path(args.output_root)
    trait_dir = output_root / "trait"
    emo_dir = output_root / "emobench"
    trait_path = trait_dir / "trait_train.jsonl"
    emo_path = emo_dir / "emobench_test.jsonl"

    if trait_path.exists() and emo_path.exists() and not args.force:
        print("Cached benchmark data already exists. Use --force to refresh.")
        print(f"TRAIT rows: {len(_read_jsonl(trait_path))}")
        print(f"EmoBench rows: {len(_read_jsonl(emo_path))}")
        return 0

    def fetch_any(dataset: str, config: str, split: str, max_rows: int) -> List[Dict[str, Any]]:
        split_norm = (split or "").strip().lower()
        cfg = None if not config or config == "default" else config

        if split_norm == "all":
            return fetch_rows_via_datasets_lib(dataset=dataset, split="all", max_rows=max_rows, config=cfg)

        if args.fetch_backend == "datasets-server":
            return fetch_rows(dataset=dataset, config=config, split=split, max_rows=max_rows, timeout=args.timeout)
        if args.fetch_backend == "datasets-lib":
            return fetch_rows_via_datasets_lib(dataset=dataset, split=split, max_rows=max_rows, config=cfg)

        # auto: try datasets-server first, fallback to datasets lib.
        try:
            return fetch_rows(dataset=dataset, config=config, split=split, max_rows=max_rows, timeout=args.timeout)
        except Exception as exc:
            print(f"datasets-server failed for {dataset} ({exc}); trying datasets library fallback...")
            return fetch_rows_via_datasets_lib(dataset=dataset, split=split, max_rows=max_rows, config=cfg)

    try:
        raw_trait = fetch_any(
            dataset=args.trait_dataset,
            config=args.trait_config,
            split=args.trait_split,
            max_rows=args.trait_max_rows,
        )
        trait_rows = normalize_trait_rows(raw_trait)

        raw_emo = fetch_any(
            dataset=args.emobench_dataset,
            config=args.emobench_config,
            split=args.emobench_split,
            max_rows=args.emobench_max_rows,
        )
        emo_rows = normalize_emobench_rows(raw_emo)
    except Exception as exc:
        print(f"Failed to download datasets: {exc}")
        print("Check internet access, HF dataset permissions, or install datasets package.")
        return 1

    if not trait_rows:
        print("Downloaded TRAIT data but normalization produced 0 rows.")
        if raw_trait:
            print(f"Sample TRAIT row keys: {sorted(list(raw_trait[0].keys()))[:30]}")
        return 2
    if not emo_rows:
        print("Downloaded EmoBench data but normalization produced 0 rows.")
        if raw_emo:
            print(f"Sample EmoBench row keys: {sorted(list(raw_emo[0].keys()))[:30]}")
        return 3

    _write_jsonl(trait_rows, trait_path)
    _write_jsonl(emo_rows, emo_path)
    _save_metadata(
        trait_dir / "metadata.json",
        {
            "dataset": args.trait_dataset,
            "split": args.trait_split,
            "raw_rows": len(raw_trait),
            "normalized_rows": len(trait_rows),
            "fetch_backend": args.fetch_backend,
        },
    )
    _save_metadata(
        emo_dir / "metadata.json",
        {
            "dataset": args.emobench_dataset,
            "split": args.emobench_split,
            "raw_rows": len(raw_emo),
            "normalized_rows": len(emo_rows),
            "fetch_backend": args.fetch_backend,
        },
    )

    print(f"Saved TRAIT rows: {len(trait_rows)} -> {trait_path}")
    print(f"Saved EmoBench rows: {len(emo_rows)} -> {emo_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
