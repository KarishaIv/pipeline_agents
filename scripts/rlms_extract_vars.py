#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

from pandas.io.stata import StataReader


def _to_jsonable_value_labels(value_labels: dict) -> dict:
    """Convert value-label mappings to JSON-friendly strings."""
    out = {}
    for label_name, mapping in value_labels.items():
        if not isinstance(mapping, dict):
            out[label_name] = mapping
            continue
        out[label_name] = {str(k): v for k, v in mapping.items()}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract RLMS variable labels (and optional value labels) from a .dta file."
    )
    parser.add_argument(
        "--dta",
        default="data/benchmark/rlms/RLMS_IND_1994_2024_eng.dta",
        help="Path to RLMS .dta file",
    )
    parser.add_argument(
        "--out-vars",
        default="data/benchmark/rlms/rlms_variables.csv",
        help="Output CSV with variable names and labels",
    )
    parser.add_argument(
        "--out-value-labels",
        default="data/benchmark/rlms/rlms_value_labels.json",
        help="Optional JSON output with value label dictionaries",
    )
    parser.add_argument(
        "--no-value-labels",
        action="store_true",
        help="Skip exporting value labels",
    )
    args = parser.parse_args()

    dta_path = Path(args.dta)
    out_vars = Path(args.out_vars)
    out_value_labels = Path(args.out_value_labels)

    if not dta_path.exists():
        raise FileNotFoundError(f"RLMS .dta not found: {dta_path}")

    out_vars.parent.mkdir(parents=True, exist_ok=True)
    if not args.no_value_labels:
        out_value_labels.parent.mkdir(parents=True, exist_ok=True)

    reader = StataReader(dta_path.as_posix(), convert_categoricals=False)

    var_labels = reader.variable_labels()
    with out_vars.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["variable", "label"])
        for var, label in var_labels.items():
            writer.writerow([var, label])

    if not args.no_value_labels:
        value_labels = reader.value_labels()
        jsonable = _to_jsonable_value_labels(value_labels)
        with out_value_labels.open("w", encoding="utf-8") as f:
            json.dump(jsonable, f, ensure_ascii=False, indent=2)

    print(f"Saved variables to: {out_vars}")
    if not args.no_value_labels:
        print(f"Saved value labels to: {out_value_labels}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
