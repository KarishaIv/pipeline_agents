#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from scipy import stats


def _load_latest_synthetic(path_glob: str) -> Path:
    candidates = sorted(Path().glob(path_glob), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No synthetic personas found for glob: {path_glob}")
    return candidates[0]


def _normalize_text(x: Optional[str]) -> str:
    if x is None:
        return ""
    return str(x).strip().lower()


def _parse_age(value: object) -> Optional[int]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value).strip()
    if s == "":
        return None
    s = s.replace("+", "")
    if s.startswith("<"):
        return 24
    if s.startswith(">="):
        return 80
    if "-" in s:
        try:
            lo, hi = s.split("-", 1)
            return int((int(lo) + int(hi)) / 2)
        except ValueError:
            return None
    try:
        return int(float(s))
    except ValueError:
        return None


def _parse_age_group(value: object) -> Optional[int]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value).strip()
    if s == "":
        return None
    if "-" in s:
        return _parse_age(s)
    try:
        # age_group stored as bin index
        v = int(float(s))
        return v * 5 + 2
    except ValueError:
        return None


def _normalize_gender(value: object) -> Optional[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    # numeric codes
    try:
        code = int(float(value))
        if code == 1:
            return "male"
        if code == 2:
            return "female"
    except (TypeError, ValueError):
        pass
    s = _normalize_text(value)
    if s in {"male", "m", "man", "мужской", "мужчина"}:
        return "male"
    if s in {"female", "f", "woman", "женский", "женщина"}:
        return "female"
    return None


def _normalize_marital(value: object) -> Optional[str]:
    s = _normalize_text(value)
    if s in {"married", "женат", "замужем", "in a registered marriage", "registered, not living together"}:
        return "married"
    if s in {"single", "не женат", "не замужем", "never married"}:
        return "single"
    if s in {"divorced", "divorsed and not remarried", "разведен", "разведена", "separated"}:
        return "divorced"
    if s in {"widowed", "widower or widow", "вдовец", "вдова"}:
        return "widowed"
    if "living together" in s and "not registered" in s:
        return "married"
    return None


def _normalize_education(value: object) -> Optional[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    # numeric education years (RLMS educ)
    try:
        years = int(float(value))
        if years <= 9:
            return "low"
        if 10 <= years <= 11:
            return "mid"
        if years >= 12:
            return "high"
    except (TypeError, ValueError):
        pass
    s = _normalize_text(value)
    if s == "":
        return None
    # High
    if any(k in s for k in ["master", "магистр", "магистрат", "аспиран", "phd", "graduate", "postgraduate"]):
        return "high"
    if any(k in s for k in ["bachelor", "бакалавр", "бакалавриат", "специалитет", "высшее", "higher"]):
        return "high"
    # Mid
    if any(k in s for k in ["college", "secondary vocational", "среднее спец", "профессион", "professional"]):
        return "mid"
    if any(k in s for k in ["secondary", "high school", "среднее"]):
        return "mid"
    # Low
    if any(k in s for k in ["primary", "basic", "неполное", "less than"]):
        return "low"
    # try to extract grade count from label
    m = re.search(r"(\d+)", s)
    if m:
        years = int(m.group(1))
        if years <= 9:
            return "low"
        if 10 <= years <= 11:
            return "mid"
        if years >= 12:
            return "high"
    return None


def _normalize_children(value: object) -> Optional[int]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value).strip()
    if s == "":
        return None
    # RLMS missing sentinels
    try:
        v = float(s)
        if v >= 99999990:
            return None
    except ValueError:
        pass
    if s.endswith("+"):
        try:
            return int(s[:-1])
        except ValueError:
            return None
    try:
        return int(float(s))
    except ValueError:
        return None


def _normalize_income(value: object) -> Optional[float]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value).strip().replace(" ", "")
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        # extract digits
        m = re.findall(r"\d+\.?\d*", s)
        if not m:
            return None
        try:
            return float(m[0])
        except ValueError:
            return None


def _parse_income_level(value: object) -> Optional[float]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value).strip()
    if s == "":
        return None
    # Normalize like "123 500-250000" -> "123500-250000"
    s = s.replace(" ", "")
    nums = re.findall(r"\d+", s)
    if not nums:
        return None
    try:
        vals = [float(n) for n in nums]
    except ValueError:
        return None
    if len(vals) >= 2:
        return float(sum(vals[:2]) / 2)
    return float(vals[0])


def _parse_income_range(value: object) -> Optional[Tuple[float, float]]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value).strip()
    if s == "":
        return None
    s = s.replace(" ", "")
    nums = re.findall(r"\d+", s)
    if not nums:
        return None
    try:
        vals = [float(n) for n in nums]
    except ValueError:
        return None
    if len(vals) >= 2:
        return float(vals[0]), float(vals[1])
    if len(vals) == 1:
        return float(vals[0]), float(vals[0])
    return None


def _decode_value(value: object, labels: Optional[Dict[str, str]]) -> object:
    if labels is None:
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    # try numeric key
    try:
        key = str(int(float(value)))
    except (TypeError, ValueError):
        key = str(value).strip()
    return labels.get(key, value)


def _suggest_mapping(vars_csv: Path) -> Dict[str, List[Dict[str, str]]]:
    df = pd.read_csv(vars_csv)
    df["label"] = df["label"].fillna("")
    df["label_lc"] = df["label"].str.lower()

    keywords = {
        "year": ["year", "wave"],
        "age": ["age"],
        "gender": ["sex", "gender"],
        "education": ["education", "school", "degree"],
        "marital_status": ["marital", "marriage"],
        "children": ["children", "child"],
        "income": ["income", "wage", "earnings", "salary"],
        "occupation": ["occupation", "job", "employment"],
        "region": ["region", "oblast", "republic", "federal", "settlement", "city"],
    }

    suggestions: Dict[str, List[Dict[str, str]]] = {}
    for field, keys in keywords.items():
        scored = []
        for _, row in df.iterrows():
            label = row["label_lc"]
            score = sum(1 for k in keys if k in label)
            if score > 0:
                scored.append((score, row["variable"], row["label"]))
        scored.sort(reverse=True, key=lambda x: x[0])
        suggestions[field] = [
            {"variable": var, "label": label, "score": score} for score, var, label in scored[:5]
        ]
    return suggestions


def _load_mapping(mapping_path: Path, suggestions_path: Path) -> Dict[str, str]:
    if mapping_path.exists():
        with mapping_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return {k: v for k, v in data.items() if not k.startswith("__")}

    suggestions = _suggest_mapping(Path("data/benchmark/rlms/rlms_variables.csv"))
    suggestions_path.parent.mkdir(parents=True, exist_ok=True)
    with suggestions_path.open("w", encoding="utf-8") as f:
        json.dump(suggestions, f, ensure_ascii=False, indent=2)

    # Create auto mapping with top suggestion
    auto_mapping = {k: (v[0]["variable"] if v else None) for k, v in suggestions.items()}
    auto_mapping["__auto__"] = True
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    with mapping_path.open("w", encoding="utf-8") as f:
        json.dump(auto_mapping, f, ensure_ascii=False, indent=2)

    print(f"Mapping file created: {mapping_path}")
    print(f"Suggestions file created: {suggestions_path}")
    print("Please review mapping before trusting results.")
    return {k: v for k, v in auto_mapping.items() if not k.startswith("__")}


def _prepare_synthetic(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    if "age" in df.columns:
        out["age"] = df["age"].apply(_parse_age)
    elif "age_group" in df.columns:
        out["age"] = df["age_group"].apply(_parse_age_group)
    out["gender"] = df.get("gender", pd.Series([None] * len(df))).apply(_normalize_gender)
    out["education"] = df.get("education", pd.Series([None] * len(df))).apply(_normalize_education)
    out["marital_status"] = df.get("marital_status", pd.Series([None] * len(df))).apply(_normalize_marital)
    if "children" in df.columns:
        out["children"] = df["children"].apply(_normalize_children)
    elif "children_group" in df.columns:
        out["children"] = df["children_group"].apply(_normalize_children)
    else:
        out["children"] = pd.Series([None] * len(df))

    if "income" in df.columns:
        out["income"] = df["income"].apply(_normalize_income)
    elif "income_level" in df.columns:
        out["income"] = df["income_level"].apply(_parse_income_level)
    else:
        out["income"] = pd.Series([None] * len(df))
    out["region"] = df.get("region", pd.Series([None] * len(df)))
    return out


def _age_range_from_evidence(age_value: Any) -> Optional[Tuple[int, int]]:
    if age_value is None:
        return None
    try:
        age = int(float(age_value))
    except (TypeError, ValueError):
        return None
    bins = np.arange(0, 100, 5)
    bin_idx = int(np.digitize(age, bins))
    return bin_idx * 5, bin_idx * 5 + 4


def _children_filter(value: Any) -> Optional[Dict[str, int]]:
    if value is None:
        return None
    s = str(value).strip()
    if s == "":
        return None
    if s.endswith("+"):
        try:
            return {"min": int(s[:-1])}
        except ValueError:
            return None
    try:
        return {"exact": int(float(s))}
    except ValueError:
        return None


def _load_evidence(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Evidence file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return [d for d in data if isinstance(d, dict)]
    raise ValueError("Evidence file must be a dict or list of dicts.")


def _build_evidence_filters(evidence_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    filters: List[Dict[str, Any]] = []
    for ev in evidence_list:
        f: Dict[str, Any] = {}
        age_range = _age_range_from_evidence(ev.get("age_group"))
        if age_range:
            f["age_range"] = age_range

        gender = _normalize_gender(ev.get("gender"))
        if gender:
            f["gender"] = gender

        marital = _normalize_marital(ev.get("marital_status"))
        if marital:
            f["marital_status"] = marital

        children = _children_filter(ev.get("children_group", ev.get("children")))
        if children:
            f["children"] = children

        education = _normalize_education(ev.get("education"))
        if education:
            f["education"] = education

        region = ev.get("region")
        if region:
            f["region"] = str(region).strip()

        income_range = _parse_income_range(ev.get("income_level", ev.get("income")))
        if income_range:
            f["income_range"] = income_range

        if f:
            filters.append(f)
    return filters


def _apply_evidence_filters(df: pd.DataFrame, filters: List[Dict[str, Any]]) -> pd.DataFrame:
    if not filters:
        return df
    mask = pd.Series(False, index=df.index)
    for f in filters:
        m = pd.Series(True, index=df.index)
        if "age_range" in f:
            lo, hi = f["age_range"]
            m &= df.get("age").between(lo, hi)
        if "gender" in f:
            m &= df.get("gender") == f["gender"]
        if "marital_status" in f:
            m &= df.get("marital_status") == f["marital_status"]
        if "children" in f:
            ch = f["children"]
            if "exact" in ch:
                m &= df.get("children") == ch["exact"]
            elif "min" in ch:
                m &= df.get("children") >= ch["min"]
        if "education" in f:
            m &= df.get("education") == f["education"]
        if "region" in f:
            m &= df.get("region") == f["region"]
        if "income_range" in f:
            lo, hi = f["income_range"]
            m &= df.get("income").between(lo, hi)
        mask |= m
    return df[mask]


def _prepare_rlms(df: pd.DataFrame, mapping: Dict[str, str], value_labels: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    out = pd.DataFrame()
    if mapping.get("age") and mapping["age"] in df.columns:
        out["age"] = df[mapping["age"]].apply(_parse_age)
    if mapping.get("gender") and mapping["gender"] in df.columns:
        labels = value_labels.get(mapping["gender"])
        out["gender"] = df[mapping["gender"]].apply(lambda v: _normalize_gender(_decode_value(v, labels)))
    if mapping.get("education") and mapping["education"] in df.columns:
        # education is numeric years in RLMS, avoid value labels
        out["education"] = df[mapping["education"]].apply(_normalize_education)
    if mapping.get("marital_status") and mapping["marital_status"] in df.columns:
        labels = value_labels.get(mapping["marital_status"])
        out["marital_status"] = df[mapping["marital_status"]].apply(lambda v: _normalize_marital(_decode_value(v, labels)))
    if mapping.get("children") and mapping["children"] in df.columns:
        out["children"] = df[mapping["children"]].apply(_normalize_children)
    if mapping.get("income") and mapping["income"] in df.columns:
        out["income"] = df[mapping["income"]].apply(_normalize_income)
    if mapping.get("region") and mapping["region"] in df.columns:
        labels = value_labels.get(mapping["region"])
        out["region"] = df[mapping["region"]].apply(lambda v: _decode_value(v, labels))
    return out


def _year_codes_for_target(
    target_year: int, year_col: str, value_labels: Dict[str, Dict[str, str]]
) -> Optional[set]:
    if year_col not in value_labels:
        return None
    labels = value_labels.get(year_col, {})
    codes = set()
    for code, label in labels.items():
        m = re.search(r"(\d{4})", str(label))
        if not m:
            continue
        if int(m.group(1)) == target_year:
            try:
                codes.add(int(code))
            except ValueError:
                continue
    return codes if codes else None


def _year_codes_for_range(
    start_year: int, end_year: int, year_col: str, value_labels: Dict[str, Dict[str, str]]
) -> Optional[set]:
    codes: set = set()
    for year in range(start_year, end_year + 1):
        year_codes = _year_codes_for_target(year, year_col, value_labels)
        if year_codes:
            codes.update(year_codes)
    return codes if codes else None


def _filter_year(
    df: pd.DataFrame,
    year_series: pd.Series,
    target_year: int,
    year_col: str,
    value_labels: Dict[str, Dict[str, str]],
) -> pd.DataFrame:
    codes = _year_codes_for_target(target_year, year_col, value_labels)
    if codes:
        year_num = pd.to_numeric(year_series, errors="coerce")
        return df[year_num.isin(codes)]

    year_digits = year_series.astype(str).str.extract(r"(\d{4})")[0]
    year_num = pd.to_numeric(year_digits, errors="coerce")
    return df[year_num == target_year]


def _filter_year_range(
    df: pd.DataFrame,
    year_series: pd.Series,
    start_year: int,
    end_year: int,
    year_col: str,
    value_labels: Dict[str, Dict[str, str]],
) -> pd.DataFrame:
    codes = _year_codes_for_range(start_year, end_year, year_col, value_labels)
    if codes:
        year_num = pd.to_numeric(year_series, errors="coerce")
        return df[year_num.isin(codes)]

    year_digits = year_series.astype(str).str.extract(r"(\d{4})")[0]
    year_num = pd.to_numeric(year_digits, errors="coerce")
    return df[(year_num >= start_year) & (year_num <= end_year)]


def _latest_year_from_codes(
    codes: set, year_col: str, value_labels: Dict[str, Dict[str, str]]
) -> Optional[int]:
    labels = value_labels.get(year_col, {})
    if not labels:
        # If no labels, assume codes are years directly
        try:
            return max(int(c) for c in codes)
        except ValueError:
            return None
    years = []
    for code in codes:
        label = labels.get(str(int(code)))
        if label is None:
            continue
        m = re.search(r"(\d{4})", str(label))
        if not m:
            continue
        try:
            years.append(int(m.group(1)))
        except ValueError:
            continue
    return max(years) if years else None


def _available_years_from_codes(
    codes: set, year_col: str, value_labels: Dict[str, Dict[str, str]]
) -> List[int]:
    labels = value_labels.get(year_col, {})
    years = []
    if not labels:
        for code in codes:
            try:
                years.append(int(code))
            except ValueError:
                continue
        return sorted(set(years))
    for code in codes:
        label = labels.get(str(int(code)))
        if label is None:
            continue
        m = re.search(r"(\d{4})", str(label))
        if not m:
            continue
        try:
            years.append(int(m.group(1)))
        except ValueError:
            continue
    return sorted(set(years))


def _compare_numeric(a: pd.Series, b: pd.Series) -> Dict[str, float]:
    a = a.dropna().astype(float)
    b = b.dropna().astype(float)
    if len(a) < 30 or len(b) < 30:
        return {"n_a": int(len(a)), "n_b": int(len(b))}
    ks = stats.ks_2samp(a, b)
    return {"n_a": int(len(a)), "n_b": int(len(b)), "ks_stat": float(ks.statistic), "ks_pvalue": float(ks.pvalue)}


def _compare_categorical(a: pd.Series, b: pd.Series) -> Dict[str, float]:
    a = a.dropna().astype(str)
    b = b.dropna().astype(str)
    if len(a) < 30 or len(b) < 30:
        return {"n_a": int(len(a)), "n_b": int(len(b))}
    categories = sorted(set(a.unique()).union(set(b.unique())))
    a_counts = [int((a == c).sum()) for c in categories]
    b_counts = [int((b == c).sum()) for c in categories]
    table = np.vstack([a_counts, b_counts])
    chi2, p, _, _ = stats.chi2_contingency(table)
    return {"n_a": int(len(a)), "n_b": int(len(b)), "chi2": float(chi2), "chi2_pvalue": float(p)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark RLMS demographics vs synthetic personas.")
    parser.add_argument(
        "--rlms-path",
        default="data/benchmark/rlms/RLMS_IND_1994_2024_eng.dta",
        help="Path to RLMS individuals .dta file.",
    )
    parser.add_argument(
        "--mapping",
        default="data/benchmark/rlms/rlms_mapping.json",
        help="JSON with RLMS -> canonical field mapping.",
    )
    parser.add_argument(
        "--synthetic-path",
        default="outputs/all_replicated_personas_*.csv",
        help="Glob to synthetic personas CSV.",
    )
    parser.add_argument("--year", type=int, default=2024, help="Target RLMS year.")
    parser.add_argument(
        "--year-range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Target RLMS year range (inclusive). Overrides --year when set.",
    )
    parser.add_argument("--chunksize", type=int, default=200_000, help="Stata read chunksize.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional cap on RLMS rows.")
    parser.add_argument("--out-dir", default="outputs/benchmarks/rlms", help="Output directory.")
    parser.add_argument("--evidence", default="data/evidence.json", help="Evidence JSON file.")
    parser.add_argument(
        "--filter-evidence",
        action="store_true",
        help="Filter RLMS and synthetic samples by evidence conditions before comparing.",
    )
    args = parser.parse_args()

    rlms_path = Path(args.rlms_path)
    if not rlms_path.exists():
        raise FileNotFoundError(f"RLMS file not found: {rlms_path}")

    mapping_path = Path(args.mapping)
    suggestions_path = Path("data/benchmark/rlms/rlms_mapping_suggestions.json")
    mapping = _load_mapping(mapping_path, suggestions_path)
    value_labels_path = Path("data/benchmark/rlms/rlms_value_labels.json")
    value_labels = {}
    if value_labels_path.exists():
        with value_labels_path.open("r", encoding="utf-8") as f:
            value_labels = json.load(f)

    # Read RLMS in chunks with selected columns
    cols = [v for v in mapping.values() if v]
    if not cols:
        raise ValueError("No mapped columns found. Check rlms_mapping.json.")

    year_col = mapping.get("year")
    if year_col and year_col not in cols:
        cols.append(year_col)

    rlms_rows = []
    total = 0
    seen_year_codes = set()
    year_range = None
    if args.year_range:
        start_year, end_year = args.year_range
        if start_year > end_year:
            start_year, end_year = end_year, start_year
        year_range = (start_year, end_year)
    for chunk in pd.read_stata(
        rlms_path,
        columns=cols,
        chunksize=args.chunksize,
        convert_categoricals=False,
    ):
        if year_col and year_col in chunk.columns:
            # Track seen year codes for fallback
            year_num = pd.to_numeric(chunk[year_col], errors="coerce").dropna().astype(int)
            seen_year_codes.update(year_num.unique().tolist())
            if year_range:
                chunk = _filter_year_range(
                    chunk, chunk[year_col], year_range[0], year_range[1], year_col, value_labels
                )
            else:
                chunk = _filter_year(chunk, chunk[year_col], args.year, year_col, value_labels)
        if len(chunk) == 0:
            continue
        rlms_rows.append(chunk)
        total += len(chunk)
        if args.max_rows and total >= args.max_rows:
            break

    if not rlms_rows:
        latest_year = _latest_year_from_codes(seen_year_codes, year_col, value_labels) if year_col else None
        if year_range:
            available = _available_years_from_codes(seen_year_codes, year_col, value_labels) if year_col else []
            if available:
                raise ValueError(
                    f"No RLMS rows for year range {year_range[0]}-{year_range[1]}. "
                    f"Detected available years: {available[0]}-{available[-1]}."
                )
        elif latest_year and latest_year != args.year:
            raise ValueError(
                f"No RLMS rows for year {args.year}. "
                f"Detected latest available year: {latest_year}. "
                f"Rerun with --year {latest_year}."
            )
        raise ValueError("No RLMS rows after filtering. Check year mapping.")

    rlms_df = pd.concat(rlms_rows, ignore_index=True)

    # Synthetic personas
    synth_path = _load_latest_synthetic(args.synthetic_path)
    synth_df = pd.read_csv(synth_path)

    rlms_prepared = _prepare_rlms(rlms_df, mapping, value_labels)
    synth_prepared = _prepare_synthetic(synth_df)

    if args.filter_evidence:
        evidence = _load_evidence(Path(args.evidence))
        filters = _build_evidence_filters(evidence)
        if not filters:
            raise ValueError("No usable evidence filters built from evidence file.")
        rlms_before = len(rlms_prepared)
        synth_before = len(synth_prepared)
        rlms_prepared = _apply_evidence_filters(rlms_prepared, filters)
        synth_prepared = _apply_evidence_filters(synth_prepared, filters)
        print(
            f"Filtered RLMS rows: {rlms_before} -> {len(rlms_prepared)}; "
            f"Synthetic rows: {synth_before} -> {len(synth_prepared)}"
        )
        if len(rlms_prepared) == 0 or len(synth_prepared) == 0:
            raise ValueError("Evidence filtering left no rows. Relax evidence or check mappings.")

    metrics = {}
    metrics["age"] = _compare_numeric(rlms_prepared.get("age", pd.Series(dtype=float)), synth_prepared.get("age", pd.Series(dtype=float)))
    metrics["income"] = _compare_numeric(rlms_prepared.get("income", pd.Series(dtype=float)), synth_prepared.get("income", pd.Series(dtype=float)))
    metrics["children"] = _compare_numeric(rlms_prepared.get("children", pd.Series(dtype=float)), synth_prepared.get("children", pd.Series(dtype=float)))

    metrics["gender"] = _compare_categorical(rlms_prepared.get("gender", pd.Series(dtype=str)), synth_prepared.get("gender", pd.Series(dtype=str)))
    metrics["education"] = _compare_categorical(rlms_prepared.get("education", pd.Series(dtype=str)), synth_prepared.get("education", pd.Series(dtype=str)))
    metrics["marital_status"] = _compare_categorical(rlms_prepared.get("marital_status", pd.Series(dtype=str)), synth_prepared.get("marital_status", pd.Series(dtype=str)))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / "rlms_vs_synthetic_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"Saved metrics: {metrics_path}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
