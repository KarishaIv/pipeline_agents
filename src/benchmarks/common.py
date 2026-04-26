from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Type

import numpy as np
import pandas as pd

from src.utils import robust_llm_call


class QuotaExceeded(RuntimeError):
    """Raised when run-time or call budget is exhausted."""


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def find_latest_synthetic_csv(root: Optional[Path] = None) -> Path:
    base = root or project_root()
    candidates = sorted((base / "outputs").glob("all_replicated_personas_*.csv"))
    if not candidates:
        raise FileNotFoundError("No synthetic personas found in outputs/all_replicated_personas_*.csv")
    return candidates[-1]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def deterministic_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n <= 0 or n >= len(df):
        return df.reset_index(drop=True)
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def balanced_sample(df: pd.DataFrame, group_col: str, n: int, seed: int) -> pd.DataFrame:
    if n <= 0 or n >= len(df) or group_col not in df.columns:
        return deterministic_sample(df, n, seed)

    rng = np.random.default_rng(seed)
    groups = [g for g in sorted(df[group_col].dropna().unique())]
    if not groups:
        return deterministic_sample(df, n, seed)

    per_group = max(1, n // len(groups))
    chunks: List[pd.DataFrame] = []

    for idx, group in enumerate(groups):
        part = df[df[group_col] == group]
        take = min(len(part), per_group)
        if idx == len(groups) - 1:
            # Fill the remaining budget in the last group.
            already = sum(len(c) for c in chunks)
            take = min(len(part), max(0, n - already))
        if take > 0:
            chunks.append(part.sample(n=take, random_state=int(rng.integers(0, 2**31 - 1))))

    out = pd.concat(chunks, axis=0) if chunks else pd.DataFrame(columns=df.columns)
    if len(out) < n:
        used_idx = set(out.index.tolist())
        remaining = df.drop(index=list(used_idx), errors="ignore")
        if len(remaining) > 0:
            extra = deterministic_sample(remaining, min(n - len(out), len(remaining)), seed + 997)
            out = pd.concat([out, extra], axis=0)
    return out.head(n).reset_index(drop=True)


def as_dict(value: Any) -> Dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    if isinstance(value, dict):
        return value
    raise TypeError(f"Cannot convert type {type(value)} to dict.")


def parse_choice(value: Any, allowed: Sequence[str], default: str) -> str:
    if value is None:
        return default
    text = str(value).strip().upper()
    for item in allowed:
        if text == item.upper():
            return item
    if text and text[0] in {x[0].upper() for x in allowed}:
        first = text[0]
        for item in allowed:
            if item[0].upper() == first:
                return item
    return default


def parse_int_choice(value: Any, lo: int, hi: int, default: int) -> int:
    try:
        val = int(str(value).strip())
    except (ValueError, TypeError):
        return default
    if val < lo or val > hi:
        return default
    return val


def git_commit_hash(root: Optional[Path] = None) -> str:
    cwd = str(root or project_root())
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd, text=True)
        return out.strip()
    except Exception:
        return "unknown"


def load_personas(synthetic_path: Optional[str], persona_sample: int, seed: int) -> pd.DataFrame:
    if synthetic_path:
        path = Path(synthetic_path)
    else:
        path = find_latest_synthetic_csv()
    if not path.exists():
        raise FileNotFoundError(f"Synthetic personas file not found: {path}")
    df = pd.read_csv(path)
    if len(df) == 0:
        raise ValueError("Synthetic personas CSV is empty.")
    return deterministic_sample(df, persona_sample, seed)


def persona_id(row: pd.Series, fallback_idx: int) -> str:
    for key in ("name", "persona_id", "id"):
        if key in row and pd.notna(row[key]):
            return str(row[key])
    return f"persona_{fallback_idx}"


def hash_payload(payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def load_cache(path: Path) -> Dict[str, Any]:
    data = load_json(path, default={})
    return data if isinstance(data, dict) else {}


def save_cache(path: Path, cache: Dict[str, Any]) -> None:
    save_json(cache, path)


@dataclass
class QuotaGuard:
    max_calls: int
    max_runtime_min: float
    started_at: float = field(default_factory=time.time)
    calls_made: int = 0

    def _runtime_seconds(self) -> float:
        return time.time() - self.started_at

    def snapshot(self) -> Dict[str, Any]:
        return {
            "calls_made": int(self.calls_made),
            "max_calls": int(self.max_calls),
            "elapsed_seconds": round(self._runtime_seconds(), 3),
            "max_runtime_seconds": float(self.max_runtime_min * 60.0),
        }

    def check(self) -> None:
        if self.calls_made >= self.max_calls:
            raise QuotaExceeded(f"Call budget exhausted ({self.calls_made}/{self.max_calls}).")
        if self._runtime_seconds() > self.max_runtime_min * 60.0:
            raise QuotaExceeded(
                f"Runtime budget exhausted ({self._runtime_seconds():.1f}s/{self.max_runtime_min * 60.0:.1f}s)."
            )

    def consume(self, calls: int = 1) -> None:
        for _ in range(max(1, calls)):
            self.check()
            self.calls_made += 1


async def call_structured(
    prompt: str,
    schema: Type[Any],
    guard: QuotaGuard,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    guard.consume(1)
    raw = await robust_llm_call(prompt, structured_output=schema, temperature=temperature)
    data = as_dict(raw)
    if hasattr(schema, "model_validate"):
        parsed = schema.model_validate(data)
        return parsed.model_dump()
    parsed = schema(**data)
    return parsed.dict()


async def call_text(prompt: str, guard: QuotaGuard, temperature: float = 0.0) -> str:
    guard.consume(1)
    raw = await robust_llm_call(prompt, temperature=temperature)
    return str(raw).strip()


async def gather_limited(
    items: Sequence[Any],
    worker,
    concurrency: int,
) -> List[Any]:
    sem = asyncio.Semaphore(max(1, concurrency))
    results: List[Any] = [None] * len(items)

    async def run_one(idx: int, item: Any) -> None:
        async with sem:
            results[idx] = await worker(idx, item)

    tasks = [asyncio.create_task(run_one(i, item)) for i, item in enumerate(items)]
    for task in asyncio.as_completed(tasks):
        await task
    return results


GENDER_EN = {"Мужской": "Male", "Женский": "Female"}
MARITAL_EN = {
    "Женат": "Married",
    "Замужем": "Married",
    "Не женат": "Single",
    "Не замужем": "Single",
    "Разведен": "Divorced",
    "Разведена": "Divorced",
    "Вдовец": "Widowed",
    "Вдова": "Widowed",
}
EDUCATION_EN = {
    "среднего профессионального образования": "vocational secondary",
    "магистратура": "master's degree",
    "бакалавриат": "bachelor's degree",
    "аспирантура": "postgraduate degree",
    "специалитет": "specialist degree",
}
REGION_EN = {
    "Регион миллионник": "large metropolitan region",
    "Москва": "Moscow",
    "Санкт-Петербург": "Saint Petersburg",
}
INCOME_EN = {
    "Низкий": "low income",
    "Выше МРОТ": "lower-middle income",
    "Средний": "middle income",
    "Выше_среднего": "upper-middle income",
    "Высокий": "high income",
    "Очень_высокий": "very high income",
    "Ultima": "ultra high income",
}
OCEAN_EN = {
    "умеренно открыт новому": "moderately open to new experiences",
    "открыт новому опыту и идеям": "open to new experiences and ideas",
    "обычно ответственен и организован": "usually responsible and organized",
    "дисциплинирован, пунктуален, ответственен": "disciplined, punctual, and responsible",
    "очень организован, планирует всё до деталей": "highly organized and detail-oriented",
    "склонен к одиночеству, избегает социума": "introverted and avoids social interaction",
    "умеренно общителен, зависит от настроения": "moderately social depending on mood",
    "общительный, энергичный, легко находит друзей": "sociable, energetic, and makes friends easily",
    "очень общителен, постоянно ищет социального взаимодействия": "highly extroverted and seeks social interaction",
    "альтруист, эмпатичен, готов помочь": "altruistic, empathic, and helpful",
    "очень эмпатичен, ставит нужды других выше своих": "highly empathic and prioritizes others' needs",
    "обычно спокоен, но может реагировать на сложности": "usually calm but can react under stress",
}


def _income_to_en(value: Any) -> str:
    if value is None:
        return "unknown income"
    text = str(value)
    for key, mapped in INCOME_EN.items():
        if key in text:
            return mapped
    return text


def translate_persona_to_en(profile: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(profile)
    if "gender" in out:
        out["gender"] = GENDER_EN.get(str(out["gender"]), out["gender"])
    if "marital_status" in out:
        out["marital_status"] = MARITAL_EN.get(str(out["marital_status"]), out["marital_status"])
    if "education" in out:
        out["education"] = EDUCATION_EN.get(str(out["education"]), out["education"])
    if "region" in out:
        out["region"] = REGION_EN.get(str(out["region"]), out["region"])
    if "income_level" in out:
        out["income_level"] = _income_to_en(out["income_level"])
    if "children_group" in out:
        out["children_group"] = str(out["children_group"])
    for key in ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"):
        if key in out:
            out[key] = OCEAN_EN.get(str(out[key]), out[key])
    return out
