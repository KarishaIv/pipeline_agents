import json
from pathlib import Path
from datetime import datetime
import asyncio
from typing import Any, Dict, Iterable, Optional, List
import logging
import pandas as pd
from pandas import DataFrame
from pydantic.dataclasses import dataclass

logger = logging.getLogger(__name__)


class StorageManager:
    """
    Асинхронный менеджер сохранения результатов.
    - save_json_async: сохраняет dict -> файл (в executor)
    - save_stream: дозапись логов/стримов
    - save_batch: сохраняет набор результатов агрегированно
    - append_parquet_async: добавляет данные в parquet файл
    """

    @staticmethod
    async def save_json_async(obj: Any, path: Path, ensure_ascii: bool = False, indent: int = 2) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        loop = asyncio.get_event_loop()
        data = json.dumps(obj, ensure_ascii=ensure_ascii, indent=indent)
        await loop.run_in_executor(None, StorageManager._write_text_sync, path, data, "w", "utf-8")
        logger.debug(f"Saved JSON -> {path}")

    @staticmethod
    def _write_text_sync(path: Path, data: str, mode: str = "w", encoding: str = "utf-8"):
        with open(path, mode, encoding=encoding) as f:
            f.write(data)

    @staticmethod
    async def append_line_async(path: Path, line: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, StorageManager._append_line_sync, path, line)

    @staticmethod
    def _append_line_sync(path: Path, line: str):
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    @staticmethod
    async def save_batch(results: Iterable[Dict], out_dir: Path, prefix: str = "batch", per_file: int = 100):
        out_dir.mkdir(parents=True, exist_ok=True)
        results = list(results)
        total = len(results)
        for i in range(0, total, per_file):
            batch = results[i: i + per_file]
            filename = out_dir / f"{prefix}_{i // per_file + 1}.json"
            await StorageManager.save_json_async(batch, filename)
            logger.info(f"Saved batch {i // per_file + 1} ({len(batch)} items) -> {filename}")

    @staticmethod
    async def append_parquet_async(data: list[Dict] | DataFrame, path: Path, check_columns: bool = True) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, StorageManager._append_parquet_sync, data, path, check_columns)
        logger.debug(f"Saved Parquet -> {path}")

    @staticmethod
    def _append_parquet_sync(new_data: list[Dict] | DataFrame, path: Path, check_columns: bool = True):
        # Приведение к DataFrame
        if not new_data:
            return
        if isinstance(new_data, list):
            if not isinstance(new_data[0], dict):
                raise TypeError(f"Expected list[dict], got list[{type(new_data[0]).__name__}]")
            new_data = pd.DataFrame(new_data)
        elif not isinstance(new_data, pd.DataFrame):
            raise TypeError(f"Expected list[dict] or DataFrame, got {type(new_data).__name__}")

        # Слияние данных
        if path.exists():
            existing_data = pd.read_parquet(path)

            if check_columns:
                missing = set(existing_data.columns) ^ set(new_data.columns)
                assert not missing, (
                    f"Column mismatch: existing={sorted(existing_data.columns)}, "
                    f"new={sorted(new_data.columns)}, diff={sorted(missing)}"
                )
            new_data = pd.concat([existing_data, new_data])

        # Deduplicate by UUID, keeping the last occurrence (most recent write wins)
        if "UUID" in new_data.columns:
            before = len(new_data)
            new_data = new_data.drop_duplicates(subset=["UUID"], keep="last")
            dropped = before - len(new_data)
            if dropped:
                logger.debug("Dropped %d duplicate UUID(s) in %s", dropped, path)

        # Сохранение данных
        new_data.to_parquet(path, index=False)