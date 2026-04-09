"""
Инструменты агента-аналитика:
  - ComputeStatsTool   — описательная статистика по JSON
  - ExecuteCodeTool    — безопасное выполнение Python в песочнице
  - CreateChartTool    — построение графиков matplotlib
  - SummarizeTextsTool — резюме и инсайты через языковую модель
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import math
import os
import statistics as _stats
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import List, Literal, TYPE_CHECKING

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openai import AsyncOpenAI
from pydantic import Field

from sgr_agent_core.base_tool import BaseTool
from config import PROJECT_ROOT, YANDEX_BASE_URL, get_model_uri

if TYPE_CHECKING:
    from sgr_agent_core.models import AgentContext
    from sgr_agent_core.agent_definition import AgentConfig

logger = logging.getLogger("meta_agent.analyzer")

CHARTS_DIR = PROJECT_ROOT / "charts"
CODE_TIMEOUT = 30


# ---------------------------------------------------------------------------
# Песочница для ExecuteCodeTool
# ---------------------------------------------------------------------------

_SAFE_BUILTINS: dict = {
    "print": print,
    "range": range,
    "len": len,
    "sum": sum,
    "min": min,
    "max": max,
    "abs": abs,
    "round": round,
    "sorted": sorted,
    "enumerate": enumerate,
    "zip": zip,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "type": type,
    "isinstance": isinstance,
    "hasattr": hasattr,
    "getattr": getattr,
    "repr": repr,
    "format": format,
    "map": map,
    "filter": filter,
    "any": any,
    "all": all,
    "iter": iter,
    "next": next,
    "reversed": reversed,
    "vars": vars,
    "Exception": Exception,
    "ValueError": ValueError,
    "KeyError": KeyError,
    "TypeError": TypeError,
}


def _make_sandbox(stdout_buf: io.StringIO, saved_charts: list) -> dict:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    def _save_chart(filename: str | None = None) -> str:
        name = filename or f"chart_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
        path = str(CHARTS_DIR / name)
        plt.savefig(path, bbox_inches="tight", dpi=150)
        plt.close()
        saved_charts.append(path)
        return path

    builtins_patched = {
        **_SAFE_BUILTINS,
        "print": lambda *a, **kw: print(*a, **kw, file=stdout_buf),
    }

    return {
        "__builtins__": builtins_patched,
        "np": np,
        "pd": pd,
        "plt": plt,
        "math": math,
        "json": json,
        "stats": _stats,
        "save_chart": _save_chart,
    }


def _run_code(code: str) -> tuple[str, str]:
    stdout_buf = io.StringIO()
    saved_charts: list = []
    namespace = _make_sandbox(stdout_buf, saved_charts)
    try:
        exec(compile(code, "<analyzer>", "exec"), namespace)  # noqa: S102
        output = stdout_buf.getvalue()
        if saved_charts:
            output += f"\nСохранённые графики: {', '.join(saved_charts)}"
        return output.strip(), ""
    except Exception:
        return stdout_buf.getvalue().strip(), traceback.format_exc()


async def _execute_safely(code: str) -> tuple[str, str]:
    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(executor, _run_code, code),
            timeout=CODE_TIMEOUT,
        )
    except asyncio.TimeoutError:
        return "", f"Превышено время выполнения ({CODE_TIMEOUT} с)"
    finally:
        executor.shutdown(wait=False)


# ---------------------------------------------------------------------------
# Инструменты
# ---------------------------------------------------------------------------

class ComputeStatsTool(BaseTool):
    """Описательная статистика по JSON: среднее, медиана, разброс, квартили, асимметрия, эксцесс, пропуски, корреляции."""

    tool_name = "compute_stats"
    description = (
        "Посчитать описательную статистику по JSON-данным: среднее, медиана, стандартное отклонение, "
        "квартили, асимметрия, эксцесс, число пропусков и попарные корреляции числовых столбцов."
    )

    reasoning: str = Field(description="Зачем нужны эти показатели")
    data_json: str = Field(
        description="JSON-массив чисел или массив объектов (записей)"
    )
    columns: List[str] = Field(
        default=[],
        description=(
            "Какие числовые столбцы анализировать, если данные — массив объектов. "
            "Пустой список — все числовые столбцы."
        ),
    )

    async def __call__(self, context: "AgentContext", config: "AgentConfig", **_) -> str:
        try:
            raw = json.loads(self.data_json)
        except json.JSONDecodeError as exc:
            return json.dumps({"error": f"Некорректный JSON: {exc}"}, ensure_ascii=False)

        try:
            if raw and isinstance(raw[0], dict):
                df = pd.DataFrame(raw)
            else:
                df = pd.DataFrame({"value": raw})

            if self.columns:
                df = df[[c for c in self.columns if c in df.columns]]

            numeric = df.select_dtypes(include="number")
            if numeric.empty:
                return json.dumps({"error": "Числовые столбцы не найдены"}, ensure_ascii=False)

            result = {
                "describe": numeric.describe().round(4).to_dict(),
                "skewness": numeric.skew().round(4).to_dict(),
                "kurtosis": numeric.kurtosis().round(4).to_dict(),
                "null_counts": numeric.isnull().sum().to_dict(),
                "correlation": (
                    numeric.corr().round(4).to_dict()
                    if len(numeric.columns) > 1
                    else {}
                ),
            }
            return json.dumps(result, ensure_ascii=False, default=str)

        except Exception as exc:
            return json.dumps({"error": str(exc)}, ensure_ascii=False)


class ExecuteCodeTool(BaseTool):
    """Безопасное выполнение Python-кода для произвольного анализа.

    В песочнице: np (numpy), pd (pandas), plt (matplotlib), math, json, stats, save_chart().
    Вызов save_chart('имя.png') сохраняет текущий график matplotlib.
    Запрещены произвольный доступ к файлам и внешние import.
    """

    tool_name = "execute_code"
    description = (
        "Написать и безопасно выполнить код на Python для анализа. "
        "В песочнице доступны: np (numpy), pd (pandas), plt (matplotlib), math, json, stats, save_chart(). "
        "Нельзя читать/писать файлы и подключать внешние модули."
    )

    reasoning: str = Field(description="Что вычисляет код и зачем")
    code: str = Field(
        description=(
            "Корректный код на Python. "
            "Уже импортированы: np (numpy), pd (pandas), plt (matplotlib), math, json, stats. "
            "Для сохранения графика вызови save_chart('файл.png'). "
            "Вывод — через print(); stdout возвращается в ответе инструмента."
        )
    )

    async def __call__(self, context: "AgentContext", config: "AgentConfig", **_) -> str:
        stdout, error = await _execute_safely(self.code)
        result: dict = {}
        if stdout:
            result["output"] = stdout
        if error:
            result["error"] = error
        if not result:
            result["output"] = "(нет вывода)"
        return json.dumps(result, ensure_ascii=False)


class CreateChartTool(BaseTool):
    """Построить и сохранить график matplotlib по JSON-данным; вернуть путь к PNG."""

    tool_name = "create_chart"
    description = (
        "Построить и сохранить график matplotlib (bar, line, scatter, histogram, pie, box, heatmap) по JSON-данным."
    )

    reasoning: str = Field(description="Что показывает график и зачем он нужен")
    chart_type: Literal["bar", "line", "scatter", "histogram", "pie", "box", "heatmap"] = Field(
        description="Тип графика (bar, line, scatter, histogram, pie, box, heatmap)"
    )
    data_json: str = Field(
        description="JSON-массив чисел или массив объектов (записей)"
    )
    title: str = Field(description="Заголовок графика")
    x_column: str = Field(default="", description="Столбец для оси X / подписей")
    y_column: str = Field(default="", description="Столбец для оси Y / значений")
    x_label: str = Field(default="", description="Подпись оси X")
    y_label: str = Field(default="", description="Подпись оси Y")

    async def __call__(self, context: "AgentContext", config: "AgentConfig", **_) -> str:
        try:
            raw = json.loads(self.data_json)
        except json.JSONDecodeError as exc:
            return json.dumps({"error": f"Некорректный JSON: {exc}"}, ensure_ascii=False)

        CHARTS_DIR.mkdir(parents=True, exist_ok=True)

        try:
            df = pd.DataFrame(raw) if raw and isinstance(raw[0], dict) else pd.DataFrame({"value": raw})

            fig, ax = plt.subplots(figsize=(10, 6))

            match self.chart_type:
                case "bar":
                    if self.x_column and self.y_column:
                        ax.bar(df[self.x_column].astype(str), df[self.y_column])
                    else:
                        df.select_dtypes(include="number").plot(kind="bar", ax=ax)

                case "line":
                    if self.x_column and self.y_column:
                        ax.plot(df[self.x_column], df[self.y_column], marker="o")
                    else:
                        df.select_dtypes(include="number").plot(kind="line", ax=ax)

                case "scatter":
                    x_col = self.x_column or df.select_dtypes(include="number").columns[0]
                    y_col = self.y_column or df.select_dtypes(include="number").columns[1]
                    ax.scatter(df[x_col], df[y_col], alpha=0.7)

                case "histogram":
                    col = self.x_column or df.select_dtypes(include="number").columns[0]
                    ax.hist(df[col].dropna(), bins=20, edgecolor="white")

                case "pie":
                    label_col = self.x_column or df.columns[0]
                    val_col = self.y_column or (df.columns[1] if len(df.columns) > 1 else df.columns[0])
                    ax.pie(df[val_col], labels=df[label_col], autopct="%1.1f%%", startangle=90)

                case "box":
                    cols = [self.y_column] if self.y_column else list(df.select_dtypes(include="number").columns)
                    ax.boxplot([df[c].dropna().tolist() for c in cols], labels=cols)

                case "heatmap":
                    numeric = df.select_dtypes(include="number")
                    corr = numeric.corr()
                    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
                    ax.set_xticks(range(len(corr.columns)))
                    ax.set_yticks(range(len(corr.columns)))
                    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
                    ax.set_yticklabels(corr.columns)
                    plt.colorbar(im, ax=ax)

            ax.set_title(self.title)
            if self.x_label:
                ax.set_xlabel(self.x_label)
            if self.y_label:
                ax.set_ylabel(self.y_label)

            plt.tight_layout()
            filename = f"chart_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
            path = str(CHARTS_DIR / filename)
            plt.savefig(path, bbox_inches="tight", dpi=150)
            plt.close(fig)

            return json.dumps(
                {"chart_saved": path, "title": self.title, "type": self.chart_type},
                ensure_ascii=False,
            )

        except Exception as exc:
            plt.close("all")
            return json.dumps(
                {"error": str(exc), "traceback": traceback.format_exc()},
                ensure_ascii=False,
            )


class SummarizeTextsTool(BaseTool):
    """Краткое изложение или извлечение инсайтов из списка текстов через языковую модель.

    Удобно для рассуждений агентов, ответов опроса, длинных записей и любых текстовых массивов.
    """

    tool_name = "summarize_texts"
    description = (
        "Отправить список текстов в языковую модель с инструкцией и получить "
        "краткое резюме или список инсайтов."
        "Применяй для рассуждений агентов, ответов респондентов и любых текстовых корпусов."
    )

    reasoning: str = Field(description="Зачем нужно резюме и какие инсайты извлечь")
    texts: List[str] = Field(
        description="Список текстов целиком для обработки (рассуждения, ответы опроса, записи и т.д.)",
        min_length=1,
    )
    instruction: str = Field(
        default=(
            "Кратко изложи тексты и выдели главные инсайты, закономерности и выводы. "
            "Отвечай на русском языке."
        ),
        description=(
            "Инструкция на естественном языке: что извлечь или как обобщить "
            "(например: «Выдели ключевые причины», «Тезисы по основным темам»)."
        ),
    )
    max_tokens: int = Field(
        default=1024,
        ge=64,
        le=4096,
        description="Максимум токенов в ответе языковой модели.",
    )

    async def __call__(self, context: "AgentContext", config: "AgentConfig", **_) -> str:
        if not self.texts:
            return json.dumps({"error": "Список текстов пуст"}, ensure_ascii=False)

        numbered = "\n\n".join(
            f"[{i + 1}] {t.strip()}" for i, t in enumerate(self.texts) if t.strip()
        )
        user_message = (
            f"{self.instruction}\n\n"
            f"Тексты (всего {len(self.texts)}):\n\n{numbered}"
        )

        try:
            client = AsyncOpenAI(
                api_key=os.getenv("YANDEX_API_KEY", ""),
                base_url=YANDEX_BASE_URL,
            )
            response = await client.chat.completions.create(
                model=get_model_uri(),
                messages=[{"role": "user", "content": user_message}],
                max_tokens=self.max_tokens,
                temperature=0.3,
            )
            summary = response.choices[0].message.content or ""
            return json.dumps(
                {
                    "summary": summary,
                    "texts_count": len(self.texts),
                    "instruction": self.instruction,
                },
                ensure_ascii=False,
            )
        except Exception as exc:
            logger.warning("SummarizeTextsTool завершился ошибкой: %s", exc)
            return json.dumps({"error": str(exc)}, ensure_ascii=False)
