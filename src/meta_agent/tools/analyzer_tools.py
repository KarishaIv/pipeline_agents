"""
Analytical tools for the analyzer agent:
  - ComputeStatsTool   — descriptive statistics on JSON data
  - ExecuteCodeTool    — safe sandboxed Python execution
  - CreateChartTool    — matplotlib chart generation
  - SummarizeTextsTool — LLM-based summarisation / insight extraction
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
# Sandbox for ExecuteCodeTool
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
            output += f"\nSaved charts: {', '.join(saved_charts)}"
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
        return "", f"Execution timed out after {CODE_TIMEOUT}s"
    finally:
        executor.shutdown(wait=False)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

class ComputeStatsTool(BaseTool):
    """Compute descriptive statistics on a JSON dataset: mean, median, std,
    quartiles, skewness, kurtosis, null counts, and pairwise correlation."""

    tool_name = "compute_stats"
    description = (
        "Compute descriptive statistics on a JSON dataset: mean, median, std, "
        "quartiles, skewness, kurtosis, null counts, and pairwise correlation."
    )

    reasoning: str = Field(description="Why these statistics are needed")
    data_json: str = Field(
        description="JSON array of numbers OR array of objects (records)"
    )
    columns: List[str] = Field(
        default=[],
        description=(
            "Specific numeric columns to analyse when data is an array of objects. "
            "Leave empty to analyse all numeric columns."
        ),
    )

    async def __call__(self, context: "AgentContext", config: "AgentConfig", **_) -> str:
        try:
            raw = json.loads(self.data_json)
        except json.JSONDecodeError as exc:
            return json.dumps({"error": f"Invalid JSON: {exc}"}, ensure_ascii=False)

        try:
            if raw and isinstance(raw[0], dict):
                df = pd.DataFrame(raw)
            else:
                df = pd.DataFrame({"value": raw})

            if self.columns:
                df = df[[c for c in self.columns if c in df.columns]]

            numeric = df.select_dtypes(include="number")
            if numeric.empty:
                return json.dumps({"error": "No numeric columns found"}, ensure_ascii=False)

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
    """Write and safely execute Python code for custom analysis.

    Available in the sandbox: np, pd, plt, math, json, stats, save_chart().
    Call save_chart('name.png') to persist a matplotlib figure.
    No file I/O or external imports are allowed.
    """

    tool_name = "execute_code"
    description = (
        "Write and safely execute Python code for custom analysis. "
        "Available in sandbox: np, pd, plt, math, json, stats, save_chart(). "
        "No file I/O or external imports allowed."
    )

    reasoning: str = Field(description="What this code computes and why")
    code: str = Field(
        description=(
            "Valid Python code. "
            "Pre-imported: np, pd, plt, math, json, stats. "
            "Call save_chart('filename.png') to save a plot. "
            "Use print() for output — stdout is captured and returned."
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
            result["output"] = "(no output)"
        return json.dumps(result, ensure_ascii=False)


class CreateChartTool(BaseTool):
    """Create and save a matplotlib chart from a JSON dataset.
    Returns the path to the saved PNG file."""

    tool_name = "create_chart"
    description = "Create and save a matplotlib chart (bar, line, scatter, histogram, pie, box, heatmap) from a JSON dataset."

    reasoning: str = Field(description="What this chart visualises and why")
    chart_type: Literal["bar", "line", "scatter", "histogram", "pie", "box", "heatmap"] = Field(
        description="Chart type"
    )
    data_json: str = Field(
        description="JSON array of numbers OR array of objects (records)"
    )
    title: str = Field(description="Chart title")
    x_column: str = Field(default="", description="Column for x-axis / labels")
    y_column: str = Field(default="", description="Column for y-axis / values")
    x_label: str = Field(default="", description="x-axis label")
    y_label: str = Field(default="", description="y-axis label")

    async def __call__(self, context: "AgentContext", config: "AgentConfig", **_) -> str:
        try:
            raw = json.loads(self.data_json)
        except json.JSONDecodeError as exc:
            return json.dumps({"error": f"Invalid JSON: {exc}"}, ensure_ascii=False)

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
    """Summarise or extract insights from a list of texts using the LLM.

    Useful for distilling agents' reasoning logs, survey responses, textual
    records, or any collection of free-form strings into a compact summary or
    structured bullet-point insights.
    """

    tool_name = "summarize_texts"
    description = (
        "Send a list of texts to the LLM with a custom instruction and return a "
        "concise summary or bullet-point insights. "
        "Use for summarising agents' reasonings, survey answers, or any text corpus."
    )

    reasoning: str = Field(description="Why summarisation is needed and what insights to extract")
    texts: List[str] = Field(
        description="List of texts to summarise (e.g. reasoning logs, survey responses, records)",
        min_length=1,
    )
    instruction: str = Field(
        default=(
            "Summarise the provided texts and extract the most important insights, "
            "patterns, and conclusions. Reply in Russian."
        ),
        description=(
            "Natural-language instruction for the LLM describing WHAT to extract "
            "or HOW to summarise (e.g. 'Extract key reasons', 'Bullet-point main themes')."
        ),
    )
    max_tokens: int = Field(
        default=1024,
        ge=64,
        le=4096,
        description="Maximum tokens the LLM may use for its response.",
    )

    async def __call__(self, context: "AgentContext", config: "AgentConfig", **_) -> str:
        if not self.texts:
            return json.dumps({"error": "texts list is empty"}, ensure_ascii=False)

        numbered = "\n\n".join(
            f"[{i + 1}] {t.strip()}" for i, t in enumerate(self.texts) if t.strip()
        )
        user_message = (
            f"{self.instruction}\n\n"
            f"Texts ({len(self.texts)} total):\n\n{numbered}"
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
            logger.warning("SummarizeTextsTool failed: %s", exc)
            return json.dumps({"error": str(exc)}, ensure_ascii=False)
