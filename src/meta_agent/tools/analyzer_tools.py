"""
Инструменты агента-аналитика:
  - ComputeStatsTool   — описательная статистика по DTO
  - CreateChartTool    — построение графиков matplotlib по DTO
  - SummarizeTextsTool — резюме и инсайты по DTO через языковую модель
"""

from __future__ import annotations

import json
import logging
import traceback
from typing import List, Literal, TYPE_CHECKING

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pydantic import Field

from sgr_agent_core.base_tool import BaseTool
from config import get_model_uri
from src.utils import make_openai_client
from src.meta_agent.configs import CHARTS_DIR, SUMMARIZE_TEXTS_TEMPERATURE
from src.meta_agent.tools.dto_tools import dto_summary_view, resolve_dto_or_error
from src.meta_agent.utils.json_responses import json_error, serialize_tool_result
from src.meta_agent.output_models import AgentArtifact

if TYPE_CHECKING:
    from sgr_agent_core.models import AgentContext
    from sgr_agent_core.agent_definition import AgentConfig

logger = logging.getLogger("meta_agent.analyzer")


# ---------------------------------------------------------------------------
# Инструменты
# ---------------------------------------------------------------------------

class ComputeStatsTool(BaseTool):
    """Описательная статистика по DTO: среднее, медиана, разброс, квартили, асимметрия, эксцесс, пропуски, корреляции."""

    tool_name = "compute_stats"
    description = (
        "Посчитать описательную статистику по данным DTO: среднее, медиана, стандартное отклонение, "
        "квартили, асимметрия, эксцесс, число пропусков и попарные корреляции числовых столбцов."
    )

    reasoning: str = Field(description="Зачем нужны эти показатели")
    dto_name: str = Field(description="Имя DTO для анализа")
    columns: List[str] = Field(
        default_factory=list,
        description=(
            "Какие числовые столбцы анализировать. "
            "Пустой список — все числовые столбцы."
        ),
    )

    async def __call__(self, context: "AgentContext", config: "AgentConfig", **_) -> str:
        df, dto_payload, error = resolve_dto_or_error(context, self.dto_name)
        if error:
            return error
        assert df is not None and dto_payload is not None

        try:
            if self.columns:
                df = df[[c for c in self.columns if c in df.columns]]

            numeric = df.select_dtypes(include="number")
            if numeric.empty:
                return json_error("Числовые столбцы не найдены", error_type="validation_error")

            result = {
                "dto": dto_summary_view(self.dto_name, dto_payload),
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
            return serialize_tool_result(result)

        except Exception as exc:
            return json_error(str(exc), error_type="computation_error")


class CreateChartTool(BaseTool):
    """Построить и сохранить график matplotlib по данным DTO; вернуть путь к PNG."""

    tool_name = "create_chart"
    description = (
        "Построить matplotlib-график по данным DTO, сохранить PNG через ChartService "
        "и зарегистрировать созданный файл как артефакт агента."
    )

    reasoning: str = Field(description="Что показывает график и зачем он нужен")
    chart_type: Literal["bar", "line", "scatter", "histogram", "pie", "box", "heatmap"] = Field(
        description="Тип графика (bar, line, scatter, histogram, pie, box, heatmap)"
    )
    dto_name: str = Field(description="Имя DTO, по которому строится график")
    title: str = Field(description="Заголовок графика")
    x_column: str = Field(default="", description="Столбец для оси X или подписей; пусто — выбрать автоматически")
    y_column: str = Field(default="", description="Столбец для оси Y или значений; пусто — выбрать автоматически")
    x_label: str = Field(default="", description="Подпись оси X")
    y_label: str = Field(default="", description="Подпись оси Y")

    def _validate_chart_inputs(self, df: pd.DataFrame) -> str | None:
        """Проверить входные данные графика; вернуть JSON-ошибку или None."""
        if df.empty:
            return json_error("DTO пуст или содержит 0 строк", error_type="validation_error")

        numeric_cols = set(df.select_dtypes(include="number").columns)

        if self.chart_type == "scatter":
            if len(numeric_cols) < 2 and (not self.x_column or not self.y_column):
                return json_error(
                    "Для scatter нужны минимум два числовых столбца или явно указанные x_column и y_column",
                    error_type="validation_error",
                )
            if self.x_column and self.x_column not in df.columns:
                return json_error(
                    f"Столбец '{self.x_column}' не найден в DTO",
                    error_type="validation_error",
                )
            if self.y_column and self.y_column not in df.columns:
                return json_error(
                    f"Столбец '{self.y_column}' не найден в DTO",
                    error_type="validation_error",
                )

        elif self.chart_type == "histogram":
            if not numeric_cols and not self.x_column:
                return json_error(
                    "Для histogram нужен числовой столбец",
                    error_type="validation_error",
                )
            if self.x_column and self.x_column not in df.columns:
                return json_error(
                    f"Столбец '{self.x_column}' не найден в DTO",
                    error_type="validation_error",
                )

        elif self.chart_type == "bar" or self.chart_type == "line":
            if self.x_column and self.x_column not in df.columns:
                return json_error(
                    f"Столбец '{self.x_column}' не найден в DTO",
                    error_type="validation_error",
                )
            if self.y_column and self.y_column not in df.columns:
                return json_error(
                    f"Столбец '{self.y_column}' не найден в DTO",
                    error_type="validation_error",
                )

        elif self.chart_type == "pie":
            if len(df.columns) < 1:
                return json_error(
                    "DTO должен иметь минимум один столбец для pie",
                    error_type="validation_error",
                )
            label_col = self.x_column if self.x_column else df.columns[0]
            val_col = self.y_column if self.y_column else (df.columns[1] if len(df.columns) > 1 else df.columns[0])
            if label_col not in df.columns:
                return json_error(
                    f"Столбец '{label_col}' не найден в DTO",
                    error_type="validation_error",
                )
            if val_col not in df.columns:
                return json_error(
                    f"Столбец '{val_col}' не найден в DTO",
                    error_type="validation_error",
                )
            if val_col not in numeric_cols:
                return json_error(
                    f"Столбец '{val_col}' должен содержать числовые значения для pie графика",
                    error_type="validation_error",
                )

        elif self.chart_type == "box":
            if not numeric_cols:
                return json_error(
                    "Для box plot нужны числовые столбцы",
                    error_type="validation_error",
                )
            if self.y_column and self.y_column not in numeric_cols:
                return json_error(
                    f"Столбец '{self.y_column}' должен содержать числовые значения для box plot",
                    error_type="validation_error",
                )

        elif self.chart_type == "heatmap":
            if len(numeric_cols) < 2:
                return json_error(
                    "Для heatmap нужны минимум два числовых столбца для корреляционной матрицы",
                    error_type="validation_error",
                )

        return None

    async def __call__(self, context: "AgentContext", config: "AgentConfig", **_) -> str:
        df, dto_payload, error = resolve_dto_or_error(context, self.dto_name)
        if error:
            return error
        assert df is not None and dto_payload is not None

        if validation_error := self._validate_chart_inputs(df):
            return validation_error

        CHARTS_DIR.mkdir(parents=True, exist_ok=True)

        try:
            from src.meta_agent.services.chart import ChartService

            chart_service = ChartService(CHARTS_DIR)
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
                    numeric_cols = list(df.select_dtypes(include="number").columns)
                    x_col = self.x_column or numeric_cols[0]
                    y_col = self.y_column or numeric_cols[1]
                    ax.scatter(df[x_col], df[y_col], alpha=0.7)

                case "histogram":
                    numeric_cols = list(df.select_dtypes(include="number").columns)
                    col = self.x_column or numeric_cols[0]
                    ax.hist(df[col].dropna(), bins=20, edgecolor="white")

                case "pie":
                    label_col = self.x_column or df.columns[0]
                    val_col = self.y_column or (df.columns[1] if len(df.columns) > 1 else df.columns[0])
                    ax.pie(df[val_col], labels=df[label_col], autopct="%1.1f%%", startangle=90)

                case "box":
                    cols = [self.y_column] if self.y_column else list(df.select_dtypes(include="number").columns)
                    ax.boxplot([df[c].dropna().tolist() for c in cols], tick_labels=cols)

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

            # Сохраняем через сервис, чтобы получить единый формат метаданных.
            chart_path, artifact_metadata = chart_service.save_chart()

            if not hasattr(context, 'custom_context'):
                context.custom_context = {}
            if 'artifacts' not in context.custom_context:
                context.custom_context['artifacts'] = []

            artifact_metadata['chart_type'] = self.chart_type
            artifact_metadata['dto_name'] = self.dto_name
            agent_artifact = AgentArtifact(
                id=artifact_metadata['id'],
                kind=artifact_metadata['kind'],
                path=artifact_metadata['path'],
                filename=artifact_metadata['filename'],
                mime_type=artifact_metadata['mime_type'],
                caption=self.title,
                metadata=artifact_metadata['metadata'] | {'chart_type': self.chart_type, 'dto_name': self.dto_name}
            )

            context.custom_context['artifacts'].append(agent_artifact)

            return serialize_tool_result({
                "chart_saved": chart_path,
                "title": self.title,
                "type": self.chart_type,
                "dto": dto_summary_view(self.dto_name, dto_payload),
            })

        except Exception as exc:
            plt.close("all")
            return json_error(
                str(exc),
                error_type="chart_rendering_error",
                details={"traceback": traceback.format_exc()},
            )


class SummarizeTextsTool(BaseTool):
    """Краткое изложение или извлечение инсайтов из списка текстов через языковую модель.

    Удобно для рассуждений агентов, ответов опроса, длинных записей и любых текстовых массивов.
    """

    tool_name = "summarize_texts"
    description = (
        "Извлечь тексты из DTO, отправить их в языковую модель с инструкцией и получить "
        "краткое резюме или список инсайтов. "
        "Если text_columns не заданы, используются object-колонки; если их нет, строки DTO передаются как JSON. "
        "Вызывай по dto_name, а не передавай полный массив текстов в аргументах."
    )

    reasoning: str = Field(description="Зачем нужно резюме и какие инсайты извлечь")
    dto_name: str = Field(
        description="Имя DTO, из которого извлекаются тексты",
    )
    text_columns: List[str] = Field(
        default_factory=list,
        description=(
            "Список колонок DTO, содержащих текст. "
            "Если пусто, берутся object-колонки; если их нет, используется JSON каждой строки."
        ),
    )
    max_items: int = Field(
        default=200,
        ge=1,
        le=2000,
        description="Ограничение на количество строк DTO для отправки в LLM.",
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
        df, dto_payload, error = resolve_dto_or_error(context, self.dto_name)
        if error:
            return error
        assert df is not None and dto_payload is not None

        columns = [col for col in self.text_columns if col in df.columns]
        if not columns:
            columns = [col for col in df.columns if df[col].dtype == "object"]

        texts: list[str] = []
        limited_df = df.head(self.max_items)
        if columns:
            for _, row in limited_df.iterrows():
                values = [
                    str(row.get(col, "")).strip()
                    for col in columns
                    if str(row.get(col, "")).strip() not in ("", "nan", "None")
                ]
                if values:
                    texts.append("\n".join(values))
        else:
            for _, row in limited_df.iterrows():
                row_json = json.dumps(row.to_dict(), ensure_ascii=False, default=str)
                if row_json.strip():
                    texts.append(row_json)

        if not texts:
            return json_error(
                f"В DTO '{self.dto_name}' не найдено текстов для резюме",
                error_type="validation_error",
            )

        numbered = "\n\n".join(
            f"[{i + 1}] {t.strip()}" for i, t in enumerate(texts) if t.strip()
        )
        user_message = (
            f"{self.instruction}\n\n"
            f"Тексты (всего {len(texts)}):\n\n{numbered}"
        )

        try:
            client = make_openai_client()
            response = await client.chat.completions.create(
                model=get_model_uri(),
                messages=[{"role": "user", "content": user_message}],
                max_tokens=self.max_tokens,
                temperature=SUMMARIZE_TEXTS_TEMPERATURE,
            )
            summary = response.choices[0].message.content or ""
            return serialize_tool_result({
                "summary": summary,
                "texts_count": len(texts),
                "dto": dto_summary_view(self.dto_name, dto_payload, 50),
                "text_columns": columns,
                "instruction": self.instruction,
            })
        except Exception as exc:
            logger.warning("SummarizeTextsTool завершился ошибкой: %s", exc)
            return json_error(str(exc), error_type="llm_error")
