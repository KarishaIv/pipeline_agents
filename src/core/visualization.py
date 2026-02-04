from pathlib import Path
import asyncio
import logging
from typing import Iterable, Dict, Any
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

class EmotionalVisualizer:
    """
    Класс для визуализации эмоциональных динамик агентов.

    Основная логика синхронной визуализации реализована в приватных методах этого класса.
    Публичный асинхронный метод plot_async позволяет строить график и сохранять его в HTML
    при помощи выполнения визуализации в отдельном потоке (executor).

    Методы:
      - _build_figure: собирает интерактивный график динамики эмоций для набора агентов.
      - _save_html: сохраняет построенный график в HTML-файл по заданному пути.
      - plot_async: асинхронно строит график и сохраняет его, чтобы не блокировать основной поток выполнения.
    """

    @staticmethod
    def _build_figure(viz_data: Iterable[Dict[str, Any]]):
        fig = go.Figure()

        groups = {}
        for item in viz_data:
            key = (item.get("agent", "Persona"), item.get("emotion"))
            groups.setdefault(key, []).append(item)
        for (agent, emo), rows in groups.items():
            x = [r["step"] for r in rows]
            y = [r["intensity"] for r in rows]
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name=f"{agent} — {emo}", line=dict(shape="spline")))
        fig.update_layout(title="Emotional dynamics", xaxis_title="Step", yaxis_title="Intensity", template="plotly_white", hovermode="x unified")
        return fig

    @staticmethod
    def _save_html(fig, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(path))
        logger.debug(f"Saved visualization -> {path}")

    @staticmethod
    async def plot_async(viz_data: Iterable[Dict[str, Any]], save_path: Path):
        loop = asyncio.get_event_loop()
        fig = await loop.run_in_executor(None, EmotionalVisualizer._build_figure, list(viz_data))
        await loop.run_in_executor(None, EmotionalVisualizer._save_html, fig, save_path)
