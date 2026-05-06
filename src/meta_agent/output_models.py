"""Структурированные модели результатов и артефактов meta-agent.

Задаёт контракт пользовательских результатов (TextOutput, JsonOutput,
ImageOutput, FileOutput) и внутренней метаинформации артефактов.
"""

from typing import Any, Optional, Literal
from pydantic import BaseModel, Field
from uuid import uuid4


class TextOutput(BaseModel):
    """Текстовый результат ответа."""

    type: Literal["text"] = Field(default="text", description="Идентификатор типа результата")
    text: str = Field(..., description="Содержимое текстового ответа")


class JsonOutput(BaseModel):
    """JSON-результат со структурированными данными."""

    type: Literal["json"] = Field(default="json", description="Идентификатор типа результата")
    data: dict[str, Any] = Field(..., description="Структурированные JSON-данные")
    caption: Optional[str] = Field(
        default=None, description="Опциональная подпись или метаданные для JSON"
    )


class ImageOutput(BaseModel):
    """Результат-изображение: график, визуализация и т.п."""

    type: Literal["image"] = Field(default="image", description="Идентификатор типа результата")
    url: str = Field(..., description="URL изображения, отдаваемого API")
    caption: Optional[str] = Field(default=None, description="Подпись или заголовок изображения")
    alt_text: Optional[str] = Field(default=None, description="Альтернативный текст для доступности")
    mime_type: str = Field(default="image/png", description="MIME-тип изображения")


class FileOutput(BaseModel):
    """Файловый результат: CSV, PDF и другие документы."""

    type: Literal["file"] = Field(default="file", description="Идентификатор типа результата")
    filename: str = Field(..., description="Рекомендуемое имя файла")
    mime_type: str = Field(..., description="MIME-тип файла, например application/pdf")
    download_url: str = Field(
        ...,
        description="URL для скачивания файла с API-сервера",
    )
    caption: Optional[str] = Field(
        default=None, description="Опциональная подпись или описание"
    )


# Union-тип для всех вариантов результата.
AgentOutput = TextOutput | JsonOutput | ImageOutput | FileOutput


class AgentArtifact(BaseModel):
    """Внутренняя метаинформация созданного файлового артефакта."""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Уникальный идентификатор артефакта")
    kind: Literal["chart", "csv", "pdf", "data", "file"] = Field(
        description="Тип артефакта: chart, csv, pdf, data или file"
    )
    path: str = Field(description="Локальный путь к файлу артефакта")
    filename: str = Field(description="Рекомендуемое имя файла для скачивания или показа")
    mime_type: str = Field(description="MIME-тип артефакта")
    caption: Optional[str] = Field(
        default=None, description="Опциональная подпись или заголовок артефакта"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Дополнительные метаданные: тип графика, размеры, имя DTO и т.п.",
    )
