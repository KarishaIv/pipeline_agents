"""Сервисы мета-агента.

Содержит бизнес-логику, не связанную напрямую с инструментами LangGraph (например, клиент Qdrant).
"""

from .qdrant import QdrantService

__all__ = ["QdrantService"]
