"""Configuration package for meta-agent.

Unified access to runtime settings, catalog definitions, worker configurations,
and Telegram bot settings.
"""

# Runtime configuration
from src.meta_agent.configs.runtime import (
    BIG_MODEL,
    CHARTS_DIR,
    CHECKPOINT_DB_PATH,
    CODE_TIMEOUT,
    HISTORY_SUMMARY_MAX_TOKENS,
    HISTORY_SUMMARY_MODEL,
    LLM_MODEL,
    MAX_AGENT_ITERATIONS,
    MAX_DELEGATED_ATTEMPTS,
    MAX_HISTORY_CHARS,
    MAX_SUPERVISOR_ITERATIONS,
    QDRANT_HOST,
    QDRANT_PORT,
    SUMMARY_RECENT_MESSAGES,
)

# Catalog
from src.meta_agent.configs.catalog import (
    AVAILABLE_COLLECTIONS,
    COLLECTION_DESCRIPTIONS,
    COLLECTION_ENUM_DESC,
    CollectionName,
    get_collection_catalog,
)

# Workers
from src.meta_agent.configs.workers import (
    WORKER_DEFINITIONS,
    WorkerDefinition,
    WorkerName,
)

# Telegram
from src.meta_agent.configs.telegram import TelegramBotConfig

__all__ = [
    # Runtime
    "BIG_MODEL",
    "CHARTS_DIR",
    "CHECKPOINT_DB_PATH",
    "CODE_TIMEOUT",
    "HISTORY_SUMMARY_MAX_TOKENS",
    "HISTORY_SUMMARY_MODEL",
    "LLM_MODEL",
    "MAX_AGENT_ITERATIONS",
    "MAX_DELEGATED_ATTEMPTS",
    "MAX_HISTORY_CHARS",
    "MAX_SUPERVISOR_ITERATIONS",
    "QDRANT_HOST",
    "QDRANT_PORT",
    "SUMMARY_RECENT_MESSAGES",
    # Catalog
    "AVAILABLE_COLLECTIONS",
    "COLLECTION_DESCRIPTIONS",
    "COLLECTION_ENUM_DESC",
    "CollectionName",
    "get_collection_catalog",
    # Workers
    "WORKER_DEFINITIONS",
    "WorkerDefinition",
    "WorkerName",
    # Telegram
    "TelegramBotConfig",
]
