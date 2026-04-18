"""Конфигурационные константы пайплайна meta_agent."""

import os
from pathlib import Path

from config import PROJECT_ROOT

MAX_SUPERVISOR_ITERATIONS = 10
MAX_AGENT_ITERATIONS = 20
MAX_HISTORY_CHARS = 12_000
MAX_DELEGATED_ATTEMPTS = 6

# Qdrant configuration (overridable via env vars for deployment flexibility)
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

# Charts and code execution
CHARTS_DIR: Path = PROJECT_ROOT / "charts"
CODE_TIMEOUT = 30
