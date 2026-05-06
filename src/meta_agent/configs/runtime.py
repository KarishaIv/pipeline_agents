"""Runtime configuration for meta-agent: limits, paths, model names, and Qdrant settings.

All values may be overridden via environment variables.
"""

import os
from pathlib import Path

from config import PROJECT_ROOT

# Iteration limits
MAX_SUPERVISOR_ITERATIONS = 10
MAX_AGENT_ITERATIONS = 20
MAX_HISTORY_CHARS = 10_000
MAX_DELEGATED_ATTEMPTS = 6
SUMMARY_RECENT_MESSAGES = 5
HISTORY_SUMMARY_MAX_TOKENS = 3000

# Qdrant configuration (overridable via env)
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

# Paths and timeouts
CHARTS_DIR: Path = PROJECT_ROOT / "data" / "charts"
CHECKPOINT_DB_PATH: Path = PROJECT_ROOT / "data" / "meta_agent_checkpoints.sqlite3"
CODE_TIMEOUT = 30

# LLM models (BIG_MODEL is used for code_writer)
LLM_MODEL = "aliceai-llm"
BIG_MODEL = "aliceai-llm"
HISTORY_SUMMARY_MODEL = "aliceai-llm"
