import logging
import os

PIPELINE_PATH = os.getenv("PIPELINE_PATH", "/Users/msamorodova/pipeline_agents")
NEWS_SYSTEM_PATH = os.getenv("NEWS_SYSTEM_PATH", "/Users/msamorodova/Downloads/pipeline_agents-multiagent_system_for_context/multi_agent_rag")

TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "YOUR_BOT_TOKEN")
YANDEX_FOLDER_ID = os.getenv("YANDEX_FOLDER_ID", "YOUR_FOLDER_ID")
YANDEX_API_KEY = os.getenv("YANDEX_API_KEY", "YOUR_API_KEY")
YANDEX_GPT_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
YANDEX_MODEL_URI = f"gpt://{YANDEX_FOLDER_ID}/yandexgpt/latest"

ANALYSIS_TIMEOUT = 900
META_AGENT_URL = os.getenv("META_AGENT_URL", "http://localhost:8000")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
