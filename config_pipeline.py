import os

YANDEX_GPT_CONFIG = {
    "folder_id": "",
    "api_key": "",
    "model_uri": "",
}

def set_yandex_config(folder_id: str, api_key: str):
    """Set YandexGPT credentials (Api-Key auth)"""
    YANDEX_GPT_CONFIG["folder_id"] = folder_id
    YANDEX_GPT_CONFIG["api_key"] = api_key
    YANDEX_GPT_CONFIG["model_uri"] = f"gpt://{folder_id}/yandexgpt/latest"
    os.environ["YANDEX_FOLDER_ID"] = folder_id
    os.environ["YANDEX_API_KEY"] = api_key
