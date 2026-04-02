"""
Telegram-бот: произвольный запрос - парсинг через YandexGPT - новостной контекст - мультиагентная симуляция - ответ пользователю.

Переменные окружения:
  TG_BOT_TOKEN  — токен бота
  YANDEX_FOLDER_ID — ID каталога в Yandex Cloud
  YANDEX_API_KEY — API-ключ YandexGPT
  PIPELINE_PATH — путь к папке с базовым пайплайном
  NEWS_SYSTEM_PATH — путь к папке с агентом новостного контекста
"""

import asyncio
import logging

from aiogram import Bot, Dispatcher
from prometheus_client import start_http_server

from config import TG_BOT_TOKEN
from handlers import router

from formatters import _format_news_block, format_result_simple, format_reasoning_message, format_news_message, _make_archive  # noqa
from keyboards import keyboard_confirm_ta, keyboard_select_count, keyboard_select_ratio, keyboard_result  # noqa
from llm import _extract_json  # noqa
from pipeline import distribute_personas  # noqa


async def main() -> None:
    start_http_server(8001)
    logging.info("✅ Метрики доступны на http://localhost:8001/metrics")
    bot = Bot(token=TG_BOT_TOKEN)
    dp  = Dispatcher()
    dp.include_router(router)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
