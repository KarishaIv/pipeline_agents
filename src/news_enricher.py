"""
Адаптер новостного мультиагента для общего пайплайна.

Загружает модули новостной системы изолированно через importlib, чтобы избежать конфликта имён.
"""

import sys
import os
import json
import importlib.util
import logging
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger(__name__)

NEWS_LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "news_context")


def _save_context_to_file(ta_name: str, context: dict):
    """Сохраняет структурированный новостной контекст в JSON файл."""
    os.makedirs(NEWS_LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = ta_name.replace(" ", "_")[:30]

    json_path = os.path.join(NEWS_LOG_DIR, f"{slug}_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(context, f, ensure_ascii=False, indent=2)

    logger.info(f"[News] Контекст сохранён: {json_path}")


def _load_news_module(
    alias: str,
    filepath: str,
    preload: Dict[str, object] = None
) -> object:
    """
    Загружает Python-файл как модуль под уникальным именем alias.

    preload — словарь {имя: модуль}, которые временно регистрируются
    в sys.modules во время exec_module, чтобы внутренние импорты
    новостных файлов разрешались корректно.
    После загрузки временные записи убираются.
    """
    if alias in sys.modules:
        return sys.modules[alias]

    spec = importlib.util.spec_from_file_location(alias, filepath)
    if spec is None:
        raise ImportError(f"Не удалось загрузить модуль из {filepath}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module

    saved: Dict[str, Optional[object]] = {}
    if preload:
        for name, mod in preload.items():
            saved[name] = sys.modules.get(name)
            sys.modules[name] = mod

    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(alias, None)
        raise
    finally:
        for name, old_mod in saved.items():
            if old_mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_mod

    return module


class NewsContextEnricher:
    """
    Получает новостной контекст из мультиагентной системы.

    Инициализируется один раз за запуск пайплайна.
    Кэширует результаты по паре (ta_name, question) - повторные вызовы с теми же параметрами не делают новых LLM-запросов.

    Args:
        news_system_path: абсолютный или относительный путь к папке multi_agent_rag 2
        api_key: YandexGPT API-ключ (если не задан, берётся из config новостной системы)
    """

    def __init__(self, news_system_path: str, iam_token: Optional[str] = None):
        self.news_system_path = os.path.abspath(news_system_path)
        self.iam_token = iam_token
        self._orchestrator = None
        self._cache: Dict[str, str] = {}
        self._initialized = False

    def _fp(self, filename: str) -> str:
        return os.path.join(self.news_system_path, filename)

    async def initialize(self):
        """Загружает модули новостной системы и создаёт оркестратор."""
        if self._initialized:
            return

        logger.info("Инициализация новостной системы...")

        news_config = _load_news_module("_news_config", self._fp("config.py"))

        if self.iam_token:
            news_config.YANDEX_GPT_CONFIG["iam_token"] = self.iam_token

        rel_cache = news_config.YANDEX_GPT_CONFIG.get("cache_dir", "cache")
        news_config.YANDEX_GPT_CONFIG["cache_dir"] = os.path.join(
            self.news_system_path, rel_cache
        )

        search_rag = _load_news_module("_news_search_rag", self._fp("search_rag.py"))

        rag_system_mod = _load_news_module(
            "_news_rag_system",
            self._fp("rag_system.py"),
            preload={
                "config": news_config,
                "search_rag": search_rag,
            },
        )

        agents_mod = _load_news_module(
            "_news_agents",
            self._fp("agents.py"),
            preload={"config": news_config},
        )

        orchestrator_mod = _load_news_module(
            "_news_orchestrator",
            self._fp("orchestrator.py"),
            preload={
                "config": news_config,
                "agents": agents_mod,
            },
        )

        RAGSystem = rag_system_mod.RAGSystem
        DomainAgent = agents_mod.DomainAgent
        YandexGPTClient = agents_mod.YandexGPTClient
        Orchestrator = orchestrator_mod.Orchestrator
        CATEGORIES = news_config.CATEGORIES

        llm_client = YandexGPTClient()

        rag_system = RAGSystem(collection_name="telegram_news", local=True)
        rag_system.storage_path = os.path.join(self.news_system_path, "qdrant_data")

        agents = [DomainAgent(name, rag_system, llm_client) for name in CATEGORIES]
        self._orchestrator = Orchestrator(agents, llm_client)
        self._initialized = True

        logger.info(f"Новостная система инициализирована ({len(CATEGORIES)} агентов)")

    async def get_news_context(self, question: str, ta_name: str) -> dict:
        """
        Возвращает структурированный новостной контекст для заданной ЦА и вопроса.

        Поля: snapshot_id, generated_at, audience, question, overall_reaction,
              impact_horizon, summary_text, factors, risks, opportunities, evidence, fictional_warning.

        Повторные вызовы с теми же (ta_name, question) возвращают кэшированный результат.
        При любой ошибке возвращает пустой dict — симуляция продолжается без контекста.
        """
        if not self._initialized:
            await self.initialize()

        cache_key = f"{ta_name}||{question}"
        if cache_key in self._cache:
            logger.debug(f"[News] Cache hit для '{ta_name}'")
            return self._cache[cache_key]

        logger.info(f"[News] Запрос новостного контекста: ЦА='{ta_name}', вопрос='{question[:60]}...'")
        try:
            report = await self._orchestrator.run(question, ta_name)
            context = report.get("context_for_simulation", {})
            self._cache[cache_key] = context
            logger.info(
                f"[News]Контекст получен для '{ta_name}': "
                f"reaction={context.get('overall_reaction')} | "
                f"horizon={context.get('impact_horizon')} | "
                f"evidence={len(context.get('evidence', []))} новостей"
            )
            if context.get('factors'):
                logger.info(f"[News] Факторы: {'; '.join(context['factors'])}")
            if context.get('risks'):
                logger.info(f"[News] Риски: {'; '.join(context['risks'])}")
            if context.get('opportunities'):
                logger.info(f"[News] Возможности: {'; '.join(context['opportunities'])}")
            _save_context_to_file(ta_name, context)
            return context
        except Exception as e:
            logger.warning(f"[News] Не удалось получить контекст для '{ta_name}': {e}")
            self._cache[cache_key] = {}
            return {}
