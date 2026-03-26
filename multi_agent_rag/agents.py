import aiohttp
import asyncio
import os
import json
import hashlib
from typing import List, Optional
from config import YANDEX_GPT_CONFIG, CATEGORIES, CATEGORY_SLUG


class YandexGPTClient:
    """Клиент для YandexGPT API с кэшированием"""

    def __init__(self):
        self.folder_id = YANDEX_GPT_CONFIG["folder_id"]
        self.iam_token = YANDEX_GPT_CONFIG["iam_token"]
        self.model_uri = YANDEX_GPT_CONFIG["model_uri"]
        self.temperature = YANDEX_GPT_CONFIG["temperature"]

        self.rag_max_tokens = YANDEX_GPT_CONFIG.get("rag_max_tokens", 1000)
        self.final_max_tokens = YANDEX_GPT_CONFIG.get("final_max_tokens", 2000)

        self.cache_dir = YANDEX_GPT_CONFIG.get("cache_dir", "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.stats = {"requests_made": 0, "cache_hits": 0, "tokens_used": 0}

    def _get_cache_key(self, prompt: str, system_prompt: str) -> str:
        import hashlib
        content = f"{system_prompt}|||{prompt}"
        return hashlib.md5(content.encode('utf-8', errors='ignore')).hexdigest()

    async def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Генерация ответа через YandexGPT с кэшированием"""
        cache_key = self._get_cache_key(prompt, system_prompt)
        cache_path = f"{self.cache_dir}/{cache_key}.json"

        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.stats["cache_hits"] += 1
                    return data["result"]
            except:
                pass

        self.stats["requests_made"] += 1
        print(f"Запрос к API #{self.stats['requests_made']}")

        url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.iam_token}",
            "x-folder-id": self.folder_id
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "text": system_prompt})
        messages.append({"role": "user", "text": prompt})

        payload = {
            "modelUri": self.model_uri,
            "completionOptions": {
                "temperature": self.temperature,
                "maxTokens": self.rag_max_tokens
            },
            "messages": messages
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status != 200:
                    raise Exception(f"YandexGPT API error: {response.status}")
                result = await response.json()
                text = result["result"]["alternatives"][0]["message"]["text"]

                try:
                    with open(cache_path, 'w', encoding='utf-8') as f:
                        json.dump({"result": text}, f, ensure_ascii=False, indent=2)
                except:
                    pass
                return text

    async def generate_final(self, prompt: str, system_prompt: str = "") -> str:
        original = self.rag_max_tokens
        self.rag_max_tokens = self.final_max_tokens
        result = await self.generate(prompt, system_prompt)
        self.rag_max_tokens = original
        return result

    def print_stats(self):
        total = self.stats["requests_made"] + self.stats["cache_hits"]
        if total > 0:
            savings = (self.stats["cache_hits"] / total) * 100
            cost = (self.stats["requests_made"] * 3000 / 1000 * 0.8)
            print(f"\nСтатистика YandexGPT:")
            print(f"   Всего обращений: {total}")
            print(f"   Запросов к API: {self.stats['requests_made']}")
            print(f"   Кэш-хитов: {self.stats['cache_hits']}")
            print(f"   Экономия: {savings:.1f}%")
            print(f"   Примерная стоимость: {cost:.0f}₽")


class DomainAgent:
    """Агент по конкретной категории новостей"""

    def __init__(self, name: str, rag_system, llm_client: YandexGPTClient):
        self.name = name
        self.slug = CATEGORY_SLUG.get(name, name.lower())
        self.rag_system = rag_system
        self.llm_client = llm_client
        self.local_knowledge: str = ""
        self.consultation_history: List[dict] = []

    async def process_query(self, question: str, target_audience: str) -> str:
        """Первичная обработка вопроса (RAG + LLM)"""
        context_snippets = self.rag_system.get_relevant_context(self.name, question)
        full_results = self.rag_system.get_cached_results(self.name, question)

        if not context_snippets:
            self.local_knowledge = f"По категории '{self.name}' нет релевантных новостей для анализа реакции ЦА."
            return self.local_knowledge

        context_parts = []
        for i, (snippet, result) in enumerate(zip(context_snippets, full_results), 1):
            metadata = result.get("metadata", {})
            score = result.get("score", 0)
            context_parts.append(
                f"[{i}] (score={score:.3f}) {snippet}\n"
                f"Источник: {metadata.get('source', 'unknown')}, "
                f"Просмотры: {metadata.get('views', 'N/A')}"
            )
        context = "\n\n".join(context_parts)

        #реакция
        prompt = f"""
Ты эксперт по категории "{self.name}".
Целевая аудитория: {target_audience}
Вопрос пользователя (для контекста): {question}

Контекст новостей (найден по семантическому поиску):
{context}

Задача: Проанализируй, как эта целевая аудитория может ВОСПРИНЯТЬ указанные новости в контексте вопроса.
НЕ отвечай на вопрос "будут/не будут". Твоя задача — описать реакцию на новости.

Укажи в ответе:
1. Эмоциональный тон реакции (настороженность / интерес / безразличие / доверие)
2. Какие аспекты новостей могут вызвать опасения или, наоборот, доверие
3. Какие факторы из новостей наиболее релевантны для этой ЦА

Ответ (3-4 предложения, без выводов о решении):
"""
        self.local_knowledge = await self.llm_client.generate(prompt)
        return self.local_knowledge

    async def consult(self, other_agent: 'DomainAgent', query: str) -> str:
        """Консультация с другим агентом"""
        if other_agent.name == self.name:
            raise ValueError("Агент не может обратиться к самому себе!")
        response = await other_agent._answer_consultation(query, self.name)
        self.consultation_history.append({
            "to": other_agent.name,
            "query": query,
            "response": response
        })
        return response

    async def _answer_consultation(self, query: str, from_agent: str) -> str:
        """Ответ на запрос от другого агента"""
        prompt = f"""
Ты эксперт по категории "{self.name}".
Агент из категории "{from_agent}" спрашивает: "{query}"

Твои знания по теме:
{self.local_knowledge}

Задача: Помоги другому агенту понять, как целевая аудитория может интерпретировать новости.
НЕ предлагай решение. Опиши только:
- Как новости из твоей категории могут усилить или ослабить реакцию ЦА
- Какие пересечения с другими категориями стоит учесть

Ответ (2-3 предложения, фокус на восприятии, не на решении):
"""
        return await self.llm_client.generate(prompt)

    def get_full_opinion(self) -> str:
        """Возвращает полное мнение агента (включая консультации)"""
        opinion = f"=== {self.name} ===\n{self.local_knowledge}\n"
        if self.consultation_history:
            opinion += "Консультации:\n"
            for c in self.consultation_history:
                opinion += f"{c['to']}: {c['query']}\n    Ответ: {c['response']}\n"
        return opinion