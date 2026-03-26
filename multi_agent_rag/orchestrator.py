import asyncio
import json
from typing import List, Dict, Union
from agents import DomainAgent, YandexGPTClient
from config import (
    COMMUNICATION_RULES,
    ORCHESTRATOR_PLANNING_PROMPT,
    FINAL_OUTPUT_PROMPT,
    CATEGORIES,
    CATEGORY_SLUG
)


class Orchestrator:
    """Оркестратор мультиагентной системы"""

    def __init__(self, agents: List[DomainAgent], llm_client: YandexGPTClient):
        self.agents = {agent.name: agent for agent in agents}
        self.llm_client = llm_client
        self.max_rounds = COMMUNICATION_RULES["max_rounds"]

    async def run(self, question: str, target_audience: str) -> dict:
        """Запускает полный цикл обработки и возвращает контекст для Strata"""

        # Этап 1: Параллельный сбор локальных знаний
        print("Сбор локальных знаний агентов")
        tasks = [
            agent.process_query(question, target_audience)
            for agent in self.agents.values()
        ]
        await asyncio.gather(*tasks)

        # Этап 2: Консультации между агентами
        print("Консультации между агентами")
        for round_num in range(1, self.max_rounds + 1):
            print(f"Раунд {round_num}/{self.max_rounds}")
            consultations = await self._plan_consultations(question, target_audience)

            if not consultations:
                print("Консультации не требуются")
                break

            consultation_tasks = []
            for consult in consultations:
                from_agent = self.agents.get(consult["from_title"])
                to_agent = self.agents.get(consult["to_title"])
                if from_agent and to_agent:
                    task = from_agent.consult(to_agent, consult["query"])
                    consultation_tasks.append(task)

            if consultation_tasks:
                await asyncio.gather(*consultation_tasks)

        # Этап 3: Финальный вывод (контекст реакции)
        print("Формирование итогового вывода")
        final_data = await self._generate_final_output(question, target_audience)

        # Обработка результата (dict или str)
        if isinstance(final_data, dict):
            structured = final_data
            summary_text = final_data.get("summary_text", str(final_data))
        else:
            clean = self._clean_json_response(final_data)
            try:
                structured = json.loads(clean)
                summary_text = structured.get("summary_text", final_data)
            except:
                structured = {
                    "sentiment": self._extract_sentiment(final_data),
                    "key_triggers": self._extract_triggers(final_data),
                    "barriers": self._extract_barriers(final_data),
                    "enablers": self._extract_enablers(final_data),
                    "summary_text": final_data,
                    "fictional_warning": "АУДИТОРИЯ НЕРЕАЛИСТИЧНА" in final_data
                }
                summary_text = final_data

        detailed_report = {
            "target_audience": target_audience,
            "question": question,
            "context_for_simulation": {
                "sentiment": structured.get("sentiment", "смешанный"),
                "key_triggers": structured.get("key_triggers", []),
                "barriers": structured.get("barriers", []),
                "enablers": structured.get("enablers", []),
                "raw_analysis": summary_text,
                "fictional_warning": structured.get("fictional_warning", False)
            },
            "agent_opinions": {},
            "final_summary": structured
        }

        for name, agent in self.agents.items():
            detailed_report["agent_opinions"][name] = {
                "local_knowledge": agent.local_knowledge,
                "consultations": agent.consultation_history
            }

        return detailed_report

    def _extract_sentiment(self, text: str) -> str:
        """Извлекает общий сентимент из текста (Fallback)"""
        if "насторожен" in text.lower() or "опасен" in text.lower():
            return "настороженный"
        elif "заинтересован" in text.lower() or "возможность" in text.lower():
            return "заинтересованный"
        elif "нейтральн" in text.lower():
            return "нейтральный"
        else:
            return "смешанный"

    def _extract_triggers(self, text: str) -> List[str]:
        """Извлекает ключевые триггеры (Fallback)"""
        triggers = []
        if "ограничение наличных" in text.lower():
            triggers.append("ограничение наличных через банкоматы")
        if "процентн" in text.lower() and "ставк" in text.lower():
            triggers.append("процентные ставки по продуктам")
        if "довер" in text.lower():
            triggers.append("уровень доверия к банку")
        return triggers if triggers else ["недостаточно данных для выделения триггеров"]

    def _extract_barriers(self, text: str) -> List[str]:
        """Извлекает барьеры (Fallback)"""
        barriers = []
        if "недостаточн" in text.lower() and "информ" in text.lower():
            barriers.append("недостаточная информированность")
        if "опасен" in text.lower() or "риск" in text.lower():
            barriers.append("опасения относительно безопасности")
        return barriers if barriers else ["недостаточно данных для выделения барьеров"]

    def _extract_enablers(self, text: str) -> List[str]:
        """Извлекает стимулы (Fallback)"""
        enablers = []
        if "доход" in text.lower() or "выгод" in text.lower():
            enablers.append("возможность получить дополнительный доход")
        if "прозрачн" in text.lower() or "надежн" in text.lower():
            enablers.append("прозрачность и надежность условий")
        return enablers if enablers else ["недостаточно данных для выделения стимулов"]

    async def _plan_consultations(self, question: str, target_audience: str) -> List[Dict]:
        """LLM решает, каким агентам проконсультироваться"""
        agent_knowledge = "\n\n".join([
            f"{name}: {agent.local_knowledge[:200]}..."
            for name, agent in self.agents.items()
        ])

        categories_en = ", ".join(CATEGORY_SLUG.values())

        prompt = ORCHESTRATOR_PLANNING_PROMPT.format(
            categories=categories_en,
            question=question,
            target_audience=target_audience,
            agent_knowledge=agent_knowledge
        )

        try:
            response = await self.llm_client.generate(
                prompt,
                system_prompt="Верни ТОЛЬКО валидный JSON массив. Без пояснений."
            )

            start_idx = response.find("[")
            end_idx = response.rfind("]") + 1
            if start_idx != -1 and end_idx > start_idx:
                consultations = json.loads(response[start_idx:end_idx])

                slug_to_name = {v: k for k, v in CATEGORY_SLUG.items()}

                valid = []
                for c in consultations:
                    from_slug = c.get("from", "").lower()
                    to_slug = c.get("to", "").lower()

                    if from_slug != to_slug and from_slug in slug_to_name and to_slug in slug_to_name:
                        valid.append({
                            "from": from_slug,
                            "to": to_slug,
                            "from_title": slug_to_name[from_slug],
                            "to_title": slug_to_name[to_slug],
                            "query": c.get("query", "")
                        })

                return valid[:3]
            return []
        except Exception as e:
            print(f"Ошибка планирования: {e}")
            return []

    def _clean_json_response(self, text: str) -> str:
        """Удаляет markdown-разметку из ответа LLM"""
        text = text.replace("```json", "").replace("```", "").strip()
        start_idx = text.find("{")
        end_idx = text.rfind("}") + 1
        if start_idx != -1 and end_idx > start_idx:
            return text[start_idx:end_idx]
        return text

    async def _generate_final_output(self, question: str, target_audience: str) -> Union[dict, str]:
        """Генерирует финальный вывод в формате JSON"""

        agent_opinions = "\n\n".join([
            agent.get_full_opinion()
            for agent in self.agents.values()
        ])

        consultations_text = ""
        for agent in self.agents.values():
            for c in agent.consultation_history:
                consultations_text += f"{agent.name} -> {c['to']}: {c['query']}\n"
                consultations_text += f"Ответ: {c['response']}\n\n"

        if not consultations_text.strip():
            consultations_text = "Консультаций не было."

        prompt = FINAL_OUTPUT_PROMPT.format(
            target_audience=target_audience,
            question=question,
            agent_opinions=agent_opinions,
            consultations=consultations_text
        )

        response = await self.llm_client.generate_final(prompt)

        clean_response = self._clean_json_response(response)
        try:
            data = json.loads(clean_response)
            if data.get("fictional_warning") or "АУДИТОРИЯ НЕРЕАЛИСТИЧНА" in data.get("summary_text", ""):
                data["fictional_warning"] = True
                print("Обнаружена нереалистичная целевая аудитория")
            print("JSON успешно распарсен")
            return data
        except json.JSONDecodeError as e:
            print(f"Ошибка парсинга JSON: {e}, используем fallback")
            return response