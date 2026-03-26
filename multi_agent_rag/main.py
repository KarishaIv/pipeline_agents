import asyncio
import os
import json
from datetime import datetime
from rag_system import RAGSystem
from agents import DomainAgent, YandexGPTClient
from orchestrator import Orchestrator
from config import CATEGORIES


async def main():
    print("Инициализация системы")
    llm_client = YandexGPTClient()
    rag_system = RAGSystem(collection_name="telegram_news", local=True)

    print("\nСоздание агентов")
    agents = [DomainAgent(name, rag_system, llm_client) for name in CATEGORIES]
    for agent in agents:
        print(f"{agent.name} {agent.slug}")

    orchestrator = Orchestrator(agents, llm_client)

    print("Система готова к работе\n")

    target_audience = input("Целевая аудитория: ")
    question = input("Вопрос: ")

    print("\nОбработка\n")
    report = await orchestrator.run(question, target_audience)

    os.makedirs("outputs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"outputs/result_{timestamp}.txt"
    json_path = f"outputs/result_{timestamp}.json"

    # Текстовый отчет для человека
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"Целевая аудитория: {report['target_audience']}\n")
        f.write(f"Вопрос: {report['question']}\n")
        f.write(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

        f.write("Рассуждения агентов:\n")

        for agent_name, opinion in report['agent_opinions'].items():
            f.write(f"| {agent_name}  " + "─" * (55 - len(agent_name)) + "\n")
            f.write(f"|\n│Первоначальный анализ:\n")
            for line in opinion['local_knowledge'].split('\n'):
                f.write(f"| {line}\n")
            f.write(f"|\n")
            if opinion['consultations']:
                f.write(f"|Консультации:\n")
                for c in opinion['consultations']:
                    f.write(f"|{c['to']}: {c['query']}\n|     Ответ: {c['response']}\n")
            else:
                f.write(f"│Консультаций не было\n")
            f.write(f"│\n└" + "─" * 60 + "\n\n")

        # Контекст реакции ЦА на новости
        f.write("Контекст реакции ЦА на новости:\n")
        final = report['final_summary']
        if isinstance(final, dict):
            summary = final.get('summary_text', 'Нет данных')
            if final.get('fictional_warning'):
                f.write(
                    "ВНИМАНИЕ: Целевая аудитория является нереалистичной. Анализ представлен как гипотетический сценарий.\n\n")
            f.write(summary)
        else:
            if "АУДИТОРИЯ НЕРЕАЛИСТИЧНА" in str(final):
                f.write("ВНИМАНИЕ: Целевая аудитория является нереалистичной.\n\n")
            f.write(str(final))

        # Структурированный контекст для симуляции
        f.write("\n\nСтруктурированный контекст для дальнейшей симуляции:\n")
        ctx = report['context_for_simulation']

        if ctx.get('fictional_warning'):
            f.write("ПРЕДУПРЕЖДЕНИЕ: АУДИТОРИЯ НЕРЕАЛИСТИЧНА\n")

        f.write(f"Сентимент: {ctx['sentiment']}\n")

        def safe_join(field):
            val = ctx.get(field, [])
            if isinstance(val, list):
                return ', '.join(val) if val else 'не определено'
            return str(val) if val else 'не определено'

        f.write(f"Триггеры: {safe_join('key_triggers')}\n")
        f.write(f"Барьеры: {safe_join('barriers')}\n")
        f.write(f"Стимулы: {safe_join('enablers')}\n")

    # JSON отчет
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\nРезультат сохранен:")
    print(f"{output_path} (текст)")
    print(f"{json_path} (JSON)")

    print("\nКонтекст реакции:")
    final = report['final_summary']
    if isinstance(final, dict):
        if final.get('fictional_warning'):
            print("АУДИТОРИЯ НЕРЕАЛИСТИЧНА (гипотетический анализ)")
        print(final.get('summary_text', 'Нет данных'))
    else:
        if "АУДИТОРИЯ НЕРЕАЛИСТИЧНА" in str(final):
            print("АУДИТОРИЯ НЕРЕАЛИСТИЧНА")
        print(final)

    llm_client.print_stats()

    print("\nОбработка завершена успешно")


if __name__ == "__main__":
    asyncio.run(main())