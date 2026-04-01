import json
import logging
from textwrap import dedent

import aiohttp

from config import YANDEX_API_KEY, YANDEX_FOLDER_ID, YANDEX_GPT_URL, YANDEX_MODEL_URI

log = logging.getLogger(__name__)

SYSTEM_PROMPT = dedent("""\
    Ты — ассистент-аналитик. Пользователь присылает запрос о финансовом поведении людей.
    Извлеки целевые аудитории, вопрос и соотношение. Верни ТОЛЬКО валидный JSON без пояснений:
    {
      "audiences": ["<ЦА 1>", "<ЦА 2>"],
      "question": "<суть финансового вопроса кратко>",
      "ratios": [<число>, <число>]
    }

    Правила для audiences:
    - Список реальных групп людей (любое количество)
    - Если одна группа — список из одного элемента
    - Если субъект вымышленный — пустой список []
    - Если субъект бессмысленный — пустой список []
    - Если ЦА не упомянута вообще — пустой список []

    Правила для ratios:
    - Если упомянуто соотношение (3 к 1, 2:1 и т.д.) — отрази в ratios
    - Если соотношение не указано — все значения равны 1

    Если вопрос не про финансы — question: "не финансовый вопрос"
""")

FALLBACK_TA_PROMPT = dedent("""\
    Пользователь задал вопрос о финансовом поведении, но целевая аудитория не была явно указана.
    Попробуй предположить наиболее вероятную реальную группу людей для этого вопроса.

    Верни ТОЛЬКО валидный JSON:
    {
      "target_audience": "<реальная группа людей>",
      "inferred": true
    }

    Правила:
    - Если субъект вымышленный — верни {"target_audience": "не удалось", "inferred": false}
    - Если субъект бессмысленный — верни {"target_audience": "не удалось", "inferred": false}
    - Если вопрос совсем не про финансы — верни {"target_audience": "не удалось", "inferred": false}
    - Иначе — предложи разумное обобщение: "работающие граждане России", "жители России старше 18 лет" и т.д.
""")

DEMOGRAPHICS_PROMPT = dedent("""\
    Из описания целевой аудитории извлеки демографические параметры типичного представителя.
    Верни ТОЛЬКО валидный JSON без пояснений:
    {
      "age_group": <число — типичный возраст>,
      "gender": "<Мужской|Женский>",
      "marital_status": "<Женат|Замужем|Не женат|Не замужем|Разведен|Разведена|Вдовец|Вдова>",
      "education": "<среднего профессионального образования|бакалавриат|специалитет|магистратура|аспирантура>"
    }
    Если параметр неизвестен — используй наиболее вероятное значение для России.
""")


async def _yandex_call(messages: list, temperature: float = 0.1, max_tokens: int = 500) -> str:
    payload = {
        "modelUri": YANDEX_MODEL_URI,
        "completionOptions": {"stream": False, "temperature": temperature, "maxTokens": max_tokens},
        "messages": messages,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Api-Key {YANDEX_API_KEY}",
        "x-folder-id": YANDEX_FOLDER_ID,
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(YANDEX_GPT_URL, json=payload, headers=headers) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise Exception(f"YandexGPT {resp.status}: {body}")
            data = await resp.json()
            return data["result"]["alternatives"][0]["message"]["text"]


def _extract_json(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("```", 1)[0]
    cleaned = cleaned.strip()
    start = cleaned.find("{")
    end   = cleaned.rfind("}") + 1
    if start == -1:
        return {}
    try:
        return json.loads(cleaned[start:end])
    except json.JSONDecodeError:
        return {}


async def parse_query_with_llm(user_text: str) -> dict:
    try:
        raw = await _yandex_call([
            {"role": "system", "text": SYSTEM_PROMPT},
            {"role": "user",   "text": user_text},
        ])
        return _extract_json(raw)
    except Exception as exc:
        log.error("parse_query_with_llm: %s", exc)
        return {"error": str(exc)}


async def infer_ta_from_query(user_text: str) -> dict:
    try:
        raw = await _yandex_call([
            {"role": "system", "text": FALLBACK_TA_PROMPT},
            {"role": "user",   "text": user_text},
        ], temperature=0.2)
        return _extract_json(raw)
    except Exception as exc:
        log.warning("infer_ta_from_query: %s", exc)
    return {"target_audience": "не удалось", "inferred": False}


async def extract_demographics_with_llm(target_audience: str) -> dict:
    try:
        raw = await _yandex_call([
            {"role": "system", "text": DEMOGRAPHICS_PROMPT},
            {"role": "user",   "text": target_audience},
        ], max_tokens=200)
        return _extract_json(raw)
    except Exception as exc:
        log.warning("extract_demographics_with_llm: %s", exc)
    return {}
