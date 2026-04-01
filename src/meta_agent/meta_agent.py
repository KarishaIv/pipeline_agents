import json
import os

from openai import OpenAI

from src.meta_agent.tools import TOOL_DEFINITIONS, execute_tool

META_AGENT_SYSTEM_PROMPT = (
    "Ты мета-агент, который отвечает на вопросы, "
    "используя инструменты для поиска вопросов в базе данных."
)

FOLDER_ID = os.getenv("YANDEX_FOLDER_ID", "")
MODEL_URI = f"gpt://{FOLDER_ID}/yandexgpt/latest"

client = OpenAI(
    api_key=os.getenv("YANDEX_API_KEY"),
    base_url="https://llm.api.cloud.yandex.net/v1",
)


def invoke(question: str) -> str:
    """Run the agentic tool-calling loop and return the final answer."""
    messages = [
        {"role": "system", "content": META_AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    while True:
        response = client.chat.completions.create(
            model=MODEL_URI,
            messages=messages,
            tools=TOOL_DEFINITIONS,
        )
        msg = response.choices[0].message

        # Append assistant turn (preserving tool_calls if present)
        assistant_turn: dict = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            assistant_turn["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in msg.tool_calls
            ]
        messages.append(assistant_turn)

        if not msg.tool_calls:
            return msg.content or ""

        for call in msg.tool_calls:
            raw_args = json.loads(call.function.arguments)
            result = execute_tool(call.function.name, raw_args)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result,
                }
            )
