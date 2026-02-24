"""FastAPI-сервер для мета-агента.

Запуск:
    uvicorn src.scripts.serve_meta_agent:app --host 0.0.0.0 --port 8000
"""

import asyncio

from fastapi import FastAPI, Query
from langchain_core.messages import HumanMessage
from tenacity import retry, stop_after_attempt, wait_exponential

from src.meta_agent.meta_agent import graph

app = FastAPI(title="Meta Agent API")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
def _invoke_graph(messages):
    return graph.invoke(messages)


@app.get("/ask")
async def ask(q: str = Query(..., description="Вопрос для мета-агента")):
    result = await asyncio.to_thread(
        _invoke_graph, {"messages": [HumanMessage(content=q)]}
    )
    answer = result["messages"][-1].content
    return {"answer": answer}
