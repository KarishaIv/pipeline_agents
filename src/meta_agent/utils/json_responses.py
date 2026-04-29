"""Unified JSON response helpers for tools and agents.

Standardizes response formats across all tools and nodes to ensure consistent
error handling, structured payloads, and improved error clarity.
"""

import json
import traceback
from typing import Any


def json_success(data: dict | list, **extra: Any) -> str:
    """Serialize successful tool result to JSON string.

    Args:
        data: The successful result (dict or list).
        **extra: Additional fields to include in the response.

    Returns:
        JSON string with success structure.
    """
    payload = {
        "success": True,
        **(data if isinstance(data, dict) else {"data": data}),
        **extra,
    }
    return json.dumps(payload, ensure_ascii=False, default=str)


def json_error(
    message: str,
    *,
    error_type: str = "unknown_error",
    details: dict | None = None,
    **extra: Any,
) -> str:
    """Serialize error to standardized JSON format.

    Args:
        message: User-friendly error message.
        error_type: Classification of error (e.g., "validation_error", "not_found", "qdrant_error").
        details: Additional context (e.g., operation, traceback, available options).
        **extra: Additional fields to include in the response.

    Returns:
        JSON string with error structure.
    """
    payload = {
        "success": False,
        "error": message,
        "error_type": error_type,
    }
    if details:
        payload["details"] = details
    payload.update(extra)
    return json.dumps(payload, ensure_ascii=False, default=str)


def serialize_tool_result(result: Any) -> str:
    """Serialize arbitrary result to JSON string (for success paths).

    **Important:** This function ALWAYS wraps the result with "success": true.
    All callers must expect responses in the format:
        {"success": true, ...payload...}
    
    This ensures consistent response schemas across all tools

    If result is a dict or list, passes to json_success.
    If result is a Pydantic model, converts to dict first.
    Otherwise wraps in dict with 'result' key.

    Args:
        result: The result to serialize.

    Returns:
        JSON string with "success": true and the payload.
    """
    def convert_value(val: Any) -> Any:
        """Recursively convert Pydantic models to dicts."""
        if hasattr(val, "model_dump"):
            return val.model_dump()
        elif isinstance(val, dict):
            return {k: convert_value(v) for k, v in val.items()}
        elif isinstance(val, list):
            return [convert_value(item) for item in val]
        return val
    
    result = convert_value(result)
    
    if isinstance(result, dict):
        return json_success(result)
    if isinstance(result, list):
        return json_success(result)
    return json_success({"result": result})


def json_node_failure(
    *,
    worker: str,
    raw_output: str,
    expected_tool: str,
    parse_error: Exception | None = None,
) -> str:
    """Serialize node/worker failure to structured JSON format.

    Used for graph node failures when agent output cannot be parsed.

    Args:
        worker: The worker name that failed.
        raw_output: The unparseable output from the worker.
        expected_tool: The tool/report format that was expected.
        parse_error: The exception that occurred during parsing (if any).

    Returns:
        JSON string with failure structure.
    """
    payload = {
        "success": False,
        "status": "failed",
        "worker": worker,
        "error_type": "report_parse_error" if parse_error else "unexpected_output",
        "expected_report_tool": expected_tool,
        "message": (
            f"Не удалось распарсить отчёт инструмента {expected_tool}"
            if parse_error
            else "Воркер вернул неожиданный формат ответа"
        ),
        "details": str(parse_error) if parse_error else "",
        "raw_output": raw_output,
    }
    return json.dumps(payload, ensure_ascii=False)
