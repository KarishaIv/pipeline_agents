"""Structured Pydantic models for meta-agent API requests and responses.

Defines the contract for POST /ask endpoint, supporting text, JSON, and file
outputs extensible for future artifact types.
"""

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Request body for POST /ask."""

    question: str = Field(
        ...,
        description="User question for meta-agent",
        min_length=1,
        max_length=10000,
    )
    thread_id: Optional[str] = Field(
        default=None,
        description=(
            "Session thread ID for conversation continuity. "
            "null or omitted = new session, "
            "-1 = explicit new session, "
            "otherwise continue existing session"
        ),
    )


class TextOutput(BaseModel):
    """Text output from meta-agent."""

    type: Literal["text"] = Field(default="text", description="Output type identifier")
    text: str = Field(..., description="Text response content")


class JsonOutput(BaseModel):
    """JSON data output from meta-agent."""

    type: Literal["json"] = Field(default="json", description="Output type identifier")
    data: dict[str, Any] = Field(..., description="Structured JSON data")
    caption: Optional[str] = Field(
        default=None, description="Optional caption or metadata for the JSON"
    )


class FileOutput(BaseModel):
    """File/document output from meta-agent."""

    type: Literal["file"] = Field(default="file", description="Output type identifier")
    filename: str = Field(..., description="Suggested filename")
    mime_type: str = Field(..., description="MIME type (e.g., application/pdf)")
    download_url: str = Field(
        ...,
        description="URL where the file can be downloaded from the API server",
    )
    caption: Optional[str] = Field(
        default=None, description="Optional caption or description"
    )


AgentOutput = TextOutput | JsonOutput | FileOutput


class MetaAgentApiResponse(BaseModel):
    """Structured response from POST /ask endpoint."""

    thread_id: str = Field(
        ..., description="Session thread ID for future message continuity"
    )
    outputs: list[AgentOutput] = Field(
        default_factory=list,
        description="Ordered list of outputs (text, JSON, files, etc.)",
    )


class ErrorResponse(BaseModel):
    """Error response from API."""

    error: str = Field(..., description="User-friendly error message")
    error_type: str = Field(
        default="unknown_error",
        description="Error classification (validation_error, timeout_error, etc.)",
    )
    details: Optional[dict[str, Any]] = Field(
        default=None, description="Additional error context"
    )
