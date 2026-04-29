"""HTTP client for meta-agent API communication."""

from typing import Optional
import httpx
import logging

from meta_agent import AskRequest, MetaAgentApiResponse

logger = logging.getLogger(__name__)


class MetaAgentClient:
    """HTTP client for communicating with meta-agent API."""

    def __init__(self, base_url: str, request_timeout: float = 60.0):
        """Initialize meta-agent client.

        Args:
            base_url: Base URL of meta-agent API (e.g., http://localhost:8000).
            request_timeout: HTTP request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.request_timeout = request_timeout
        self.client = httpx.AsyncClient(timeout=request_timeout)

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()

    async def ask(
        self, question: str, thread_id: Optional[str] = None
    ) -> MetaAgentApiResponse:
        """Ask meta-agent a question.

        Args:
            question: User question.
            thread_id: Optional thread ID for session continuity.

        Returns:
            MetaAgentApiResponse with outputs and thread_id.

        Raises:
            httpx.HTTPError: On HTTP errors.
            ValueError: On invalid response format.
        """
        request = AskRequest(question=question, thread_id=thread_id)

        try:
            response = await self.client.post(
                f"{self.base_url}/ask",
                json=request.model_dump(exclude_none=True),
            )
            response.raise_for_status()

            data = response.json()
            return MetaAgentApiResponse.model_validate(data)
        except httpx.HTTPError as e:
            logger.error("HTTP error calling meta-agent: %s", e)
            raise
        except ValueError as e:
            logger.error("Invalid response format from meta-agent: %s", e)
            raise
