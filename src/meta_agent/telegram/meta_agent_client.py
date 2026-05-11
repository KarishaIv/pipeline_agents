"""HTTP client for meta-agent API communication."""

from typing import Optional
import httpx
import logging

from src.meta_agent import AskRequest, MetaAgentApiResponse

logger = logging.getLogger(__name__)


class MetaAgentClient:
    """HTTP client for communicating with meta-agent API."""

    def __init__(self, base_url: str, request_timeout: float = 300.0):
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

    async def fetch_artifact_bytes(self, artifact_url: str) -> tuple[bytes, str, str]:
        """Fetch artifact bytes from a relative artifact URL.

        Resolves only relative artifact URLs like `/artifacts/chart.png` against base_url.
        Absolute URLs or external URLs are rejected for security.

        Args:
            artifact_url: Relative artifact URL (e.g., `/artifacts/chart.png`).

        Returns:
            Tuple of (content_bytes, mime_type, filename).

        Raises:
            ValueError: If artifact_url is not relative or malformed.
            httpx.HTTPError: On HTTP errors.
        """
        if not artifact_url.startswith("/artifacts/"):
            raise ValueError(f"Only relative artifact URLs are allowed: {artifact_url}")

        try:
            full_url = f"{self.base_url}{artifact_url}"
            response = await self.client.get(full_url)
            response.raise_for_status()

            content = response.content
            mime_type = response.headers.get("content-type", "application/octet-stream")

            # Extract filename from content-disposition or URL
            content_disposition = response.headers.get("content-disposition", "")
            filename = "artifact"
            if 'filename=' in content_disposition:
                filename = content_disposition.split('filename="')[1].split('"')[0]
            else:
                # Extract from URL path
                filename = artifact_url.split("/")[-1] or "artifact"

            return content, mime_type, filename
        except httpx.HTTPError as e:
            logger.error("HTTP error fetching artifact from %s: %s", artifact_url, e)
            raise
