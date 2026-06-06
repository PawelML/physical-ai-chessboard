from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LLMResponse:
    content: str
    stop_reason: str | None = None
    raw_response: dict[str, Any] | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class LLMService(ABC):
    @abstractmethod
    async def complete(self, *, model: str, prompt: str) -> LLMResponse:
        """Return a normalized response for one prompt."""
