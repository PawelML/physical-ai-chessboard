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
    thinking: str | None = None
    thinking_used: bool = False


@dataclass(frozen=True)
class GenerationOptions:
    temperature: float | None = None
    top_p: float | None = None
    num_predict: int | None = None
    think: str | None = None


class LLMService(ABC):
    @abstractmethod
    async def complete(
        self,
        *,
        model: str,
        prompt: str,
        options: GenerationOptions | None = None,
    ) -> LLMResponse:
        """Return a normalized response for one prompt."""
