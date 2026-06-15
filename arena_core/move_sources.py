from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MoveProposal:
    raw_response: str
    latency_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    thinking: str | None = None
    thinking_used: bool = False
    metadata: dict[str, Any] | None = None
