from dataclasses import dataclass


@dataclass(frozen=True)
class MoveProposal:
    raw_response: str
    source: str
    latency_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    thinking: str | None = None
    thinking_used: bool = False
