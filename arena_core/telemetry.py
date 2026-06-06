from dataclasses import dataclass


@dataclass(frozen=True)
class TokenEstimate:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_context_window: int
    estimated_context_remaining: int
    truncation_applied: bool = False


@dataclass(frozen=True)
class ModelFootprint:
    model: str
    vram_gb: float | None


@dataclass(frozen=True)
class PairFootprint:
    models: tuple[str, str]
    footprints: tuple[ModelFootprint, ModelFootprint]
    total_vram_gb: float | None
    budget_gb: float
    exceeds_budget: bool
    unknown_models: tuple[str, ...]


OLLAMA_MODEL_FOOTPRINTS_GB: dict[str, float] = {
    "qwen3.5:9b": 6.6,
    "gemma3n:e4b": 5.0,
    "qwen3-vl:8b": 6.1,
    "nanbeige4.1:3b": 4.2,
    "qwen3.5-27b-ud": 17.0,
    "qwen3.5:35b-a3b": 23.0,
    "nemotron-3-nano": 24.0,
}


def estimate_tokens(text: str) -> int:
    """Cheap cross-model token estimate.

    Local model tokenizers differ, so this is intentionally approximate and stored
    as telemetry, not billing truth.
    """

    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def estimate_usage(prompt: str, response: str, context_window: int) -> TokenEstimate:
    prompt_tokens = estimate_tokens(prompt)
    completion_tokens = estimate_tokens(response)
    total_tokens = prompt_tokens + completion_tokens
    remaining = max(context_window - total_tokens, 0)
    return TokenEstimate(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        estimated_context_window=context_window,
        estimated_context_remaining=remaining,
        truncation_applied=False,
    )


def estimate_pair_footprint(model_a: str, model_b: str, budget_gb: float) -> PairFootprint:
    footprint_a = ModelFootprint(model_a, OLLAMA_MODEL_FOOTPRINTS_GB.get(model_a))
    footprint_b = ModelFootprint(model_b, OLLAMA_MODEL_FOOTPRINTS_GB.get(model_b))
    known = [value for value in (footprint_a.vram_gb, footprint_b.vram_gb) if value is not None]
    total = sum(known) if len(known) == 2 else None
    unknown = tuple(
        footprint.model for footprint in (footprint_a, footprint_b) if footprint.vram_gb is None
    )
    return PairFootprint(
        models=(model_a, model_b),
        footprints=(footprint_a, footprint_b),
        total_vram_gb=total,
        budget_gb=budget_gb,
        exceeds_budget=total is not None and total > budget_gb,
        unknown_models=unknown,
    )
