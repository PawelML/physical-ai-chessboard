from dataclasses import dataclass
from typing import Any

import httpx

from arena_core.config import Settings
from arena_core.llm.base import LLMResponse, LLMService
from arena_core.llm.ollama import OllamaLLMService


@dataclass(frozen=True)
class ProviderModel:
    provider: str
    model: str


class ProviderDisabledError(ValueError):
    pass


class OpenAICompatibleLLMService(LLMService):
    def __init__(self, *, base_url: str, api_key: str, timeout_seconds: float = 120.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout_seconds

    async def complete(self, *, model: str, prompt: str) -> LLMResponse:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self._base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                },
            )
            response.raise_for_status()
        payload = response.json()
        choice = payload.get("choices", [{}])[0]
        message = choice.get("message", {})
        usage = payload.get("usage", {})
        return LLMResponse(
            content=str(message.get("content", "")),
            stop_reason=choice.get("finish_reason"),
            raw_response=payload,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
        )


class AnthropicLLMService(LLMService):
    def __init__(self, *, api_key: str, timeout_seconds: float = 120.0) -> None:
        self._api_key = api_key
        self._timeout = timeout_seconds

    async def complete(self, *, model: str, prompt: str) -> LLMResponse:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self._api_key,
                    "anthropic-version": "2023-06-01",
                },
                json={
                    "model": model,
                    "max_tokens": 512,
                    "temperature": 0,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            response.raise_for_status()
        payload: dict[str, Any] = response.json()
        blocks = payload.get("content", [])
        content = "".join(block.get("text", "") for block in blocks if isinstance(block, dict))
        usage = payload.get("usage", {})
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        total_tokens = (
            input_tokens + output_tokens
            if input_tokens is not None and output_tokens is not None
            else None
        )
        return LLMResponse(
            content=content,
            stop_reason=payload.get("stop_reason"),
            raw_response=payload,
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=total_tokens,
        )


class GeminiLLMService(LLMService):
    def __init__(self, *, api_key: str, timeout_seconds: float = 120.0) -> None:
        self._api_key = api_key
        self._timeout = timeout_seconds

    async def complete(self, *, model: str, prompt: str) -> LLMResponse:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={self._api_key}"
        )
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                url,
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0},
                },
            )
            response.raise_for_status()
        payload: dict[str, Any] = response.json()
        candidates = payload.get("candidates", [])
        parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
        content = "".join(part.get("text", "") for part in parts if isinstance(part, dict))
        usage = payload.get("usageMetadata", {})
        prompt_tokens = usage.get("promptTokenCount")
        completion_tokens = usage.get("candidatesTokenCount")
        total_tokens = usage.get("totalTokenCount")
        return LLMResponse(
            content=content,
            raw_response=payload,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )


def parse_provider_model(value: str) -> ProviderModel:
    if ":" not in value:
        return ProviderModel("local", value)
    provider, model = value.split(":", 1)
    if provider in {"openai", "anthropic", "gemini", "local"}:
        return ProviderModel(provider, model)
    return ProviderModel("local", value)


def llm_service_for(value: str, settings: Settings) -> tuple[str, LLMService]:
    provider_model = parse_provider_model(value)
    if provider_model.provider == "local":
        return (
            provider_model.model,
            OllamaLLMService(
                base_url=settings.ollama_base_url,
                timeout_seconds=settings.ollama_timeout_seconds,
                temperature=settings.ollama_temperature,
                top_p=settings.ollama_top_p,
                num_ctx=settings.ollama_num_ctx,
                num_predict=settings.ollama_num_predict,
                think=settings.ollama_think,
            ),
        )
    if not settings.api_providers_enabled:
        raise ProviderDisabledError(
            "API providers are disabled; set ARENA_API_PROVIDERS_ENABLED=true"
        )
    if provider_model.provider == "openai":
        if not settings.openai_api_key:
            raise ProviderDisabledError("ARENA_OPENAI_API_KEY is required")
        return (
            provider_model.model,
            OpenAICompatibleLLMService(
                base_url="https://api.openai.com/v1",
                api_key=settings.openai_api_key,
            ),
        )
    if provider_model.provider == "anthropic":
        if not settings.anthropic_api_key:
            raise ProviderDisabledError("ARENA_ANTHROPIC_API_KEY is required")
        return provider_model.model, AnthropicLLMService(api_key=settings.anthropic_api_key)
    if provider_model.provider == "gemini":
        if not settings.gemini_api_key:
            raise ProviderDisabledError("ARENA_GEMINI_API_KEY is required")
        return provider_model.model, GeminiLLMService(api_key=settings.gemini_api_key)
    raise ProviderDisabledError(f"Unsupported provider: {provider_model.provider}")
