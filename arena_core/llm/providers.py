import asyncio
from dataclasses import dataclass
from typing import Any

import httpx

from arena_core.config import Settings
from arena_core.llm.base import GenerationOptions, LLMResponse, LLMService
from arena_core.llm.ollama import OllamaLLMService


@dataclass(frozen=True)
class ProviderModel:
    provider: str
    model: str


class ProviderDisabledError(ValueError):
    pass


class OpenAICompatibleLLMService(LLMService):
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout_seconds: float = 120.0,
        max_retries: int = 2,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout_seconds
        self._max_retries = max_retries

    async def complete(
        self,
        *,
        model: str,
        prompt: str,
        options: GenerationOptions | None = None,
    ) -> LLMResponse:
        temperature = options.temperature if options and options.temperature is not None else 0
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if options and options.top_p is not None:
            payload["top_p"] = options.top_p
        if options and options.num_predict is not None:
            payload["max_tokens"] = options.num_predict
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await _post_with_retries(
                client,
                f"{self._base_url}/chat/completions",
                max_retries=self._max_retries,
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=payload,
            )
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
    def __init__(
        self,
        *,
        api_key: str,
        timeout_seconds: float = 120.0,
        max_retries: int = 2,
    ) -> None:
        self._api_key = api_key
        self._timeout = timeout_seconds
        self._max_retries = max_retries

    async def complete(
        self,
        *,
        model: str,
        prompt: str,
        options: GenerationOptions | None = None,
    ) -> LLMResponse:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await _post_with_retries(
                client,
                "https://api.anthropic.com/v1/messages",
                max_retries=self._max_retries,
                headers={
                    "x-api-key": self._api_key,
                    "anthropic-version": "2023-06-01",
                },
                json={
                    "model": model,
                    "max_tokens": (
                        options.num_predict if options and options.num_predict is not None else 512
                    ),
                    "temperature": (
                        options.temperature if options and options.temperature is not None else 0
                    ),
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
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
    def __init__(
        self,
        *,
        api_key: str,
        timeout_seconds: float = 120.0,
        max_retries: int = 2,
    ) -> None:
        self._api_key = api_key
        self._timeout = timeout_seconds
        self._max_retries = max_retries

    async def complete(
        self,
        *,
        model: str,
        prompt: str,
        options: GenerationOptions | None = None,
    ) -> LLMResponse:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={self._api_key}"
        )
        generation_config: dict[str, float | int] = {
            "temperature": options.temperature if options and options.temperature is not None else 0
        }
        if options and options.top_p is not None:
            generation_config["topP"] = options.top_p
        if options and options.num_predict is not None:
            generation_config["maxOutputTokens"] = options.num_predict
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await _post_with_retries(
                client,
                url,
                max_retries=self._max_retries,
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": generation_config,
                },
            )
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


async def _post_with_retries(
    client: httpx.AsyncClient,
    url: str,
    *,
    max_retries: int,
    **kwargs: Any,
) -> httpx.Response:
    for retry in range(max_retries + 1):
        try:
            response = await client.post(url, **kwargs)
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as exc:
            if retry >= max_retries or not _retryable_status(exc.response.status_code):
                raise
            await asyncio.sleep(_retry_delay_seconds(retry))
        except (httpx.TimeoutException, httpx.TransportError):
            if retry >= max_retries:
                raise
            await asyncio.sleep(_retry_delay_seconds(retry))
    raise RuntimeError("unreachable provider retry state")


def _retryable_status(status_code: int) -> bool:
    return status_code == 429 or 500 <= status_code < 600


def _retry_delay_seconds(retry: int) -> float:
    return float(min(2**retry, 5))


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
                num_gpu=settings.ollama_num_gpu,
                min_num_gpu=settings.ollama_cpu_offload_min_gpu_layers,
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
                timeout_seconds=settings.api_timeout_seconds,
                max_retries=settings.api_max_retries,
            ),
        )
    if provider_model.provider == "anthropic":
        if not settings.anthropic_api_key:
            raise ProviderDisabledError("ARENA_ANTHROPIC_API_KEY is required")
        return (
            provider_model.model,
            AnthropicLLMService(
                api_key=settings.anthropic_api_key,
                timeout_seconds=settings.api_timeout_seconds,
                max_retries=settings.api_max_retries,
            ),
        )
    if provider_model.provider == "gemini":
        if not settings.gemini_api_key:
            raise ProviderDisabledError("ARENA_GEMINI_API_KEY is required")
        return (
            provider_model.model,
            GeminiLLMService(
                api_key=settings.gemini_api_key,
                timeout_seconds=settings.api_timeout_seconds,
                max_retries=settings.api_max_retries,
            ),
        )
    raise ProviderDisabledError(f"Unsupported provider: {provider_model.provider}")
