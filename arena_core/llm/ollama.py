from dataclasses import dataclass
from typing import Any

import httpx

from arena_core.llm.base import LLMResponse, LLMService


@dataclass(frozen=True)
class OllamaModelMetadata:
    name: str
    digest: str | None
    family: str | None
    parameter_size: str | None
    quantization: str | None
    context_window: int | None
    runtime_version: str | None
    modified_at: str | None
    size_bytes: int | None


class OllamaLLMService(LLMService):
    def __init__(self, *, base_url: str, timeout_seconds: float) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds

    async def complete(self, *, model: str, prompt: str) -> LLMResponse:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self._base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "think": False,
                    "options": {"temperature": 0},
                },
            )
            response.raise_for_status()
        payload = response.json()
        prompt_eval_count = payload.get("prompt_eval_count")
        eval_count = payload.get("eval_count")
        total_tokens = None
        if prompt_eval_count is not None and eval_count is not None:
            total_tokens = prompt_eval_count + eval_count
        return LLMResponse(
            content=str(payload.get("response", "")),
            stop_reason="done" if payload.get("done") else None,
            raw_response=payload,
            prompt_tokens=prompt_eval_count,
            completion_tokens=eval_count,
            total_tokens=total_tokens,
        )


async def fetch_ollama_model_metadata(
    *,
    model: str,
    base_url: str,
    timeout_seconds: float,
) -> OllamaModelMetadata | None:
    base_url = base_url.rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            version_response = await client.get(f"{base_url}/api/version")
            version_response.raise_for_status()
            tags_response = await client.get(f"{base_url}/api/tags")
            tags_response.raise_for_status()
            show_response = await client.post(f"{base_url}/api/show", json={"model": model})
            show_response.raise_for_status()
    except httpx.HTTPError:
        return None

    version_payload: dict[str, Any] = version_response.json()
    tags_payload: dict[str, Any] = tags_response.json()
    show_payload: dict[str, Any] = show_response.json()
    tag = _find_tag(tags_payload, model)
    details = _dict_value(show_payload.get("details")) or _dict_value(tag.get("details"))
    model_info = _dict_value(show_payload.get("model_info")) or {}

    return OllamaModelMetadata(
        name=model,
        digest=_str_value(tag.get("digest")),
        family=_str_value(details.get("family")),
        parameter_size=_str_value(details.get("parameter_size")),
        quantization=_str_value(details.get("quantization_level")),
        context_window=_context_window(model_info),
        runtime_version=_str_value(version_payload.get("version")),
        modified_at=_str_value(tag.get("modified_at") or show_payload.get("modified_at")),
        size_bytes=_int_value(tag.get("size")),
    )


def _find_tag(tags_payload: dict[str, Any], model: str) -> dict[str, Any]:
    for item in tags_payload.get("models", []):
        if not isinstance(item, dict):
            continue
        if item.get("name") == model or item.get("model") == model:
            return item
    return {}


def _context_window(model_info: dict[str, Any]) -> int | None:
    for key, value in model_info.items():
        if key.endswith(".context_length"):
            return _int_value(value)
    return None


def _dict_value(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _str_value(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _int_value(value: object) -> int | None:
    return value if isinstance(value, int) else None
