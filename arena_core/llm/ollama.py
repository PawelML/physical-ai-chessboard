from dataclasses import dataclass
from typing import Any

import httpx

from arena_core.llm.base import LLMResponse, LLMService


class OllamaServiceError(RuntimeError):
    pass


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
    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: float,
        temperature: float = 0.0,
        top_p: float | None = None,
        num_ctx: int | None = None,
        num_predict: int | None = None,
        num_gpu: int | None = None,
        min_num_gpu: int | None = None,
        think: str = "off",
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds
        self._temperature = temperature
        self._top_p = top_p
        self._num_ctx = num_ctx
        self._num_predict = num_predict
        self._num_gpu = num_gpu
        self._min_num_gpu = min_num_gpu
        self._think = think
        self._working_num_gpu = num_gpu

    async def complete(self, *, model: str, prompt: str) -> LLMResponse:
        options: dict[str, float | int] = {"temperature": self._temperature}
        if self._top_p is not None:
            options["top_p"] = self._top_p
        if self._num_ctx is not None:
            options["num_ctx"] = self._num_ctx
        if self._num_predict is not None:
            options["num_predict"] = self._num_predict
        if self._num_gpu is not None:
            options["num_gpu"] = self._num_gpu

        request_payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "think": self._should_think(model),
            "options": options,
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = None
                for candidate_num_gpu in self._num_gpu_candidates():
                    if candidate_num_gpu is None:
                        options.pop("num_gpu", None)
                    else:
                        options["num_gpu"] = candidate_num_gpu
                    response = await self._post_generate(client, request_payload)
                    if _is_gpu_memory_error_response(response) and candidate_num_gpu not in {
                        None,
                        0,
                    }:
                        continue
                    self._working_num_gpu = candidate_num_gpu
                    break
                if response is None:
                    raise OllamaServiceError("Ollama generate failed: no request was attempted")
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    raise OllamaServiceError(_ollama_error_message(exc.response)) from exc
        except httpx.TimeoutException as exc:
            raise OllamaServiceError(
                f"Ollama generate timed out after {self._timeout:.0f}s for model {model!r}. "
                "A cold model load (large models can take >100s) likely exceeded the timeout; "
                "raise ARENA_OLLAMA_TIMEOUT_SECONDS or keep the model resident."
            ) from exc
        response_payload = response.json()
        content = str(response_payload.get("response") or response_payload.get("thinking") or "")
        thinking_text = response_payload.get("thinking")
        prompt_eval_count = response_payload.get("prompt_eval_count")
        eval_count = response_payload.get("eval_count")
        total_tokens = None
        if prompt_eval_count is not None and eval_count is not None:
            total_tokens = prompt_eval_count + eval_count
        return LLMResponse(
            content=content,
            stop_reason="done" if response_payload.get("done") else None,
            raw_response=response_payload,
            prompt_tokens=prompt_eval_count,
            completion_tokens=eval_count,
            total_tokens=total_tokens,
            thinking=str(thinking_text) if thinking_text else None,
            thinking_used=bool(request_payload.get("think")),
        )

    async def _post_generate(
        self,
        client: httpx.AsyncClient,
        payload: dict[str, object],
    ) -> httpx.Response:
        response = await client.post(f"{self._base_url}/api/generate", json=payload)
        if _is_unsupported_thinking_response(response) and payload["think"]:
            payload["think"] = False
            response = await client.post(f"{self._base_url}/api/generate", json=payload)
        return response

    def _num_gpu_candidates(self) -> list[int | None]:
        if self._num_gpu is None:
            return [None]
        if self._working_num_gpu is not None and self._working_num_gpu != self._num_gpu:
            return [self._working_num_gpu]
        minimum = self._min_num_gpu if self._min_num_gpu is not None else 0
        minimum = min(minimum, self._num_gpu)
        candidates: list[int | None] = list(range(self._num_gpu, max(minimum, 1) - 1, -8))
        if minimum == 0 and 0 not in candidates:
            candidates.append(0)
        return candidates

    def _should_think(self, model: str) -> bool:
        if self._think == "on":
            return True
        if self._think == "auto":
            return "qwen" in model.lower()
        return False


def _ollama_error_message(response: httpx.Response) -> str:
    detail = response.text.strip()
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if isinstance(payload, dict) and isinstance(payload.get("error"), str):
        detail = payload["error"].strip()
    if detail:
        return f"Ollama generate failed ({response.status_code}): {detail}"
    return f"Ollama generate failed ({response.status_code})"


def _is_unsupported_thinking_response(response: httpx.Response) -> bool:
    if response.status_code != 400:
        return False
    try:
        payload = response.json()
    except ValueError:
        return False
    error = payload.get("error") if isinstance(payload, dict) else None
    return isinstance(error, str) and "does not support thinking" in error


def _is_gpu_memory_error_response(response: httpx.Response) -> bool:
    if response.status_code < 500:
        return False
    try:
        payload = response.json()
    except ValueError:
        error = response.text
    else:
        raw_error = payload.get("error") if isinstance(payload, dict) else None
        error = raw_error if isinstance(raw_error, str) else response.text
    normalized = error.lower()
    return (
        ("cuda" in normalized and "out of memory" in normalized)
        or "memory layout cannot be allocated" in normalized
        or "failed to allocate" in normalized
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
