import httpx
import pytest

from arena_core.config import Settings
from arena_core.llm.ollama import OllamaLLMService, _ollama_error_message
from arena_core.llm.providers import (
    GeminiLLMService,
    ProviderDisabledError,
    llm_service_for,
    parse_provider_model,
)


def test_parse_provider_model_defaults_to_local() -> None:
    parsed = parse_provider_model("qwen3.5:9b")

    assert parsed.provider == "local"
    assert parsed.model == "qwen3.5:9b"


def test_parse_provider_model_accepts_explicit_provider() -> None:
    parsed = parse_provider_model("openai:gpt-4.1-mini")

    assert parsed.provider == "openai"
    assert parsed.model == "gpt-4.1-mini"


def test_api_provider_is_disabled_by_default() -> None:
    with pytest.raises(ProviderDisabledError):
        llm_service_for("openai:gpt-4.1-mini", Settings())


def test_ollama_error_message_includes_response_error_body() -> None:
    response = httpx.Response(
        500,
        json={"error": "CUDA error: out of memory"},
        request=httpx.Request("POST", "http://localhost:11434/api/generate"),
    )

    assert _ollama_error_message(response) == (
        "Ollama generate failed (500): CUDA error: out of memory"
    )


async def test_ollama_retries_without_thinking_when_model_rejects_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    class FakeAsyncClient:
        def __init__(self, *, timeout: float) -> None:
            self.timeout = timeout

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def post(self, url: str, json: dict[str, object]) -> httpx.Response:
            calls.append(json.copy())
            request = httpx.Request("POST", url)
            if json["think"]:
                return httpx.Response(
                    400,
                    json={"error": '"qwen3.5-27b-ud:latest" does not support thinking'},
                    request=request,
                )
            return httpx.Response(
                200,
                json={"response": '{"move":"e7e5"}', "done": True},
                request=request,
            )

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    service = OllamaLLMService(
        base_url="http://localhost:11434",
        timeout_seconds=120,
        think="auto",
    )

    response = await service.complete(model="qwen3.5-27b-ud:latest", prompt="prompt")

    assert response.content == '{"move":"e7e5"}'
    assert [call["think"] for call in calls] == [True, False]


async def test_gemini_retries_timeout_before_success(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    class FakeAsyncClient:
        def __init__(self, *, timeout: float) -> None:
            self.timeout = timeout

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def post(self, url: str, **kwargs: object) -> httpx.Response:
            nonlocal calls
            calls += 1
            if calls < 3:
                raise httpx.ReadTimeout("timed out")
            return httpx.Response(
                200,
                json={
                    "candidates": [{"content": {"parts": [{"text": '{"move":"e2e4"}'}]}}],
                    "usageMetadata": {
                        "promptTokenCount": 12,
                        "candidatesTokenCount": 5,
                        "totalTokenCount": 17,
                    },
                },
                request=httpx.Request("POST", url),
            )

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr("arena_core.llm.providers.asyncio.sleep", lambda _delay: _noop())
    service = GeminiLLMService(api_key="test-key", timeout_seconds=10, max_retries=2)

    response = await service.complete(model="gemini-test", prompt="prompt")

    assert response.content == '{"move":"e2e4"}'
    assert response.total_tokens == 17
    assert calls == 3


async def test_gemini_does_not_retry_bad_request(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    class FakeAsyncClient:
        def __init__(self, *, timeout: float) -> None:
            self.timeout = timeout

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def post(self, url: str, **kwargs: object) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(400, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    service = GeminiLLMService(api_key="test-key", timeout_seconds=10, max_retries=2)

    with pytest.raises(httpx.HTTPStatusError):
        await service.complete(model="gemini-test", prompt="prompt")

    assert calls == 1


async def _noop() -> None:
    return None


@pytest.mark.parametrize(
    "error_message",
    [
        "CUDA error: out of memory",
        "memory layout cannot be allocated with num_gpu = 48",
    ],
)
async def test_ollama_retries_with_fewer_gpu_layers_after_gpu_memory_error(
    monkeypatch: pytest.MonkeyPatch,
    error_message: str,
) -> None:
    calls: list[int] = []

    class FakeAsyncClient:
        def __init__(self, *, timeout: float) -> None:
            self.timeout = timeout

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def post(self, url: str, json: dict[str, object]) -> httpx.Response:
            options = json["options"]
            assert isinstance(options, dict)
            num_gpu = options["num_gpu"]
            assert isinstance(num_gpu, int)
            calls.append(num_gpu)
            request = httpx.Request("POST", url)
            if num_gpu > 32:
                return httpx.Response(
                    500,
                    json={"error": error_message},
                    request=request,
                )
            return httpx.Response(
                200,
                json={"response": '{"move":"e7e5"}', "done": True},
                request=request,
            )

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    service = OllamaLLMService(
        base_url="http://localhost:11434",
        timeout_seconds=120,
        num_gpu=48,
        min_num_gpu=8,
    )

    first_response = await service.complete(model="qwen3.5-27b-ud:latest", prompt="prompt")
    second_response = await service.complete(model="qwen3.5-27b-ud:latest", prompt="prompt")

    assert first_response.content == '{"move":"e7e5"}'
    assert second_response.content == '{"move":"e7e5"}'
    assert calls == [48, 40, 32, 32]
