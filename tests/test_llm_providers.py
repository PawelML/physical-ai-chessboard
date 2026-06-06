import pytest

from arena_core.config import Settings
from arena_core.llm.providers import ProviderDisabledError, llm_service_for, parse_provider_model


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
