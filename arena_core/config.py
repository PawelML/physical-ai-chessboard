from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings shared by the CLI and backend shell."""

    model_config = SettingsConfigDict(
        env_prefix="ARENA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = "sqlite+aiosqlite:///./arena.db"
    prompt_retention_enabled: bool = True
    prompt_version: str = "strict-v7"
    max_retries: int = Field(default=3, ge=0, le=10)
    ollama_base_url: str = "http://localhost:11434"
    ollama_timeout_seconds: float = Field(default=600.0, gt=0)
    ollama_temperature: float = Field(default=0.0, ge=0.0)
    ollama_top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    ollama_num_ctx: int | None = Field(default=None, gt=0)
    ollama_num_predict: int | None = Field(default=None, gt=0)
    ollama_num_gpu: int | None = Field(default=None, ge=0)
    ollama_cpu_offload_gpu_layers: int = Field(default=48, ge=1)
    ollama_cpu_offload_min_gpu_layers: int = Field(default=8, ge=0)
    ollama_think: str = "off"
    api_providers_enabled: bool = False
    api_timeout_seconds: float = Field(default=120.0, gt=0)
    api_max_retries: int = Field(default=2, ge=0, le=5)
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    gemini_api_key: str | None = None
    gemini_model: str | None = None
    default_context_window: int = Field(default=8192, gt=0)
    stockfish_path: str | None = None
    stockfish_nodes: int = Field(default=200_000, gt=0)
    stockfish_threads: int = Field(default=1, gt=0)
    stockfish_hash_mb: int = Field(default=128, gt=0)
    stockfish_skill: int | None = Field(default=None, ge=0, le=20)
    stockfish_limit_strength: bool | None = None
    stockfish_target_elo: int | None = Field(default=None, ge=1320, le=3190)
    ollama_vram_budget_gb: float = Field(default=24.0, gt=0)
    reranker_n_candidates: int = Field(default=5, ge=1, le=32)
    reranker_temperature: float = Field(default=0.8, ge=0.0)
    reranker_veto_cpl_threshold: int = Field(default=300, ge=0)
    reranker_veto_nodes: int = Field(default=50_000, gt=0)
    reranker_scorer: str = "stockfish"
    deliberation_mode: str = "revise"
    deliberation_n_candidates: int = Field(default=5, ge=1, le=16)
    deliberation_candidate_temperature: float = Field(default=0.7, ge=0.0)
    deliberation_critic_temperature: float = Field(default=0.0, ge=0.0)
    deliberation_final_temperature: float = Field(default=0.0, ge=0.0)
    deliberation_max_opponent_replies: int = Field(default=40, ge=0)
    deliberation_max_analysis_tokens: int = Field(default=1024, gt=0)
    deliberation_max_final_tokens: int = Field(default=64, gt=0)
    deliberation_max_pairwise_tokens: int = Field(default=24, gt=0)
    deliberation_pairwise_min_vote_margin: int = Field(default=0, ge=0)
    deliberation_pairwise_critic_model: str | None = None
    deliberation_persist_intermediate_prompts: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
