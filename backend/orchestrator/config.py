"""Orchestrator settings (env vars, see .env.example)."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", case_sensitive=False)

    # --- wiring -----------------------------------------------------------
    redis_url: str = "redis://redis:6379/0"
    internal_token: str  # shared secret between frontend <-> backend (required)
    vllm_url: str = "http://vllm:8000"
    media_url: str = "http://video:1376"
    asr_url: str = "http://audio:8000"
    vllm_container: str = "container_vllm"
    compose_project: str = "aistack"

    # --- storage ----------------------------------------------------------
    data_dir: str = "/data"
    llm_config_path: str = "/config/llm.env"

    # --- public /api compatibility endpoint ------------------------------
    api_keys: str = ""  # comma separated bearer tokens; empty = /api disabled
    public_base_url: str = ""  # e.g. https://example.com (used in /api image URLs)

    # --- limits & scheduling ---------------------------------------------
    max_jobs_per_user: int = 2
    max_video_jobs_per_user: int = 1
    chat_rpm: int = 30  # chat requests per user per minute
    chat_max_tokens: int = 8192
    chat_wait_timeout: float = 900  # how long a chat may wait for the GPU
    drain_timeout: float = 60  # wait for running chats before sleeping vLLM
    batch_max_jobs: int = 4  # same-type jobs run back-to-back before yielding
    batch_max_seconds: float = 120
    llm_min_window: float = 20  # min seconds chats get between media batches
    job_ttl: int = 24 * 3600
    llm_wake_timeout: float = 600
    llm_deploy_timeout: float = 3600

    @property
    def api_key_set(self) -> set[str]:
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
