from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Database
    postgres_user: str = "ai_validator"
    postgres_password: str = "ai_validator_dev_pass"
    postgres_db: str = "ai_validator_db"
    postgres_host: str = "localhost"
    postgres_port: int = 5433
    database_url: str

    # LLM & Search Providers
    openai_api_key: str = ""
    gemini_api_key: str = ""
    groq_api_key: str = ""
    tavily_api_key: str = ""

    # Configurable Model Names (never hardcoded in agent logic)
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    gemini_model: str = "gemini-3.5-flash-lite"
    gemini_fallback_models: list[str] = [
        "gemini-3.1-flash-lite",
        "gemini-3.8-flash",
    ]
    groq_model: str = "qwen/qwen3.8-27b"

    # App Config
    app_env: str = "development"
    log_level: str = "INFO"
    validator_api_key: str = "aiv_dev_secret_123"


@lru_cache
def get_settings() -> Settings:
    return Settings()