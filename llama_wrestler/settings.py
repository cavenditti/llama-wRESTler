from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str
    openai_base_url: str | None = None
    openai_model: str = "gpt-4o"
    openai_weak_model: str = (
        "gpt-4o-mini"  # Weaker/cheaper model for per-step data generation
    )

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
