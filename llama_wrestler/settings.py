from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str
    openai_base_url: str | None = None
    openai_model: str = "gpt-4o"  # Used for test plan generation (preliminary phase)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # Ignore extra env vars not defined in the model
    )


settings = Settings()
