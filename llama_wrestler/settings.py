from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings with flexible LLM provider configuration.

    Model strings follow pydantic-ai format: "provider:model-name"

    Supported providers:
    - openai: OpenAI models (gpt-4o, gpt-4o-mini, etc.)
    - anthropic: Anthropic models (claude-sonnet-4-5, claude-3-5-haiku-latest, etc.)
    - google: Google models (gemini-2.5-pro, gemini-2.0-flash, etc.)
    - huggingface: Hugging Face models (Qwen/Qwen3-235B-A22B, etc.)
    - groq: Groq models (llama-3.3-70b-versatile, etc.)
    - mistral: Mistral models (mistral-large-latest, etc.)
    - together: Together AI models (meta-llama/Llama-3.3-70B-Instruct-Turbo-Free, etc.)
    - cerebras: Cerebras models (llama3.3-70b, etc.)
    - fireworks: Fireworks AI models

    OpenAI-compatible providers (use openai: prefix with OPENAI_BASE_URL):
    - OpenRouter: Set OPENAI_BASE_URL=https://openrouter.ai/api/v1
    - Ollama: Set OPENAI_BASE_URL=http://localhost:11434/v1
    - LiteLLM: Set OPENAI_BASE_URL=http://localhost:4000
    - Any OpenAI-compatible API

    Each provider requires its corresponding API key environment variable:
    - OPENAI_API_KEY for openai: (also used for OpenRouter, Ollama, etc.)
    - ANTHROPIC_API_KEY for anthropic:
    - GOOGLE_API_KEY or GEMINI_API_KEY for google:
    - HUGGINGFACE_API_KEY for huggingface:
    - GROQ_API_KEY for groq:
    - MISTRAL_API_KEY for mistral:
    - TOGETHER_API_KEY for together:
    - CEREBRAS_API_KEY for cerebras:
    - FIREWORKS_API_KEY for fireworks:

    Examples:
        # OpenAI (default)
        MODEL=openai:gpt-4o
        WEAK_MODEL=openai:gpt-4o-mini

        # Anthropic
        MODEL=anthropic:claude-sonnet-4-5
        WEAK_MODEL=anthropic:claude-3-5-haiku-latest

        # Google
        MODEL=google:gemini-2.5-pro
        WEAK_MODEL=google:gemini-2.0-flash

        # OpenRouter (access 100+ models via one API)
        MODEL=openai:anthropic/claude-sonnet-4
        WEAK_MODEL=openai:openai/gpt-4o-mini
        OPENAI_API_KEY=sk-or-v1-...
        OPENAI_BASE_URL=https://openrouter.ai/api/v1

        # Mixed providers
        MODEL=anthropic:claude-sonnet-4-5
        WEAK_MODEL=openai:gpt-4o-mini
    """

    # Primary model for test plan generation and refinement (complex reasoning tasks)
    model: str = "openai:gpt-4o"

    # Weak model for summarization/recap tasks (faster, cheaper)
    weak_model: str = "openai:gpt-4o-mini"

    # Maximum number of passes to attempt for full endpoint coverage in preliminary phase
    max_coverage_passes: int = 3

    # Legacy OpenAI-specific settings (kept for backwards compatibility)
    # These are only used if the model string doesn't include a provider prefix
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    openai_model: str | None = None  # Deprecated: use MODEL instead
    openai_weak_model: str | None = None  # Deprecated: use WEAK_MODEL instead

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # Ignore extra env vars not defined in the model
    )

    def get_model(self) -> str:
        """Get the primary model string for pydantic-ai."""
        # Check for legacy setting first for backwards compatibility
        if self.openai_model:
            # If it already has a provider prefix, use as-is
            if ":" in self.openai_model:
                return self.openai_model
            # Otherwise, assume it's an OpenAI model
            return f"openai:{self.openai_model}"
        return self.model

    def get_weak_model(self) -> str:
        """Get the weak model string for pydantic-ai."""
        # Check for legacy setting first for backwards compatibility
        if self.openai_weak_model:
            # If it already has a provider prefix, use as-is
            if ":" in self.openai_weak_model:
                return self.openai_weak_model
            # Otherwise, assume it's an OpenAI model
            return f"openai:{self.openai_weak_model}"
        return self.weak_model


settings = Settings()
