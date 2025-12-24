from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LiteLLM model string, e.g. "openai/gpt-4.1-mini"
    OPENAI_MODEL: str = "openai/gpt-4.1-mini"

    # MCP endpoint for Alpha Vantage tools
    MCP_ALPHA_URL: str = "http://localhost:8787/sse"

    QUOTE_CACHE_TTL_SEC: int = 20
    MAX_HISTORY_DAYS: int = 3650
    REQUESTS_PER_MIN_PER_USER: int = 30


settings = Settings()

