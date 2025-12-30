from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LiteLLM model string, e.g. "openai/gpt-4.1-mini"
    OPENAI_MODEL: str = "openai/gpt-4.1-mini"

    # MCP endpoint for Alpha Vantage tools
    MCP_ALPHA_URL: str = "http://localhost:8787/sse"
    MCP_SHEETS_URL: str = "http://localhost:8790/sse"

    # Default spreadsheet to use when the caller does not provide one explicitly
    GOOGLE_SHEETS_SPREADSHEET_ID: str = ""

    QUOTE_CACHE_TTL_SEC: int = 20
    MAX_HISTORY_DAYS: int = 3650
    REQUESTS_PER_MIN_PER_USER: int = 30


settings = Settings()

