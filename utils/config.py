import os
from functools import lru_cache
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr, ValidationError

"""
We are using this class to help us import secrets from the .env to our project files.

An example of how we use this can be found in the `dialogue/dialoge_manager.py` file
"""
class Settings(BaseModel):
    # OpenAI / Whisper
    openai_api_key: SecretStr = Field(alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    whisper_model: str = Field(default="whisper-1", alias="WHISPER_MODEL")
    whisper_language: str = Field(default="en", alias="WHISPER_LANGUAGE")
    whisper_temperature: float = Field(
        default=0.0, alias="WHISPER_TEMPERATURE", ge=0.0, le=2.0
    )

    # Generation controls
    max_tokens: int = Field(default=50, alias="MAX_TOKENS", ge=1, le=200000)
    temperature: float = Field(default=0.7, alias="TEMPERATURE", ge=0.0, le=2.0)

    # App behavior
    max_conversation_history: int = Field(
        default=5, alias="MAX_CONVERSATION_HISTORY", ge=0, le=100000
    )
    enable_caching: bool = Field(default=False, alias="ENABLE_CACHING")

    model_config = {
        "populate_by_name": True,
        "extra": "ignore",
        "frozen": True,
    }


def load_env_from_conf() -> None:
    # Load conf/.env relative to project root (two dirs up from this file)
    env_path = Path(__file__).resolve().parent.parent / "conf" / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)


@lru_cache
def get_settings() -> Settings:
    try:
        load_env_from_conf()
        return Settings(
            OPENAI_API_KEY=os.getenv("OPENAI_API_KEY"),
            OPENAI_MODEL=os.getenv("OPENAI_MODEL"),
            WHISPER_MODEL=os.getenv("WHISPER_MODEL"),
            WHISPER_LANGUAGE=os.getenv("WHISPER_LANGUAGE"),
            WHISPER_TEMPERATURE=os.getenv("WHISPER_TEMPERATURE"),
            MAX_TOKENS=os.getenv("MAX_TOKENS"),
            TEMPERATURE=os.getenv("TEMPERATURE"),
            MAX_CONVERSATION_HISTORY=os.getenv("MAX_CONVERSATION_HISTORY"),
            ENABLE_CACHING=os.getenv("ENABLE_CACHING"),
        )
    except ValidationError as e:
        raise SystemExit(f"Invalid configuration:\n{e}") from e