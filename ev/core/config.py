from pathlib import Path
from decouple import Config, RepositoryEnv
from pydantic_settings import BaseSettings
import os
from typing import Literal

config = Config(RepositoryEnv(".env"))


class Settings(BaseSettings):
    EVALS_ROOT: Path = Path("evals")
    KEY_SOURCE: Literal["file", "env"] = "file"

    class Config:
        case_sensitive = True

    @property
    def OPENAI_API_KEY(self) -> str:
        source = self.KEY_SOURCE
        if source == "env":
            value = os.getenv("OPENAI_API_KEY")
            if not value:
                raise RuntimeError(
                    "OPENAI_API_KEY not set in environment (KEY_SOURCE='env')"
                )
            return value
        if source == "file":
            return config("OPENAI_API_KEY", cast=str)
        raise RuntimeError(f"Unsupported KEY_SOURCE '{source}'")

    @property
    def GROQ_API_KEY(self) -> str:
        source = self.KEY_SOURCE
        if source == "env":
            value = os.getenv("GROQ_API_KEY")
            if not value:
                raise RuntimeError(
                    "GROQ_API_KEY not set in environment (KEY_SOURCE='env')"
                )
            return value
        if source == "file":
            return config("GROQ_API_KEY", cast=str)
        raise RuntimeError(f"Unsupported KEY_SOURCE '{source}'")


settings = Settings()


def configure_key_source(source: str) -> None:
    normalized = (source or "").lower()
    if normalized not in ("file", "env"):
        raise ValueError("key source must be 'file' or 'env'")
    settings.KEY_SOURCE = normalized
