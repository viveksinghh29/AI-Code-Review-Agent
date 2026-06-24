"""
Configuration Management for AI Code Review Agent.

Loads all settings from environment variables (.env file).
Provides a typed, singleton Config object used throughout the app.
"""

import os
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

# Load .env automatically when this module is first imported
load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# Sub-configs (grouped by concern)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LLMConfig:
    """Settings for the AI / LLM provider."""
    provider: str = field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "anthropic")
    )
    anthropic_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY")
    )
    openai_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )
    anthropic_model: str = field(
        default_factory=lambda: os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
    )
    openai_model: str = field(
        default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    )
    max_tokens: int = field(
        default_factory=lambda: int(os.getenv("MAX_TOKENS", "4096"))
    )
    temperature: float = field(
        default_factory=lambda: float(os.getenv("TEMPERATURE", "0.2"))
    )
    max_retries: int = field(
        default_factory=lambda: int(os.getenv("MAX_RETRIES", "3"))
    )
    chunk_size: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_SIZE", "3000"))
    )

    @property
    def active_model(self) -> str:
        """Return the model name for the currently active provider."""
        if self.provider == "anthropic":
            return self.anthropic_model
        return self.openai_model

    @property
    def active_api_key(self) -> Optional[str]:
        """Return the API key for the currently active provider."""
        if self.provider == "anthropic":
            return self.anthropic_api_key
        return self.openai_api_key


@dataclass
class GitHubConfig:
    """Settings for GitHub access and repository cloning."""
    token: Optional[str] = field(
        default_factory=lambda: os.getenv("GITHUB_TOKEN")
    )
    clone_dir: str = field(
        default_factory=lambda: os.getenv("CLONE_DIR", "/tmp/ai_code_review_repos")
    )
    max_file_size_kb: int = field(
        default_factory=lambda: int(os.getenv("MAX_FILE_SIZE_KB", "500"))
    )
    max_files: int = field(
        default_factory=lambda: int(os.getenv("MAX_FILES", "50"))
    )
    timeout_seconds: int = field(
        default_factory=lambda: int(os.getenv("CLONE_TIMEOUT", "120"))
    )


@dataclass
class AppConfig:
    """General application settings."""
    debug: bool = field(
        default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true"
    )
    log_level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )
    reports_dir: str = field(
        default_factory=lambda: os.getenv("REPORTS_DIR", "reports")
    )
    max_parallel_reviews: int = field(
        default_factory=lambda: int(os.getenv("MAX_PARALLEL_REVIEWS", "5"))
    )
    cache_enabled: bool = field(
        default_factory=lambda: os.getenv("CACHE_ENABLED", "true").lower() == "true"
    )
    db_url: str = field(
        default_factory=lambda: os.getenv("DATABASE_URL", "sqlite:///reviews.db")
    )


# ─────────────────────────────────────────────────────────────────────────────
# Root config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    """
    Root configuration object.
    Instantiated once and reused throughout the application.
    """
    llm: LLMConfig = field(default_factory=LLMConfig)
    github: GitHubConfig = field(default_factory=GitHubConfig)
    app: AppConfig = field(default_factory=AppConfig)

    def validate(self) -> list[str]:
        """
        Validate required settings.
        Returns a list of human-readable error messages.
        An empty list means the config is valid.
        """
        errors: list[str] = []

        provider = self.llm.provider.lower()
        if provider not in ("anthropic", "openai"):
            errors.append(
                f"LLM_PROVIDER must be 'anthropic' or 'openai', got '{provider}'."
            )
        if provider == "anthropic" and not self.llm.anthropic_api_key:
            errors.append(
                "ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic."
            )
        if provider == "openai" and not self.llm.openai_api_key:
            errors.append(
                "OPENAI_API_KEY is required when LLM_PROVIDER=openai."
            )

        return errors

    def is_github_enabled(self) -> bool:
        """True when a GitHub token is configured."""
        return bool(self.github.token)

    def summary(self) -> dict:
        """Return a safe (no secrets) config summary for logging/display."""
        return {
            "llm_provider": self.llm.provider,
            "llm_model": self.llm.active_model,
            "llm_key_set": bool(self.llm.active_api_key),
            "github_token_set": self.is_github_enabled(),
            "clone_dir": self.github.clone_dir,
            "max_files": self.github.max_files,
            "debug": self.app.debug,
            "log_level": self.app.log_level,
            "cache_enabled": self.app.cache_enabled,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Singleton accessor
# ─────────────────────────────────────────────────────────────────────────────

_config: Optional[Config] = None


def get_config() -> Config:
    """Return the global Config singleton, creating it if necessary."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def reload_config() -> Config:
    """Force-reload config from environment variables (useful in tests)."""
    global _config
    load_dotenv(override=True)
    _config = Config()
    return _config
