"""Ingestion package — repository cloning and file discovery."""

from backend.ingestion.repo_ingestion import (
    RepositoryIngestion,
    RepositoryMetadata,
    FileInfo,
    CommitInfo,
    URLValidator,
    IngestionError,
    InvalidRepositoryURLError,
    RepositoryNotFoundError,
    AuthenticationError,
    CloneTimeoutError,
    SUPPORTED_EXTENSIONS,
)

__all__ = [
    "RepositoryIngestion",
    "RepositoryMetadata",
    "FileInfo",
    "CommitInfo",
    "URLValidator",
    "IngestionError",
    "InvalidRepositoryURLError",
    "RepositoryNotFoundError",
    "AuthenticationError",
    "CloneTimeoutError",
    "SUPPORTED_EXTENSIONS",
]
