"""Handles GitHub repository cloning, metadata extraction, and source file discovery."""

import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Generator, Optional

import git
from git import GitCommandError, Repo

from backend.utils.config import get_config
from backend.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Maps file extension → language name used throughout the pipeline
SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".py":   "python",
    ".js":   "javascript",
    ".ts":   "typescript",
    ".jsx":  "javascript",
    ".tsx":  "typescript",
    ".java": "java",
    ".go":   "go",
    ".rs":   "rust",
    ".cpp":  "cpp",
    ".c":    "c",
    ".cs":   "csharp",
    ".rb":   "ruby",
    ".php":  "php",
}

# Directories to skip during file discovery
IGNORED_DIRS: set[str] = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    "env", ".env", "dist", "build", ".next", "target",
    ".idea", ".vscode", "coverage", ".pytest_cache", ".mypy_cache",
    "htmlcov", "site-packages", "migrations", ".tox",
}

# Individual filenames to always skip
IGNORED_FILES: set[str] = {
    ".gitignore", ".gitattributes", "package-lock.json",
    "yarn.lock", "poetry.lock", "Pipfile.lock", "Cargo.lock",
}


# ─────────────────────────────────────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FileInfo:
    """
    Represents a single source file ready for parsing and review.
    """
    path: str             # absolute path on disk
    relative_path: str    # path relative to repo root (used as display name)
    language: str         # e.g. "python", "typescript"
    size_bytes: int
    content: str
    line_count: int

    @property
    def size_kb(self) -> float:
        return round(self.size_bytes / 1024, 2)

    def __repr__(self) -> str:
        return f"FileInfo({self.relative_path!r}, {self.language}, {self.line_count} lines)"


@dataclass
class CommitInfo:
    """Metadata for a single git commit."""
    sha: str
    message: str
    author: str
    date: str


@dataclass
class RepositoryMetadata:
    """
    All metadata extracted from a cloned repository.
    Passed downstream to the dashboard and reports.
    """
    url: str
    name: str
    owner: str
    default_branch: str
    clone_path: str

    # Commit info
    last_commit: CommitInfo

    # File statistics
    total_files_on_disk: int    # all files including non-source
    supported_files: int        # source files the pipeline will review
    skipped_files: int          # files skipped (too large, unsupported, etc.)

    # Code statistics
    total_lines: int
    languages: dict[str, int]   # language → file count

    # Timestamps
    cloned_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    @property
    def primary_language(self) -> str:
        """The language with the most files."""
        if not self.languages:
            return "unknown"
        return max(self.languages, key=self.languages.get)  # type: ignore

    @property
    def repo_full_name(self) -> str:
        return f"{self.owner}/{self.name}"

    def to_dict(self) -> dict:
        """Serialise to plain dict for JSON export."""
        return {
            "url": self.url,
            "name": self.name,
            "owner": self.owner,
            "full_name": self.repo_full_name,
            "default_branch": self.default_branch,
            "clone_path": self.clone_path,
            "last_commit": {
                "sha": self.last_commit.sha,
                "message": self.last_commit.message,
                "author": self.last_commit.author,
                "date": self.last_commit.date,
            },
            "stats": {
                "total_files_on_disk": self.total_files_on_disk,
                "supported_files": self.supported_files,
                "skipped_files": self.skipped_files,
                "total_lines": self.total_lines,
                "languages": self.languages,
                "primary_language": self.primary_language,
            },
            "cloned_at": self.cloned_at,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Custom Exceptions
# ─────────────────────────────────────────────────────────────────────────────

class IngestionError(Exception):
    """Base exception for all ingestion failures."""
    pass


class InvalidRepositoryURLError(IngestionError):
    """Raised when the provided URL is not a valid GitHub repo URL."""
    pass


class RepositoryNotFoundError(IngestionError):
    """Raised when GitHub returns 404 or the repo does not exist."""
    pass


class AuthenticationError(IngestionError):
    """Raised when a GitHub token is required but missing or invalid."""
    pass


class CloneTimeoutError(IngestionError):
    """Raised when the clone operation exceeds the configured timeout."""
    pass


# ─────────────────────────────────────────────────────────────────────────────
# URL Validator
# ─────────────────────────────────────────────────────────────────────────────

class URLValidator:
    """
    Validates and normalises GitHub repository URLs.
    Handles HTTPS and SSH formats.
    """

    # Matches:  https://github.com/owner/repo  (with optional .git suffix)
    _HTTPS_RE = re.compile(
        r"^https?://github\.com/(?P<owner>[\w\-\.]+)/(?P<repo>[\w\-\.]+?)(?:\.git)?/?$"
    )
    # Matches:  git@github.com:owner/repo.git
    _SSH_RE = re.compile(
        r"^git@github\.com:(?P<owner>[\w\-\.]+)/(?P<repo>[\w\-\.]+?)(?:\.git)?$"
    )

    @classmethod
    def is_valid(cls, url: str) -> bool:
        url = url.strip()
        return bool(cls._HTTPS_RE.match(url) or cls._SSH_RE.match(url))

    @classmethod
    def extract_owner_repo(cls, url: str) -> tuple[str, str]:
        """
        Extract (owner, repo_name) from any supported GitHub URL format.
        Raises InvalidRepositoryURLError if the URL cannot be parsed.
        """
        url = url.strip()
        for pattern in (cls._HTTPS_RE, cls._SSH_RE):
            m = pattern.match(url)
            if m:
                return m.group("owner"), m.group("repo")
        raise InvalidRepositoryURLError(
            f"Cannot parse owner/repo from URL: {url!r}\n"
            "Expected format: https://github.com/owner/repository"
        )

    @classmethod
    def to_https(cls, url: str, token: Optional[str] = None) -> str:
        """
        Convert any GitHub URL to a clean HTTPS clone URL.
        Injects a token if provided (for private repos).
        """
        owner, repo = cls.extract_owner_repo(url)
        if token:
            return f"https://{token}@github.com/{owner}/{repo}.git"
        return f"https://github.com/{owner}/{repo}.git"


# ─────────────────────────────────────────────────────────────────────────────
# Clone Progress Adapter
# ─────────────────────────────────────────────────────────────────────────────

class _CloneProgress(git.RemoteProgress):
    """
    Adapts GitPython's progress callbacks to our progress_callback signature.
    progress_callback(message: str) -> None
    """

    def __init__(self, callback: Optional[Callable[[str], None]] = None):
        super().__init__()
        self._callback = callback

    def update(self, op_code, cur_count, max_count=None, message=""):
        if self._callback and message:
            pct = ""
            if max_count:
                pct = f" {int(cur_count / max_count * 100)}%"
            self._callback(f"Cloning{pct} — {message}")


# ─────────────────────────────────────────────────────────────────────────────
# Main Ingestion Class
# ─────────────────────────────────────────────────────────────────────────────

class RepositoryIngestion:
    """
    Handles the full repository ingestion pipeline:

      1. Validate URL
      2. Clone repository (shallow, depth=1)
      3. Extract metadata
      4. Discover all reviewable source files

    Usage:
        ingestion = RepositoryIngestion()
        metadata, files = ingestion.ingest("https://github.com/owner/repo")
    """

    def __init__(self):
        self.config = get_config()
        self._clone_base = Path(self.config.github.clone_dir)
        self._clone_base.mkdir(parents=True, exist_ok=True)
        self.validator = URLValidator()

    # ── Public API ────────────────────────────────────────────────────────────

    def validate_url(self, url: str) -> None:
        """
        Validate the URL and raise InvalidRepositoryURLError if invalid.
        Does NOT make a network request.
        """
        if not self.validator.is_valid(url.strip()):
            raise InvalidRepositoryURLError(
                f"'{url}' is not a valid GitHub repository URL.\n"
                "Valid formats:\n"
                "  • https://github.com/owner/repository\n"
                "  • https://github.com/owner/repository.git\n"
                "  • git@github.com:owner/repository.git"
            )

    def clone(
        self,
        url: str,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> tuple[Repo, Path]:
        """
        Clone the repository to a local directory.

        Args:
            url:               GitHub repository URL.
            progress_callback: Optional callable(message) for UI progress updates.

        Returns:
            (Repo, clone_path) tuple.

        Raises:
            InvalidRepositoryURLError, RepositoryNotFoundError,
            AuthenticationError, CloneTimeoutError, IngestionError
        """
        self.validate_url(url)
        owner, repo_name = self.validator.extract_owner_repo(url)
        clone_path = self._clone_base / f"{owner}_{repo_name}"

        # Always start fresh to avoid stale state
        if clone_path.exists():
            logger.info(f"Removing stale clone: {clone_path}")
            shutil.rmtree(clone_path, ignore_errors=True)

        clone_url = self.validator.to_https(url, token=self.config.github.token)
        safe_url  = self.validator.to_https(url)   # no token — safe to log

        logger.info(f"Cloning {owner}/{repo_name} → {clone_path}")
        if progress_callback:
            progress_callback(f"Starting clone of {owner}/{repo_name}…")

        try:
            progress = _CloneProgress(progress_callback) if progress_callback else None
            repo = Repo.clone_from(
                clone_url,
                str(clone_path),
                depth=1,          # shallow — we only need the latest snapshot
                progress=progress,
            )
            logger.info(f"Clone successful: {owner}/{repo_name}")
            if progress_callback:
                progress_callback("Clone complete ✓")
            return repo, clone_path

        except GitCommandError as exc:
            err = str(exc).lower()
            shutil.rmtree(clone_path, ignore_errors=True)

            if "not found" in err or "repository not found" in err or "does not exist" in err:
                raise RepositoryNotFoundError(
                    f"Repository not found: {safe_url}\n"
                    "• Check that the URL is correct.\n"
                    "• For private repos, provide GITHUB_TOKEN in your .env file."
                ) from exc

            if "authentication" in err or "could not read" in err or "403" in err or "401" in err:
                raise AuthenticationError(
                    "GitHub authentication failed.\n"
                    "• Add a valid GITHUB_TOKEN to your .env file.\n"
                    "• Token must have 'repo' scope for private repositories."
                ) from exc

            if "timed out" in err or "timeout" in err:
                raise CloneTimeoutError(
                    f"Clone timed out after {self.config.github.timeout_seconds}s.\n"
                    "The repository may be very large. Try increasing CLONE_TIMEOUT."
                ) from exc

            raise IngestionError(f"Git clone failed: {exc}") from exc

    def extract_metadata(
        self, repo: Repo, url: str, clone_path: Path
    ) -> RepositoryMetadata:
        """
        Extract structured metadata from a freshly cloned repository.
        """
        owner, repo_name = self.validator.extract_owner_repo(url)

        # ── Commit info ──────────────────────────────────────────────────────
        try:
            head    = repo.head.commit
            commit  = CommitInfo(
                sha     = head.hexsha[:8],
                message = head.message.strip()[:200],
                author  = str(head.author),
                date    = datetime.fromtimestamp(head.committed_date).isoformat(),
            )
        except Exception:
            commit = CommitInfo("unknown", "unknown", "unknown", "unknown")

        # ── Branch ──────────────────────────────────────────────────────────
        try:
            default_branch = repo.active_branch.name
        except TypeError:
            default_branch = "main"

        # ── File stats ───────────────────────────────────────────────────────
        all_paths   = [p for p in clone_path.rglob("*") if p.is_file()]
        total_files = len(all_paths)

        source_files = list(self._discover_files(clone_path))
        skipped      = total_files - len(source_files)

        languages:    dict[str, int] = {}
        total_lines   = 0
        for fi in source_files:
            languages[fi.language] = languages.get(fi.language, 0) + 1
            total_lines += fi.line_count

        return RepositoryMetadata(
            url                 = url,
            name                = repo_name,
            owner               = owner,
            default_branch      = default_branch,
            clone_path          = str(clone_path),
            last_commit         = commit,
            total_files_on_disk = total_files,
            supported_files     = len(source_files),
            skipped_files       = max(0, skipped),
            total_lines         = total_lines,
            languages           = languages,
        )

    def get_source_files(self, clone_path: str) -> list[FileInfo]:
        """
        Return all reviewable source files from a cloned repository path.
        This is the primary output consumed by the parser.
        """
        return list(self._discover_files(Path(clone_path)))

    def ingest(
        self,
        url: str,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> tuple[RepositoryMetadata, list[FileInfo]]:
        """
        Full ingestion pipeline entry point.

        Steps:
          1. Clone repository
          2. Extract metadata
          3. Discover source files

        Returns:
            (RepositoryMetadata, list[FileInfo])
        """
        repo, clone_path = self.clone(url, progress_callback)

        if progress_callback:
            progress_callback("Extracting repository metadata…")
        metadata = self.extract_metadata(repo, url, clone_path)

        if progress_callback:
            progress_callback("Discovering source files…")
        files = self.get_source_files(str(clone_path))

        logger.info(
            f"Ingestion complete — {metadata.repo_full_name} | "
            f"{len(files)} files | {metadata.total_lines} lines | "
            f"languages: {metadata.languages}"
        )
        return metadata, files

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _discover_files(self, root: Path) -> Generator[FileInfo, None, None]:
        """
        Walk the repository tree and yield FileInfo for each reviewable file.

        Rules:
          • Skips directories listed in IGNORED_DIRS
          • Skips files listed in IGNORED_FILES
          • Only yields files with extensions in SUPPORTED_EXTENSIONS
          • Skips files larger than max_file_size_kb
          • Stops after max_files files
          • Skips empty files and files with read errors
        """
        max_size_bytes  = self.config.github.max_file_size_kb * 1024
        max_files       = self.config.github.max_files
        file_count      = 0

        for file_path in sorted(root.rglob("*")):
            if file_count >= max_files:
                logger.warning(
                    f"Reached max file limit ({max_files}). "
                    "Increase MAX_FILES in .env to review more files."
                )
                break

            if not file_path.is_file():
                continue

            # ── Skip ignored directories ─────────────────────────────────────
            relative_parts = set(
                file_path.relative_to(root).parts[:-1]  # exclude filename
            )
            if relative_parts & IGNORED_DIRS:
                continue

            # ── Skip ignored filenames ───────────────────────────────────────
            if file_path.name in IGNORED_FILES:
                continue

            # ── Check extension ──────────────────────────────────────────────
            ext = file_path.suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue

            # ── Size guard ───────────────────────────────────────────────────
            try:
                size = file_path.stat().st_size
            except OSError:
                continue

            if size == 0:
                continue

            if size > max_size_bytes:
                logger.debug(
                    f"Skipping large file: {file_path.relative_to(root)} "
                    f"({size / 1024:.1f} KB > {self.config.github.max_file_size_kb} KB)"
                )
                continue

            # ── Read content ─────────────────────────────────────────────────
            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
            except Exception as exc:
                logger.warning(f"Cannot read {file_path}: {exc}")
                continue

            relative = str(file_path.relative_to(root))
            yield FileInfo(
                path          = str(file_path),
                relative_path = relative,
                language      = SUPPORTED_EXTENSIONS[ext],
                size_bytes    = size,
                content       = content,
                line_count    = content.count("\n") + 1,
            )
            file_count += 1
            logger.debug(f"Discovered: {relative} ({SUPPORTED_EXTENSIONS[ext]})")
