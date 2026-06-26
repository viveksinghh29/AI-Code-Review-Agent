"""Unit tests for the repository ingestion module and related components."""

import tempfile
from pathlib import Path

import pytest

from backend.ingestion.repo_ingestion import (
    FileInfo,
    IngestionError,
    InvalidRepositoryURLError,
    RepositoryIngestion,
    RepositoryMetadata,
    URLValidator,
    SUPPORTED_EXTENSIONS,
    IGNORED_DIRS,
)


# ─────────────────────────────────────────────────────────────────────────────
# URLValidator
# ─────────────────────────────────────────────────────────────────────────────

class TestURLValidator:

    def test_valid_https_url(self):
        assert URLValidator.is_valid("https://github.com/pallets/flask") is True

    def test_valid_https_with_git_suffix(self):
        assert URLValidator.is_valid("https://github.com/pallets/flask.git") is True

    def test_valid_https_with_trailing_slash(self):
        assert URLValidator.is_valid("https://github.com/django/django/") is True

    def test_valid_ssh_url(self):
        assert URLValidator.is_valid("git@github.com:pallets/flask.git") is True

    def test_valid_org_with_dashes(self):
        assert URLValidator.is_valid("https://github.com/my-org/my-repo") is True

    def test_invalid_gitlab_url(self):
        assert URLValidator.is_valid("https://gitlab.com/owner/repo") is False

    def test_invalid_plain_string(self):
        assert URLValidator.is_valid("not-a-url") is False

    def test_invalid_owner_only(self):
        assert URLValidator.is_valid("https://github.com/only-owner") is False

    def test_invalid_empty_string(self):
        assert URLValidator.is_valid("") is False

    def test_extract_owner_repo_https(self):
        owner, repo = URLValidator.extract_owner_repo("https://github.com/pallets/flask")
        assert owner == "pallets"
        assert repo  == "flask"

    def test_extract_owner_repo_ssh(self):
        owner, repo = URLValidator.extract_owner_repo("git@github.com:django/django.git")
        assert owner == "django"
        assert repo  == "django"

    def test_extract_owner_repo_git_suffix_stripped(self):
        owner, repo = URLValidator.extract_owner_repo("https://github.com/a/b.git")
        assert repo == "b"

    def test_extract_invalid_url_raises(self):
        with pytest.raises(InvalidRepositoryURLError):
            URLValidator.extract_owner_repo("not-a-url")

    def test_to_https_no_token(self):
        url = URLValidator.to_https("https://github.com/a/b")
        assert url == "https://github.com/a/b.git"
        assert "token" not in url

    def test_to_https_with_token(self):
        url = URLValidator.to_https("https://github.com/a/b", token="ghp_abc")
        assert "ghp_abc@github.com" in url

    def test_to_https_from_ssh(self):
        url = URLValidator.to_https("git@github.com:a/b.git")
        assert "github.com/a/b.git" in url
        assert url.startswith("https://")


# ─────────────────────────────────────────────────────────────────────────────
# FileInfo
# ─────────────────────────────────────────────────────────────────────────────

class TestFileInfo:

    def test_size_kb_property(self):
        fi = FileInfo("a.py", "a.py", "python", 2048, "x=1\n", 1)
        assert fi.size_kb == 2.0

    def test_repr(self):
        fi = FileInfo("a.py", "a.py", "python", 100, "x=1\n", 1)
        assert "a.py" in repr(fi)
        assert "python" in repr(fi)


# ─────────────────────────────────────────────────────────────────────────────
# RepositoryIngestion — validate_url
# ─────────────────────────────────────────────────────────────────────────────

class TestRepositoryIngestionValidation:

    def setup_method(self):
        self.ingestion = RepositoryIngestion()

    def test_valid_url_does_not_raise(self):
        self.ingestion.validate_url("https://github.com/pallets/flask")

    def test_invalid_url_raises(self):
        with pytest.raises(InvalidRepositoryURLError):
            self.ingestion.validate_url("not-a-url")

    def test_gitlab_url_raises(self):
        with pytest.raises(InvalidRepositoryURLError):
            self.ingestion.validate_url("https://gitlab.com/owner/repo")


# ─────────────────────────────────────────────────────────────────────────────
# RepositoryIngestion — file discovery
# ─────────────────────────────────────────────────────────────────────────────

class TestFileDiscovery:

    def setup_method(self):
        self.ingestion = RepositoryIngestion()

    def _make_repo(self, tmp_path: Path, files: dict[str, str]) -> Path:
        """Create a fake repo directory with the given {relative_path: content}."""
        for rel, content in files.items():
            p = tmp_path / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
        return tmp_path

    def test_discovers_python_files(self, tmp_path):
        self._make_repo(tmp_path, {"main.py": "x=1\n", "utils.py": "y=2\n"})
        files = self.ingestion.get_source_files(str(tmp_path))
        names = [f.relative_path for f in files]
        assert "main.py"  in names
        assert "utils.py" in names

    def test_discovers_multiple_languages(self, tmp_path):
        self._make_repo(tmp_path, {
            "app.py":  "x=1\n",
            "app.js":  "var x=1;\n",
            "app.ts":  "const x:number=1;\n",
        })
        files = self.ingestion.get_source_files(str(tmp_path))
        langs = {f.language for f in files}
        assert "python"     in langs
        assert "javascript" in langs
        assert "typescript" in langs

    def test_ignores_unsupported_extensions(self, tmp_path):
        self._make_repo(tmp_path, {
            "main.py":   "x=1\n",
            "README.md": "# readme",
            "data.csv":  "a,b,c",
            "img.png":   "binary",
        })
        files = self.ingestion.get_source_files(str(tmp_path))
        names = [f.relative_path for f in files]
        assert "README.md" not in names
        assert "data.csv"  not in names
        assert "img.png"   not in names
        assert "main.py"   in names

    def test_ignores_node_modules(self, tmp_path):
        self._make_repo(tmp_path, {
            "src/app.py":                "x=1\n",
            "node_modules/lib.js":       "module.exports={};\n",
            "node_modules/deep/lib2.js": "x=1;\n",
        })
        files = self.ingestion.get_source_files(str(tmp_path))
        names = [f.relative_path for f in files]
        assert not any("node_modules" in n for n in names)
        assert "src/app.py" in names

    def test_ignores_venv_directory(self, tmp_path):
        self._make_repo(tmp_path, {
            "app.py":           "x=1\n",
            "venv/lib/util.py": "y=2\n",
        })
        files = self.ingestion.get_source_files(str(tmp_path))
        names = [f.relative_path for f in files]
        assert not any("venv" in n for n in names)

    def test_ignores_empty_files(self, tmp_path):
        self._make_repo(tmp_path, {"empty.py": "", "main.py": "x=1\n"})
        files = self.ingestion.get_source_files(str(tmp_path))
        names = [f.relative_path for f in files]
        assert "empty.py" not in names
        assert "main.py"  in names

    def test_skips_oversized_files(self, tmp_path):
        from backend.utils.config import get_config
        cfg      = get_config()
        big_size = cfg.github.max_file_size_kb * 1024 + 1
        big_path = tmp_path / "huge.py"
        big_path.write_bytes(b"x" * big_size)
        (tmp_path / "small.py").write_text("x=1\n")

        files = self.ingestion.get_source_files(str(tmp_path))
        names = [f.relative_path for f in files]
        assert "huge.py"  not in names
        assert "small.py" in names

    def test_file_info_attributes(self, tmp_path):
        content = "def hello():\n    return 'hi'\n"
        self._make_repo(tmp_path, {"hello.py": content})
        files = self.ingestion.get_source_files(str(tmp_path))
        assert len(files) == 1
        fi = files[0]
        assert fi.language   == "python"
        assert fi.line_count == content.count("\n") + 1
        assert fi.content    == content
        assert fi.size_bytes  > 0

    def test_nested_file_discovery(self, tmp_path):
        self._make_repo(tmp_path, {
            "a/b/c/deep.py": "x=1\n",
            "top.py":         "y=2\n",
        })
        files = self.ingestion.get_source_files(str(tmp_path))
        names = [f.relative_path for f in files]
        assert "top.py"     in names
        assert any("deep.py" in n for n in names)

    def test_supported_extensions_coverage(self):
        expected = {".py", ".js", ".ts", ".java", ".go", ".rs",
                    ".cpp", ".c", ".cs", ".rb", ".php", ".jsx", ".tsx"}
        for ext in expected:
            assert ext in SUPPORTED_EXTENSIONS, f"{ext} not in SUPPORTED_EXTENSIONS"


# ─────────────────────────────────────────────────────────────────────────────
# RepositoryMetadata
# ─────────────────────────────────────────────────────────────────────────────

class TestRepositoryMetadata:

    def _make_meta(self, languages: dict) -> RepositoryMetadata:
        from backend.ingestion.repo_ingestion import CommitInfo
        return RepositoryMetadata(
            url="https://github.com/a/b", name="b", owner="a",
            default_branch="main", clone_path="/tmp/a_b",
            last_commit=CommitInfo("abc", "msg", "author", "2025-01-01"),
            total_files_on_disk=10, supported_files=5,
            skipped_files=5, total_lines=200, languages=languages,
        )

    def test_primary_language_single(self):
        meta = self._make_meta({"python": 5})
        assert meta.primary_language == "python"

    def test_primary_language_most_files(self):
        meta = self._make_meta({"python": 8, "javascript": 3})
        assert meta.primary_language == "python"

    def test_primary_language_empty(self):
        meta = self._make_meta({})
        assert meta.primary_language == "unknown"

    def test_repo_full_name(self):
        meta = self._make_meta({})
        assert meta.repo_full_name == "a/b"

    def test_to_dict_keys(self):
        meta = self._make_meta({"python": 2})
        d = meta.to_dict()
        assert "url"    in d
        assert "stats"  in d
        assert "last_commit" in d
        assert d["stats"]["primary_language"] == "python"
