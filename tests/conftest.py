"""
Shared pytest fixtures for the AI Code Review Agent test suite.
"""

import json
import pytest
from unittest.mock import patch

from backend.ingestion.repo_ingestion import CommitInfo, FileInfo, RepositoryMetadata
from backend.parser.ast_parser import PythonASTParser
from backend.reviewer.ai_reviewer import FileReview, ReviewComment, ReviewReport


# ─────────────────────────────────────────────────────────────────────────────
# Common file fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def simple_file():
    return FileInfo(
        path="simple.py", relative_path="simple.py",
        language="python", size_bytes=50,
        content="x = 1\ny = 2\n",
        line_count=2,
    )


@pytest.fixture
def documented_file():
    content = (
        '"""Module docstring."""\n\n'
        "def add(x: int, y: int) -> int:\n"
        '    """Return x + y."""\n'
        "    return x + y\n\n"
        "class Calculator:\n"
        '    """Simple calculator."""\n'
        "    def multiply(self, a: int, b: int) -> int:\n"
        '        """Multiply a by b."""\n'
        "        return a * b\n"
    )
    return FileInfo(
        path="calc.py", relative_path="calc.py",
        language="python", size_bytes=len(content),
        content=content, line_count=content.count("\n") + 1,
    )


@pytest.fixture
def dangerous_file():
    content = (
        "import pickle\n\n"
        "def load(data):\n"
        "    obj = pickle.loads(data)\n"
        "    result = eval(str(obj))\n"
        "    password = 'hunter2'\n"
        "    return result\n"
    )
    return FileInfo(
        path="dangerous.py", relative_path="dangerous.py",
        language="python", size_bytes=len(content),
        content=content, line_count=content.count("\n") + 1,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Parser fixture
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def parser():
    return PythonASTParser()


# ─────────────────────────────────────────────────────────────────────────────
# Repository metadata fixture
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def repo_metadata():
    return RepositoryMetadata(
        url="https://github.com/acme/webapp",
        name="webapp", owner="acme",
        default_branch="main",
        clone_path="/tmp/acme_webapp",
        last_commit=CommitInfo(
            sha="abc123ef",
            message="Add authentication module",
            author="Alice Dev",
            date="2025-06-01T10:00:00",
        ),
        total_files_on_disk=20,
        supported_files=5,
        skipped_files=15,
        total_lines=350,
        languages={"python": 4, "javascript": 1},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Review fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_review_comment():
    return ReviewComment(
        file_name="auth.py", line_number=15,
        issue_type="eval_use", severity="Critical",
        confidence_score=97,
        explanation="eval() executes arbitrary code passed as input.",
        suggested_fix="Remove eval() and use a safe alternative.",
        category="Security",
        is_ast_detected=False,
    )


@pytest.fixture
def sample_file_review(sample_review_comment):
    fr = FileReview(
        file_name="auth.py",
        language="python",
        line_count=80,
    )
    fr.comments      = [sample_review_comment]
    fr.summary       = "Critical security vulnerability in authentication."
    fr.overall_score = 25
    return fr


@pytest.fixture
def sample_review_report(sample_file_review):
    return ReviewReport(
        file_reviews=[sample_file_review],
        total_comments=1,
        critical_count=1,
        high_count=0,
        medium_count=0,
        low_count=0,
        avg_confidence=97.0,
        avg_quality_score=25.0,
        files_with_errors=0,
        llm_provider="anthropic",
        llm_model="claude-sonnet-4-20250514",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Mock LLM patcher
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_llm_clean():
    """Patch LLM to return a 'clean file' response."""
    response = json.dumps({
        "comments": [],
        "summary":  "The code is clean and well-written.",
        "overall_score": 92,
    })
    with patch(
        "backend.reviewer.ai_reviewer.LLMClient.call",
        return_value=response,
    ):
        yield response


@pytest.fixture
def mock_llm_critical():
    """Patch LLM to return a critical-issue response."""
    response = json.dumps({
        "comments": [
            {
                "line_number": 5,
                "issue_type": "sql_injection",
                "severity": "Critical",
                "confidence_score": 95,
                "category": "Security",
                "explanation": "SQL injection vulnerability detected.",
                "suggested_fix": "Use parameterised queries.",
            }
        ],
        "summary":  "Critical security issue found.",
        "overall_score": 15,
    })
    with patch(
        "backend.reviewer.ai_reviewer.LLMClient.call",
        return_value=response,
    ):
        yield response


# ─────────────────────────────────────────────────────────────────────────────
# Mock ingestion patcher
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_ingest(repo_metadata, simple_file):
    """Patch RepositoryIngestion.ingest to return mock data."""
    with patch(
        "backend.ingestion.repo_ingestion.RepositoryIngestion.ingest",
        return_value=(repo_metadata, [simple_file]),
    ):
        yield (repo_metadata, [simple_file])
