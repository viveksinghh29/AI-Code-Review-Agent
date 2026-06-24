"""
Unit Tests — Orchestrator + GitHub Integration (Phase 6 & 8)
Tests: ResultCache, ReportBuilder, Orchestrator, GitHubClient,
       PRReviewPoster, GitHubIntegration
"""

import json
import pytest
from unittest.mock import patch, MagicMock

from backend.ingestion.repo_ingestion import CommitInfo, FileInfo, RepositoryMetadata
from backend.orchestration import (
    Orchestrator,
    PipelineResult,
    PipelineStatus,
    ProgressEvent,
    ReportBuilder,
    ResultCache,
)
from backend.parser.code_analyzer import (
    AnalysisReport, AggregateMetrics, ImportAnomalies,
)
from backend.reviewer.ai_reviewer import (
    FileReview, ReviewComment, ReviewReport,
)
from backend.github import (
    GitHubClient,
    GitHubIntegration,
    GitHubIntegrationError,
    NotAuthenticatedError,
    PRFile,
    PRNotFoundError,
    PRReviewResult,
    PullRequestInfo,
)
from backend.github.github_integration import PRReviewPoster


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures / helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_file(path="a.py", content="x = 1\n"):
    return FileInfo(path=path, relative_path=path, language="python",
                    size_bytes=len(content), content=content,
                    line_count=content.count("\n") + 1)


def make_meta(name="repo", owner="acme"):
    return RepositoryMetadata(
        url=f"https://github.com/{owner}/{name}",
        name=name, owner=owner, default_branch="main",
        clone_path=f"/tmp/{owner}_{name}",
        last_commit=CommitInfo("abc123", "Initial commit", "dev", "2025-01-01"),
        total_files_on_disk=5, supported_files=2,
        skipped_files=3, total_lines=50,
        languages={"python": 2},
    )


def make_review_report():
    fr = FileReview("a.py", "python", 10)
    fr.comments = [
        ReviewComment("a.py", 1, "eval_use", "Critical", 95,
                      "eval() is dangerous.", "Remove eval.", "Security"),
    ]
    fr.overall_score = 30
    return ReviewReport(
        file_reviews=[fr], total_comments=1, critical_count=1,
        high_count=0, medium_count=0, low_count=0,
        avg_confidence=95.0, avg_quality_score=30.0,
        files_with_errors=0, llm_provider="anthropic",
        llm_model="claude-sonnet-4-20250514",
    )


def make_empty_analysis():
    agg = AggregateMetrics(
        total_files=0, total_lines=0, total_functions=0, total_classes=0,
        total_imports=0, total_smells=0, avg_function_length=0.0,
        avg_complexity=0.0, max_complexity=0, max_complexity_fn="",
        doc_coverage_pct=0.0, type_hint_pct=0.0, avg_quality_score=0.0,
        smell_breakdown={}, severity_breakdown={}, language_breakdown={},
        files_with_errors=0,
    )
    return AnalysisReport(
        parsed_files=[], quality_scores=[], aggregate=agg,
        duplicates=[], naming_issues=[],
        import_anomalies=ImportAnomalies(0, 0, 0, 0, [], []),
        analysis_errors=[],
    )


def make_pipeline_result(status=PipelineStatus.COMPLETE):
    from backend.orchestration.orchestrator import Orchestrator as O
    if status == PipelineStatus.FAILED:
        return O._failed_result(
            "https://github.com/x/y", "2025-01-01T00:00:00",
            "2025-01-01T00:00:05", "Test error",
        )
    from backend.ingestion.repo_ingestion import FileInfo
    files  = [make_file()]
    meta   = make_meta()
    review = make_review_report()
    from backend.parser.code_analyzer import CodeAnalyzer
    analysis = CodeAnalyzer().analyze_all(files)
    return PipelineResult(
        repo_url="https://github.com/acme/repo",
        started_at="2025-01-01T00:00:00",
        ended_at="2025-01-01T00:00:10",
        metadata=meta, files=files,
        analysis_report=analysis,
        review_report=review,
        status=status,
    )


MOCK_LLM = json.dumps({"comments": [], "summary": "Clean.", "overall_score": 85})


# ─────────────────────────────────────────────────────────────────────────────
# ResultCache
# ─────────────────────────────────────────────────────────────────────────────

class TestResultCache:
    def setup_method(self):
        self.cache = ResultCache(max_size=3)

    def test_set_and_get(self):
        self.cache.set("https://github.com/a/b", "value1")
        assert self.cache.get("https://github.com/a/b") == "value1"

    def test_miss_returns_none(self):
        assert self.cache.get("https://github.com/x/y") is None

    def test_url_normalisation_git_suffix(self):
        self.cache.set("https://github.com/a/b.git", "v")
        assert self.cache.get("https://github.com/a/b") == "v"

    def test_url_normalisation_trailing_slash(self):
        self.cache.set("https://github.com/a/b/", "v")
        assert self.cache.get("https://github.com/a/b") == "v"

    def test_lru_eviction(self):
        self.cache.set("https://github.com/a/r1", "v1")
        self.cache.set("https://github.com/a/r2", "v2")
        self.cache.set("https://github.com/a/r3", "v3")
        # Access r1 to make it recently used
        self.cache.get("https://github.com/a/r1")
        # Add r4 — r2 should be evicted (LRU)
        self.cache.set("https://github.com/a/r4", "v4")
        assert self.cache.get("https://github.com/a/r1") == "v1"
        assert self.cache.get("https://github.com/a/r2") is None
        assert self.cache.get("https://github.com/a/r4") == "v4"

    def test_invalidate(self):
        self.cache.set("https://github.com/a/b", "v")
        self.cache.invalidate("https://github.com/a/b")
        assert self.cache.get("https://github.com/a/b") is None

    def test_clear(self):
        self.cache.set("https://github.com/a/b", "v")
        self.cache.clear()
        assert self.cache.size == 0

    def test_size_property(self):
        assert self.cache.size == 0
        self.cache.set("https://github.com/a/b", "v")
        assert self.cache.size == 1


# ─────────────────────────────────────────────────────────────────────────────
# ReportBuilder
# ─────────────────────────────────────────────────────────────────────────────

class TestReportBuilder:
    def setup_method(self):
        self.builder = ReportBuilder()
        self.result  = make_pipeline_result()
        self.result.markdown_report = ""
        self.result.json_report     = ""

    def test_markdown_contains_header(self):
        md = self.builder.build_markdown(self.result)
        assert "# AI Code Review Report" in md

    def test_markdown_contains_repo_name(self):
        md = self.builder.build_markdown(self.result)
        assert "repo" in md or "acme" in md

    def test_markdown_contains_summary_table(self):
        md = self.builder.build_markdown(self.result)
        assert "Executive Summary" in md
        assert "Files reviewed"    in md

    def test_markdown_contains_per_file_section(self):
        md = self.builder.build_markdown(self.result)
        assert "Per-File Review" in md

    def test_markdown_contains_critical_section(self):
        md = self.builder.build_markdown(self.result)
        # Has at least one critical comment from make_review_report()
        assert "Critical" in md

    def test_json_is_valid(self):
        js = self.builder.build_json(self.result)
        data = json.loads(js)
        assert "meta"     in data
        assert "analysis" in data
        assert "review"   in data

    def test_json_meta_fields(self):
        js   = self.builder.build_json(self.result)
        meta = json.loads(js)["meta"]
        assert "repo_name"       in meta
        assert "llm_provider"    in meta
        assert "elapsed_seconds" in meta

    def test_json_review_summary(self):
        js     = self.builder.build_json(self.result)
        review = json.loads(js)["review"]
        assert "summary"      in review
        assert "file_reviews" in review


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class TestOrchestrator:
    def setup_method(self):
        self.orch = Orchestrator(use_cache=False)

    def _patch_pipeline(self):
        return (
            patch("backend.ingestion.repo_ingestion.RepositoryIngestion.ingest",
                  return_value=(make_meta(), [make_file()])),
            patch("backend.reviewer.ai_reviewer.LLMClient.call",
                  return_value=MOCK_LLM),
        )

    def test_validate_url_valid(self):
        ok, err = self.orch.validate_url("https://github.com/pallets/flask")
        assert ok  is True
        assert err == ""

    def test_validate_url_invalid(self):
        ok, err = self.orch.validate_url("not-a-url")
        assert ok  is False
        assert err != ""

    def test_run_returns_pipeline_result(self):
        p1, p2 = self._patch_pipeline()
        with p1, p2:
            result = self.orch.run("https://github.com/pallets/flask")
        assert isinstance(result, PipelineResult)

    def test_run_success(self):
        p1, p2 = self._patch_pipeline()
        with p1, p2:
            result = self.orch.run("https://github.com/pallets/flask")
        assert result.status  == PipelineStatus.COMPLETE
        assert result.success is True

    def test_run_failed_url(self):
        result = self.orch.run("https://github.com/totally-fake-xyz/norepo-abc123")
        assert result.status  == PipelineStatus.FAILED
        assert result.success is False
        assert result.error   is not None

    def test_progress_callback_called(self):
        events = []
        p1, p2 = self._patch_pipeline()
        with p1, p2:
            self.orch.run(
                "https://github.com/pallets/flask",
                progress_callback=lambda e: events.append(e),
            )
        assert len(events) > 0
        assert events[-1].status == PipelineStatus.COMPLETE

    def test_progress_pct_monotonic(self):
        events = []
        p1, p2 = self._patch_pipeline()
        with p1, p2:
            self.orch.run(
                "https://github.com/pallets/flask",
                progress_callback=lambda e: events.append(e),
            )
        pcts = [e.pct for e in events]
        assert pcts == sorted(pcts)
        assert pcts[-1] == 100.0

    def test_cache_hit_skips_pipeline(self):
        call_count = [0]
        original_ingest = None

        def counting_ingest(self_inner, url, progress_callback=None):
            call_count[0] += 1
            return make_meta(), [make_file()]

        orch = Orchestrator(use_cache=True)
        p2   = patch("backend.reviewer.ai_reviewer.LLMClient.call",
                     return_value=MOCK_LLM)
        with patch("backend.ingestion.repo_ingestion.RepositoryIngestion.ingest",
                   counting_ingest), p2:
            orch.run("https://github.com/pallets/flask")
            orch.run("https://github.com/pallets/flask")  # should hit cache

        assert call_count[0] == 1, "Pipeline ran twice — cache not working"

    def test_force_refresh_bypasses_cache(self):
        call_count = [0]
        def counting_ingest(self_inner, url, progress_callback=None):
            call_count[0] += 1
            return make_meta(), [make_file()]

        orch = Orchestrator(use_cache=True)
        p2   = patch("backend.reviewer.ai_reviewer.LLMClient.call",
                     return_value=MOCK_LLM)
        with patch("backend.ingestion.repo_ingestion.RepositoryIngestion.ingest",
                   counting_ingest), p2:
            orch.run("https://github.com/pallets/flask")
            orch.run("https://github.com/pallets/flask", force_refresh=True)

        assert call_count[0] == 2

    def test_reports_generated(self):
        p1, p2 = self._patch_pipeline()
        with p1, p2:
            result = self.orch.run("https://github.com/pallets/flask")
        assert len(result.markdown_report) > 0
        assert len(result.json_report)     > 0
        data = json.loads(result.json_report)
        assert "meta" in data

    def test_failed_result_properties(self):
        result = Orchestrator._failed_result(
            "https://github.com/x/y",
            "2025-01-01T10:00:00",
            "2025-01-01T10:00:05",
            "Test error",
        )
        assert result.status          == PipelineStatus.FAILED
        assert result.success         is False
        assert result.elapsed_seconds == 5.0
        assert result.error           == "Test error"

    def test_no_cache_mode(self):
        orch = Orchestrator(use_cache=False)
        assert orch._cache is None

    def test_progress_event_dataclass(self):
        evt = ProgressEvent(
            PipelineStatus.PARSING, "Parsing", "Parsing f.py", 3, 10, 35.0
        )
        assert evt.progress_label == "Parsing (3/10)"
        assert evt.pct            == 35.0
        assert evt.timestamp      != ""


# ─────────────────────────────────────────────────────────────────────────────
# GitHubClient
# ─────────────────────────────────────────────────────────────────────────────

class TestGitHubClient:
    def test_extract_full_name_https(self):
        assert GitHubClient.extract_full_name(
            "https://github.com/pallets/flask") == "pallets/flask"

    def test_extract_full_name_ssh(self):
        assert GitHubClient.extract_full_name(
            "git@github.com:django/django.git") == "django/django"

    def test_extract_full_name_trailing_slash(self):
        assert GitHubClient.extract_full_name(
            "https://github.com/a/b/") == "a/b"

    def test_extract_invalid_url_raises(self):
        with pytest.raises(GitHubIntegrationError):
            GitHubClient.extract_full_name("https://notgithub.com/a/b")

    def test_no_token_raises_on_get_github(self):
        import os
        os.environ.pop("GITHUB_TOKEN", None)
        from backend.utils.config import reload_config
        reload_config()
        client = GitHubClient(token=None)
        with pytest.raises(NotAuthenticatedError):
            client._get_github()

    def test_is_authenticated_false(self):
        gh = GitHubIntegration(token=None)
        assert gh.is_authenticated is False

    def test_is_authenticated_true(self):
        gh = GitHubIntegration(token="ghp_fake")
        assert gh.is_authenticated is True


# ─────────────────────────────────────────────────────────────────────────────
# PRReviewPoster — formatting
# ─────────────────────────────────────────────────────────────────────────────

class TestPRReviewPosterFormatting:
    def setup_method(self):
        self.poster = PRReviewPoster(None)

    def test_inline_comment_has_severity(self):
        c = ReviewComment("f.py", 5, "eval_use", "Critical", 95,
                          "eval is dangerous.", "Remove it.", "Security")
        body = self.poster._format_inline_comment(c)
        assert "Critical"  in body
        assert "eval_use"  in body
        assert "Security"  in body
        assert "95%"       in body

    def test_inline_comment_ast_source(self):
        c = ReviewComment("f.py", 1, "x", "High", 80, "E.", "F.", "Bug Risk",
                          is_ast_detected=True)
        body = self.poster._format_inline_comment(c)
        assert "AST Analysis" in body

    def test_inline_comment_ai_source(self):
        c = ReviewComment("f.py", 1, "x", "High", 80, "E.", "F.", "Bug Risk",
                          is_ast_detected=False)
        body = self.poster._format_inline_comment(c)
        assert "AI Review" in body

    def test_inline_comment_code_block_for_fix(self):
        c = ReviewComment("f.py", 1, "x", "High", 80, "E.",
                          "def fixed():\n    return 1", "Security")
        body = self.poster._format_inline_comment(c)
        assert "```python" in body

    def test_summary_comment_structure(self):
        report = make_review_report()
        body   = self.poster._format_summary_comment(report, "acme/app", 7)
        assert "# 🔍 AI Code Review Summary" in body
        assert "acme/app"                     in body
        assert "#7"                           in body
        assert "Critical"                     in body

    def test_summary_comment_table(self):
        report = make_review_report()
        body   = self.poster._format_summary_comment(report, "org/repo", 1)
        assert "| Total comments |" in body
        assert "| 🔴 Critical |"    in body


# ─────────────────────────────────────────────────────────────────────────────
# GitHubIntegration — dry_run
# ─────────────────────────────────────────────────────────────────────────────

class TestGitHubIntegrationDryRun:
    def setup_method(self):
        self.gh     = GitHubIntegration(token="ghp_fake")
        self.pr_info = PullRequestInfo(
            number=5, title="Test PR", author="dev", state="open",
            base_branch="main", head_branch="feat/x",
            url="https://github.com/t/r/pull/5",
            created_at="", updated_at="", body="",
            files=[PRFile("a.py", "modified", 10, 2, 12, "")],
        )

    def _patch_get_pr(self):
        from backend.github.github_integration import PRFetcher
        return patch.object(PRFetcher, "get_pr", return_value=self.pr_info)

    def test_dry_run_returns_result(self):
        report = make_review_report()
        with self._patch_get_pr():
            result = self.gh.post_pr_review(
                "https://github.com/t/r", 5, report, dry_run=True
            )
        assert isinstance(result, PRReviewResult)

    def test_dry_run_no_api_calls(self):
        """dry_run should not call any real GitHub API."""
        report = make_review_report()
        gh_mock = MagicMock()
        with self._patch_get_pr():
            with patch("backend.github.github_integration.GitHubClient.get_repo",
                       return_value=gh_mock):
                result = self.gh.post_pr_review(
                    "https://github.com/t/r", 5, report, dry_run=True
                )
        # get_repo should NOT have been called in dry_run mode
        gh_mock.get_pull.assert_not_called()

    def test_dry_run_summary_id_minus_one(self):
        report = make_review_report()
        with self._patch_get_pr():
            result = self.gh.post_pr_review(
                "https://github.com/t/r", 5, report, dry_run=True
            )
        assert result.summary_comment_id == -1

    def test_file_filter_applied(self):
        """Comments for files not in the PR should be excluded."""
        from backend.reviewer.ai_reviewer import FileReview
        fr_unrelated = FileReview("other/unrelated.py", "python", 20)
        fr_unrelated.comments = [
            ReviewComment("other/unrelated.py", 1, "x", "Critical", 90,
                          "E.", "F.", "Security"),
        ]
        fr_unrelated.overall_score = 50

        fr_in_pr = FileReview("a.py", "python", 10)
        fr_in_pr.comments = [
            ReviewComment("a.py", 1, "eval_use", "Critical", 95,
                          "eval.", "Remove.", "Security"),
        ]
        fr_in_pr.overall_score = 30

        report = ReviewReport(
            file_reviews=[fr_in_pr, fr_unrelated],
            total_comments=2, critical_count=2,
            high_count=0, medium_count=0, low_count=0,
            avg_confidence=92.5, avg_quality_score=40.0,
            files_with_errors=0, llm_provider="anthropic",
            llm_model="claude-sonnet-4-20250514",
        )
        with self._patch_get_pr():
            result = self.gh.post_pr_review(
                "https://github.com/t/r", 5, report, dry_run=True
            )
        posted_files = {c.file_path for c in result.posted_comments}
        assert "other/unrelated.py" not in posted_files
        assert "a.py"                in posted_files

    def test_generate_pr_summary(self):
        report  = make_review_report()
        summary = self.gh.generate_pr_summary(
            report, "https://github.com/acme/app", 42
        )
        assert "# 🔍 AI Code Review Summary" in summary
        assert "acme/app"                     in summary
        assert "#42"                          in summary
