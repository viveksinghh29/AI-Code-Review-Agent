"""Orchestrates the end-to-end AI code review pipeline from repository ingestion to report generation."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Optional

from backend.ingestion.repo_ingestion import (
    FileInfo,
    RepositoryIngestion,
    RepositoryMetadata,
)
from backend.parser.code_analyzer import AnalysisReport, CodeAnalyzer
from backend.reviewer.ai_reviewer import AIReviewer, ReviewReport
from backend.utils.config import get_config
from backend.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline Status
# ─────────────────────────────────────────────────────────────────────────────

class PipelineStatus(str, Enum):
    """Lifecycle state of a pipeline run."""
    IDLE       = "idle"
    INGESTING  = "ingesting"
    PARSING    = "parsing"
    ANALYSING  = "analysing"
    REVIEWING  = "reviewing"
    REPORTING  = "reporting"
    COMPLETE   = "complete"
    FAILED     = "failed"


# ─────────────────────────────────────────────────────────────────────────────
# Progress Event
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ProgressEvent:
    """
    A single progress update broadcast to the UI during a pipeline run.
    The dashboard collects these to update its progress bar and log.
    """
    status:     PipelineStatus
    stage:      str          # human-readable stage name
    message:    str          # detail message
    current:    int  = 0     # items processed so far
    total:      int  = 0     # total items in this stage
    pct:        float = 0.0  # 0.0 – 100.0 overall pipeline progress
    timestamp:  str  = field(default_factory=lambda: datetime.utcnow().isoformat())

    @property
    def progress_label(self) -> str:
        if self.total > 0:
            return f"{self.stage} ({self.current}/{self.total})"
        return self.stage


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline Result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """
    Complete output of one Orchestrator.run() call.
    This is the single object the Streamlit dashboard receives and renders.
    """
    # Inputs
    repo_url:   str
    started_at: str
    ended_at:   str

    # Phase outputs
    metadata:         RepositoryMetadata
    files:            list[FileInfo]
    analysis_report:  AnalysisReport
    review_report:    ReviewReport

    # Status
    status:     PipelineStatus
    error:      Optional[str] = None      # set if status == FAILED

    # Generated artefacts (populated by ReportBuilder)
    markdown_report: str = ""
    json_report:     str = ""

    @property
    def elapsed_seconds(self) -> float:
        try:
            start = datetime.fromisoformat(self.started_at)
            end   = datetime.fromisoformat(self.ended_at)
            return round((end - start).total_seconds(), 1)
        except Exception:
            return 0.0

    @property
    def success(self) -> bool:
        return self.status == PipelineStatus.COMPLETE

    @property
    def total_comments(self) -> int:
        return self.review_report.total_comments

    @property
    def avg_quality(self) -> float:
        return self.review_report.avg_quality_score


# ─────────────────────────────────────────────────────────────────────────────
# Report Builder
# ─────────────────────────────────────────────────────────────────────────────

class ReportBuilder:
    """
    Assembles human-readable Markdown and machine-readable JSON reports
    from a completed PipelineResult.
    """

    def build_markdown(self, result: PipelineResult) -> str:
        """Generate a full Markdown review report."""
        meta  = result.metadata
        agg   = result.analysis_report.aggregate
        rev   = result.review_report
        lines: list[str] = []

        # ── Header ──────────────────────────────────────────────────────────
        lines += [
            "# AI Code Review Report",
            "",
            f"> **Repository:** [{meta.repo_full_name}]({meta.url})  ",
            f"> **Branch:** `{meta.default_branch}`  ",
            f"> **Reviewed at:** {result.started_at[:19].replace('T',' ')} UTC  ",
            f"> **Duration:** {result.elapsed_seconds}s  ",
            f"> **LLM:** {rev.llm_provider} / `{rev.llm_model}`  ",
            "",
        ]

        # ── Executive Summary ────────────────────────────────────────────────
        lines += [
            "## Executive Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Files reviewed | {agg.total_files} |",
            f"| Total lines | {agg.total_lines:,} |",
            f"| Functions | {agg.total_functions} |",
            f"| Classes | {agg.total_classes} |",
            f"| Review comments | {rev.total_comments} |",
            f"| Critical issues | 🔴 {rev.critical_count} |",
            f"| High issues | 🟠 {rev.high_count} |",
            f"| Medium issues | 🟡 {rev.medium_count} |",
            f"| Low issues | 🟢 {rev.low_count} |",
            f"| Avg quality score | {rev.avg_quality_score}/100 |",
            f"| Avg confidence | {rev.avg_confidence}% |",
            f"| Doc coverage | {agg.doc_coverage_pct}% |",
            f"| Type hint coverage | {agg.type_hint_pct}% |",
            f"| Avg cyclomatic complexity | {agg.avg_complexity} |",
            f"| Repo health | **{agg.health_label}** |",
            "",
        ]

        # ── Language breakdown ───────────────────────────────────────────────
        if agg.language_breakdown:
            lines += ["## Language Breakdown", ""]
            for lang, count in sorted(
                agg.language_breakdown.items(), key=lambda x: -x[1]
            ):
                lines.append(f"- **{lang}**: {count} file(s)")
            lines.append("")

        # ── Critical & High findings ─────────────────────────────────────────
        critical_high = [
            c for c in result.review_report.all_comments()
            if c.severity in ("Critical", "High")
        ]
        if critical_high:
            lines += ["## 🚨 Critical & High Priority Findings", ""]
            for c in critical_high[:20]:
                icon = "🔴" if c.severity == "Critical" else "🟠"
                src  = "AST" if c.is_ast_detected else "AI"
                lines += [
                    f"### {icon} `{c.issue_type}` — `{c.file_name}` L{c.line_number}",
                    "",
                    f"- **Severity:** {c.severity}  ",
                    f"- **Category:** {c.category}  ",
                    f"- **Confidence:** {c.confidence_score}%  ",
                    f"- **Source:** {src}  ",
                    "",
                    f"**Problem:** {c.explanation}",
                    "",
                    f"**Fix:** {c.suggested_fix}",
                    "",
                    "---",
                    "",
                ]

        # ── Per-file summaries ───────────────────────────────────────────────
        lines += ["## Per-File Review", ""]
        for file_review in result.review_report.file_reviews:
            score = result.analysis_report.get_score(file_review.file_name)
            grade = score.grade if score else "?"
            lines += [
                f"### `{file_review.file_name}`",
                "",
                "| | |",
                "|--|--|",
                f"| Language | {file_review.language} |",
                f"| Lines | {file_review.line_count} |",
                f"| Quality score | {file_review.overall_score}/100 ({grade}) |",
                f"| Comments | {len(file_review.comments)} |",
                f"| Avg confidence | {file_review.avg_confidence}% |",
                "",
            ]
            if file_review.summary:
                lines += [f"> {file_review.summary}", ""]

            if file_review.comments:
                lines.append("#### Findings")
                lines.append("")
                for c in file_review.comments:
                    sev_icon = {
                        "Critical": "🔴", "High": "🟠",
                        "Medium":  "🟡", "Low":  "🟢",
                    }.get(c.severity, "⚪")
                    lines.append(
                        f"- {sev_icon} **L{c.line_number}** "
                        f"[{c.severity}] `{c.issue_type}` — "
                        f"{c.explanation[:100]}"
                    )
                lines.append("")

        # ── Code smell breakdown ─────────────────────────────────────────────
        if agg.smell_breakdown:
            lines += ["## Code Smell Breakdown", ""]
            lines += ["| Smell Type | Count |", "|-----------|-------|"]
            for smell, count in sorted(
                agg.smell_breakdown.items(), key=lambda x: -x[1]
            ):
                lines.append(f"| `{smell}` | {count} |")
            lines.append("")

        # ── Footer ───────────────────────────────────────────────────────────
        lines += [
            "---",
            f"*Report generated by AI Code Review Agent · "
            f"{result.started_at[:10]}*",
        ]

        return "\n".join(lines)

    def build_json(self, result: PipelineResult) -> str:
        """Generate a structured JSON report."""
        payload = {
            "meta": {
                "repo_url":        result.repo_url,
                "repo_name":       result.metadata.repo_full_name,
                "branch":          result.metadata.default_branch,
                "last_commit":     result.metadata.last_commit.sha,
                "started_at":      result.started_at,
                "ended_at":        result.ended_at,
                "elapsed_seconds": result.elapsed_seconds,
                "llm_provider":    result.review_report.llm_provider,
                "llm_model":       result.review_report.llm_model,
            },
            "analysis": result.analysis_report.to_summary_dict(),
            "review":   result.review_report.to_dict(),
        }
        return json.dumps(payload, indent=2, default=str)


# ─────────────────────────────────────────────────────────────────────────────
# In-memory Result Cache
# ─────────────────────────────────────────────────────────────────────────────

class ResultCache:
    """
    Simple LRU cache keyed on normalised repo URL.
    Prevents redundant clones + API calls for the same repo
    within the same session.

    Max size: 5 entries (configurable).
    """

    def __init__(self, max_size: int = 5):
        self._max   = max_size
        self._store: dict[str, PipelineResult] = {}
        self._order: list[str] = []

    def _normalise(self, url: str) -> str:
        return url.strip().rstrip("/").lower().removesuffix(".git")

    def get(self, url: str) -> Optional[PipelineResult]:
        key = self._normalise(url)
        result = self._store.get(key)
        if result:
            # Move to front (most-recently-used)
            self._order.remove(key)
            self._order.insert(0, key)
            logger.info(f"Cache hit for {url}")
        return result

    def set(self, url: str, result: PipelineResult) -> None:
        key = self._normalise(url)
        if key in self._store:
            self._order.remove(key)
        elif len(self._store) >= self._max:
            evict = self._order.pop()
            del self._store[evict]
            logger.debug(f"Cache evicted: {evict}")
        self._store[key] = result
        self._order.insert(0, key)

    def invalidate(self, url: str) -> None:
        key = self._normalise(url)
        if key in self._store:
            del self._store[key]
            self._order.remove(key)

    def clear(self) -> None:
        self._store.clear()
        self._order.clear()

    @property
    def size(self) -> int:
        return len(self._store)


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

# Overall pipeline stage weights used to calculate pct (0-100)
_STAGE_PCT = {
    PipelineStatus.INGESTING:  (0,   20),
    PipelineStatus.PARSING:    (20,  40),
    PipelineStatus.ANALYSING:  (40,  55),
    PipelineStatus.REVIEWING:  (55,  90),
    PipelineStatus.REPORTING:  (90, 100),
}


class Orchestrator:
    """
    Central pipeline coordinator for the AI Code Review Agent.

    Usage:
        orch   = Orchestrator()
        result = orch.run(
            url               = "https://github.com/owner/repo",
            progress_callback = lambda evt: print(evt.message),
        )

    The progress_callback receives ProgressEvent objects in real-time,
    allowing the Streamlit dashboard to update its UI without blocking.

    Args:
        use_cache: If True (default), return a cached result for repos
                   that were reviewed in the same session.
    """

    def __init__(self, use_cache: bool = True):
        self.config    = get_config()
        self._cache    = ResultCache() if use_cache else None
        self._ingestion = RepositoryIngestion()
        self._analyzer  = CodeAnalyzer()
        self._reviewer  = AIReviewer()
        self._report_builder = ReportBuilder()

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        url:               str,
        progress_callback: Optional[Callable[[ProgressEvent], None]] = None,
        force_refresh:     bool = False,
    ) -> PipelineResult:
        """
        Execute the full review pipeline for a GitHub repository URL.

        Steps:
          1. Validate URL
          2. Check cache (unless force_refresh)
          3. Clone + extract metadata      [Phase 2]
          4. Parse all source files        [Phase 3]
          5. Analyse quality metrics       [Phase 4]
          6. AI review every file          [Phase 5]
          7. Build Markdown + JSON reports

        Args:
            url:               GitHub repository URL.
            progress_callback: Optional callable(ProgressEvent).
            force_refresh:     Skip cache and re-run from scratch.

        Returns:
            PipelineResult (cached or freshly computed).

        Raises:
            Never raises — all errors are captured in PipelineResult.status
        """
        started_at = datetime.utcnow().isoformat()
        emit = self._make_emitter(progress_callback)

        # ── 0. Cache lookup ──────────────────────────────────────────────────
        if self._cache and not force_refresh:
            cached = self._cache.get(url)
            if cached:
                emit(PipelineStatus.COMPLETE, "Cache", "Loaded from cache ✓", pct=100)
                return cached

        logger.info(f"Pipeline started for {url}")

        try:
            # ── 1. Ingestion ─────────────────────────────────────────────────
            emit(PipelineStatus.INGESTING, "Ingestion",
                 "Cloning repository…", pct=2)

            metadata, files = self._ingestion.ingest(
                url,
                progress_callback=lambda msg: emit(
                    PipelineStatus.INGESTING, "Ingestion", msg, pct=10
                ),
            )

            emit(
                PipelineStatus.INGESTING, "Ingestion",
                f"Cloned {metadata.repo_full_name} — "
                f"{len(files)} source files, "
                f"{metadata.total_lines:,} lines",
                pct=20,
            )

            # ── 2. Parsing ───────────────────────────────────────────────────
            emit(PipelineStatus.PARSING, "Parsing",
                 f"Parsing {len(files)} files with AST…", pct=22)

            def parse_progress(cur: int, tot: int, fname: str) -> None:
                pct = 20 + int(cur / max(tot, 1) * 20)
                emit(PipelineStatus.PARSING, "Parsing",
                     f"Parsing {fname}…", cur, tot, pct)

            analysis_report = self._analyzer.analyze_all(
                files, progress_callback=parse_progress
            )

            agg = analysis_report.aggregate
            emit(
                PipelineStatus.ANALYSING, "Analysis",
                f"Analysis complete — "
                f"{agg.total_functions} functions, "
                f"{agg.total_smells} smells detected, "
                f"avg score {agg.avg_quality_score}/100",
                pct=55,
            )

            # ── 3. AI Review ─────────────────────────────────────────────────
            emit(PipelineStatus.REVIEWING, "AI Review",
                 f"Starting AI review of {len(files)} files…", pct=57)

            def review_progress(cur: int, tot: int, fname: str) -> None:
                pct = 55 + int(cur / max(tot, 1) * 35)
                emit(PipelineStatus.REVIEWING, "AI Review",
                     f"Reviewing {fname}…", cur, tot, pct)

            review_report = self._reviewer.review_all_files(
                files              = files,
                parsed_files       = analysis_report.parsed_files,
                analysis_report    = analysis_report,
                progress_callback  = review_progress,
            )

            emit(
                PipelineStatus.REVIEWING, "AI Review",
                f"AI review complete — "
                f"{review_report.total_comments} comments "
                f"({review_report.critical_count} critical, "
                f"{review_report.high_count} high)",
                pct=90,
            )

            # ── 4. Report generation ─────────────────────────────────────────
            emit(PipelineStatus.REPORTING, "Reports",
                 "Generating Markdown & JSON reports…", pct=92)

            ended_at = datetime.utcnow().isoformat()
            result   = PipelineResult(
                repo_url         = url,
                started_at       = started_at,
                ended_at         = ended_at,
                metadata         = metadata,
                files            = files,
                analysis_report  = analysis_report,
                review_report    = review_report,
                status           = PipelineStatus.COMPLETE,
            )

            result.markdown_report = self._report_builder.build_markdown(result)
            result.json_report     = self._report_builder.build_json(result)

            # ── 5. Cache + finalise ──────────────────────────────────────────
            if self._cache:
                self._cache.set(url, result)

            emit(
                PipelineStatus.COMPLETE, "Complete",
                f"✓ Pipeline finished in {result.elapsed_seconds}s — "
                f"{review_report.total_comments} comments across "
                f"{len(files)} files",
                pct=100,
            )
            logger.info(
                f"Pipeline complete for {metadata.repo_full_name} — "
                f"{result.elapsed_seconds}s"
            )
            return result

        except Exception as exc:
            ended_at = datetime.utcnow().isoformat()
            logger.error(f"Pipeline failed for {url}: {exc}", exc_info=True)
            err_msg  = str(exc)

            emit(PipelineStatus.FAILED, "Error",
                 f"Pipeline failed: {err_msg}", pct=0)

            # Return a minimal failed result so the UI can show the error
            return self._failed_result(url, started_at, ended_at, err_msg)

    def validate_url(self, url: str) -> tuple[bool, str]:
        """
        Quick URL validation without cloning.
        Returns (is_valid, error_message).
        """
        try:
            self._ingestion.validate_url(url)
            return True, ""
        except Exception as exc:
            return False, str(exc)

    def invalidate_cache(self, url: str) -> None:
        """Remove a cached result for the given URL."""
        if self._cache:
            self._cache.invalidate(url)

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _make_emitter(
        callback: Optional[Callable[[ProgressEvent], None]]
    ) -> Callable:
        """Return a typed emit function that safely calls the callback."""
        def emit(
            status:  PipelineStatus,
            stage:   str,
            message: str,
            current: int   = 0,
            total:   int   = 0,
            pct:     float = 0.0,
        ) -> None:
            event = ProgressEvent(
                status  = status,
                stage   = stage,
                message = message,
                current = current,
                total   = total,
                pct     = pct,
            )
            logger.debug(f"[{status.value}] {message}")
            if callback:
                try:
                    callback(event)
                except Exception as cb_exc:
                    logger.warning(f"Progress callback raised: {cb_exc}")
        return emit

    @staticmethod
    def _failed_result(
        url: str,
        started_at: str,
        ended_at: str,
        error: str,
    ) -> PipelineResult:
        """Construct a minimal PipelineResult representing a failed run."""
        from backend.ingestion.repo_ingestion import (
            CommitInfo, RepositoryMetadata,
        )
        from backend.parser.code_analyzer import (
            AggregateMetrics, AnalysisReport, ImportAnomalies,
        )
        from backend.reviewer.ai_reviewer import ReviewReport

        dummy_meta = RepositoryMetadata(
            url="", name="error", owner="error",
            default_branch="main", clone_path="",
            last_commit=CommitInfo("","","",""),
            total_files_on_disk=0, supported_files=0,
            skipped_files=0, total_lines=0, languages={},
        )
        dummy_agg = AggregateMetrics(
            total_files=0, total_lines=0, total_functions=0,
            total_classes=0, total_imports=0, total_smells=0,
            avg_function_length=0.0, avg_complexity=0.0,
            max_complexity=0, max_complexity_fn="",
            doc_coverage_pct=0.0, type_hint_pct=0.0,
            avg_quality_score=0.0, smell_breakdown={},
            severity_breakdown={}, language_breakdown={},
            files_with_errors=0,
        )
        dummy_analysis = AnalysisReport(
            parsed_files=[], quality_scores=[], aggregate=dummy_agg,
            duplicates=[], naming_issues=[],
            import_anomalies=ImportAnomalies(0,0,0,0,[],[]),
            analysis_errors=[error],
        )
        cfg = get_config()
        dummy_review = ReviewReport(
            file_reviews=[], total_comments=0, critical_count=0,
            high_count=0, medium_count=0, low_count=0,
            avg_confidence=0.0, avg_quality_score=0.0,
            files_with_errors=0,
            llm_provider=cfg.llm.provider,
            llm_model=cfg.llm.active_model,
        )
        return PipelineResult(
            repo_url        = url,
            started_at      = started_at,
            ended_at        = ended_at,
            metadata        = dummy_meta,
            files           = [],
            analysis_report = dummy_analysis,
            review_report   = dummy_review,
            status          = PipelineStatus.FAILED,
            error           = error,
        )
