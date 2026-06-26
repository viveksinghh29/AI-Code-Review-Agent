"""Analyzes repository source code to generate file-level and repository-wide quality insights."""

import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Optional

from backend.ingestion.repo_ingestion import FileInfo
from backend.parser.ast_parser import (
    CodeSmell,
    ParsedFile,
    PythonASTParser,
)
from backend.utils.config import get_config
from backend.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Naming-convention patterns
# ─────────────────────────────────────────────────────────────────────────────

_SNAKE_CASE   = re.compile(r"^[a-z_][a-z0-9_]*$")
_PASCAL_CASE  = re.compile(r"^[A-Z][a-zA-Z0-9]*$")
_UPPER_SNAKE  = re.compile(r"^[A-Z_][A-Z0-9_]*$")          # CONSTANT
_DUNDER       = re.compile(r"^__[a-z_]+__$")                # __init__ etc.
_PRIVATE_OK   = re.compile(r"^_[a-z_][a-z0-9_]*$")         # _private


def _is_valid_function_name(name: str) -> bool:
    return bool(
        _SNAKE_CASE.match(name)
        or _DUNDER.match(name)
        or _PRIVATE_OK.match(name)
    )


def _is_valid_class_name(name: str) -> bool:
    return bool(_PASCAL_CASE.match(name))


def _is_valid_variable_name(name: str) -> bool:
    return bool(
        _SNAKE_CASE.match(name)
        or _UPPER_SNAKE.match(name)
        or _DUNDER.match(name)
        or _PRIVATE_OK.match(name)
        or name in ("i", "j", "k", "x", "y", "z", "n", "e", "f", "v")
    )


# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FileQualityScore:
    """
    0-100 quality score for a single file, broken down by dimension.
    Higher = better quality.
    """
    file_path:        str
    overall:          int          # weighted aggregate
    documentation:    int          # docstring coverage
    complexity:       int          # inverse of cyclomatic complexity
    maintainability:  int          # inverse of smell count + long functions
    security:         int          # inverse of critical/high security smells
    style:            int          # naming + line length + TODO density
    grade: str = ""                # A/B/C/D/F letter grade

    def __post_init__(self):
        self.grade = _score_to_grade(self.overall)


def _score_to_grade(score: int) -> str:
    if score >= 90: return "A"
    if score >= 80: return "B"
    if score >= 70: return "C"
    if score >= 60: return "D"
    return "F"


@dataclass
class DuplicatePattern:
    """Two functions/classes with the same name in different files."""
    name:       str
    kind:       str          # "function" | "class"
    locations:  list[str]    # list of "file_path:line" strings


@dataclass
class NamingIssue:
    """A name that violates the language's naming convention."""
    name:        str
    kind:        str          # "function" | "class" | "variable"
    file_path:   str
    line_number: int
    expected:    str          # description of expected convention


@dataclass
class ImportAnomalies:
    """Cross-file import statistics."""
    total_imports:     int
    stdlib_count:      int
    third_party_count: int
    internal_count:    int
    wildcard_files:    list[str]         # files with "from x import *"
    heavy_importers:   list[tuple[str, int]]  # (file, import_count) top 5


@dataclass
class AggregateMetrics:
    """Repository-level aggregate quality metrics."""
    total_files:          int
    total_lines:          int
    total_functions:      int
    total_classes:        int
    total_imports:        int
    total_smells:         int

    avg_function_length:  float
    avg_complexity:       float
    max_complexity:       int
    max_complexity_fn:    str   # "file:function_name"

    doc_coverage_pct:     float  # % of public fns+classes with docstrings
    type_hint_pct:        float  # % of functions with type hints
    avg_quality_score:    float

    smell_breakdown:      dict[str, int]   # smell_type → count
    severity_breakdown:   dict[str, int]   # severity   → count
    language_breakdown:   dict[str, int]   # language   → file count
    files_with_errors:    int

    @property
    def health_label(self) -> str:
        s = self.avg_quality_score
        if s >= 85: return "Excellent"
        if s >= 70: return "Good"
        if s >= 55: return "Fair"
        if s >= 40: return "Poor"
        return "Critical"


@dataclass
class AnalysisReport:
    """
    Complete analysis of a repository.
    Primary output of CodeAnalyzer.analyze_all().
    Consumed by AIReviewer (Phase 5) and Dashboard (Phase 7).
    """
    parsed_files:       list[ParsedFile]
    quality_scores:     list[FileQualityScore]
    aggregate:          AggregateMetrics
    duplicates:         list[DuplicatePattern]
    naming_issues:      list[NamingIssue]
    import_anomalies:   ImportAnomalies
    analysis_errors:    list[str]           # non-fatal errors during analysis

    # Index helpers (built post-init for fast lookup)
    _score_by_path: dict[str, FileQualityScore] = field(
        default_factory=dict, repr=False
    )
    _parsed_by_path: dict[str, ParsedFile] = field(
        default_factory=dict, repr=False
    )

    def __post_init__(self):
        self._score_by_path  = {s.file_path: s for s in self.quality_scores}
        self._parsed_by_path = {p.file_path: p for p in self.parsed_files}

    def get_score(self, file_path: str) -> Optional[FileQualityScore]:
        return self._score_by_path.get(file_path)

    def get_parsed(self, file_path: str) -> Optional[ParsedFile]:
        return self._parsed_by_path.get(file_path)

    def files_sorted_by_score(self) -> list[FileQualityScore]:
        return sorted(self.quality_scores, key=lambda s: s.overall)

    def worst_files(self, n: int = 5) -> list[FileQualityScore]:
        return self.files_sorted_by_score()[:n]

    def all_smells(self) -> list[tuple[str, CodeSmell]]:
        """Return (file_path, smell) pairs sorted by severity."""
        pairs = [
            (pf.file_path, smell)
            for pf in self.parsed_files
            for smell in pf.code_smells
        ]
        rank = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
        return sorted(pairs, key=lambda t: rank.get(t[1].severity, 4))

    def to_summary_dict(self) -> dict:
        """Serialisable summary for JSON export."""
        return {
            "aggregate": {
                "total_files":       self.aggregate.total_files,
                "total_lines":       self.aggregate.total_lines,
                "total_functions":   self.aggregate.total_functions,
                "total_classes":     self.aggregate.total_classes,
                "total_smells":      self.aggregate.total_smells,
                "avg_quality_score": round(self.aggregate.avg_quality_score, 1),
                "avg_complexity":    round(self.aggregate.avg_complexity, 2),
                "doc_coverage_pct":  round(self.aggregate.doc_coverage_pct, 1),
                "type_hint_pct":     round(self.aggregate.type_hint_pct, 1),
                "health_label":      self.aggregate.health_label,
                "smell_breakdown":   self.aggregate.smell_breakdown,
                "severity_breakdown": self.aggregate.severity_breakdown,
                "language_breakdown": self.aggregate.language_breakdown,
            },
            "file_scores": [
                {
                    "file":    s.file_path,
                    "overall": s.overall,
                    "grade":   s.grade,
                }
                for s in self.quality_scores
            ],
            "duplicates":   len(self.duplicates),
            "naming_issues": len(self.naming_issues),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Quality Scorer
# ─────────────────────────────────────────────────────────────────────────────

class QualityScorer:
    """
    Converts a ParsedFile into a FileQualityScore (0-100 per dimension).

    Scoring model (all dimensions start at 100, penalties deducted):

    documentation  : -20 per missing public docstring (max -100)
    complexity     : -10 per function above threshold (max -80)
    maintainability: -5  per Medium smell, -10 per High, -15 per Critical
    security       : -25 per Critical dangerous pattern, -15 per High
    style          : -2  per naming issue, -1 per long line, -3 per TODO

    overall = weighted average:
        doc(20%) + complexity(25%) + maintainability(25%) + security(20%) + style(10%)
    """

    _WEIGHTS = {
        "documentation":   0.20,
        "complexity":      0.25,
        "maintainability": 0.25,
        "security":        0.20,
        "style":           0.10,
    }

    def score(self, parsed: ParsedFile) -> FileQualityScore:
        doc   = self._score_documentation(parsed)
        comp  = self._score_complexity(parsed)
        maint = self._score_maintainability(parsed)
        sec   = self._score_security(parsed)
        style = self._score_style(parsed)

        overall = int(
            doc   * self._WEIGHTS["documentation"]
            + comp  * self._WEIGHTS["complexity"]
            + maint * self._WEIGHTS["maintainability"]
            + sec   * self._WEIGHTS["security"]
            + style * self._WEIGHTS["style"]
        )
        overall = max(0, min(100, overall))

        return FileQualityScore(
            file_path       = parsed.file_path,
            overall         = overall,
            documentation   = doc,
            complexity      = comp,
            maintainability = maint,
            security        = sec,
            style           = style,
        )

    # ── Dimension scorers ─────────────────────────────────────────────────────

    def _score_documentation(self, p: ParsedFile) -> int:
        score = 100
        for fn in p.functions:
            if not fn.name.startswith("_") and not fn.has_docstring:
                score -= 15
        for cls in p.classes:
            if not cls.name.startswith("_") and not cls.has_docstring:
                score -= 10
        if p.module_docstring is None and (p.functions or p.classes):
            score -= 5
        return max(0, score)

    def _score_complexity(self, p: ParsedFile) -> int:
        score = 100
        from backend.parser.ast_parser import MAX_COMPLEXITY, MAX_NESTING_DEPTH
        for fn in p.functions:
            if fn.complexity > MAX_COMPLEXITY:
                excess = fn.complexity - MAX_COMPLEXITY
                score -= min(30, excess * 5)
            if fn.max_nesting > MAX_NESTING_DEPTH:
                score -= 8
        return max(0, score)

    def _score_maintainability(self, p: ParsedFile) -> int:
        score = 100
        severity_penalty = {"Low": 1, "Medium": 5, "High": 10, "Critical": 15}
        # Only count non-security, non-style smells here
        security_types = {"dangerous_pattern"}
        style_types    = {"long_line", "missing_docstring"}
        for smell in p.code_smells:
            if smell.smell_type in security_types | style_types:
                continue
            score -= severity_penalty.get(smell.severity, 3)
        return max(0, score)

    def _score_security(self, p: ParsedFile) -> int:
        score = 100
        for smell in p.code_smells:
            if smell.smell_type != "dangerous_pattern":
                continue
            if smell.severity == "Critical":
                score -= 25
            elif smell.severity == "High":
                score -= 15
            elif smell.severity == "Medium":
                score -= 8
        return max(0, score)

    def _score_style(self, p: ParsedFile) -> int:
        score = 100
        long_lines = sum(
            1 for s in p.code_smells if s.smell_type == "long_line"
        )
        score -= min(20, long_lines * 2)
        score -= min(15, len(p.todo_comments) * 3)
        return max(0, score)


# ─────────────────────────────────────────────────────────────────────────────
# Cross-File Analyzer
# ─────────────────────────────────────────────────────────────────────────────

class CrossFileAnalyzer:
    """
    Detects patterns that span multiple files:
      - Duplicate function / class names
      - Naming convention violations
      - Import anomalies (heavy importers, wildcard files)
    """

    def analyze(
        self,
        parsed_files: list[ParsedFile],
        files: list[FileInfo],
    ) -> tuple[list[DuplicatePattern], list[NamingIssue], ImportAnomalies]:

        duplicates    = self._find_duplicates(parsed_files)
        naming_issues = self._check_naming_conventions(parsed_files)
        import_stats  = self._analyze_imports(parsed_files, files)

        return duplicates, naming_issues, import_stats

    # ─────────────────────────────────────────────────────────────────────────

    def _find_duplicates(
        self, parsed_files: list[ParsedFile]
    ) -> list[DuplicatePattern]:
        """
        Find function and class names that appear in more than one file.
        Ignores dunder names (__init__, __str__ …) and private names.
        """
        fn_locations:  defaultdict[str, list[str]] = defaultdict(list)
        cls_locations: defaultdict[str, list[str]] = defaultdict(list)

        for pf in parsed_files:
            for fn in pf.functions:
                if fn.name.startswith("_"):
                    continue
                fn_locations[fn.name].append(
                    f"{pf.file_path}:{fn.start_line}"
                )
            for cls in pf.classes:
                if cls.name.startswith("_"):
                    continue
                cls_locations[cls.name].append(
                    f"{pf.file_path}:{cls.start_line}"
                )

        duplicates: list[DuplicatePattern] = []
        for name, locs in fn_locations.items():
            if len(locs) > 1:
                duplicates.append(DuplicatePattern(
                    name      = name,
                    kind      = "function",
                    locations = locs,
                ))
        for name, locs in cls_locations.items():
            if len(locs) > 1:
                duplicates.append(DuplicatePattern(
                    name      = name,
                    kind      = "class",
                    locations = locs,
                ))

        return duplicates

    # ─────────────────────────────────────────────────────────────────────────

    def _check_naming_conventions(
        self, parsed_files: list[ParsedFile]
    ) -> list[NamingIssue]:
        """
        Check function names (snake_case), class names (PascalCase),
        and module-level variable names (snake_case or UPPER_CASE).
        """
        issues: list[NamingIssue] = []

        for pf in parsed_files:
            if pf.language != "python":
                continue

            for fn in pf.functions:
                if not _is_valid_function_name(fn.name):
                    issues.append(NamingIssue(
                        name        = fn.name,
                        kind        = "function",
                        file_path   = pf.file_path,
                        line_number = fn.start_line,
                        expected    = "snake_case (e.g. my_function)",
                    ))

            for cls in pf.classes:
                if not _is_valid_class_name(cls.name):
                    issues.append(NamingIssue(
                        name        = cls.name,
                        kind        = "class",
                        file_path   = pf.file_path,
                        line_number = cls.start_line,
                        expected    = "PascalCase (e.g. MyClass)",
                    ))

            for var in pf.global_variables:
                if not _is_valid_variable_name(var):
                    issues.append(NamingIssue(
                        name        = var,
                        kind        = "variable",
                        file_path   = pf.file_path,
                        line_number = 0,
                        expected    = "snake_case or UPPER_CASE constant",
                    ))

        return issues

    # ─────────────────────────────────────────────────────────────────────────

    def _analyze_imports(
        self,
        parsed_files: list[ParsedFile],
        files: list[FileInfo],
    ) -> ImportAnomalies:
        """Compute import statistics across the full repository."""
        total = stdlib = third_party = internal = 0
        wildcard_files: list[str] = []
        import_counts:  dict[str, int] = {}

        # Build a set of relative paths for internal import detection
        relative_paths = {f.relative_path for f in files}
        # Module names are inferred from the path stem
        internal_modules = {
            p.replace("/", ".").replace("\\", ".").removesuffix(".py")
            for p in relative_paths
        }

        for pf in parsed_files:
            file_import_count = 0
            has_wildcard = False

            for imp in pf.imports:
                total += 1
                file_import_count += 1

                if imp.is_wildcard:
                    has_wildcard = True

                if imp.is_stdlib:
                    stdlib += 1
                elif any(
                    imp.module.startswith(m) for m in internal_modules
                ):
                    internal += 1
                else:
                    third_party += 1

            if has_wildcard:
                wildcard_files.append(pf.file_path)
            import_counts[pf.file_path] = file_import_count

        heavy = sorted(
            import_counts.items(), key=lambda x: x[1], reverse=True
        )[:5]

        return ImportAnomalies(
            total_imports     = total,
            stdlib_count      = stdlib,
            third_party_count = third_party,
            internal_count    = internal,
            wildcard_files    = wildcard_files,
            heavy_importers   = heavy,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate Metrics Builder
# ─────────────────────────────────────────────────────────────────────────────

class MetricsBuilder:
    """Computes repository-level aggregate statistics from all ParsedFiles."""

    def build(
        self,
        parsed_files: list[ParsedFile],
        quality_scores: list[FileQualityScore],
        files: list[FileInfo],
    ) -> AggregateMetrics:

        total_functions  = sum(len(p.functions) for p in parsed_files)
        total_classes    = sum(len(p.classes)   for p in parsed_files)
        total_imports    = sum(len(p.imports)   for p in parsed_files)
        total_smells     = sum(len(p.code_smells) for p in parsed_files)
        total_lines      = sum(f.line_count for f in files)

        # Average function length
        all_fn_lengths = [
            fn.line_count
            for p in parsed_files
            for fn in p.functions
        ]
        avg_fn_length = (
            sum(all_fn_lengths) / len(all_fn_lengths)
            if all_fn_lengths else 0.0
        )

        # Complexity stats
        all_complexities = [
            fn.complexity
            for p in parsed_files
            for fn in p.functions
        ]
        avg_complexity = (
            sum(all_complexities) / len(all_complexities)
            if all_complexities else 0.0
        )
        max_complexity = max(all_complexities) if all_complexities else 0

        # Which function has max complexity?
        max_complexity_fn = ""
        for p in parsed_files:
            for fn in p.functions:
                if fn.complexity == max_complexity:
                    max_complexity_fn = f"{p.file_path}:{fn.name}"
                    break

        # Doc coverage
        pub_fns  = [fn for p in parsed_files for fn in p.functions
                    if not fn.name.startswith("_")]
        pub_cls  = [c  for p in parsed_files for c  in p.classes
                    if not c.name.startswith("_")]
        all_pub  = len(pub_fns) + len(pub_cls)
        all_docs = (
            sum(1 for f in pub_fns if f.has_docstring)
            + sum(1 for c in pub_cls if c.has_docstring)
        )
        doc_pct  = round(all_docs / all_pub * 100, 1) if all_pub else 100.0

        # Type hint coverage
        all_fns_count  = sum(len(p.functions) for p in parsed_files)
        typed_count    = sum(
            1 for p in parsed_files for fn in p.functions if fn.has_type_hints
        )
        type_hint_pct  = (
            round(typed_count / all_fns_count * 100, 1)
            if all_fns_count else 100.0
        )

        # Smell breakdowns
        smell_breakdown:    dict[str, int] = {}
        severity_breakdown: dict[str, int] = {
            "Critical": 0, "High": 0, "Medium": 0, "Low": 0
        }
        for p in parsed_files:
            for s in p.code_smells:
                smell_breakdown[s.smell_type] = (
                    smell_breakdown.get(s.smell_type, 0) + 1
                )
                severity_breakdown[s.severity] = (
                    severity_breakdown.get(s.severity, 0) + 1
                )

        # Language breakdown (from FileInfo)
        lang_breakdown: dict[str, int] = {}
        for f in files:
            lang_breakdown[f.language] = lang_breakdown.get(f.language, 0) + 1

        avg_quality = (
            sum(s.overall for s in quality_scores) / len(quality_scores)
            if quality_scores else 0.0
        )

        return AggregateMetrics(
            total_files         = len(files),
            total_lines         = total_lines,
            total_functions     = total_functions,
            total_classes       = total_classes,
            total_imports       = total_imports,
            total_smells        = total_smells,
            avg_function_length = round(avg_fn_length, 1),
            avg_complexity      = round(avg_complexity, 2),
            max_complexity      = max_complexity,
            max_complexity_fn   = max_complexity_fn,
            doc_coverage_pct    = doc_pct,
            type_hint_pct       = type_hint_pct,
            avg_quality_score   = round(avg_quality, 1),
            smell_breakdown     = smell_breakdown,
            severity_breakdown  = severity_breakdown,
            language_breakdown  = lang_breakdown,
            files_with_errors   = sum(1 for p in parsed_files if p.has_errors),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main CodeAnalyzer
# ─────────────────────────────────────────────────────────────────────────────

class CodeAnalyzer:
    """
    Orchestrates AST parsing and quality analysis for an entire repository.

    Usage:
        analyzer = CodeAnalyzer()
        report   = analyzer.analyze_all(files, progress_callback=cb)

    The returned AnalysisReport is everything downstream modules need:
      - Phase 5 (AI Reviewer) uses parsed_files + quality_scores
      - Phase 7 (Dashboard) uses the full report including aggregate metrics
    """

    def __init__(self):
        self.config         = get_config()
        self._parser        = PythonASTParser()
        self._scorer        = QualityScorer()
        self._cross         = CrossFileAnalyzer()
        self._metrics       = MetricsBuilder()

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze_file(self, file_info: FileInfo) -> ParsedFile:
        """
        Parse and analyse a single file.
        Safe to call from multiple threads.
        """
        try:
            return self._parser.parse(
                file_info.relative_path,
                file_info.content,
                file_info.language,
            )
        except Exception as exc:
            logger.error(f"Parse error on {file_info.relative_path}: {exc}")
            return ParsedFile(
                file_path   = file_info.relative_path,
                language    = file_info.language,
                line_count  = file_info.line_count,
                parse_error = str(exc),
            )

    def analyze_all(
        self,
        files: list[FileInfo],
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> AnalysisReport:
        """
        Full analysis pipeline.

        Steps:
          1. Parse every file (parallel)
          2. Score every file (sequential, fast)
          3. Cross-file analysis (duplicate names, naming, imports)
          4. Build aggregate metrics

        Args:
            files:             List of FileInfo from Phase 2.
            progress_callback: Optional callable(current, total, filename).

        Returns:
            AnalysisReport
        """
        if not files:
            logger.warning("analyze_all() called with empty file list.")
            return self._empty_report()

        logger.info(f"Starting analysis of {len(files)} files …")

        # ── Step 1: Parse (parallel) ──────────────────────────────────────
        parsed_files, errors = self._parse_parallel(files, progress_callback)

        # ── Step 2: Score each file ───────────────────────────────────────
        quality_scores = [self._scorer.score(p) for p in parsed_files]

        # ── Step 3: Cross-file analysis ───────────────────────────────────
        duplicates, naming_issues, import_stats = self._cross.analyze(
            parsed_files, files
        )

        # ── Step 4: Aggregate metrics ─────────────────────────────────────
        aggregate = self._metrics.build(parsed_files, quality_scores, files)

        report = AnalysisReport(
            parsed_files     = parsed_files,
            quality_scores   = quality_scores,
            aggregate        = aggregate,
            duplicates       = duplicates,
            naming_issues    = naming_issues,
            import_anomalies = import_stats,
            analysis_errors  = errors,
        )

        logger.info(
            f"Analysis complete — "
            f"{aggregate.total_files} files | "
            f"{aggregate.total_functions} functions | "
            f"{aggregate.total_smells} smells | "
            f"avg score {aggregate.avg_quality_score}/100 "
            f"({aggregate.health_label})"
        )
        return report

    # ── Internal ──────────────────────────────────────────────────────────────

    def _parse_parallel(
        self,
        files: list[FileInfo],
        progress_callback: Optional[Callable[[int, int, str], None]],
    ) -> tuple[list[ParsedFile], list[str]]:
        """Parse all files using a thread pool; return (parsed, errors)."""
        max_workers = min(
            self.config.app.max_parallel_reviews, len(files), 8
        )
        results: list[ParsedFile] = []
        errors:  list[str]        = []
        completed = 0
        total     = len(files)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_map = {
                pool.submit(self.analyze_file, f): f for f in files
            }
            for future in as_completed(future_map):
                fi = future_map[future]
                try:
                    parsed = future.result()
                    results.append(parsed)
                    if parsed.parse_error:
                        errors.append(
                            f"{fi.relative_path}: {parsed.parse_error}"
                        )
                except Exception as exc:
                    err_msg = f"{fi.relative_path}: unexpected error — {exc}"
                    logger.error(err_msg)
                    errors.append(err_msg)
                finally:
                    completed += 1
                    if progress_callback:
                        progress_callback(
                            completed, total, fi.relative_path
                        )

        # Sort results to match input order for deterministic output
        order = {f.relative_path: i for i, f in enumerate(files)}
        results.sort(key=lambda p: order.get(p.file_path, 9999))
        return results, errors

    @staticmethod
    def _empty_report() -> AnalysisReport:
        """Return a valid but empty AnalysisReport when no files exist."""
        empty_agg = AggregateMetrics(
            total_files=0, total_lines=0, total_functions=0,
            total_classes=0, total_imports=0, total_smells=0,
            avg_function_length=0.0, avg_complexity=0.0,
            max_complexity=0, max_complexity_fn="",
            doc_coverage_pct=100.0, type_hint_pct=100.0,
            avg_quality_score=0.0,
            smell_breakdown={}, severity_breakdown={},
            language_breakdown={}, files_with_errors=0,
        )
        return AnalysisReport(
            parsed_files=[], quality_scores=[], aggregate=empty_agg,
            duplicates=[], naming_issues=[],
            import_anomalies=ImportAnomalies(0, 0, 0, 0, [], []),
            analysis_errors=[],
        )
