"""
AI Review Engine
================
Sends source code + AST context to an LLM and returns structured,
confidence-rated review comments.

Supports:
  - Anthropic Claude  (claude-sonnet-4-20250514)
  - OpenAI GPT        (gpt-4o-mini)

Review pipeline per file:
  1. Convert AST-detected code smells → ReviewComments (instant, no API cost)
  2. Build a rich prompt from FileInfo + ParsedFile context
  3. Call LLM with retry/backoff
  4. Parse JSON response → ReviewComments
  5. Merge, de-duplicate, sort by severity + confidence

Output:  FileReview  (per file)
         ReviewReport (full repo — all files combined)

Design:
  - LLMClient        : thin wrapper over Anthropic / OpenAI SDK
  - PromptBuilder    : builds the system + user prompt pair
  - ResponseParser   : parses and validates LLM JSON output
  - AIReviewer       : orchestrates the pipeline for all files
"""

import json
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from backend.ingestion.repo_ingestion import FileInfo
from backend.parser.ast_parser import CodeSmell, ParsedFile
from backend.parser.code_analyzer import AnalysisReport, FileQualityScore
from backend.utils.config import get_config
from backend.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

VALID_SEVERITIES  = {"Low", "Medium", "High", "Critical"}
VALID_CATEGORIES  = {
    "Security", "Performance", "Readability",
    "Maintainability", "Scalability", "Best Practices", "Bug Risk",
}

SEVERITY_RANK = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}

# Files shorter than this get a lightweight review (fewer tokens)
SHORT_FILE_THRESHOLD = 30


# ─────────────────────────────────────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ReviewComment:
    """
    A single review finding for one location in a file.
    May originate from AST analysis (is_ast_detected=True)
    or from the LLM (is_ast_detected=False).
    """
    file_name:        str
    line_number:      int
    issue_type:       str        # short snake_case label
    severity:         str        # Low | Medium | High | Critical
    confidence_score: int        # 0–100  (100 = certain)
    explanation:      str        # human-readable problem description
    suggested_fix:    str        # concrete fix / recommendation
    category:         str        # Security | Performance | …
    is_ast_detected:  bool = False

    @property
    def severity_rank(self) -> int:
        return SEVERITY_RANK.get(self.severity, 4)

    def to_dict(self) -> dict:
        return {
            "file_name":        self.file_name,
            "line_number":      self.line_number,
            "issue_type":       self.issue_type,
            "severity":         self.severity,
            "confidence_score": self.confidence_score,
            "explanation":      self.explanation,
            "suggested_fix":    self.suggested_fix,
            "category":         self.category,
            "is_ast_detected":  self.is_ast_detected,
        }


@dataclass
class FileReview:
    """Complete review result for a single file."""
    file_name:     str
    language:      str
    line_count:    int
    comments:      list[ReviewComment] = field(default_factory=list)
    summary:       str = ""
    overall_score: int = 0        # 0-100 quality score from LLM
    review_error:  Optional[str] = None   # set if LLM call failed

    @property
    def critical_count(self) -> int:
        return sum(1 for c in self.comments if c.severity == "Critical")

    @property
    def high_count(self) -> int:
        return sum(1 for c in self.comments if c.severity == "High")

    @property
    def has_errors(self) -> bool:
        return self.review_error is not None

    @property
    def avg_confidence(self) -> float:
        if not self.comments:
            return 0.0
        return round(sum(c.confidence_score for c in self.comments)
                     / len(self.comments), 1)

    def comments_by_severity(self, severity: str) -> list[ReviewComment]:
        return [c for c in self.comments if c.severity == severity]

    def to_dict(self) -> dict:
        return {
            "file_name":     self.file_name,
            "language":      self.language,
            "line_count":    self.line_count,
            "overall_score": self.overall_score,
            "summary":       self.summary,
            "comment_count": len(self.comments),
            "critical":      self.critical_count,
            "high":          self.high_count,
            "avg_confidence": self.avg_confidence,
            "comments":      [c.to_dict() for c in self.comments],
            "review_error":  self.review_error,
        }


@dataclass
class ReviewReport:
    """
    Full review of an entire repository.
    Primary output of AIReviewer.review_all_files().
    Consumed by Orchestrator (Phase 6) and Dashboard (Phase 7).
    """
    file_reviews:      list[FileReview]
    total_comments:    int
    critical_count:    int
    high_count:        int
    medium_count:      int
    low_count:         int
    avg_confidence:    float
    avg_quality_score: float
    files_with_errors: int
    llm_provider:      str
    llm_model:         str

    # Fast lookup
    _by_file: dict[str, FileReview] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self._by_file = {r.file_name: r for r in self.file_reviews}

    def get_file_review(self, file_name: str) -> Optional[FileReview]:
        return self._by_file.get(file_name)

    def all_comments(self) -> list[ReviewComment]:
        """All comments across all files, sorted by severity then confidence."""
        comments = [c for r in self.file_reviews for c in r.comments]
        return sorted(comments,
                      key=lambda c: (c.severity_rank, -c.confidence_score))

    def comments_by_category(self) -> dict[str, list[ReviewComment]]:
        result: dict[str, list[ReviewComment]] = {}
        for c in self.all_comments():
            result.setdefault(c.category, []).append(c)
        return result

    def to_dict(self) -> dict:
        return {
            "summary": {
                "total_files":       len(self.file_reviews),
                "total_comments":    self.total_comments,
                "critical":          self.critical_count,
                "high":              self.high_count,
                "medium":            self.medium_count,
                "low":               self.low_count,
                "avg_confidence":    self.avg_confidence,
                "avg_quality_score": self.avg_quality_score,
                "files_with_errors": self.files_with_errors,
                "llm_provider":      self.llm_provider,
                "llm_model":         self.llm_model,
            },
            "file_reviews": [r.to_dict() for r in self.file_reviews],
        }


# ─────────────────────────────────────────────────────────────────────────────
# LLM System Prompt
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are an expert software engineer and code reviewer with deep knowledge of:
- Security vulnerabilities (OWASP Top 10, CWE)
- Performance optimisation patterns
- Clean code principles (SOLID, DRY, KISS)
- Language-specific best practices
- Design patterns and anti-patterns

You will receive source code with structural context and must produce a
thorough, actionable JSON code review.

## Response Format

Return ONLY valid JSON — no markdown fences, no prose outside JSON.

{
  "comments": [
    {
      "line_number":      <integer — the line where the issue starts>,
      "issue_type":       <snake_case label, e.g. "missing_input_validation">,
      "severity":         <"Low" | "Medium" | "High" | "Critical">,
      "confidence_score": <integer 0-100>,
      "category":         <"Security"|"Performance"|"Readability"|"Maintainability"|"Scalability"|"Best Practices"|"Bug Risk">,
      "explanation":      <clear, specific description of the problem>,
      "suggested_fix":    <concrete code snippet or actionable recommendation>
    }
  ],
  "summary":       <2-3 sentence overall assessment of the file>,
  "overall_score": <integer 0-100 representing code quality>
}

## Severity Guide
- Critical : Security vulnerability, data loss risk, crash-inducing bug
- High     : Logic error, serious performance issue, bad practice
- Medium   : Maintainability concern, moderate performance issue
- Low      : Style, minor readability, small improvements

## Confidence Guide
- 90-100 : Certain — obvious problem with no ambiguity
- 70-89  : Likely — strong evidence but context may change things
- 50-69  : Possible — pattern suggests a problem, needs verification
- 0-49   : Speculative — flagging for human review

## Rules
1. Be specific — reference exact line numbers and variable/function names
2. Every suggested_fix must be actionable (code snippet preferred)
3. Skip trivial issues already flagged in the pre-detected smells context
4. If the code is high quality, say so — don't invent issues
5. Return raw JSON only — no ```json fences"""


# ─────────────────────────────────────────────────────────────────────────────
# Prompt Builder
# ─────────────────────────────────────────────────────────────────────────────

class PromptBuilder:
    """
    Constructs the user-side LLM prompt from FileInfo + ParsedFile.
    Keeps prompts within the configured chunk_size to avoid token overruns.
    """

    def __init__(self):
        self.config = get_config()

    def build(self, file_info: FileInfo, parsed: ParsedFile) -> str:
        """Build the complete user message for the LLM."""
        context = self._build_context_block(file_info, parsed)
        code    = self._truncate_code(file_info)
        return (
            f"{context}\n\n"
            f"## Source Code\n"
            f"```{file_info.language}\n{code}\n```\n\n"
            f"Review this code and return JSON as specified."
        )

    def _build_context_block(
        self, file_info: FileInfo, parsed: ParsedFile
    ) -> str:
        lines = [
            "## File Context",
            f"- **File**     : `{file_info.relative_path}`",
            f"- **Language** : {file_info.language}",
            f"- **Lines**    : {file_info.line_count}",
        ]

        if parsed.language == "python":
            if parsed.functions:
                fn_summary = ", ".join(
                    f"`{f.name}` (complexity={f.complexity}, "
                    f"lines={f.line_count})"
                    for f in parsed.functions[:8]
                )
                lines.append(f"- **Functions**: {fn_summary}")

            if parsed.classes:
                cls_summary = ", ".join(
                    f"`{c.name}` ({len(c.methods)} methods)"
                    for c in parsed.classes[:5]
                )
                lines.append(f"- **Classes**  : {cls_summary}")

            lines.append(
                f"- **Doc coverage** : {parsed.doc_coverage:.0%}"
            )
            lines.append(
                f"- **Type hints**   : {parsed.type_hint_coverage:.0%}"
            )
            lines.append(
                f"- **Complexity**   : {parsed.complexity_score} total"
            )

            if parsed.code_smells:
                smell_lines = []
                for s in parsed.code_smells[:10]:
                    smell_lines.append(
                        f"  - L{s.line_number} [{s.severity}] "
                        f"`{s.smell_type}`: {s.description[:80]}"
                    )
                lines.append(
                    "- **Pre-detected smells** (skip these in your review):\n"
                    + "\n".join(smell_lines)
                )

            if parsed.todo_comments:
                todos = ", ".join(
                    f"L{ln}" for ln, _ in parsed.todo_comments[:5]
                )
                lines.append(f"- **TODOs** : {todos}")

        return "\n".join(lines)

    def _truncate_code(self, file_info: FileInfo) -> str:
        """Truncate source code to stay within chunk_size characters."""
        max_chars = self.config.llm.chunk_size * 4   # chars ≈ 4× tokens
        code = file_info.content
        if len(code) <= max_chars:
            return code
        truncated = code[:max_chars]
        omitted   = len(code) - max_chars
        return (
            truncated
            + f"\n\n# … [{omitted} characters truncated — "
            f"review the visible portion only]"
        )


# ─────────────────────────────────────────────────────────────────────────────
# LLM Client
# ─────────────────────────────────────────────────────────────────────────────

class LLMClient:
    """
    Thin wrapper over Anthropic / OpenAI SDK.
    Implements exponential-backoff retry on transient failures.
    """

    def __init__(self):
        self.config    = get_config()
        self._provider = self.config.llm.provider.lower()
        self._client   = None   # lazily initialised

    def call(self, user_message: str) -> str:
        """
        Send a message to the configured LLM and return the text response.
        Raises RuntimeError after all retries are exhausted.
        """
        client       = self._get_client()
        max_retries  = self.config.llm.max_retries
        last_error: Optional[Exception] = None

        for attempt in range(max_retries):
            try:
                return self._dispatch(client, user_message)

            except Exception as exc:
                last_error = exc
                is_last    = attempt == max_retries - 1

                if is_last:
                    logger.error(
                        f"LLM call failed after {max_retries} attempts: {exc}"
                    )
                    break

                wait = 2 ** attempt        # 1 s, 2 s, 4 s …
                logger.warning(
                    f"LLM attempt {attempt + 1}/{max_retries} failed "
                    f"({type(exc).__name__}). Retrying in {wait}s…"
                )
                time.sleep(wait)

        raise RuntimeError(
            f"LLM call exhausted {max_retries} retries. "
            f"Last error: {last_error}"
        )

    def _dispatch(self, client, user_message: str) -> str:
        """Route to the correct provider API."""
        if self._provider == "anthropic":
            response = client.messages.create(
                model       = self.config.llm.anthropic_model,
                max_tokens  = self.config.llm.max_tokens,
                temperature = self.config.llm.temperature,
                system      = _SYSTEM_PROMPT,
                messages    = [{"role": "user", "content": user_message}],
            )
            return response.content[0].text

        if self._provider == "openai":
            response = client.chat.completions.create(
                model       = self.config.llm.openai_model,
                max_tokens  = self.config.llm.max_tokens,
                temperature = self.config.llm.temperature,
                messages    = [
                    {"role": "system",  "content": _SYSTEM_PROMPT},
                    {"role": "user",    "content": user_message},
                ],
                response_format = {"type": "json_object"},
            )
            return response.choices[0].message.content

        raise RuntimeError(
            f"Unknown LLM provider: '{self._provider}'. "
            "Set LLM_PROVIDER=anthropic or LLM_PROVIDER=openai in .env"
        )

    def _get_client(self):
        """Lazily initialise the SDK client for the configured provider."""
        if self._client is not None:
            return self._client

        if self._provider == "anthropic":
            try:
                import anthropic
            except ImportError:
                raise RuntimeError(
                    "anthropic package not found. Run: pip install anthropic"
                )
            key = self.config.llm.anthropic_api_key
            if not key:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY is not set. "
                    "Add it to your .env file."
                )
            self._client = anthropic.Anthropic(api_key=key)

        elif self._provider == "openai":
            try:
                import openai
            except ImportError:
                raise RuntimeError(
                    "openai package not found. Run: pip install openai"
                )
            key = self.config.llm.openai_api_key
            if not key:
                raise RuntimeError(
                    "OPENAI_API_KEY is not set. "
                    "Add it to your .env file."
                )
            self._client = openai.OpenAI(api_key=key)

        else:
            raise RuntimeError(
                f"Unknown LLM_PROVIDER: '{self._provider}'"
            )

        return self._client


# ─────────────────────────────────────────────────────────────────────────────
# Response Parser
# ─────────────────────────────────────────────────────────────────────────────

class ResponseParser:
    """
    Parses and validates the raw LLM JSON response.
    Returns (comments, summary, overall_score).
    Gracefully handles malformed JSON and partial responses.
    """

    def parse(
        self, raw: str, file_name: str
    ) -> tuple[list[ReviewComment], str, int]:
        """
        Parse raw LLM text into structured ReviewComments.

        Returns:
            (comments, summary, overall_score)
            — empty list / defaults on parse failure.
        """
        data = self._extract_json(raw)
        if data is None:
            logger.warning(
                f"Could not extract JSON from LLM response for {file_name}"
            )
            return [], "Could not parse LLM response.", 50

        raw_comments = data.get("comments", [])
        if not isinstance(raw_comments, list):
            raw_comments = []

        comments: list[ReviewComment] = []
        for item in raw_comments:
            comment = self._parse_comment(item, file_name)
            if comment:
                comments.append(comment)

        summary = str(data.get("summary", "")).strip()[:1000]
        try:
            score = max(0, min(100, int(data.get("overall_score", 50))))
        except (TypeError, ValueError):
            score = 50

        return comments, summary, score

    # ── Internal ──────────────────────────────────────────────────────────────

    def _extract_json(self, raw: str) -> Optional[dict]:
        """Try several strategies to extract a JSON object from raw text."""
        # Strategy 1: direct parse
        cleaned = raw.strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Strategy 2: strip accidental ```json … ``` fences
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$",          "", cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Strategy 3: find the outermost { … } block
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        return None

    def _parse_comment(
        self, item: dict, file_name: str
    ) -> Optional[ReviewComment]:
        """Convert a raw dict from the LLM into a ReviewComment."""
        if not isinstance(item, dict):
            return None

        try:
            severity = str(item.get("severity", "Low")).strip()
            if severity not in VALID_SEVERITIES:
                severity = "Low"

            category = str(item.get("category", "Best Practices")).strip()
            if category not in VALID_CATEGORIES:
                category = "Best Practices"

            confidence = item.get("confidence_score", 50)
            try:
                confidence = max(0, min(100, int(confidence)))
            except (TypeError, ValueError):
                confidence = 50

            line_number = item.get("line_number", 1)
            try:
                line_number = max(1, int(line_number))
            except (TypeError, ValueError):
                line_number = 1

            explanation  = str(item.get("explanation",   "")).strip()
            suggested    = str(item.get("suggested_fix", "")).strip()
            issue_type   = str(item.get("issue_type",    "issue")).strip()

            # Skip empty / useless comments
            if not explanation:
                return None

            return ReviewComment(
                file_name        = file_name,
                line_number      = line_number,
                issue_type       = issue_type,
                severity         = severity,
                confidence_score = confidence,
                explanation      = explanation,
                suggested_fix    = suggested,
                category         = category,
                is_ast_detected  = False,
            )
        except Exception as exc:
            logger.debug(f"Skipping malformed comment item: {exc}")
            return None


# ─────────────────────────────────────────────────────────────────────────────
# AST → ReviewComment converter
# ─────────────────────────────────────────────────────────────────────────────

class SmellConverter:
    """
    Converts AST-detected CodeSmells into ReviewComments.
    These are always included — they don't consume LLM tokens.
    """

    # Maps smell_type → review category
    _CATEGORY_MAP: dict[str, str] = {
        "dangerous_pattern":       "Security",
        "mutable_default_argument": "Bug Risk",
        "bare_except":             "Bug Risk",
        "long_function":           "Maintainability",
        "too_many_arguments":      "Maintainability",
        "high_complexity":         "Maintainability",
        "deep_nesting":            "Maintainability",
        "god_class":               "Maintainability",
        "wildcard_import":         "Best Practices",
        "missing_docstring":       "Readability",
        "long_line":               "Readability",
    }

    # Confidence levels per severity  (AST is deterministic → high confidence)
    _CONFIDENCE: dict[str, int] = {
        "Critical": 95,
        "High":     88,
        "Medium":   80,
        "Low":      72,
    }

    # Stock suggested fixes per smell type
    _FIX_MAP: dict[str, str] = {
        "dangerous_pattern":
            "Replace with a safer alternative — see the explanation for details.",
        "mutable_default_argument":
            "Use `None` as default and assign inside the function:\n"
            "  def f(items=None):\n      items = items or []",
        "bare_except":
            "Catch specific exceptions:\n"
            "  except (ValueError, TypeError) as e:\n      ...",
        "long_function":
            "Split into smaller, single-purpose helper functions.",
        "too_many_arguments":
            "Group related parameters into a dataclass or config object.",
        "high_complexity":
            "Extract branches into clearly named helper functions "
            "or use a strategy/lookup pattern.",
        "deep_nesting":
            "Flatten with early returns:\n"
            "  if not condition: return\n  # happy path continues",
        "god_class":
            "Apply the Single Responsibility Principle — "
            "split into focused, cohesive classes.",
        "wildcard_import":
            "Replace with explicit imports:\n"
            "  from module import ClassA, function_b",
        "missing_docstring":
            "Add a docstring:\n"
            '  """Brief description.\n\n  Args: ...\n  Returns: ...\n  """',
        "long_line":
            "Break the line using parentheses or a backslash continuation.",
    }

    def convert(
        self, smells: list[CodeSmell], file_name: str
    ) -> list[ReviewComment]:
        return [self._to_comment(s, file_name) for s in smells]

    def _to_comment(self, smell: CodeSmell, file_name: str) -> ReviewComment:
        category = self._CATEGORY_MAP.get(smell.smell_type, "Best Practices")
        confidence = self._CONFIDENCE.get(smell.severity, 75)
        fix = self._FIX_MAP.get(smell.smell_type,
                                 "Refactor to address this issue.")
        return ReviewComment(
            file_name        = file_name,
            line_number      = smell.line_number,
            issue_type       = smell.smell_type,
            severity         = smell.severity,
            confidence_score = confidence,
            explanation      = smell.description,
            suggested_fix    = fix,
            category         = category,
            is_ast_detected  = True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main AIReviewer
# ─────────────────────────────────────────────────────────────────────────────

class AIReviewer:
    """
    Orchestrates the full AI review pipeline for one or many files.

    Usage (single file):
        reviewer = AIReviewer()
        review   = reviewer.review_file(file_info, parsed_file)

    Usage (full repo):
        report = reviewer.review_all_files(
            files, parsed_files, progress_callback=cb
        )
    """

    def __init__(self):
        self.config    = get_config()
        self._llm      = LLMClient()
        self._prompt   = PromptBuilder()
        self._parser   = ResponseParser()
        self._converter = SmellConverter()

    # ── Public API ────────────────────────────────────────────────────────────

    def review_file(
        self,
        file_info: FileInfo,
        parsed:    ParsedFile,
        score:     Optional[FileQualityScore] = None,
    ) -> FileReview:
        """
        Review a single file:
          1. Convert AST smells → comments (always)
          2. Call LLM for deeper analysis (skipped for empty/trivial files)
          3. Merge, de-duplicate, sort

        Args:
            file_info : The raw file (content, path, language).
            parsed    : AST analysis result from Phase 3/4.
            score     : Optional quality score from Phase 4.

        Returns:
            FileReview with all comments and an overall score.
        """
        review = FileReview(
            file_name  = file_info.relative_path,
            language   = file_info.language,
            line_count = file_info.line_count,
        )

        # Step 1 — AST-detected smells (free, always run)
        ast_comments = self._converter.convert(
            parsed.code_smells, file_info.relative_path
        )
        review.comments.extend(ast_comments)

        # Step 2 — Skip LLM for trivially small or non-reviewable files
        if file_info.line_count < 3:
            review.summary       = "File too small for AI review."
            review.overall_score = 95
            self._sort_comments(review)
            return review

        # Step 3 — LLM review
        try:
            prompt       = self._prompt.build(file_info, parsed)
            raw_response = self._llm.call(prompt)
            llm_comments, summary, llm_score = self._parser.parse(
                raw_response, file_info.relative_path
            )

            review.comments.extend(llm_comments)
            review.summary       = summary
            review.overall_score = llm_score

        except RuntimeError as exc:
            # LLM failed after all retries — degrade gracefully
            logger.error(
                f"LLM review failed for {file_info.relative_path}: {exc}"
            )
            review.review_error  = str(exc)
            review.summary       = (
                "AI review unavailable — AST-based findings shown only. "
                "Check your API key and network connection."
            )
            # Estimate score from quality scorer result or AST penalties
            if score:
                review.overall_score = score.overall
            else:
                penalty = min(60, len(ast_comments) * 5)
                review.overall_score = max(10, 80 - penalty)

        # Step 4 — Remove near-duplicates and sort
        review.comments = self._deduplicate(review.comments)
        self._sort_comments(review)

        logger.info(
            f"  {file_info.relative_path}: "
            f"{len(review.comments)} comments "
            f"(AST={len(ast_comments)}, "
            f"LLM={len(review.comments) - len(ast_comments)}), "
            f"score={review.overall_score}"
        )
        return review

    def review_all_files(
        self,
        files:             list[FileInfo],
        parsed_files:      list[ParsedFile],
        analysis_report:   Optional[AnalysisReport] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> ReviewReport:
        """
        Review every file in the repository sequentially.

        LLM APIs have rate limits — we call them one at a time.
        AST analysis is still fast because it runs in-process.

        Args:
            files            : FileInfo list from Phase 2.
            parsed_files     : ParsedFile list from Phase 3/4.
            analysis_report  : Optional AnalysisReport from Phase 4
                               (used for quality scores).
            progress_callback: Optional callable(current, total, filename).

        Returns:
            ReviewReport aggregating all FileReviews.
        """
        # Build fast-lookup maps
        parsed_map: dict[str, ParsedFile] = {
            p.file_path: p for p in parsed_files
        }
        score_map: dict[str, FileQualityScore] = {}
        if analysis_report:
            score_map = {
                s.file_path: s for s in analysis_report.quality_scores
            }

        file_reviews: list[FileReview] = []
        total = len(files)

        logger.info(
            f"Starting AI review of {total} files "
            f"[provider={self.config.llm.provider}, "
            f"model={self.config.llm.active_model}]"
        )

        for idx, file_info in enumerate(files, 1):
            if progress_callback:
                progress_callback(idx, total, file_info.relative_path)

            parsed = parsed_map.get(file_info.relative_path)
            if parsed is None:
                logger.warning(
                    f"No parsed data for {file_info.relative_path} — skipping"
                )
                continue

            score = score_map.get(file_info.relative_path)
            review = self.review_file(file_info, parsed, score)
            file_reviews.append(review)

        return self._build_report(file_reviews)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _deduplicate(
        self, comments: list[ReviewComment]
    ) -> list[ReviewComment]:
        """
        Remove near-duplicate comments.
        Two comments are considered duplicates if they share the same
        (issue_type, line_number) pair — keep the higher-confidence one.
        """
        seen: dict[tuple[str, int], ReviewComment] = {}
        for comment in comments:
            key = (comment.issue_type, comment.line_number)
            existing = seen.get(key)
            if existing is None:
                seen[key] = comment
            elif comment.confidence_score > existing.confidence_score:
                seen[key] = comment
        return list(seen.values())

    @staticmethod
    def _sort_comments(review: FileReview) -> None:
        """Sort in-place: Critical → High → Medium → Low, then by confidence."""
        review.comments.sort(
            key=lambda c: (c.severity_rank, -c.confidence_score)
        )

    @staticmethod
    def _build_report(file_reviews: list[FileReview]) -> ReviewReport:
        """Aggregate FileReviews into a ReviewReport."""
        all_comments = [c for r in file_reviews for c in r.comments]

        sev_counts = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}
        for c in all_comments:
            sev_counts[c.severity] = sev_counts.get(c.severity, 0) + 1

        avg_conf = (
            round(sum(c.confidence_score for c in all_comments)
                  / len(all_comments), 1)
            if all_comments else 0.0
        )
        avg_score = (
            round(sum(r.overall_score for r in file_reviews)
                  / len(file_reviews), 1)
            if file_reviews else 0.0
        )

        cfg = get_config()
        return ReviewReport(
            file_reviews      = file_reviews,
            total_comments    = len(all_comments),
            critical_count    = sev_counts["Critical"],
            high_count        = sev_counts["High"],
            medium_count      = sev_counts["Medium"],
            low_count         = sev_counts["Low"],
            avg_confidence    = avg_conf,
            avg_quality_score = avg_score,
            files_with_errors = sum(1 for r in file_reviews if r.has_errors),
            llm_provider      = cfg.llm.provider,
            llm_model         = cfg.llm.active_model,
        )
