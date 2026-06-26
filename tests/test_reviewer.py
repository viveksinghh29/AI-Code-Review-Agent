"""Unit tests for the AI review engine and related review components."""

import json
import pytest
from unittest.mock import patch, MagicMock

from backend.ingestion.repo_ingestion import FileInfo
from backend.parser.ast_parser import CodeSmell, PythonASTParser
from backend.parser.code_analyzer import FileQualityScore
from backend.reviewer.ai_reviewer import (
    AIReviewer,
    FileReview,
    PromptBuilder,
    ResponseParser,
    ReviewComment,
    ReviewReport,
    SmellConverter,
    SEVERITY_RANK,
    VALID_CATEGORIES,
    VALID_SEVERITIES,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_comment(sev="High", conf=80, cat="Security", ast=False):
    return ReviewComment(
        file_name="f.py", line_number=1, issue_type="x",
        severity=sev, confidence_score=conf,
        explanation="desc", suggested_fix="fix",
        category=cat, is_ast_detected=ast,
    )


def make_file_info(content="x = 1\n", path="f.py", lang="python"):
    return FileInfo(
        path=path, relative_path=path, language=lang,
        size_bytes=len(content), content=content,
        line_count=content.count("\n") + 1,
    )


def make_review_report(comments=None):
    fr = FileReview("f.py", "python", 10)
    fr.comments      = comments or []
    fr.overall_score = 75
    return ReviewReport(
        file_reviews=[fr],
        total_comments=len(fr.comments),
        critical_count=sum(1 for c in fr.comments if c.severity == "Critical"),
        high_count=sum(1 for c in fr.comments if c.severity == "High"),
        medium_count=sum(1 for c in fr.comments if c.severity == "Medium"),
        low_count=sum(1 for c in fr.comments if c.severity == "Low"),
        avg_confidence=80.0,
        avg_quality_score=75.0,
        files_with_errors=0,
        llm_provider="anthropic",
        llm_model="claude-sonnet-4-20250514",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

class TestConstants:
    def test_severity_rank_order(self):
        assert SEVERITY_RANK["Critical"] < SEVERITY_RANK["High"]
        assert SEVERITY_RANK["High"]     < SEVERITY_RANK["Medium"]
        assert SEVERITY_RANK["Medium"]   < SEVERITY_RANK["Low"]

    def test_valid_severities_complete(self):
        assert VALID_SEVERITIES == {"Low", "Medium", "High", "Critical"}

    def test_valid_categories_complete(self):
        expected = {"Security", "Performance", "Readability",
                    "Maintainability", "Scalability", "Best Practices", "Bug Risk"}
        assert VALID_CATEGORIES == expected


# ─────────────────────────────────────────────────────────────────────────────
# ReviewComment
# ─────────────────────────────────────────────────────────────────────────────

class TestReviewComment:
    def test_severity_rank_critical(self):
        c = make_comment(sev="Critical")
        assert c.severity_rank == 0

    def test_severity_rank_low(self):
        c = make_comment(sev="Low")
        assert c.severity_rank == 3

    def test_to_dict_keys(self):
        c = make_comment()
        d = c.to_dict()
        for key in ["file_name", "line_number", "issue_type", "severity",
                    "confidence_score", "explanation", "suggested_fix",
                    "category", "is_ast_detected"]:
            assert key in d

    def test_to_dict_values(self):
        c = make_comment(sev="High", conf=85)
        d = c.to_dict()
        assert d["severity"]         == "High"
        assert d["confidence_score"] == 85
        assert d["is_ast_detected"]  is False

    def test_ast_detected_flag(self):
        c = make_comment(ast=True)
        assert c.is_ast_detected is True


# ─────────────────────────────────────────────────────────────────────────────
# FileReview
# ─────────────────────────────────────────────────────────────────────────────

class TestFileReview:
    def test_critical_count(self):
        fr = FileReview("f.py", "python", 10)
        fr.comments = [make_comment("Critical"), make_comment("High")]
        assert fr.critical_count == 1

    def test_high_count(self):
        fr = FileReview("f.py", "python", 10)
        fr.comments = [make_comment("High"), make_comment("High")]
        assert fr.high_count == 2

    def test_has_errors_false(self):
        fr = FileReview("f.py", "python", 10)
        assert fr.has_errors is False

    def test_has_errors_true(self):
        fr = FileReview("f.py", "python", 10, review_error="LLM failed")
        assert fr.has_errors is True

    def test_avg_confidence(self):
        fr = FileReview("f.py", "python", 10)
        fr.comments = [make_comment(conf=80), make_comment(conf=90)]
        assert fr.avg_confidence == 85.0

    def test_avg_confidence_empty(self):
        fr = FileReview("f.py", "python", 10)
        assert fr.avg_confidence == 0.0

    def test_comments_by_severity(self):
        fr = FileReview("f.py", "python", 10)
        fr.comments = [
            make_comment("Critical"), make_comment("High"),
            make_comment("Low"),      make_comment("Low"),
        ]
        assert len(fr.comments_by_severity("Low")) == 2
        assert len(fr.comments_by_severity("Critical")) == 1

    def test_to_dict_keys(self):
        fr = FileReview("f.py", "python", 10)
        d  = fr.to_dict()
        for key in ["file_name", "language", "line_count",
                    "overall_score", "summary", "comments"]:
            assert key in d


# ─────────────────────────────────────────────────────────────────────────────
# ReviewReport
# ─────────────────────────────────────────────────────────────────────────────

class TestReviewReport:
    def test_get_file_review_found(self):
        rpt = make_review_report()
        assert rpt.get_file_review("f.py") is not None

    def test_get_file_review_not_found(self):
        rpt = make_review_report()
        assert rpt.get_file_review("nonexistent.py") is None

    def test_all_comments_sorted_by_severity(self):
        comments = [
            make_comment("Low"), make_comment("Critical"),
            make_comment("High"), make_comment("Medium"),
        ]
        rpt    = make_review_report(comments)
        sorted_comments = rpt.all_comments()
        ranks  = [c.severity_rank for c in sorted_comments]
        assert ranks == sorted(ranks)

    def test_comments_by_category(self):
        comments = [
            make_comment(cat="Security"),
            make_comment(cat="Security"),
            make_comment(cat="Bug Risk"),
        ]
        rpt  = make_review_report(comments)
        cats = rpt.comments_by_category()
        assert len(cats["Security"]) == 2
        assert len(cats["Bug Risk"]) == 1

    def test_to_dict_keys(self):
        rpt = make_review_report()
        d   = rpt.to_dict()
        assert "summary"      in d
        assert "file_reviews" in d

    def test_to_dict_summary_values(self):
        comments = [make_comment("Critical"), make_comment("High")]
        rpt = make_review_report(comments)
        rpt.total_comments = 2
        rpt.critical_count = 1
        rpt.high_count     = 1
        d   = rpt.to_dict()
        assert d["summary"]["total_comments"] == 2


# ─────────────────────────────────────────────────────────────────────────────
# ResponseParser
# ─────────────────────────────────────────────────────────────────────────────

class TestResponseParser:
    def setup_method(self):
        self.parser = ResponseParser()

    def _valid_json(self, comments=None, summary="OK", score=80):
        return json.dumps({
            "comments": comments or [],
            "summary":  summary,
            "overall_score": score,
        })

    def test_parses_valid_json(self):
        raw = self._valid_json(score=75, summary="Looks good.")
        cs, s, sc = self.parser.parse(raw, "f.py")
        assert sc == 75
        assert s  == "Looks good."

    def test_parses_comment_fields(self):
        raw = self._valid_json(comments=[{
            "line_number": 10, "issue_type": "sql_inj", "severity": "Critical",
            "confidence_score": 95, "category": "Security",
            "explanation": "SQL injection risk.", "suggested_fix": "Use params.",
        }])
        cs, _, _ = self.parser.parse(raw, "f.py")
        assert len(cs) == 1
        assert cs[0].severity         == "Critical"
        assert cs[0].confidence_score == 95

    def test_strips_markdown_fences(self):
        raw = "```json\n" + self._valid_json(score=90) + "\n```"
        _, _, sc = self.parser.parse(raw, "f.py")
        assert sc == 90

    def test_invalid_json_returns_defaults(self):
        cs, s, sc = self.parser.parse("not json !!", "f.py")
        assert cs == []
        assert sc == 50

    def test_invalid_severity_normalised(self):
        raw = self._valid_json(comments=[{
            "line_number": 1, "issue_type": "x", "severity": "EXTREME",
            "confidence_score": 80, "category": "Security",
            "explanation": "E.", "suggested_fix": "F.",
        }])
        cs, _, _ = self.parser.parse(raw, "f.py")
        assert cs[0].severity == "Low"   # normalised to default

    def test_invalid_category_normalised(self):
        raw = self._valid_json(comments=[{
            "line_number": 1, "issue_type": "x", "severity": "High",
            "confidence_score": 80, "category": "WeirdCategory",
            "explanation": "E.", "suggested_fix": "F.",
        }])
        cs, _, _ = self.parser.parse(raw, "f.py")
        assert cs[0].category == "Best Practices"

    def test_confidence_clamped(self):
        raw = self._valid_json(comments=[{
            "line_number": 1, "issue_type": "x", "severity": "High",
            "confidence_score": 999, "category": "Security",
            "explanation": "E.", "suggested_fix": "F.",
        }])
        cs, _, _ = self.parser.parse(raw, "f.py")
        assert cs[0].confidence_score == 100

    def test_score_clamped(self):
        raw = self._valid_json(score=200)
        _, _, sc = self.parser.parse(raw, "f.py")
        assert sc == 100

    def test_empty_explanation_skipped(self):
        raw = self._valid_json(comments=[{
            "line_number": 1, "issue_type": "x", "severity": "High",
            "confidence_score": 80, "category": "Security",
            "explanation": "", "suggested_fix": "F.",
        }])
        cs, _, _ = self.parser.parse(raw, "f.py")
        assert len(cs) == 0

    def test_empty_comments_array(self):
        raw = self._valid_json(comments=[])
        cs, _, _ = self.parser.parse(raw, "f.py")
        assert cs == []


# ─────────────────────────────────────────────────────────────────────────────
# SmellConverter
# ─────────────────────────────────────────────────────────────────────────────

class TestSmellConverter:
    def setup_method(self):
        self.converter = SmellConverter()

    def test_converts_all_smells(self):
        smells = [
            CodeSmell("dangerous_pattern",        "eval()",        1, "Critical"),
            CodeSmell("long_function",             "Too long",      2, "Medium"),
            CodeSmell("mutable_default_argument",  "Mutable def",   3, "High"),
            CodeSmell("bare_except",               "Bare except",   4, "High"),
            CodeSmell("missing_docstring",         "No doc",        5, "Low"),
            CodeSmell("wildcard_import",           "Wildcard",      6, "Medium"),
            CodeSmell("god_class",                 "God class",     7, "High"),
        ]
        comments = self.converter.convert(smells, "f.py")
        assert len(comments) == 7
        assert all(c.is_ast_detected for c in comments)
        assert all(c.file_name == "f.py" for c in comments)

    def test_security_category_for_dangerous_pattern(self):
        smells   = [CodeSmell("dangerous_pattern", "eval()", 1, "Critical")]
        comments = self.converter.convert(smells, "f.py")
        assert comments[0].category == "Security"

    def test_bug_risk_for_bare_except(self):
        smells   = [CodeSmell("bare_except", "Bare except", 1, "High")]
        comments = self.converter.convert(smells, "f.py")
        assert comments[0].category == "Bug Risk"

    def test_confidence_high_for_critical(self):
        smells   = [CodeSmell("dangerous_pattern", "eval", 1, "Critical")]
        comments = self.converter.convert(smells, "f.py")
        assert comments[0].confidence_score >= 90

    def test_suggested_fix_not_empty(self):
        smells   = [CodeSmell("long_function", "Long fn", 1, "Medium")]
        comments = self.converter.convert(smells, "f.py")
        assert comments[0].suggested_fix != ""


# ─────────────────────────────────────────────────────────────────────────────
# PromptBuilder
# ─────────────────────────────────────────────────────────────────────────────

class TestPromptBuilder:
    def setup_method(self):
        self.builder = PromptBuilder()
        self.parser  = PythonASTParser()

    def _build(self, content, path="f.py", lang="python"):
        fi = make_file_info(content, path, lang)
        p  = self.parser.parse(path, content, lang)
        return self.builder.build(fi, p)

    def test_contains_filename(self):
        prompt = self._build("x = 1\n", path="mymodule.py")
        assert "mymodule.py" in prompt

    def test_contains_language(self):
        prompt = self._build("x = 1\n")
        assert "python" in prompt

    def test_contains_source_code(self):
        prompt = self._build("SECRET = 42\n")
        assert "SECRET" in prompt

    def test_contains_function_name(self):
        prompt = self._build("def my_function(): pass\n")
        assert "my_function" in prompt

    def test_contains_class_name(self):
        prompt = self._build("class MyClass: pass\n")
        assert "MyClass" in prompt

    def test_truncates_large_files(self):
        content = "x = 1\n" * 5000
        fi = make_file_info(content)
        p  = self.parser.parse("f.py", content, "python")
        prompt = self.builder.build(fi, p)
        assert "truncated" in prompt

    def test_contains_smell_context(self):
        content = "def f(x):\n    eval(x)\n"
        prompt  = self._build(content)
        assert "Pre-detected" in prompt or "smells" in prompt.lower()


# ─────────────────────────────────────────────────────────────────────────────
# AIReviewer (with mocked LLM)
# ─────────────────────────────────────────────────────────────────────────────

MOCK_LLM_RESPONSE = json.dumps({
    "comments": [
        {
            "line_number": 3, "issue_type": "unsafe_deser",
            "severity": "Critical", "confidence_score": 97,
            "category": "Security",
            "explanation": "pickle.loads executes arbitrary code.",
            "suggested_fix": "Use json.loads() instead.",
        }
    ],
    "summary":       "Critical security vulnerability found.",
    "overall_score": 20,
})


class TestAIReviewer:
    def setup_method(self):
        self.reviewer = AIReviewer()

    def _mock_call(self, response=MOCK_LLM_RESPONSE):
        return patch(
            "backend.reviewer.ai_reviewer.LLMClient.call",
            return_value=response,
        )

    def test_returns_file_review(self):
        fi = make_file_info("def f(): pass\n    pass\n    pass\n")
        p  = PythonASTParser().parse("f.py", fi.content, "python")
        with self._mock_call():
            review = self.reviewer.review_file(fi, p)
        assert isinstance(review, FileReview)

    def test_ast_comments_always_included(self):
        content = "def f(x):\n    eval(x)\n    pass\n"
        fi = make_file_info(content)
        p  = PythonASTParser().parse("f.py", content, "python")
        with self._mock_call():
            review = self.reviewer.review_file(fi, p)
        ast_comments = [c for c in review.comments if c.is_ast_detected]
        assert len(ast_comments) > 0

    def test_llm_comments_merged(self):
        content = "import pickle\ndef f(d):\n    return pickle.loads(d)\n"
        fi = make_file_info(content)
        p  = PythonASTParser().parse("f.py", content, "python")
        with self._mock_call():
            review = self.reviewer.review_file(fi, p)
        llm_comments = [c for c in review.comments if not c.is_ast_detected]
        assert len(llm_comments) >= 1

    def test_graceful_degradation_on_llm_failure(self):
        content = "import pickle\ndef f(d):\n    return pickle.loads(d)\n"
        fi = make_file_info(content)
        p  = PythonASTParser().parse("f.py", content, "python")
        with patch("backend.reviewer.ai_reviewer.LLMClient.call",
                   side_effect=RuntimeError("API error")):
            review = self.reviewer.review_file(fi, p)
        assert review.has_errors is True
        assert len(review.comments) > 0      # AST comments still present
        assert review.overall_score > 0

    def test_tiny_file_skips_llm(self):
        call_count = [0]
        def counting_call(self_inner, msg):
            call_count[0] += 1
            return MOCK_LLM_RESPONSE
        fi = make_file_info("x=1\n")   # only 1 line
        p  = PythonASTParser().parse("f.py", fi.content, "python")
        with patch("backend.reviewer.ai_reviewer.LLMClient.call", counting_call):
            review = self.reviewer.review_file(fi, p)
        assert call_count[0] == 0
        assert review.overall_score == 95

    def test_comments_sorted_by_severity(self):
        content = ("import pickle\n"
                   "def f(d):\n    return pickle.loads(d)\n"
                   "def g():\n    pass\n" * 3)
        fi = make_file_info(content)
        p  = PythonASTParser().parse("f.py", content, "python")
        with self._mock_call():
            review = self.reviewer.review_file(fi, p)
        ranks = [c.severity_rank for c in review.comments]
        assert ranks == sorted(ranks)

    def test_deduplication_keeps_higher_confidence(self):
        # Inject LLM response with same (issue_type, line) as AST smell
        content = "def f(x):\n    eval(x)\n    pass\n"
        fi = make_file_info(content)
        p  = PythonASTParser().parse("f.py", content, "python")
        # find the line where eval appears
        eval_line = next(
            (i+1 for i,l in enumerate(content.splitlines()) if "eval" in l), 2
        )
        mock_resp = json.dumps({
            "comments": [{
                "line_number": eval_line,
                "issue_type": "dangerous_pattern",   # same as AST smell type
                "severity": "Critical",
                "confidence_score": 99,
                "category": "Security",
                "explanation": "eval() is dangerous.",
                "suggested_fix": "Remove eval.",
            }],
            "summary": "Critical.", "overall_score": 10,
        })
        with self._mock_call(mock_resp):
            review = self.reviewer.review_file(fi, p)
        dp = [c for c in review.comments
              if c.issue_type == "dangerous_pattern"
              and c.line_number == eval_line]
        assert len(dp) == 1
        assert dp[0].confidence_score == 99  # LLM wins (higher)

    def test_review_all_files_returns_report(self):
        files = [make_file_info("x=1\n    pass\n    pass\n")]
        parsed = [PythonASTParser().parse(f.relative_path, f.content, f.language)
                  for f in files]
        with self._mock_call():
            report = self.reviewer.review_all_files(files, parsed)
        assert isinstance(report, ReviewReport)
        assert len(report.file_reviews) == 1

    def test_review_all_files_progress_callback(self):
        files = [make_file_info("x=1\n    pass\n    pass\n", path=f"f{i}.py")
                 for i in range(3)]
        parsed = [PythonASTParser().parse(f.relative_path, f.content, f.language)
                  for f in files]
        calls = []
        with self._mock_call():
            self.reviewer.review_all_files(
                files, parsed,
                progress_callback=lambda c,t,n: calls.append(n),
            )
        assert len(calls) == 3

    def test_review_report_provider_set(self):
        files  = [make_file_info("x=1\n    pass\n    pass\n")]
        parsed = [PythonASTParser().parse(f.relative_path, f.content, f.language)
                  for f in files]
        with self._mock_call():
            report = self.reviewer.review_all_files(files, parsed)
        assert report.llm_provider in ("anthropic", "openai")
        assert report.llm_model    != ""
