"""
Unit Tests — Code Analyzer (Phase 4)
Tests: QualityScorer, CrossFileAnalyzer, MetricsBuilder, CodeAnalyzer
"""

import pytest

from backend.ingestion.repo_ingestion import FileInfo
from backend.parser.ast_parser import PythonASTParser
from backend.parser.code_analyzer import (
    AnalysisReport,
    AggregateMetrics,
    CodeAnalyzer,
    CrossFileAnalyzer,
    DuplicatePattern,
    FileQualityScore,
    ImportAnomalies,
    MetricsBuilder,
    NamingIssue,
    QualityScorer,
    _score_to_grade,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_file(path: str, content: str, language: str = "python") -> FileInfo:
    return FileInfo(
        path=path, relative_path=path, language=language,
        size_bytes=len(content.encode()), content=content,
        line_count=content.count("\n") + 1,
    )


def parse(content: str, path: str = "f.py"):
    return PythonASTParser().parse(path, content, "python")


# ─────────────────────────────────────────────────────────────────────────────
# _score_to_grade
# ─────────────────────────────────────────────────────────────────────────────

class TestScoreToGrade:
    def test_a(self): assert _score_to_grade(95) == "A"
    def test_b(self): assert _score_to_grade(85) == "B"
    def test_c(self): assert _score_to_grade(75) == "C"
    def test_d(self): assert _score_to_grade(65) == "D"
    def test_f(self): assert _score_to_grade(50) == "F"
    def test_boundaries(self):
        assert _score_to_grade(90) == "A"
        assert _score_to_grade(89) == "B"
        assert _score_to_grade(80) == "B"
        assert _score_to_grade(79) == "C"


# ─────────────────────────────────────────────────────────────────────────────
# QualityScorer
# ─────────────────────────────────────────────────────────────────────────────

class TestQualityScorer:
    def setup_method(self):
        self.scorer = QualityScorer()

    def test_perfect_file_scores_high(self):
        src = ('"""Module."""\ndef add(x: int, y: int) -> int:\n'
               '    """Add."""\n    return x + y\n')
        p     = parse(src)
        score = self.scorer.score(p)
        assert score.overall >= 80
        assert score.grade in ("A", "B")

    def test_dangerous_code_lowers_security(self):
        src = "def f(x):\n    eval(x)\n    exec(x)\n"
        p     = parse(src)
        score = self.scorer.score(p)
        assert score.security < 60

    def test_undocumented_code_lowers_docs(self):
        src = "def a(): pass\ndef b(): pass\ndef c(): pass\n"
        p     = parse(src)
        score = self.scorer.score(p)
        assert score.documentation < 80

    def test_grade_set(self):
        src = "x = 1\n"
        p     = parse(src)
        score = self.scorer.score(p)
        assert score.grade in ("A", "B", "C", "D", "F")

    def test_score_bounded_0_100(self):
        # Worst possible file
        src = ("def f(a,b,c,d,e,f,g,h):\n"
               "    eval(a)\n    exec(b)\n    pickle_data = 'secret'\n"
               + "    if a:\n        if b:\n" * 10)
        p     = parse(src)
        score = self.scorer.score(p)
        assert 0 <= score.overall <= 100
        assert 0 <= score.security <= 100

    def test_file_path_set_on_score(self):
        p     = parse("x = 1\n", path="my/module.py")
        score = self.scorer.score(p)
        assert score.file_path == "my/module.py"


# ─────────────────────────────────────────────────────────────────────────────
# CrossFileAnalyzer
# ─────────────────────────────────────────────────────────────────────────────

class TestCrossFileAnalyzer:
    def setup_method(self):
        self.cross = CrossFileAnalyzer()

    def test_duplicate_function_detection(self):
        pa = parse("def process(data): return data\n", "a.py")
        pb = parse("def process(items): return items\n", "b.py")
        files = [make_file("a.py", ""), make_file("b.py", "")]
        dups, _, _ = self.cross.analyze([pa, pb], files)
        assert any(d.name == "process" for d in dups)

    def test_no_duplicate_for_unique_names(self):
        pa = parse("def foo(): pass\n", "a.py")
        pb = parse("def bar(): pass\n", "b.py")
        files = [make_file("a.py", ""), make_file("b.py", "")]
        dups, _, _ = self.cross.analyze([pa, pb], files)
        names = [d.name for d in dups]
        assert "foo" not in names
        assert "bar" not in names

    def test_private_names_excluded_from_duplicates(self):
        pa = parse("def _helper(): pass\n", "a.py")
        pb = parse("def _helper(): pass\n", "b.py")
        files = [make_file("a.py", ""), make_file("b.py", "")]
        dups, _, _ = self.cross.analyze([pa, pb], files)
        assert not any(d.name == "_helper" for d in dups)

    def test_naming_violation_camel_case_function(self):
        src = "def myBadFunction(): pass\n"
        p   = parse(src, "f.py")
        files = [make_file("f.py", src)]
        _, issues, _ = self.cross.analyze([p], files)
        names = [i.name for i in issues]
        assert "myBadFunction" in names

    def test_naming_ok_snake_case(self):
        src = "def good_function(): pass\n"
        p   = parse(src, "f.py")
        files = [make_file("f.py", src)]
        _, issues, _ = self.cross.analyze([p], files)
        names = [i.name for i in issues]
        assert "good_function" not in names

    def test_class_naming_violation(self):
        src = "class myBadClass: pass\n"
        p   = parse(src, "f.py")
        files = [make_file("f.py", src)]
        _, issues, _ = self.cross.analyze([p], files)
        names = [i.name for i in issues]
        assert "myBadClass" in names

    def test_class_naming_ok_pascal(self):
        src = "class GoodClass: pass\n"
        p   = parse(src, "f.py")
        files = [make_file("f.py", src)]
        _, issues, _ = self.cross.analyze([p], files)
        names = [i.name for i in issues]
        assert "GoodClass" not in names

    def test_import_anomalies_wildcard(self):
        src = "from os import *\n"
        p   = parse(src, "w.py")
        files = [make_file("w.py", src)]
        _, _, imp = self.cross.analyze([p], files)
        assert "w.py" in imp.wildcard_files

    def test_import_anomalies_heavy_importer(self):
        src = "\n".join(f"import lib{i}" for i in range(15)) + "\n"
        p   = parse(src, "heavy.py")
        files = [make_file("heavy.py", src)]
        _, _, imp = self.cross.analyze([p], files)
        assert imp.heavy_importers[0][0] == "heavy.py"
        assert imp.heavy_importers[0][1] == 15

    def test_import_counts(self):
        src = "import os\nimport sys\nfrom pathlib import Path\n"
        p   = parse(src, "f.py")
        files = [make_file("f.py", src)]
        _, _, imp = self.cross.analyze([p], files)
        assert imp.total_imports == 3
        assert imp.stdlib_count  == 3


# ─────────────────────────────────────────────────────────────────────────────
# CodeAnalyzer
# ─────────────────────────────────────────────────────────────────────────────

class TestCodeAnalyzer:
    def setup_method(self):
        self.analyzer = CodeAnalyzer()

    def test_returns_analysis_report(self):
        files = [make_file("a.py", "x = 1\n")]
        report = self.analyzer.analyze_all(files)
        assert isinstance(report, AnalysisReport)

    def test_empty_file_list(self):
        report = self.analyzer.analyze_all([])
        assert report.aggregate.total_files == 0

    def test_progress_callback_called(self):
        files = [make_file("a.py", "x=1\n"), make_file("b.py", "y=2\n")]
        calls = []
        self.analyzer.analyze_all(files, progress_callback=lambda c,t,n: calls.append(n))
        assert len(calls) == 2

    def test_aggregate_totals(self):
        files = [
            make_file("a.py", "def f(): pass\ndef g(): pass\n"),
            make_file("b.py", "class C: pass\n"),
        ]
        report = self.analyzer.analyze_all(files)
        assert report.aggregate.total_files    == 2
        assert report.aggregate.total_functions >= 2
        assert report.aggregate.total_classes  >= 1

    def test_syntax_error_counted(self):
        files = [make_file("bad.py", "def bad(\n  broken")]
        report = self.analyzer.analyze_all(files)
        assert report.aggregate.files_with_errors == 1
        assert len(report.analysis_errors)         == 1

    def test_quality_scores_generated(self):
        files = [make_file("a.py", "x = 1\n")]
        report = self.analyzer.analyze_all(files)
        assert len(report.quality_scores) == 1
        assert isinstance(report.quality_scores[0], FileQualityScore)

    def test_get_score_lookup(self):
        files = [make_file("a.py", "x = 1\n")]
        report = self.analyzer.analyze_all(files)
        assert report.get_score("a.py")    is not None
        assert report.get_score("none.py") is None

    def test_get_parsed_lookup(self):
        files = [make_file("a.py", "x = 1\n")]
        report = self.analyzer.analyze_all(files)
        assert report.get_parsed("a.py")    is not None
        assert report.get_parsed("none.py") is None

    def test_worst_files_sorted(self):
        files = [
            make_file("good.py", '"""M."""\ndef f(x: int) -> int:\n    """F."""\n    return x\n'),
            make_file("bad.py",  "def f(x):\n    eval(x)\n    exec(x)\n"),
        ]
        report = self.analyzer.analyze_all(files)
        worst  = report.worst_files(2)
        assert worst[0].overall <= worst[1].overall

    def test_all_smells_sorted_by_severity(self):
        files = [make_file("a.py",
                           "def f(x):\n    eval(x)\n"
                           "    x = 1\n" * 60 + "\n")]
        report = self.analyzer.analyze_all(files)
        smells = report.all_smells()
        ranks  = [s[1].severity_rank for s in smells]
        assert ranks == sorted(ranks)

    def test_to_summary_dict(self):
        files  = [make_file("a.py", "x = 1\n")]
        report = self.analyzer.analyze_all(files)
        d      = report.to_summary_dict()
        assert "aggregate"   in d
        assert "file_scores" in d
        assert d["aggregate"]["total_files"] == 1

    def test_language_breakdown(self):
        files = [
            make_file("a.py",  "x=1\n", "python"),
            make_file("b.py",  "y=2\n", "python"),
            make_file("c.js",  "z=3\n", "javascript"),
        ]
        report = self.analyzer.analyze_all(files)
        lb = report.aggregate.language_breakdown
        assert lb.get("python", 0)     == 2
        assert lb.get("javascript", 0) == 1
