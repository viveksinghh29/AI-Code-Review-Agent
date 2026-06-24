"""Parser package — AST analysis and code smell detection."""

from backend.parser.ast_parser import (
    PythonASTParser,
    ParsedFile,
    FunctionInfo,
    ClassInfo,
    ImportInfo,
    CodeSmell,
    DANGEROUS_PATTERNS,
    MAX_FUNCTION_LINES,
    MAX_FUNCTION_ARGS,
    MAX_COMPLEXITY,
    MAX_CLASS_METHODS,
    MAX_NESTING_DEPTH,
    MAX_LINE_LENGTH,
)

from backend.parser.code_analyzer import (
    CodeAnalyzer,
    AnalysisReport,
    FileQualityScore,
    AggregateMetrics,
    DuplicatePattern,
    NamingIssue,
    ImportAnomalies,
    QualityScorer,
    CrossFileAnalyzer,
    MetricsBuilder,
)

__all__ = [
    # ast_parser
    "PythonASTParser",
    "ParsedFile",
    "FunctionInfo",
    "ClassInfo",
    "ImportInfo",
    "CodeSmell",
    "DANGEROUS_PATTERNS",
    "MAX_FUNCTION_LINES",
    "MAX_FUNCTION_ARGS",
    "MAX_COMPLEXITY",
    "MAX_CLASS_METHODS",
    "MAX_NESTING_DEPTH",
    "MAX_LINE_LENGTH",
    # code_analyzer
    "CodeAnalyzer",
    "AnalysisReport",
    "FileQualityScore",
    "AggregateMetrics",
    "DuplicatePattern",
    "NamingIssue",
    "ImportAnomalies",
    "QualityScorer",
    "CrossFileAnalyzer",
    "MetricsBuilder",
]
