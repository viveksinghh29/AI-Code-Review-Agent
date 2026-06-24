"""
AST Parsing Engine
==================
Converts raw Python source code into a rich structural model using
Python's built-in `ast` module.

Extracts:
  - Functions  (name, args, complexity, docstring, type hints, async)
  - Classes    (name, methods, bases, docstring, class vars)
  - Imports    (module, names, aliases, wildcard detection)
  - Global variables
  - Module-level docstring
  - __main__ guard presence

Detects code smells:
  - Long functions            (> MAX_FUNCTION_LINES)
  - Too many arguments        (> MAX_FUNCTION_ARGS)
  - High cyclomatic complexity(> MAX_COMPLEXITY)
  - God classes               (> MAX_CLASS_METHODS)
  - Missing docstrings        (public functions / classes)
  - Wildcard imports          (from x import *)
  - Dangerous patterns        (eval, exec, pickle, shell=True …)
  - Long lines                (> MAX_LINE_LENGTH)
  - Deep nesting              (> MAX_NESTING_DEPTH)
  - Mutable default arguments (def f(x=[]) …)
  - Bare except clauses       (except: without type)
  - TODO / FIXME comments

Design: Stateless — parse() takes source text and returns a ParsedFile.
        Thread-safe; every call creates fresh visitor instances.
"""

import ast
import re
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Thresholds
# ─────────────────────────────────────────────────────────────────────────────

MAX_FUNCTION_LINES = 50
MAX_FUNCTION_ARGS  = 7
MAX_COMPLEXITY     = 10
MAX_CLASS_METHODS  = 20
MAX_NESTING_DEPTH  = 4
MAX_LINE_LENGTH    = 120


# ─────────────────────────────────────────────────────────────────────────────
# Dangerous source-level patterns  (regex, description, severity)
# ─────────────────────────────────────────────────────────────────────────────

DANGEROUS_PATTERNS: list[tuple[str, str, str]] = [
    (r"\beval\s*\(",
     "Use of eval() is a security risk — executes arbitrary code",
     "Critical"),

    (r"\bexec\s*\(",
     "Use of exec() is a security risk — executes arbitrary code",
     "Critical"),

    (r"\b__import__\s*\(",
     "Dynamic __import__() bypasses static analysis",
     "High"),

    (r"pickle\.loads?\s*\(",
     "Unsafe pickle deserialization — can execute arbitrary code",
     "Critical"),

    (r"marshal\.loads?\s*\(",
     "Unsafe marshal deserialization",
     "High"),

    (r"subprocess\.[^\n]+shell\s*=\s*True",
     "subprocess with shell=True enables command injection",
     "Critical"),

    (r"os\.system\s*\(",
     "os.system() is unsafe; prefer subprocess.run() with a list",
     "High"),

    (r"os\.popen\s*\(",
     "os.popen() is deprecated and less safe than subprocess",
     "High"),

    (r"hashlib\.(md5|sha1)\s*\(",
     "MD5 / SHA-1 are cryptographically broken; use SHA-256 or higher",
     "High"),

    (r"random\.(random|randint|choice|seed)\s*\(",
     "Use the `secrets` module for cryptographic randomness",
     "Medium"),

    (r"(password|passwd|secret|api_key|token)\s*=\s*[\"'][^\"']{4,}[\"']",
     "Possible hardcoded credential detected",
     "Critical"),

    (r"\.execute\s*\([^)]*\+",
     "Possible SQL injection via string concatenation in execute()",
     "Critical"),

    (r"yaml\.load\s*\([^,)]+\)",
     "yaml.load() without Loader= is unsafe; use yaml.safe_load()",
     "High"),

    (r"tempfile\.mktemp\s*\(",
     "tempfile.mktemp() has a race condition; use mkstemp() instead",
     "Medium"),

    (r"assert\s+",
     "assert is disabled with -O flag; use explicit runtime validation",
     "Low"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FunctionInfo:
    """Structural information about a single function or method."""
    name:           str
    start_line:     int
    end_line:       int
    line_count:     int
    args:           list[str]
    has_docstring:  bool
    has_type_hints: bool       # return annotation OR ≥1 param annotation
    is_async:       bool
    is_method:      bool       # True when defined inside a class body
    decorators:     list[str]
    complexity:     int        # cyclomatic complexity
    max_nesting:    int        # deepest nesting level inside the body
    has_return:     bool       # has at least one non-empty return statement
    raises:         list[str]  # names of exceptions raised

    @property
    def is_too_long(self) -> bool:
        return self.line_count > MAX_FUNCTION_LINES

    @property
    def too_many_args(self) -> bool:
        effective = [a for a in self.args if a not in ("self", "cls")]
        return len(effective) > MAX_FUNCTION_ARGS

    @property
    def is_complex(self) -> bool:
        return self.complexity > MAX_COMPLEXITY


@dataclass
class ClassInfo:
    """Structural information about a class definition."""
    name:            str
    start_line:      int
    end_line:        int
    line_count:      int
    methods:         list[str]
    base_classes:    list[str]
    has_docstring:   bool
    class_variables: list[str]
    is_dataclass:    bool      # decorated with @dataclass
    is_abstract:     bool      # inherits ABC or has abstractmethod

    @property
    def is_god_class(self) -> bool:
        return len(self.methods) > MAX_CLASS_METHODS


@dataclass
class ImportInfo:
    """A single import statement."""
    module:      str
    names:       list[str]
    alias:       Optional[str]
    line_number: int
    is_from:     bool   # True  → "from x import y"
    is_wildcard: bool   # True  → "from x import *"
    is_stdlib:   bool   # best-effort stdlib detection


@dataclass
class CodeSmell:
    """A detected static-analysis issue (produced by the parser, not the LLM)."""
    smell_type:  str
    description: str
    line_number: int
    severity:    str    # Low | Medium | High | Critical
    context:     str = ""

    @property
    def severity_rank(self) -> int:
        return {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}.get(self.severity, 4)


@dataclass
class ParsedFile:
    """
    Complete structural model of one source file.
    Output of PythonASTParser.parse() and primary input to the AI reviewer.
    """
    file_path:  str
    language:   str
    line_count: int

    # Structural elements
    functions:        list[FunctionInfo] = field(default_factory=list)
    classes:          list[ClassInfo]    = field(default_factory=list)
    imports:          list[ImportInfo]   = field(default_factory=list)
    global_variables: list[str]          = field(default_factory=list)

    # Module-level metadata
    module_docstring: Optional[str] = None
    has_main_guard:   bool          = False

    # Quality signals
    code_smells:      list[CodeSmell]           = field(default_factory=list)
    complexity_score: int                        = 0
    todo_comments:    list[tuple[int, str]]      = field(default_factory=list)

    # Error flag
    parse_error: Optional[str] = None

    # ── Convenience properties ─────────────────────────────────────────────

    @property
    def has_errors(self) -> bool:
        return self.parse_error is not None

    @property
    def critical_smells(self) -> list[CodeSmell]:
        return [s for s in self.code_smells if s.severity == "Critical"]

    @property
    def high_smells(self) -> list[CodeSmell]:
        return [s for s in self.code_smells if s.severity == "High"]

    @property
    def smell_summary(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for s in self.code_smells:
            out[s.smell_type] = out.get(s.smell_type, 0) + 1
        return out

    @property
    def doc_coverage(self) -> float:
        """Fraction of public functions + classes that have docstrings (0-1)."""
        pub_fn  = [f for f in self.functions if not f.name.startswith("_")]
        pub_cls = [c for c in self.classes   if not c.name.startswith("_")]
        total   = len(pub_fn) + len(pub_cls)
        if total == 0:
            return 1.0
        docs = sum(1 for f in pub_fn if f.has_docstring) + \
               sum(1 for c in pub_cls if c.has_docstring)
        return round(docs / total, 3)

    @property
    def type_hint_coverage(self) -> float:
        if not self.functions:
            return 1.0
        typed = sum(1 for f in self.functions if f.has_type_hints)
        return round(typed / len(self.functions), 3)

    def to_context_dict(self) -> dict:
        """Compact dict injected into LLM prompt as structured context."""
        return {
            "file":               self.file_path,
            "language":           self.language,
            "lines":              self.line_count,
            "complexity_score":   self.complexity_score,
            "doc_coverage":       self.doc_coverage,
            "type_hint_coverage": self.type_hint_coverage,
            "functions": [
                {
                    "name":           f.name,
                    "lines":          f.line_count,
                    "complexity":     f.complexity,
                    "args":           len(f.args),
                    "has_docstring":  f.has_docstring,
                    "has_type_hints": f.has_type_hints,
                }
                for f in self.functions
            ],
            "classes": [
                {
                    "name":         c.name,
                    "methods":      len(c.methods),
                    "has_docstring": c.has_docstring,
                }
                for c in self.classes
            ],
            "smells": [
                {
                    "type":     s.smell_type,
                    "line":     s.line_number,
                    "severity": s.severity,
                }
                for s in self.code_smells
            ],
        }


# ─────────────────────────────────────────────────────────────────────────────
# AST Visitors
# ─────────────────────────────────────────────────────────────────────────────

class _ComplexityVisitor(ast.NodeVisitor):
    """
    Counts cyclomatic complexity for a single function.
    Complexity = 1 + number of decision/branch points.
    """

    def __init__(self):
        self.complexity = 1

    def visit_If(self, node):
        self.complexity += 1
        self.generic_visit(node)

    def visit_While(self, node):
        self.complexity += 1
        self.generic_visit(node)

    def visit_For(self, node):
        self.complexity += 1
        self.generic_visit(node)

    def visit_AsyncFor(self, node):
        self.complexity += 1
        self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        self.complexity += 1
        self.generic_visit(node)

    def visit_With(self, node):
        self.complexity += 1
        self.generic_visit(node)

    def visit_AsyncWith(self, node):
        self.complexity += 1
        self.generic_visit(node)

    def visit_Assert(self, node):
        self.complexity += 1
        self.generic_visit(node)

    def visit_comprehension(self, node):
        self.complexity += 1
        self.generic_visit(node)

    def visit_BoolOp(self, node):
        self.complexity += len(node.values) - 1
        self.generic_visit(node)

    def visit_IfExp(self, node):
        # ternary a if cond else b
        self.complexity += 1
        self.generic_visit(node)


class _NestingVisitor(ast.NodeVisitor):
    """Measures the maximum nesting depth inside a function body."""

    def __init__(self):
        self.max_depth = 0
        self._depth    = 0

    def _enter(self, node):
        self._depth   += 1
        self.max_depth = max(self.max_depth, self._depth)
        self.generic_visit(node)
        self._depth   -= 1

    visit_If        = _enter  # type: ignore[assignment]
    visit_For       = _enter  # type: ignore[assignment]
    visit_While     = _enter  # type: ignore[assignment]
    visit_With      = _enter  # type: ignore[assignment]
    visit_Try       = _enter  # type: ignore[assignment]
    visit_AsyncFor  = _enter  # type: ignore[assignment]
    visit_AsyncWith = _enter  # type: ignore[assignment]


# ─────────────────────────────────────────────────────────────────────────────
# Known stdlib top-level names  (best-effort — Python 3.11+)
# ─────────────────────────────────────────────────────────────────────────────

_STDLIB_TOP: set[str] = {
    "abc", "ast", "asyncio", "base64", "binascii", "builtins",
    "calendar", "cgi", "cgitb", "chunk", "cmath", "cmd", "code",
    "codecs", "codeop", "collections", "colorsys", "compileall",
    "concurrent", "configparser", "contextlib", "contextvars",
    "copy", "copyreg", "csv", "ctypes", "curses", "dataclasses",
    "datetime", "dbm", "decimal", "difflib", "dis", "doctest",
    "email", "encodings", "enum", "errno", "faulthandler",
    "fcntl", "filecmp", "fileinput", "fnmatch", "fractions",
    "ftplib", "functools", "gc", "getopt", "getpass", "gettext",
    "glob", "grp", "gzip", "hashlib", "heapq", "hmac", "html",
    "http", "idlelib", "imaplib", "importlib", "inspect", "io",
    "ipaddress", "itertools", "json", "keyword", "lib2to3",
    "linecache", "locale", "logging", "lzma", "mailbox", "math",
    "mimetypes", "mmap", "modulefinder", "multiprocessing",
    "netrc", "nis", "nntplib", "numbers", "operator", "optparse",
    "os", "ossaudiodev", "pathlib", "pdb", "pickle", "pickletools",
    "pipes", "pkgutil", "platform", "plistlib", "poplib", "posix",
    "pprint", "profile", "pstats", "pty", "pwd", "py_compile",
    "pyclbr", "pydoc", "queue", "quopri", "random", "re", "readline",
    "reprlib", "resource", "rlcompleter", "runpy", "sched", "secrets",
    "select", "selectors", "shelve", "shlex", "shutil", "signal",
    "site", "smtpd", "smtplib", "sndhdr", "socket", "socketserver",
    "spwd", "sqlite3", "sre_compile", "sre_constants", "sre_parse",
    "ssl", "stat", "statistics", "string", "stringprep", "struct",
    "subprocess", "sunau", "symtable", "sys", "sysconfig", "syslog",
    "tabnanny", "tarfile", "telnetlib", "tempfile", "termios", "test",
    "textwrap", "threading", "time", "timeit", "tkinter", "token",
    "tokenize", "tomllib", "trace", "traceback", "tracemalloc",
    "tty", "turtle", "turtledemo", "types", "typing", "unicodedata",
    "unittest", "urllib", "uu", "uuid", "venv", "warnings", "wave",
    "weakref", "webbrowser", "wsgiref", "xdrlib", "xml", "xmlrpc",
    "zipapp", "zipfile", "zipimport", "zlib", "zoneinfo",
}


def _is_stdlib(module: str) -> bool:
    return module.split(".")[0] in _STDLIB_TOP


# ─────────────────────────────────────────────────────────────────────────────
# Main Parser
# ─────────────────────────────────────────────────────────────────────────────

class PythonASTParser:
    """
    Parses a Python source file into a rich ParsedFile model.

    Usage:
        parser = PythonASTParser()
        result = parser.parse("src/mymodule.py", source_code, "python")
    """

    # ── Public entry point ────────────────────────────────────────────────────

    def parse(self, file_path: str, content: str, language: str) -> ParsedFile:
        """
        Parse source code and return a fully-populated ParsedFile.

        For non-Python files, only universal text analysis is run
        (long-line detection, TODO scanning).

        Args:
            file_path : Relative path used as display identifier.
            content   : Raw source text (UTF-8 string).
            language  : Language label from SUPPORTED_EXTENSIONS.

        Returns:
            ParsedFile instance.
        """
        parsed = ParsedFile(
            file_path  = file_path,
            language   = language,
            line_count = content.count("\n") + 1,
        )

        lines = content.splitlines()

        # ── Text analysis — runs for ALL languages ────────────────────────
        self._scan_long_lines(parsed, lines)
        self._scan_todo_comments(parsed, lines)

        if language != "python":
            return parsed   # non-Python: text analysis only

        # ── Python AST analysis ───────────────────────────────────────────
        try:
            tree = ast.parse(content, filename=file_path)
        except SyntaxError as exc:
            parsed.parse_error = f"SyntaxError at line {exc.lineno}: {exc.msg}"
            # Still run dangerous-pattern scan on broken source
            self._scan_dangerous_patterns(parsed, lines)
            return parsed

        self._extract_module_docstring(parsed, tree)
        self._extract_imports(parsed, tree)
        self._extract_functions(parsed, tree)
        self._extract_classes(parsed, tree)
        self._extract_global_variables(parsed, tree)
        self._check_main_guard(parsed, tree)
        self._scan_dangerous_patterns(parsed, lines)
        self._detect_mutable_defaults(parsed, tree)
        self._detect_bare_excepts(parsed, tree)
        self._compute_complexity_score(parsed)

        return parsed

    # ── Extraction helpers ────────────────────────────────────────────────────

    def _extract_module_docstring(
        self, parsed: ParsedFile, tree: ast.Module
    ) -> None:
        ds = ast.get_docstring(tree)
        if ds:
            parsed.module_docstring = ds[:500]

    # ─────────────────────────────────────────────────────────────────────────

    def _extract_imports(self, parsed: ParsedFile, tree: ast.AST) -> None:
        """Collect all import statements; flag wildcard imports."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    parsed.imports.append(ImportInfo(
                        module      = alias.name,
                        names       = [alias.name],
                        alias       = alias.asname,
                        line_number = node.lineno,
                        is_from     = False,
                        is_wildcard = False,
                        is_stdlib   = _is_stdlib(alias.name),
                    ))

            elif isinstance(node, ast.ImportFrom):
                module      = node.module or ""
                names       = [a.name for a in node.names]
                is_wildcard = "*" in names

                parsed.imports.append(ImportInfo(
                    module      = module,
                    names       = names,
                    alias       = None,
                    line_number = node.lineno,
                    is_from     = True,
                    is_wildcard = is_wildcard,
                    is_stdlib   = _is_stdlib(module),
                ))

                if is_wildcard:
                    parsed.code_smells.append(CodeSmell(
                        smell_type  = "wildcard_import",
                        description = (
                            f"'from {module} import *' pollutes the namespace "
                            "and makes it hard to trace where names originate."
                        ),
                        line_number = node.lineno,
                        severity    = "Medium",
                    ))

    # ─────────────────────────────────────────────────────────────────────────

    def _extract_functions(self, parsed: ParsedFile, tree: ast.AST) -> None:
        """
        Walk the full AST and extract every function/method definition.
        Generates code smells for: long body, too many args,
        high complexity, deep nesting, missing docstring.
        """
        # Pre-compute which function nodes live inside a class body
        class_child_ids: set[int] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        class_child_ids.add(id(child))

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            end_line   = getattr(node, "end_lineno", node.lineno)
            line_count = end_line - node.lineno + 1
            args       = self._get_arg_names(node)

            # Decorator names
            decorators: list[str] = []
            for d in node.decorator_list:
                if isinstance(d, ast.Name):
                    decorators.append(d.id)
                elif isinstance(d, ast.Attribute):
                    decorators.append(
                        f"{d.value.id}.{d.attr}"
                        if isinstance(d.value, ast.Name) else d.attr
                    )
                elif isinstance(d, ast.Call):
                    if isinstance(d.func, ast.Name):
                        decorators.append(d.func.id)

            # Type hints: return annotation OR any param annotation
            has_type_hints = (
                node.returns is not None
                or any(
                    a.annotation is not None
                    for a in (
                        node.args.args
                        + node.args.posonlyargs
                        + node.args.kwonlyargs
                    )
                )
            )

            # Complexity and nesting
            cv = _ComplexityVisitor(); cv.visit(node)
            nv = _NestingVisitor();    nv.visit(node)

            # Return and raise detection
            has_return = any(
                isinstance(n, ast.Return) and n.value is not None
                for n in ast.walk(node)
            )
            raises = list({
                (
                    n.exc.id        if isinstance(n.exc, ast.Name)
                    else n.exc.attr if isinstance(n.exc, ast.Attribute)
                    else ""
                )
                for n in ast.walk(node)
                if isinstance(n, ast.Raise) and n.exc is not None
            } - {""})

            func_info = FunctionInfo(
                name           = node.name,
                start_line     = node.lineno,
                end_line       = end_line,
                line_count     = line_count,
                args           = args,
                has_docstring  = ast.get_docstring(node) is not None,
                has_type_hints = has_type_hints,
                is_async       = isinstance(node, ast.AsyncFunctionDef),
                is_method      = id(node) in class_child_ids,
                decorators     = decorators,
                complexity     = cv.complexity,
                max_nesting    = nv.max_depth,
                has_return     = has_return,
                raises         = raises,
            )
            parsed.functions.append(func_info)

            # ── Smell detection ───────────────────────────────────────────

            if func_info.is_too_long:
                parsed.code_smells.append(CodeSmell(
                    smell_type  = "long_function",
                    description = (
                        f"Function '{node.name}' is {line_count} lines "
                        f"(max: {MAX_FUNCTION_LINES}). "
                        "Split into smaller, single-purpose functions."
                    ),
                    line_number = node.lineno,
                    severity    = "High" if line_count > 100 else "Medium",
                ))

            effective_args = [a for a in args if a not in ("self", "cls")]
            if len(effective_args) > MAX_FUNCTION_ARGS:
                parsed.code_smells.append(CodeSmell(
                    smell_type  = "too_many_arguments",
                    description = (
                        f"Function '{node.name}' has {len(effective_args)} "
                        f"parameters (max: {MAX_FUNCTION_ARGS}). "
                        "Group related parameters into a dataclass or config object."
                    ),
                    line_number = node.lineno,
                    severity    = "Medium",
                ))

            if func_info.is_complex:
                parsed.code_smells.append(CodeSmell(
                    smell_type  = "high_complexity",
                    description = (
                        f"Function '{node.name}' has cyclomatic complexity "
                        f"{cv.complexity} (max: {MAX_COMPLEXITY}). "
                        "Reduce branching or extract sub-functions."
                    ),
                    line_number = node.lineno,
                    severity    = "High" if cv.complexity > 15 else "Medium",
                ))

            if nv.max_depth > MAX_NESTING_DEPTH:
                parsed.code_smells.append(CodeSmell(
                    smell_type  = "deep_nesting",
                    description = (
                        f"Function '{node.name}' reaches nesting depth "
                        f"{nv.max_depth} (max: {MAX_NESTING_DEPTH}). "
                        "Use early returns or extract helper functions."
                    ),
                    line_number = node.lineno,
                    severity    = "Medium",
                ))

            # Missing docstring on public functions
            if (
                not node.name.startswith("_")
                and ast.get_docstring(node) is None
            ):
                parsed.code_smells.append(CodeSmell(
                    smell_type  = "missing_docstring",
                    description = (
                        f"Public function '{node.name}' has no docstring. "
                        "Document purpose, parameters, and return value."
                    ),
                    line_number = node.lineno,
                    severity    = "Low",
                ))

    # ─────────────────────────────────────────────────────────────────────────

    def _extract_classes(self, parsed: ParsedFile, tree: ast.AST) -> None:
        """
        Extract all class definitions.
        Detects: god classes, missing class docstrings.
        """
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue

            end_line = getattr(node, "end_lineno", node.lineno)

            # Direct-child methods only (not nested class methods)
            methods = [
                child.name
                for child in ast.iter_child_nodes(node)
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]

            # Base class names
            bases: list[str] = []
            for base in node.bases:
                if isinstance(base, ast.Name):
                    bases.append(base.id)
                elif isinstance(base, ast.Attribute):
                    bases.append(
                        f"{base.value.id}.{base.attr}"
                        if isinstance(base.value, ast.Name) else base.attr
                    )

            # Class-level variable names
            class_vars: list[str] = []
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.Assign):
                    for t in child.targets:
                        if isinstance(t, ast.Name):
                            class_vars.append(t.id)
                elif isinstance(child, ast.AnnAssign):
                    if isinstance(child.target, ast.Name):
                        class_vars.append(child.target.id)

            # Decorator flags
            dec_names = []
            for d in node.decorator_list:
                if isinstance(d, ast.Name):
                    dec_names.append(d.id)
                elif isinstance(d, ast.Attribute):
                    dec_names.append(d.attr)
                elif isinstance(d, ast.Call):
                    if isinstance(d.func, ast.Name):
                        dec_names.append(d.func.id)

            is_dataclass = "dataclass" in dec_names
            is_abstract  = (
                any(b in ("ABC", "ABCMeta") for b in bases)
                or any(
                    isinstance(d, ast.Name) and d.id == "abstractmethod"
                    for m in ast.walk(node)
                    for d in getattr(m, "decorator_list", [])
                )
            )

            cls_info = ClassInfo(
                name            = node.name,
                start_line      = node.lineno,
                end_line        = end_line,
                line_count      = end_line - node.lineno + 1,
                methods         = methods,
                base_classes    = bases,
                has_docstring   = ast.get_docstring(node) is not None,
                class_variables = class_vars[:30],
                is_dataclass    = is_dataclass,
                is_abstract     = is_abstract,
            )
            parsed.classes.append(cls_info)

            if cls_info.is_god_class:
                parsed.code_smells.append(CodeSmell(
                    smell_type  = "god_class",
                    description = (
                        f"Class '{node.name}' has {len(methods)} methods "
                        f"(max: {MAX_CLASS_METHODS}). "
                        "Refactor using composition or split into focused classes."
                    ),
                    line_number = node.lineno,
                    severity    = "High",
                ))

            if not node.name.startswith("_") and ast.get_docstring(node) is None:
                parsed.code_smells.append(CodeSmell(
                    smell_type  = "missing_docstring",
                    description = (
                        f"Class '{node.name}' has no docstring. "
                        "Describe the purpose and usage of the class."
                    ),
                    line_number = node.lineno,
                    severity    = "Low",
                ))

    # ─────────────────────────────────────────────────────────────────────────

    def _extract_global_variables(
        self, parsed: ParsedFile, tree: ast.AST
    ) -> None:
        """Collect module-level variable names (excludes UPPER_CASE constants)."""
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and not t.id.isupper():
                        parsed.global_variables.append(t.id)
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name) and not node.target.id.isupper():
                    parsed.global_variables.append(node.target.id)

    def _check_main_guard(self, parsed: ParsedFile, tree: ast.AST) -> None:
        """Detect `if __name__ == '__main__':` guard."""
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                test = node.test
                if (
                    isinstance(test, ast.Compare)
                    and isinstance(test.left, ast.Name)
                    and test.left.id == "__name__"
                    and len(test.ops) == 1
                    and isinstance(test.ops[0], ast.Eq)
                ):
                    parsed.has_main_guard = True
                    return

    # ─────────────────────────────────────────────────────────────────────────
    # Smell detectors
    # ─────────────────────────────────────────────────────────────────────────

    def _detect_mutable_defaults(
        self, parsed: ParsedFile, tree: ast.AST
    ) -> None:
        """
        Flag mutable default arguments (def f(x=[], y={})).
        Classic Python gotcha — the default object is shared across all calls.
        """
        mutable_types = (ast.List, ast.Dict, ast.Set)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for default in node.args.defaults + node.args.kw_defaults:
                if default is not None and isinstance(default, mutable_types):
                    kind = type(default).__name__
                    parsed.code_smells.append(CodeSmell(
                        smell_type  = "mutable_default_argument",
                        description = (
                            f"Function '{node.name}' uses a mutable default "
                            f"argument ({kind}). "
                            "Use `None` and assign the default inside the body."
                        ),
                        line_number = node.lineno,
                        severity    = "High",
                    ))

    def _detect_bare_excepts(self, parsed: ParsedFile, tree: ast.AST) -> None:
        """
        Flag bare `except:` without specifying an exception type.
        These swallow KeyboardInterrupt, SystemExit, and all other exceptions.
        """
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                parsed.code_smells.append(CodeSmell(
                    smell_type  = "bare_except",
                    description = (
                        "Bare 'except:' catches ALL exceptions including "
                        "KeyboardInterrupt and SystemExit. "
                        "Catch specific exception types instead."
                    ),
                    line_number = node.lineno,
                    severity    = "High",
                ))

    def _scan_dangerous_patterns(
        self, parsed: ParsedFile, lines: list[str]
    ) -> None:
        """Regex scan for security-sensitive patterns on raw source lines."""
        for line_no, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue   # skip pure comment lines
            for pattern, description, severity in DANGEROUS_PATTERNS:
                if re.search(pattern, line, re.IGNORECASE):
                    parsed.code_smells.append(CodeSmell(
                        smell_type  = "dangerous_pattern",
                        description = description,
                        line_number = line_no,
                        severity    = severity,
                        context     = stripped[:120],
                    ))

    def _scan_long_lines(self, parsed: ParsedFile, lines: list[str]) -> None:
        """Flag lines exceeding MAX_LINE_LENGTH (universal — all languages)."""
        for i, line in enumerate(lines, 1):
            if len(line) > MAX_LINE_LENGTH:
                parsed.code_smells.append(CodeSmell(
                    smell_type  = "long_line",
                    description = (
                        f"Line {i} is {len(line)} characters "
                        f"(max recommended: {MAX_LINE_LENGTH})."
                    ),
                    line_number = i,
                    severity    = "Low",
                    context     = line[:80] + "…",
                ))

    def _scan_todo_comments(
        self, parsed: ParsedFile, lines: list[str]
    ) -> None:
        """Collect TODO / FIXME / HACK / BUG comment markers."""
        pattern = re.compile(
            r"#\s*(TODO|FIXME|HACK|XXX|BUG|NOTE)\b.*", re.IGNORECASE
        )
        for i, line in enumerate(lines, 1):
            m = pattern.search(line)
            if m:
                parsed.todo_comments.append((i, m.group(0).strip()))

    # ─────────────────────────────────────────────────────────────────────────

    def _compute_complexity_score(self, parsed: ParsedFile) -> None:
        """Aggregate complexity: sum of all function complexities."""
        parsed.complexity_score = sum(f.complexity for f in parsed.functions)

    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _get_arg_names(
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> list[str]:
        """Extract every argument name from a function definition."""
        args: list[str] = []
        args += [a.arg for a in node.args.posonlyargs]
        args += [a.arg for a in node.args.args]
        if node.args.vararg:
            args.append(f"*{node.args.vararg.arg}")
        args += [a.arg for a in node.args.kwonlyargs]
        if node.args.kwarg:
            args.append(f"**{node.args.kwarg.arg}")
        return args
