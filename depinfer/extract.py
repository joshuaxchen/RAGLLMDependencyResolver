"""Static extraction of imported top-level modules from a Python repository.

Replaces the string-splitting approach in the original ``rag_analyzer.py``,
which lost packages on ``import a, b`` and had no directory skip list.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import tree_sitter_python
from tree_sitter import Language, Parser

PY_LANGUAGE = Language(tree_sitter_python.language())

# Capture the module name node directly rather than slicing the whole
# statement. Relative imports (`from . import x`, `from .mod import y`) bind
# module_name to a `relative_import` node, which these patterns do not match,
# so they are skipped without special-casing.
_IMPORT_QUERY = PY_LANGUAGE.query(
    """
    (import_statement name: (dotted_name) @mod)
    (import_statement name: (aliased_import name: (dotted_name) @mod))
    (import_from_statement module_name: (dotted_name) @mod)
    """
)

# Directories that never contain first-party source worth scanning.
SKIP_DIRS = frozenset(
    {
        ".venv", "venv", ".env", "env",
        "__pycache__", ".git", ".tox", ".nox",
        "node_modules", "site-packages", "vendor", "third_party",
        "build", "dist", ".eggs", ".mypy_cache", ".pytest_cache",
    }
)

# Path segments that mark a file as test/dev rather than runtime code.
TEST_MARKERS = frozenset({"test", "tests", "testing", "conftest", "docs", "doc", "examples", "example", "benchmarks"})

PYTHON_MARKERS = ("pyproject.toml", "setup.py", "setup.cfg", "requirements.txt", "Pipfile")

# `sys.stdlib_module_names` describes the *running* interpreter (3.10 here),
# but target repositories span many versions. Without these, `tomllib` and
# friends are treated as third-party and looked up on PyPI.
EXTRA_STDLIB = frozenset(
    {
        "tomllib",        # 3.11+
        "zoneinfo",       # 3.9+
        "graphlib",       # 3.9+
        "_pytest",        # pytest internals, never a distribution
        "__pypy__",       # PyPy builtin
        # Python 2 leftovers that resolve to unrelated PyPI squatters
        "urlparse", "urllib2", "StringIO", "cStringIO", "ConfigParser",
        "Queue", "Cookie", "cPickle", "md5", "commands", "httplib",
    }
)

STDLIB = frozenset(sys.stdlib_module_names) | EXTRA_STDLIB


@dataclass
class ImportScan:
    """Imported top-level modules, split by where they were found."""

    runtime: set[str] = field(default_factory=set)
    test_only: set[str] = field(default_factory=set)
    # module -> files it was seen in, kept for error analysis
    provenance: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))

    @property
    def all_modules(self) -> set[str]:
        return self.runtime | self.test_only


def _is_test_path(rel: Path) -> bool:
    parts = {p.lower().removesuffix(".py") for p in rel.parts}
    return bool(parts & TEST_MARKERS) or rel.name.startswith("test_")


def iter_python_files(repo_path: Path):
    """Yield .py files under repo_path, skipping vendored/build directories."""
    for path in repo_path.rglob("*.py"):
        if not path.is_file():
            continue
        try:
            rel = path.relative_to(repo_path)
        except ValueError:
            continue
        if SKIP_DIRS & set(rel.parts):
            continue
        yield path, rel


def find_first_party(repo_path: Path) -> set[str]:
    """Top-level module names defined by the repository itself.

    Without this, analysing e.g. Zuehlke_ConfZ tries to resolve `confz` from
    PyPI as though it were a third-party dependency.
    """
    names: set[str] = set()
    roots = [repo_path]
    src = repo_path / "src"
    if src.is_dir():
        roots.append(src)

    for root in roots:
        if not root.is_dir():
            continue
        for child in root.iterdir():
            if child.name in SKIP_DIRS or child.name.startswith("."):
                continue
            if child.is_dir() and (child / "__init__.py").exists():
                names.add(child.name)
            elif child.suffix == ".py" and child.stem != "setup":
                names.add(child.stem)

    # Test packages are first-party too, and are a common false positive.
    for path, rel in iter_python_files(repo_path):
        if rel.parts and _is_test_path(rel):
            names.add(rel.parts[0].removesuffix(".py"))
    return names


def extract_imports(repo_path: Path) -> ImportScan:
    """Extract imported top-level module names, excluding stdlib and first-party."""
    parser = Parser(PY_LANGUAGE)
    first_party = find_first_party(repo_path)
    scan = ImportScan()

    for path, rel in iter_python_files(repo_path):
        try:
            data = path.read_bytes()
            tree = parser.parse(data)
        except Exception as exc:  # unreadable/undecodable file
            print(f"  ! skipping {rel}: {exc}", file=sys.stderr)
            continue

        captures = _IMPORT_QUERY.captures(tree.root_node)
        for nodes in captures.values():
            for node in nodes:
                try:
                    dotted = data[node.start_byte : node.end_byte].decode("utf-8")
                except UnicodeDecodeError:
                    continue
                top = dotted.split(".")[0].strip()
                if not top or top in STDLIB or top in first_party:
                    continue
                scan.provenance[top].add(str(rel))
                if _is_test_path(rel):
                    scan.test_only.add(top)
                else:
                    scan.runtime.add(top)

    # A module imported anywhere in runtime code is a runtime import.
    scan.test_only -= scan.runtime
    return scan


def find_repositories(repo_dir: str | Path) -> list[Path]:
    """Find Python repositories under repo_dir (one level of instance dirs)."""
    base = Path(repo_dir)
    repos: set[Path] = set()

    # DI-Bench layout: <base>/<instance_id>/ — check that first, it is far
    # cheaper than the recursive .git walk the original code did.
    for child in sorted(base.iterdir()) if base.is_dir() else []:
        if not child.is_dir() or child.name in SKIP_DIRS:
            continue
        if any((child / m).exists() for m in PYTHON_MARKERS) or (child / ".git").exists():
            repos.add(child)

    if repos:
        return sorted(repos)

    for git_dir in base.rglob(".git"):
        repo = git_dir.parent
        if SKIP_DIRS & set(repo.parts):
            continue
        if any((repo / m).exists() for m in PYTHON_MARKERS):
            repos.add(repo)
    return sorted(repos)
