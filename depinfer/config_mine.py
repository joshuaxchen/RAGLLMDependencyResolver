"""Mine dependencies that are declared but never imported.

94 of 103 missed dependencies in the regular subset appear in no import
statement anywhere in the repository — build backends, test plugins activated by
configuration rather than import, and tools invoked only from CI. Import
analysis cannot reach them by construction; this module reads the places they
are actually named.

DI-Bench masks only the runtime dependency section, so CI workflows, `[tool.*]`
blocks, `tox.ini` and README survive intact in the instance repositories.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import tomlkit

# `pip install foo bar==1.0 -e .` -> foo, bar
_INSTALL_CMD = re.compile(
    r"(?:pip3?|uv\s+pip|python\s+-m\s+pip)\s+install\s+([^\n|;&]+)", re.IGNORECASE
)
_POETRY_ADD = re.compile(r"poetry\s+add\s+([^\n|;&]+)", re.IGNORECASE)
_REQ_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*")

# Flags and non-package arguments that follow `pip install`.
_SKIP_TOKEN = re.compile(r"^-|^\.$|^\.\[|^/|^\$|^\{\{|^%|^~")

CONFIG_GLOBS = (
    ".github/workflows/*.yml",
    ".github/workflows/*.yaml",
    "tox.ini",
    "noxfile.py",
    "setup.cfg",
    "Makefile",
    ".pre-commit-config.yaml",
)


@dataclass
class MinedEvidence:
    """Packages named outside import statements, with where they came from."""

    packages: set[str] = field(default_factory=set)
    sources: dict[str, set[str]] = field(default_factory=dict)
    raw_text: str = ""

    def add(self, name: str, source: str) -> None:
        cleaned = name.strip().strip("'\"")
        if not cleaned:
            return
        match = _REQ_NAME.match(cleaned)
        if not match:
            return
        package = match.group(0)
        if len(package) < 2:
            return
        self.packages.add(package)
        self.sources.setdefault(package, set()).add(source)


def _split_install_args(blob: str) -> list[str]:
    out = []
    for token in blob.split():
        if _SKIP_TOKEN.match(token):
            continue
        out.append(token)
    return out


def _mine_shell_text(text: str, source: str, evidence: MinedEvidence) -> None:
    for pattern in (_INSTALL_CMD, _POETRY_ADD):
        for match in pattern.finditer(text):
            for token in _split_install_args(match.group(1)):
                evidence.add(token, source)


def _mine_pyproject(path: Path, evidence: MinedEvidence) -> None:
    try:
        doc = tomlkit.parse(path.read_text())
    except Exception:
        return

    # PEP 518 build requirements are real dependencies of the build, and turn up
    # in ground truth (setuptools was a repeated false negative).
    for entry in doc.get("build-system", {}).get("requires", []) or []:
        evidence.add(str(entry), "build-system.requires")

    tool = doc.get("tool", {})

    pytest_cfg = tool.get("pytest", {}).get("ini_options", {})
    addopts = pytest_cfg.get("addopts", "")
    if isinstance(addopts, str):
        # `--cov` only works if pytest-cov is installed.
        if "--cov" in addopts:
            evidence.add("pytest-cov", "tool.pytest.addopts")
        if "-n " in addopts or "--numprocesses" in addopts:
            evidence.add("pytest-xdist", "tool.pytest.addopts")
    for plugin in pytest_cfg.get("required_plugins", []) or []:
        evidence.add(str(plugin), "tool.pytest.required_plugins")

    if "coverage" in tool:
        evidence.add("coverage", "tool.coverage")

    # Some projects list dev tooling under tool.<name>; the section's existence
    # implies the tool, though it is usually a dev dependency and filtered later.
    for name in ("black", "isort", "mypy", "ruff", "flake8", "pylint"):
        if name in tool:
            evidence.add(name, f"tool.{name}")


def _mine_ini(path: Path, evidence: MinedEvidence) -> None:
    """tox.ini / setup.cfg `deps =` blocks are line-oriented requirement lists."""
    try:
        lines = path.read_text(errors="ignore").splitlines()
    except OSError:
        return
    in_deps = False
    for line in lines:
        stripped = line.strip()
        if re.match(r"^(deps|install_requires)\s*=", stripped):
            in_deps = True
            tail = stripped.split("=", 1)[1].strip()
            if tail:
                evidence.add(tail, path.name)
            continue
        if in_deps:
            if not line.startswith((" ", "\t")) or not stripped:
                in_deps = False
                continue
            if stripped.startswith(("-", "{")):
                continue
            evidence.add(stripped, path.name)


def _mine_readme(path: Path, evidence: MinedEvidence) -> None:
    try:
        _mine_shell_text(path.read_text(errors="ignore"), path.name, evidence)
    except OSError:
        pass


def mine_repository(repo_path: Path, collect_text: bool = False) -> MinedEvidence:
    """Collect package names from configuration, CI and docs.

    With `collect_text`, also returns the raw config text. Measured at ~5.4 KB of
    config plus ~6.0 KB of README per repo, it fits an 8k-token prompt for 96 of
    98 repos, so the LLM path can read it directly rather than retrieving it.
    """
    evidence = MinedEvidence()
    chunks: list[str] = []

    for pattern in CONFIG_GLOBS:
        for path in sorted(repo_path.glob(pattern)):
            if not path.is_file():
                continue
            try:
                text = path.read_text(errors="ignore")
            except OSError:
                continue
            rel = str(path.relative_to(repo_path))
            _mine_shell_text(text, rel, evidence)
            if path.suffix == ".ini" or path.name == "setup.cfg":
                _mine_ini(path, evidence)
            if collect_text:
                chunks.append(f"--- {rel} ---\n{text[:8000]}")

    pyproject = repo_path / "pyproject.toml"
    if pyproject.exists():
        _mine_pyproject(pyproject, evidence)

    for readme in sorted(repo_path.glob("README*")):
        if readme.is_file():
            _mine_readme(readme, evidence)
            if collect_text:
                chunks.append(f"--- {readme.name} ---\n{readme.read_text(errors='ignore')[:6000]}")

    if collect_text:
        evidence.raw_text = "\n\n".join(chunks)
    return evidence
