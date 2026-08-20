"""Read and write dependency sections of masked pyproject.toml files.

Across the 98 DI-Bench regular Python instances only two formats occur:
38 Poetry (``[tool.poetry.dependencies]``) and 60 PEP 621
(``[project] dependencies``). There are no setup.py/setup.cfg instances.

Only the *runtime* dependency section is masked by DI-Bench, so that is the
only section written here. Optional/dev groups are left untouched.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import tomlkit

POETRY = "poetry"
PEP621 = "pep621"

# Poetry treats `python` as the interpreter constraint, not a dependency.
_NOT_A_DEPENDENCY = {"python"}

_NAME_RE = re.compile(r"^\s*([A-Za-z0-9._-]+)")


@dataclass(frozen=True)
class Dependency:
    name: str
    version: str = ""

    @property
    def key(self) -> str:
        """DI-Bench comparison key."""
        return self.name.lower().replace("-", "_").strip()

    def to_pep508(self) -> str:
        return f"{self.name}{self.version}" if self.version else self.name


def parse_requirement(spec: str) -> Dependency | None:
    """Parse a PEP 508 requirement string into (name, version constraint)."""
    spec = spec.split("#", 1)[0].strip()
    if not spec:
        return None
    # Drop environment markers; they are not part of the name/version compare.
    spec = spec.split(";", 1)[0].strip()
    match = _NAME_RE.match(spec)
    if not match:
        return None
    name = match.group(1)
    rest = spec[match.end() :].strip()
    rest = re.sub(r"^\[[^\]]*\]", "", rest).strip()  # strip extras
    return Dependency(name=name, version=rest)


def detect_format(pyproject: dict) -> str | None:
    if "tool" in pyproject and "poetry" in pyproject.get("tool", {}):
        return POETRY
    if "project" in pyproject:
        return PEP621
    return None


def manifest_path(repo_path: Path) -> Path | None:
    path = repo_path / "pyproject.toml"
    return path if path.exists() else None


def read_dependencies(text: str) -> tuple[str | None, list[Dependency]]:
    """Return (format, runtime dependencies) parsed from pyproject.toml text."""
    try:
        doc = tomlkit.parse(text)
    except Exception:
        return None, []

    fmt = detect_format(doc)
    deps: list[Dependency] = []

    if fmt == POETRY:
        table = doc.get("tool", {}).get("poetry", {}).get("dependencies", {})
        for name, value in table.items():
            if name in _NOT_A_DEPENDENCY:
                continue
            if isinstance(value, str):
                version = value
            elif hasattr(value, "get"):
                version = value.get("version", "") or ""
            else:
                version = ""
            deps.append(Dependency(name=str(name), version=str(version)))

    elif fmt == PEP621:
        for entry in doc.get("project", {}).get("dependencies", []) or []:
            dep = parse_requirement(str(entry))
            if dep and dep.name not in _NOT_A_DEPENDENCY:
                deps.append(dep)

    return fmt, deps


def write_dependencies(text: str, deps: list[Dependency]) -> str:
    """Write deps into the runtime dependency section, preserving formatting."""
    doc = tomlkit.parse(text)
    fmt = detect_format(doc)

    if fmt == POETRY:
        table = doc["tool"]["poetry"].get("dependencies")
        if table is None:
            table = tomlkit.table()
            doc["tool"]["poetry"]["dependencies"] = table
        for dep in deps:
            if dep.name in _NOT_A_DEPENDENCY:
                continue
            table[dep.name] = dep.version if dep.version else "*"

    elif fmt == PEP621:
        array = tomlkit.array()
        array.multiline(True)
        existing = doc["project"].get("dependencies") or []
        seen = {
            d.key
            for d in (parse_requirement(str(e)) for e in existing)
            if d is not None
        }
        for entry in existing:
            array.append(str(entry))
        for dep in deps:
            if dep.name in _NOT_A_DEPENDENCY or dep.key in seen:
                continue
            array.append(dep.to_pep508())
        doc["project"]["dependencies"] = array

    else:
        raise ValueError("unrecognised pyproject.toml format")

    return tomlkit.dumps(doc)
