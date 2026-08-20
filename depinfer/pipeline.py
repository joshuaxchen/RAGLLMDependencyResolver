"""End-to-end dependency inference for one repository.

Two methods, so the LLM's contribution is measurable rather than assumed:

``deterministic``
    Static imports -> PyPI distribution mapping -> heuristic dev-tool filter.
    No model involved.
``llm``
    The same candidates, with the model selecting which are runtime
    dependencies.

Both then hand version selection to ``uv pip compile``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .extract import extract_imports, find_first_party
from .generate import Backend, select_runtime_dependencies
from .manifest import Dependency
from .resolve import PyPIClient, resolve_distribution, resolve_versions

# Packages that are development tooling, never runtime dependencies of the
# library itself. DI-Bench masks only the runtime section, so these are false
# positives when predicted.
DEV_TOOLING = {
    "pytest", "pytest-cov", "pytest-asyncio", "pytest-mock", "pytest-xdist",
    "tox", "nox", "coverage", "codecov",
    "flake8", "pylint", "mypy", "black", "isort", "ruff", "autopep8", "yapf",
    "pre-commit", "bandit", "pyflakes", "pycodestyle",
    "sphinx", "mkdocs", "mkdocs-material", "sphinx-rtd-theme",
    "setuptools", "wheel", "twine", "build", "hatch", "poetry", "flit",
    "mock", "responses", "freezegun", "faker", "hypothesis",
}


def is_plugin_host(tool: str, repo_name: str, first_party: list[str]) -> bool:
    """True when the repo is a plugin *for* `tool`, making it a runtime dep.

    `MartinThoma_flake8-simplify` genuinely depends on flake8 at runtime;
    blanket dev-tooling filtering costs recall on these (flake8 was the single
    largest false negative in the first baseline run).
    """
    stem = tool.split("-")[0].lower()
    haystack = [repo_name.lower().replace("_", "-")] + [p.lower() for p in first_party]
    return any(stem in h for h in haystack)


@dataclass
class InferenceResult:
    instance_id: str
    dependencies: list[Dependency] = field(default_factory=list)
    candidates: list[dict] = field(default_factory=list)
    test_only: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    first_party: list[str] = field(default_factory=list)
    method: str = "deterministic"
    resolve_error: str | None = None
    error: str | None = None


def build_candidates(repo_path: Path, client: PyPIClient) -> tuple[list[dict], list[str], list[str]]:
    """Map imported modules to PyPI distributions.

    Returns (candidates, test_only_modules, unresolved_modules).
    """
    scan = extract_imports(repo_path)
    candidates: list[dict] = []
    unresolved: list[str] = []

    for module in sorted(scan.all_modules):
        distribution = resolve_distribution(module, client)
        if not distribution:
            unresolved.append(module)
            continue
        meta = client.metadata(distribution) or {}
        candidates.append(
            {
                "module": module,
                "distribution": distribution,
                "summary": meta.get("summary", ""),
                "n_files": len(scan.provenance.get(module, ())),
                "where": "tests/docs only" if module in scan.test_only else "runtime code",
                "test_only": module in scan.test_only,
            }
        )
    return candidates, sorted(scan.test_only), unresolved


def infer_repository(
    repo_path: Path,
    client: PyPIClient,
    method: str = "deterministic",
    backend: Backend | None = None,
    pin_versions: bool = True,
) -> InferenceResult:
    instance_id = repo_path.name
    result = InferenceResult(instance_id=instance_id, method=method)

    try:
        candidates, test_only, unresolved = build_candidates(repo_path, client)
    except Exception as exc:
        result.error = f"extraction failed: {exc}"
        return result

    result.candidates = candidates
    result.test_only = test_only
    result.unresolved = unresolved
    result.first_party = sorted(find_first_party(repo_path))

    if method == "llm":
        if backend is None:
            result.error = "llm method requires a backend"
            return result
        chosen, err = select_runtime_dependencies(
            backend, candidates, test_only, instance_id
        )
        if chosen is None:
            result.error = f"selection failed: {err}"
            return result
        names = chosen
    else:
        names = sorted(
            {
                c["distribution"]
                for c in candidates
                if not c["test_only"]
                and (
                    c["distribution"].lower() not in DEV_TOOLING
                    or is_plugin_host(c["distribution"], instance_id, result.first_party)
                )
            }
        )

    if not names:
        return result

    if pin_versions:
        pinned, err = resolve_versions(names)
        result.resolve_error = err
        # On resolver failure emit unpinned names rather than inventing versions.
        result.dependencies = [
            Dependency(name=n, version=f"=={pinned[n]}" if n in pinned else "")
            for n in names
        ]
    else:
        result.dependencies = [Dependency(name=n) for n in names]

    return result
