"""LLM backends for dependency selection.

The model is asked only to *select* which candidate modules are genuine runtime
dependencies. It is never asked to produce version numbers — those come from
``uv pip compile`` in :mod:`depinfer.resolve`. Version errors account for
roughly a third of the paper's Python failures, and generation cannot do better
than guess at them.

Backends sit behind a small protocol so the model is an experimental axis
rather than a hardcoded limit.
"""

from __future__ import annotations

import json
from typing import Protocol

import requests
from pydantic import BaseModel, ValidationError

DEFAULT_MODEL = "qwen2.5-coder:7b"


class Selection(BaseModel):
    runtime_dependencies: list[str]


class Backend(Protocol):
    name: str

    def complete(self, prompt: str, schema: dict | None = None) -> str | None: ...


class OllamaBackend:
    """Local Ollama. Deterministic settings; structured JSON output."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        url: str = "http://localhost:11434",
        timeout: int = 180,
    ):
        self.model = model
        self.url = url.rstrip("/")
        self.timeout = timeout

    @property
    def name(self) -> str:
        return f"ollama:{self.model}"

    def complete(self, prompt: str, schema: dict | None = None) -> str | None:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            # Structured extraction: greedy decoding, not 0.7.
            "options": {"temperature": 0.0, "num_ctx": 8192},
            "format": schema if schema else "json",
        }
        try:
            resp = requests.post(
                f"{self.url}/api/generate", json=payload, timeout=self.timeout
            )
        except requests.RequestException as exc:
            print(f"  ! ollama request failed: {exc}")
            return None

        if resp.status_code != 200:
            print(f"  ! ollama returned {resp.status_code}: {resp.text[:200]}")
            return None
        return resp.json().get("response")


def build_prompt(
    candidates: list[dict],
    test_only: list[str],
    repo_name: str,
) -> str:
    """Prompt the model to select runtime dependencies from candidates.

    DI-Bench masks only the runtime dependency section, so test/dev-only
    packages are false positives, not omissions.
    """
    lines = []
    for c in candidates:
        summary = (c.get("summary") or "").strip()[:100]
        lines.append(
            f"- import `{c['module']}` -> PyPI package `{c['distribution']}`"
            + (f": {summary}" if summary else "")
            + f" [seen in {c['n_files']} file(s), {c['where']}]"
        )

    candidate_block = "\n".join(lines) if lines else "(none)"
    test_block = ", ".join(sorted(test_only)) if test_only else "(none)"

    return f"""You are a Python packaging expert deciding which packages belong in the \
runtime dependency list of the repository `{repo_name}`.

Candidate third-party packages found by static import analysis:
{candidate_block}

Modules imported ONLY by tests, docs, or examples:
{test_block}

Rules:
1. Include a package only if the library's own runtime code needs it to work.
2. EXCLUDE test/lint/docs tooling (pytest, tox, flake8, black, mypy, sphinx,
   coverage and similar). They belong to a dev group, not runtime dependencies.
3. EXCLUDE anything only imported by tests, docs, or examples.
4. Use the PyPI package name shown above, not the import name.
5. Do NOT include version numbers. Do NOT invent packages that are not listed.

Return only JSON: {{"runtime_dependencies": ["package-name", ...]}}"""


def select_runtime_dependencies(
    backend: Backend,
    candidates: list[dict],
    test_only: list[str],
    repo_name: str,
    retries: int = 1,
) -> tuple[list[str] | None, str | None]:
    """Ask the backend to select runtime dependencies. Returns (names, error).

    Output is validated against a schema and constrained to the candidate set,
    so the model cannot introduce packages that were never seen.
    """
    prompt = build_prompt(candidates, test_only, repo_name)
    allowed = {c["distribution"].lower(): c["distribution"] for c in candidates}
    schema = Selection.model_json_schema()

    last_error = None
    for attempt in range(retries + 1):
        raw = backend.complete(prompt, schema=schema)
        if raw is None:
            last_error = "no response from backend"
            continue
        try:
            text = raw.strip()
            if text.startswith("```"):  # strip accidental fences
                text = text.split("```")[1].removeprefix("json").strip()
            selection = Selection.model_validate_json(text)
        except (ValidationError, json.JSONDecodeError, IndexError) as exc:
            last_error = f"invalid JSON on attempt {attempt + 1}: {exc}"
            continue

        chosen, hallucinated = [], []
        for name in selection.runtime_dependencies:
            match = allowed.get(name.strip().lower())
            if match:
                chosen.append(match)
            else:
                hallucinated.append(name)
        if hallucinated:
            print(f"  ! dropped {len(hallucinated)} off-list name(s): {hallucinated[:5]}")
        return sorted(set(chosen)), None

    return None, last_error
