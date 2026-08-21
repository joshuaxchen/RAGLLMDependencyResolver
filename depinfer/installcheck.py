"""Check that a predicted dependency set actually resolves and installs.

This is a *proxy* for executability, not the paper's Exec metric. It answers
"do these packages resolve together without conflict", not "do the repository's
tests pass" — the latter needs Sysbox, which is Linux-only.

Two paths:

``compile`` (default)
    ``uv pip compile``, which performs full dependency resolution without
    creating a virtualenv. Seconds per repository.
``venv``
    Delegates to ``install_and_check()`` in the repo's existing
    ``check_requirements.py`` — a real venv, a real ``pip install``, and
    ``pip check``. Minutes per repository, but it catches problems that
    resolution alone does not (build failures, platform wheels).
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .manifest import Dependency

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class InstallResult:
    ok: bool
    mode: str
    detail: str = ""

    @property
    def short(self) -> str:
        return "ok" if self.ok else self.detail.splitlines()[0][:120] if self.detail else "failed"


def _load_check_requirements():
    """Load the existing check_requirements module by path.

    It lives at the repo root rather than inside the package, so a plain import
    depends on how the process was launched.
    """
    path = ROOT / "check_requirements.py"
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location("check_requirements", path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("check_requirements", module)
    spec.loader.exec_module(module)
    return module


def _uv_bin() -> str | None:
    try:
        import uv

        return str(uv.find_uv_bin())
    except Exception:
        return None


def check_dependencies(
    deps: list[Dependency],
    mode: str = "compile",
    python_version: str | None = None,
    timeout: int = 300,
) -> InstallResult:
    """Verify that `deps` can be installed together."""
    if not deps:
        return InstallResult(ok=True, mode=mode, detail="no dependencies to check")

    lines = [d.to_pep508() for d in deps]

    if mode == "venv":
        module = _load_check_requirements()
        if module is None:
            return InstallResult(False, "venv", "check_requirements.py not found")
        with tempfile.TemporaryDirectory(prefix="depinfer_install_") as tmp:
            req = Path(tmp) / "requirements.txt"
            req.write_text("\n".join(lines) + "\n")
            # Reuse the existing venv + pip install + pip check implementation.
            message = module.install_and_check([str(req)])
        ok = message.strip().startswith("No dependency issues found")
        return InstallResult(ok=ok, mode="venv", detail=message)

    uv_bin = _uv_bin()
    if uv_bin is None:
        return InstallResult(False, "compile", "uv not available")

    with tempfile.TemporaryDirectory(prefix="depinfer_install_") as tmp:
        req = Path(tmp) / "requirements.in"
        req.write_text("\n".join(lines) + "\n")
        cmd = [uv_bin, "pip", "compile", str(req), "--no-header"]
        if python_version:
            cmd += ["--python-version", python_version]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return InstallResult(False, "compile", f"timed out after {timeout}s")

    if proc.returncode == 0:
        return InstallResult(True, "compile")
    return InstallResult(False, "compile", (proc.stderr or proc.stdout).strip())
