"""Map imported module names to PyPI distributions, and resolve versions.

Two jobs the original ``rag_analyzer.py`` got wrong:

1. It looked up the *import* name on PyPI, so `yaml`, `sklearn`, `cv2`, `PIL`
   and `bs4` all 404'd and vanished silently.
2. It asked the LLM to invent version numbers. Version selection is a solver
   problem; here it is handed to ``uv pip compile``.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import requests

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "pypi"

# Import name -> PyPI distribution, for cases where they differ. Covers the
# common long tail; `packages_distributions()` handles anything installed
# locally, and PyPI lookup handles the identity cases.
IMPORT_TO_DISTRIBUTION: dict[str, str] = {
    "attr": "attrs",
    "attrs": "attrs",
    "bs4": "beautifulsoup4",
    "cairo": "pycairo",
    "cv2": "opencv-python",
    "dateutil": "python-dateutil",
    "dns": "dnspython",
    "docx": "python-docx",
    "dotenv": "python-dotenv",
    "fitz": "PyMuPDF",
    "git": "GitPython",
    "google": "google-api-python-client",
    "grpc": "grpcio",
    "gi": "PyGObject",
    "IPython": "ipython",
    "jose": "python-jose",
    "jwt": "PyJWT",
    "Levenshtein": "python-Levenshtein",
    "lxml": "lxml",
    "magic": "python-magic",
    "mpl_toolkits": "matplotlib",
    "msgpack": "msgpack",
    "nacl": "PyNaCl",
    "OpenSSL": "pyOpenSSL",
    "PIL": "Pillow",
    "pkg_resources": "setuptools",
    "pptx": "python-pptx",
    "psycopg2": "psycopg2-binary",
    "pylab": "matplotlib",
    "pythoncom": "pywin32",
    "pywintypes": "pywin32",
    "win32api": "pywin32",
    "win32com": "pywin32",
    "serial": "pyserial",
    "setuptools": "setuptools",
    "skimage": "scikit-image",
    "sklearn": "scikit-learn",
    "slugify": "python-slugify",
    "snowflake": "snowflake-connector-python",
    "sqlalchemy": "SQLAlchemy",
    "tkinter": "",  # stdlib but often missing from stdlib_module_names sets
    "usb": "pyusb",
    "yaml": "PyYAML",
    "zmq": "pyzmq",
    "Crypto": "pycryptodome",
    "Cryptodome": "pycryptodomex",
    "jinja2": "Jinja2",
    "markdown": "Markdown",
    "requests_toolbelt": "requests-toolbelt",
    "ruamel": "ruamel.yaml",
    "importlib_metadata": "importlib-metadata",
    "typing_extensions": "typing-extensions",
    "pkg_config": "pkgconfig",
}


def normalize(name: str) -> str:
    """DI-Bench's comparison key: lowercase, hyphens folded to underscores."""
    return name.lower().replace("-", "_").strip()


@lru_cache(maxsize=1)
def local_module_map() -> dict[str, str]:
    """Module name -> distribution, for distributions installed in this env."""
    try:
        from importlib.metadata import packages_distributions
    except ImportError:  # pragma: no cover - Python < 3.10
        return {}
    out: dict[str, str] = {}
    for module, dists in packages_distributions().items():
        if dists:
            out[module] = dists[0]
    return out


class PyPIClient:
    """PyPI metadata lookups with an on-disk cache.

    The original kept an in-memory dict only, so every re-run refetched
    everything across 98 repositories.
    """

    def __init__(self, cache_dir: Path = CACHE_DIR, timeout: int = 10):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self._mem: dict[str, dict | None] = {}

    def _cache_path(self, name: str) -> Path:
        return self.cache_dir / f"{normalize(name)}.json"

    def metadata(self, name: str) -> dict | None:
        if name in self._mem:
            return self._mem[name]

        path = self._cache_path(name)
        if path.exists():
            try:
                cached = json.loads(path.read_text())
                result = cached if cached.get("_found") else None
                self._mem[name] = result
                return result
            except (json.JSONDecodeError, OSError):
                pass

        try:
            resp = requests.get(
                f"https://pypi.org/pypi/{name}/json", timeout=self.timeout
            )
        except requests.RequestException:
            self._mem[name] = None
            return None

        if resp.status_code == 200:
            info = resp.json().get("info", {})
            record = {
                "_found": True,
                "name": info.get("name", name),
                "summary": info.get("summary") or "",
                "version": info.get("version") or "",
                "requires_dist": info.get("requires_dist") or [],
                "requires_python": info.get("requires_python") or "",
            }
        else:
            record = {"_found": False}

        try:
            path.write_text(json.dumps(record))
        except OSError:
            pass

        result = record if record.get("_found") else None
        self._mem[name] = result
        return result

    def exists(self, name: str) -> bool:
        return self.metadata(name) is not None


def resolve_distribution(
    module: str, client: PyPIClient, fallback=None
) -> str | None:
    """Best-effort import name -> PyPI distribution name.

    `fallback` is an optional object with `.lookup(module) -> list[str]` used
    only when every cheap strategy fails. See `depinfer.pypi_index`.
    """
    if module in IMPORT_TO_DISTRIBUTION:
        mapped = IMPORT_TO_DISTRIBUTION[module]
        if not mapped:
            return None
        if client.exists(mapped):
            return mapped

    local = local_module_map().get(module)
    if local and client.exists(local):
        return local

    if client.exists(module):
        meta = client.metadata(module)
        return meta["name"] if meta else module

    # Common shape: import `foo_bar`, distribution `foo-bar`.
    dashed = module.replace("_", "-")
    if dashed != module and client.exists(dashed):
        return dashed

    # Last resort: search PyPI for a plausible distribution. Every candidate is
    # still validated against the real index, so this cannot invent a package.
    if fallback is not None:
        for candidate in fallback.lookup(module):
            if client.exists(candidate):
                meta = client.metadata(candidate)
                return meta["name"] if meta else candidate

    return None


def resolve_versions(
    distributions: list[str],
    python_version: str | None = None,
    timeout: int = 180,
) -> tuple[dict[str, str], str | None]:
    """Pin a mutually consistent version set with ``uv pip compile``.

    Returns (name -> version, error). On failure the caller should fall back to
    emitting unpinned names rather than inventing versions.
    """
    if not distributions:
        return {}, None

    try:
        import uv

        uv_bin = uv.find_uv_bin()
    except Exception:
        return {}, "uv not available"

    with tempfile.TemporaryDirectory(prefix="depinfer_resolve_") as tmp:
        req_in = Path(tmp) / "requirements.in"
        req_in.write_text("\n".join(distributions) + "\n")
        # uv writes the resolution to stdout by default; `--output-file -`
        # combined with `--quiet` silently produces nothing.
        cmd = [str(uv_bin), "pip", "compile", str(req_in), "--no-header"]
        if python_version:
            cmd += ["--python-version", python_version]

        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout
            )
        except subprocess.TimeoutExpired:
            return {}, f"uv pip compile timed out after {timeout}s"

    if proc.returncode != 0:
        return {}, (proc.stderr or proc.stdout).strip()[:400]

    # uv normalizes names in its output (PyYAML -> pyyaml). Key the result by
    # the spelling the caller asked for so lookups work.
    wanted = {normalize(d): d for d in distributions}
    pinned: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or "==" not in line:
            continue
        name, _, version = line.partition("==")
        name = name.split("[", 1)[0].strip()
        requested = wanted.get(normalize(name))
        if requested:  # skip transitive dependencies
            pinned[requested] = version.strip()
    return pinned, None
