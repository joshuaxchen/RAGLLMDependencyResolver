"""Resolve import names that map to no obvious PyPI package.

`resolve_distribution()` falls back here when the curated table, the local
environment and exact-name lookup all fail. In the regular subset 23 modules
reach this point, of which 7 correspond to a real missed dependency:

    digitalocean      -> python-digitalocean
    allauth           -> django-allauth
    pythonjsonlogger  -> python-json-logger
    tagmatcher        -> tag-matcher
    airflow_client    -> apache-airflow-client
    lightstreamer     -> lightstreamer-client-lib
    djvu              -> djvulibre-python

Two strategies, deliberately comparable:

``lexical``
    Match against the full PyPI name list. The relationships above are string
    morphology — prefixes, suffixes, word splits — not semantics.
``dense``
    Embed package name + summary and retrieve by cosine similarity. The
    conventional RAG approach, included so its value is measured rather than
    assumed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import requests

from .resolve import PyPIClient, normalize

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"
NAMES_CACHE = CACHE_DIR / "pypi_names.json"
INDEX_DIR = CACHE_DIR / "pypi_index"

SIMPLE_INDEX_URL = "https://pypi.org/simple/"
SIMPLE_ACCEPT = "application/vnd.pypi.simple.v1+json"
TOP_PACKAGES_URL = "https://hugovk.dev/top-pypi-packages/top-pypi-packages.min.json"

# Ecosystem prefixes that commonly wrap an import name.
_PREFIXES = ("python-", "py-", "django-", "flask-", "pytest-", "sphinxcontrib-", "types-")
_SUFFIXES = ("-python", "-py", "-client", "-client-lib", "-lib", "-sdk", "-whl")


def load_pypi_names(refresh: bool = False) -> set[str]:
    """All PyPI project names (~600k), cached on disk. One request."""
    if NAMES_CACHE.exists() and not refresh:
        try:
            return set(json.loads(NAMES_CACHE.read_text()))
        except (json.JSONDecodeError, OSError):
            pass

    resp = requests.get(SIMPLE_INDEX_URL, headers={"Accept": SIMPLE_ACCEPT}, timeout=120)
    resp.raise_for_status()
    names = [p["name"] for p in resp.json().get("projects", [])]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    NAMES_CACHE.write_text(json.dumps(names))
    return set(names)


def _variants(module: str) -> list[str]:
    """Plausible distribution spellings for an import name."""
    base = module.lower()
    dashed = base.replace("_", "-")
    out = [base, dashed]
    for prefix in _PREFIXES:
        out.append(f"{prefix}{dashed}")
    for suffix in _SUFFIXES:
        out.append(f"{dashed}{suffix}")
    # pythonjsonlogger -> python-json-logger: split a run-on name on known words.
    for word in ("python", "json", "logger", "client", "parser", "tools", "utils"):
        if base.startswith(word) and len(base) > len(word):
            out.append(f"{word}-{base[len(word):]}")
        if base.endswith(word) and len(base) > len(word):
            out.append(f"{base[:-len(word)]}-{word}")
    # tagmatcher -> tag-matcher: try every single split point.
    for i in range(3, len(base) - 2):
        out.append(f"{base[:i]}-{base[i:]}")
    return out


class LexicalPyPIMatcher:
    """Name-list matching. No embeddings, no model."""

    def __init__(self, names: set[str] | None = None):
        self._names = names if names is not None else load_pypi_names()
        self._by_key = {}
        for name in self._names:
            self._by_key.setdefault(normalize(name), name)

    def lookup(self, module: str, limit: int = 5) -> list[str]:
        seen, out = set(), []
        for variant in _variants(module):
            key = normalize(variant)
            match = self._by_key.get(key)
            if match and match not in seen:
                seen.add(match)
                out.append(match)
                if len(out) >= limit:
                    return out

        # Last resort: distributions that contain the module name as a word.
        pattern = re.compile(rf"(^|[-_]){re.escape(normalize(module))}([-_]|$)")
        for key, name in self._by_key.items():
            if name in seen:
                continue
            if pattern.search(key):
                out.append(name)
                if len(out) >= limit:
                    break
        return out


class DensePyPIIndex:
    """Embedding index over package name + summary, built once and persisted.

    This is the only corpus in the project with real retrieval economics: too
    large for any context window, identical for every repository, and amortized
    over every query rather than rebuilt per repo.
    """

    def __init__(self, index_dir: Path = INDEX_DIR):
        self.index_dir = index_dir
        self._store = None
        self._embeddings = None

    def _get_embeddings(self):
        if self._embeddings is None:
            from langchain_huggingface import HuggingFaceEmbeddings

            self._embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        return self._embeddings

    def exists(self) -> bool:
        return (self.index_dir / "index.faiss").exists()

    def build(self, client: PyPIClient, top_n: int = 15000, verbose: bool = True) -> int:
        """Fetch top-N package metadata and build the FAISS index."""
        from langchain_community.vectorstores import FAISS
        from langchain_core.documents import Document

        resp = requests.get(TOP_PACKAGES_URL, timeout=120)
        resp.raise_for_status()
        payload = resp.json()
        rows = payload["rows"] if isinstance(payload, dict) and "rows" in payload else payload
        names = [r.get("project") or r.get("name") for r in rows][:top_n]

        docs = []
        for i, name in enumerate(names):
            if not name:
                continue
            meta = client.metadata(name)
            summary = (meta or {}).get("summary", "")
            docs.append(
                Document(
                    page_content=f"{name}. {summary}",
                    metadata={"name": (meta or {}).get("name", name)},
                )
            )
            if verbose and i % 500 == 0:
                print(f"  fetched {i}/{len(names)}")

        store = FAISS.from_documents(docs, self._get_embeddings())
        self.index_dir.mkdir(parents=True, exist_ok=True)
        store.save_local(str(self.index_dir))
        self._store = store
        return len(docs)

    def _load(self):
        if self._store is None:
            from langchain_community.vectorstores import FAISS

            self._store = FAISS.load_local(
                str(self.index_dir),
                self._get_embeddings(),
                allow_dangerous_deserialization=True,
            )
        return self._store

    def lookup(self, module: str, limit: int = 5) -> list[str]:
        if not self.exists():
            return []
        hits = self._load().similarity_search(module, k=limit)
        return [h.metadata["name"] for h in hits]
