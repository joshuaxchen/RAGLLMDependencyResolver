#!/usr/bin/env python3
"""Build the shared dense PyPI index. Run once; it persists under .cache/.

    python build_index.py [--top-n 15000] [--workers 8]

This is the slow step in the project, and deliberately so: the index is built
once and reused by every repository and every run, which is the only place in
this pipeline where retrieval has real amortization.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from depinfer.pypi_index import INDEX_DIR, TOP_PACKAGES_URL
from depinfer.resolve import PyPIClient


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-n", type=int, default=15000)
    ap.add_argument("--workers", type=int, default=8, help="parallel PyPI metadata fetches")
    args = ap.parse_args()

    print(f"fetching top-{args.top_n} package list ...", flush=True)
    payload = requests.get(TOP_PACKAGES_URL, timeout=120).json()
    rows = payload["rows"] if isinstance(payload, dict) and "rows" in payload else payload
    names = [r.get("project") or r.get("name") for r in rows][: args.top_n]
    names = [n for n in names if n]
    print(f"  {len(names)} packages", flush=True)

    client = PyPIClient()
    started = time.time()
    done = 0

    def fetch(name):
        nonlocal done
        meta = client.metadata(name)
        done += 1
        if done % 500 == 0:
            rate = done / max(time.time() - started, 1e-6)
            eta = (len(names) - done) / max(rate, 1e-6)
            print(f"  metadata {done}/{len(names)}  {rate:.0f}/s  eta {eta/60:.1f}m", flush=True)
        return name, meta

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        fetched = list(pool.map(fetch, names))
    print(f"metadata done in {(time.time()-started)/60:.1f}m", flush=True)

    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    from langchain_huggingface import HuggingFaceEmbeddings

    docs = [
        Document(
            page_content=f"{name}. {(meta or {}).get('summary', '')}",
            metadata={"name": (meta or {}).get("name", name)},
        )
        for name, meta in fetched
    ]
    print(f"embedding {len(docs)} documents ...", flush=True)
    t = time.time()
    store = FAISS.from_documents(
        docs, HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    store.save_local(str(INDEX_DIR))
    print(f"embedded + saved in {(time.time()-t)/60:.1f}m -> {INDEX_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
