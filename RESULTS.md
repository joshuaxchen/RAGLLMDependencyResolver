# Experimental results

DI-Bench regular Python subset, 98 repositories, 537 ground-truth dependencies
(mean 5.5 per repo). The paper reports 5.5 for this subset, an independent check
that ground-truth extraction is correct.

## What the metrics measure — and what they do not

The headline P/R/F1 is **set overlap of normalized package names**. It says
whether we named the same packages the maintainer wrote down. It does not say
whether the project runs.

`Zuehlke_ConfZ` scores a perfect 1.00 on names; nothing in that number checked
that `pydantic==2.13.4` falls inside the oracle's `>=1.9.0, <3.0.0`. Had the
resolver emitted `3.5.0`, the score would be identical and the project broken.
Conversely `valohai_django-allauth-2fa` scores 0.86 for missing
`django-allauth` — total failure for a package whose entire purpose is allauth
integration. The metric is not monotonic with working, and it weights a harmless
extra dependency the same as a fatal missing one.

The paper quantifies the same gap: GPT-4o reaches F1 67.2 but Exec 42.9.

Two additional columns were added to narrow it:

- **`versions`** — is the predicted constraint *satisfiable within* the oracle's
  range, via `packaging.specifiers`. This replaces a string-equality check that
  reported 1.2% and was pure formatting artifact.
- **`install proxy`** — does the set resolve and install. `--install-check
  compile` uses `uv`; `--install-check venv` does a real venv install plus
  `pip check`.

Neither is the paper's **Exec** metric, which runs each repository's CI test
suite under [Sysbox](https://github.com/nestybox/sysbox) — Linux-only, so
unavailable here. The name-level comparison against the paper's P/R/F1 columns
is valid (identical metric and normalization); nothing here is comparable to its
Exec column.

## Ablation

| Configuration | P | R | F1 | Versions OK | Perfect | Runtime |
|---|---|---|---|---|---|---|
| **`deterministic` (no LLM)** | **67.0** | 80.8 | **73.2** | 77.2% | **20/98** | 19s |
| `+ PyPI lexical fallback` | 66.6 | **81.6** | **73.3** | 76.9% | 20/98 | 12s |
| `+ PyPI dense retrieval` | 65.8 | 81.2 | 72.7 | 76.4% | 20/98 | 9s |
| `+ config/CI mining` | 45.0 | 81.2 | 57.9 | 81.8% | 1/98 | 24s |
| `llm` (qwen2.5-coder:7b) | 65.6 | 78.0 | 71.3 | 77.3% | 17/97 | 3480s |
| GPT-4o Imports-Only *(paper)* | 56.5 | 74.9 | 64.4 | — | — | — |
| GPT-4o All-In-One *(paper)* | 61.8 | 73.6 | 67.2 | — | — | — |

Install proxy on the baseline: **93.9%** under `compile`, but see the caveat
below. Under `venv` on a 6-repo sample: **83.3%**.

## Findings

### Retrieval loses to a lexical matcher

The PyPI fallback resolves import names that map to no package — 23 such cases,
of which 7 correspond to a real missed dependency (`digitalocean` →
`python-digitalocean`, `allauth` → `django-allauth`, `pythonjsonlogger` →
`python-json-logger`, and four more).

| Method | Targets recovered | F1 |
|---|---|---|
| baseline (no fallback) | 0 / 7 | 73.2 |
| **lexical name matching** | **4 / 7** | **73.3** |
| dense embedding retrieval | 2 / 7 | 72.7 |

**A ~40-line lexical matcher beat a 15,000-document FAISS index on both counts,
and the dense index scored below the no-fallback baseline.** The relationships
are string morphology — `python-` prefixes, `-client-lib` suffixes, word
splits — not semantics, so embeddings are the wrong instrument.

A cautionary detail: in a 23-package toy index with the correct answers
guaranteed present, dense retrieval scored **7/7**. At real scale with 15,000
distractors it scored 2/7. Small-corpus RAG demos are close to meaningless.

The lexical matcher reads the full PyPI name list — 875,199 names in a single
0.8s request — and resolves in under 0.4s per query with no model.

### Config/CI mining fails

94 of 103 missed dependencies appear in no import statement, so mining CI
workflows, `[tool.*]` blocks and README looked like the largest lever. It is not:
precision collapses 67.0 → 45.0 for +0.4 recall.

Only 5 of 103 misses are recoverable this way. README mining alone contributes
roughly 99 false positives against 6 true ones — install instructions name
unrelated tooling. The flag exists (`--config-mining`) and is **off by default**.

Digging further into why: of the 103 missed dependencies, **45 appear nowhere in
the repository at all** — not in source, config, docs or CI. Examples are
`xstatic_*` asset packages wired through a plugin system, vendored forks like
`termcolor_whl` and `atomicwrites_homeassistant`, and transitive dependencies
the author declared explicitly. No amount of retrieval or mining over repository
content can reach them, because the information is not there.

### Resolving to "latest" breaks projects

Version compatibility started at 73.6% — 78 of 296 constrained matches pinned
outside the oracle's allowed range, e.g. `pandas==3.0.5` against `^2.0.0`,
`pydantic==2.13.4` against `^1`.

Cause: `uv` was resolving without a Python constraint and picking the newest
release. Passing the repository's own `requires-python` floor lifted
compatibility to **77.2%**, and on the venv install proxy took a 6-repo sample
from 50% to **83.3%** — two of three failures there were `matplotlib==3.11.1`
requiring Python ≥3.11 inside a 3.10 environment.

### The compile-mode install proxy is circular

It reports 93.9%, and all 6 failures are exactly the repositories where `uv`
pinning had already failed. Since versions come *from* `uv`, re-resolving them
proves nothing. Only `venv` mode is independent — it does real builds and
catches platform and interpreter mismatches that resolution alone does not.
Treat the 93.9% as bookkeeping, not evidence.

### The LLM still does not pay for itself

Re-run with the current metrics and the python-version fix: F1 71.3 against the
deterministic 73.2, at **180x the runtime** (3480s vs 19s). Version compatibility
is a wash (77.3% vs 77.2%), so the model is not helping there either. One
instance failed outright when Ollama returned no response.

The earlier full comparison: better on 9 instances, worse on 22, identical on 67,
trading 14 true positives for 10 extra false positives.

### It generalises to the large subset

The large Python subset — 50 instances averaging 268 files and 519k tokens —
exceeds the context window, which is why the paper's All-In-One method has no
entry for it.

| Method | P | R | F1 |
|---|---|---|---|
| **`deterministic` (this repo, no LLM)** | 35.9 | 58.8 | **44.5** |
| `+ PyPI lexical fallback` | 35.2 | **59.3** | 44.2 |
| GPT-4o Imports-Only *(paper)* | **36.9** | 46.9 | 41.3 |
| GPT-4o File-Iterate *(paper)* | 19.5 | 35.3 | 25.1 |
| GPT-4o All-In-One *(paper)* | — | — | — (exceeds context) |

Static analysis beats the best published large-subset method on F1 and recall,
with no model and no retrieval. The lexical fallback does not help here either
(44.2 vs 44.5), consistent with the regular subset. Absolute numbers are much lower than on the
regular subset (F1 44.5 vs 73.2), as expected: these repositories declare more
dependencies and use more indirection.

Ground-truth extraction here required parsing four build-file formats — 14
`requirements.txt`, 13 `setup.py`, 12 `pyproject.toml`, 4 `setup.cfg` — where
the regular subset is 100% `pyproject.toml`. Mean extracted dependency count is
11.5 against the paper's reported 11.8, an independent check. Three instances
yield an empty oracle: two build `install_requires` dynamically (the AST parser
declines to guess) and one uses a bespoke `dependencies.py`.

A latent bug surfaced here: the regular dataset spells the language `"python"`
and the large dataset `"Python"`, so the exact-match filter in `load_dataset`
silently returned zero instances. Now compared case-insensitively.

### The fake rate is still 0.0 and still structural

Every candidate — imported, mined, or retrieved — is validated against PyPI
before it can be predicted, so an unresolvable name cannot be emitted. This is
not better judgment than the paper's baselines at 2.8–3.9; it is a different
pipeline shape. It should not be read as a win.

## Negative results worth keeping

1. **Transitive filtering** — drops 71 false positives and 134 true positives.
   Precision 67.0 → 67.8, recall 79.5 → **54.6**. Not implemented.
2. **Config/CI mining** — precision 67.0 → 45.0 for +0.4 recall. Off by default.
3. **Dense PyPI retrieval** — loses to lexical matching, 2/7 vs 4/7, and scores
   below baseline F1.

## Reproducing

```bash
python cli.py --method deterministic                          # ~19s
python cli.py --method deterministic --pypi-fallback lexical  # ~12s
python build_index.py                                         # ~4min, once
python cli.py --method deterministic --pypi-fallback dense
python cli.py --method deterministic --install-check venv --limit 6
python cli.py --method llm                                    # ~31min
```

Partial runs write to `results/<method>-limit<N>/` and refuse to overwrite a
larger report without `--force`.
