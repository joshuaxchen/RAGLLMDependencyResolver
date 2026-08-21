# DependencyManager

Dependency inference for Python repositories, evaluated against
[DI-Bench](https://github.com/microsoft/DI-Bench)
([paper](https://arxiv.org/abs/2501.13699)).

Given a repository whose runtime dependency declarations have been stripped from
`pyproject.toml`, reconstruct them. The benchmark's Python *regular* subset — 98
real GitHub repositories — lives in `repo-data/python/`.

## Why this is a real task

Repository-level code generation frequently produces code that cannot run, and
dependency errors are a large share of the cause: ChatDev and DevBench report
dependency issues behind **over 50%** of observed runtime errors. DI-Bench
isolates that step, and the best model in the paper reaches only **42.9%**
executability on Python.

## What is here

```
depinfer/
  extract.py     tree-sitter import extraction, stdlib/first-party filtering
  resolve.py     import name -> PyPI distribution; version pinning via uv
  manifest.py    format-preserving read/write of pyproject.toml (Poetry + PEP 621)
  generate.py    Ollama backend; the model selects names, never versions
  evaluate.py    scoring against DI-Bench ground truth
  versions.py    range-aware version comparison (packaging.specifiers)
  installcheck.py  does the predicted set actually install
  config_mine.py   mine CI/[tool.*]/README for never-imported deps
  pypi_index.py    resolve unknown import names: lexical + dense
  pipeline.py    orchestration for all methods
cli.py           run + score the benchmark
build_index.py   build the shared dense PyPI index (once, ~4min)
tests/           45 regression tests
```

Two methods, so the model's contribution is measured rather than assumed:

- `deterministic` — static imports → PyPI mapping → dev-tooling filter. No LLM.
- `llm` — same candidates, with the model selecting which are runtime deps.

Both hand version selection to `uv pip compile`, constrained to the repository's
own `requires-python` floor.

## Setup

```bash
pip install -r requirements.txt
ollama pull qwen2.5-coder:7b                       # only for --method llm
```

Ground truth (`dataset-dibench-regular.jsonl`) is tracked at the repo root, so
no download is needed. Repository data is not: extract
`dibench-regular-python.tar.gz` so instances land in
`repo-data/python/<instance_id>/`.

## Running

```bash
python cli.py --method deterministic                          # ~19s for all 98
python cli.py --method deterministic --pypi-fallback lexical  # best F1
python cli.py --method deterministic --install-check venv --limit 6
python cli.py --method llm                                    # ~31min
python build_index.py                                         # dense index, once
python -m pytest tests/ -q
```

Results are written to `results/<method>/` as `score_report.json` (per-instance
scores, missing/spurious breakdowns) and `predictions.json`. Partial runs go to
`results/<method>-limit<N>/` and refuse to overwrite a larger report without
`--force`.

## Results — DI-Bench regular Python (98 instances)

Full experimental record, error analysis and negative results: [RESULTS.md](RESULTS.md).

| Configuration | P | R | F1 | Versions OK | Perfect | Runtime |
|---|---|---|---|---|---|---|
| **`deterministic` (no LLM)** | **67.0** | 80.8 | **73.2** | 77.2% | **20/98** | 19s |
| `+ PyPI lexical fallback` | 66.6 | **81.6** | **73.3** | 76.9% | 20/98 | 12s |
| `+ PyPI dense retrieval` | 65.8 | 81.2 | 72.7 | 76.4% | 20/98 | 9s |
| `+ config/CI mining` | 45.0 | 81.2 | 57.9 | 81.8% | 1/98 | 24s |
| `llm` (qwen2.5-coder:7b) | 65.6 | 78.0 | 71.3 | 77.3% | 17/97 | 3480s |
| GPT-4o Imports-Only *(paper)* | 56.5 | 74.9 | 64.4 | — | — | — |
| GPT-4o All-In-One *(paper)* | 61.8 | 73.6 | 67.2 | — | — | — |

### Large subset (50 instances, avg 268 files / 519k tokens)

| Method | P | R | F1 |
|---|---|---|---|
| **`deterministic` (no LLM)** | 35.9 | 58.8 | **44.5** |
| GPT-4o Imports-Only *(paper)* | 36.9 | 46.9 | 41.3 |
| GPT-4o File-Iterate *(paper)* | 19.5 | 35.3 | 25.1 |
| GPT-4o All-In-One *(paper)* | — | — | — (exceeds context) |

Static analysis also beats the best published method on repositories too large
to fit in a context window.

**A ~40-line lexical matcher beats a 15,000-document FAISS index** at resolving
unknown import names — 4 of 7 target cases against 2 of 7 — and the dense index
scores below the no-fallback baseline. The relationships are string morphology,
not semantics.

**The LLM makes it worse.** Adding a 7B model on top of the deterministic
candidates costs 2.1 F1 and takes 200x longer. Per instance it is better on 9,
worse on 22, and identical on 67 — so on two thirds of repos it changes nothing,
and where it does act it is more than twice as likely to hurt as to help. Across
all 98 repos it added 28 packages and dropped 32 relative to the deterministic
set.

Scope of that claim: qwen2.5-coder:7b, textual metrics, regular subset. A
stronger model may do better, and this says nothing about the large subset,
where context limits change the problem. What it does show is that on
context-sized repos the work is being done by static analysis, name resolution,
and the version resolver — not by generation.

Two caveats that matter:

- **These are textual metrics, not executability.** The paper's headline metric
  runs each repo's CI test suite under [Sysbox](https://github.com/nestybox/sysbox),
  which is Linux-only and cannot run on macOS. The numbers above are not
  comparable to the paper's 42.9%.
- **The 0.0 fake rate is structural, not earned.** Candidates only exist if a
  PyPI lookup succeeded, so an unresolvable name cannot be emitted. It is not
  evidence of better judgment than the LLM baselines.

**Name-level F1 does not mean the project works.** `Zuehlke_ConfZ` scores a
perfect 1.00 while nothing in that number checks that `pydantic==2.13.4` falls
inside the oracle's `>=1.9.0, <3.0.0`. The `versions` column exists for that:
range-aware comparison via `packaging.specifiers`, replacing a string-equality
check that reported a meaningless 1.2%.

## Findings

**Ground truth is runtime-only.** Of ~660 dependency lines the instance patches
restore, 651 land in `dependencies = [...]` or `[tool.poetry.dependencies]` and
only 4 in `[project.optional-dependencies]`. Dev/test groups are left unmasked in
73 of 98 repos. Test-only imports are therefore *false positives*, which is why
the pipeline separates runtime from test imports.

**Transitive filtering looks obvious and is wrong.** 34% of false positives
(71/210) are packages required by another predicted package — `numpy` via
`pandas`, `botocore` via `boto3`. Filtering them out also removes 134 true
positives, because projects routinely declare a direct dependency that is *also*
transitive. Measured net effect: precision 67.0 → 67.8, recall 79.5 → **54.6**.
Not implemented, deliberately.

**Blanket dev-tooling filters cost recall.** `flake8` was the single largest
false negative: for a flake8 *plugin*, flake8 is a genuine runtime dependency.
`is_plugin_host()` in `pipeline.py` handles this.

**Resolving to "latest" breaks projects.** Version compatibility started at
73.6% — 78 of 296 matches pinned outside the oracle's range (`pandas==3.0.5`
against `^2.0.0`). `uv` was resolving without a Python constraint. Passing the
repo's own `requires-python` floor lifted it to 77.2%, and took the venv install
proxy from 50% to 83.3% on a 6-repo sample.

**Config/CI mining does not work.** 94 of 103 misses are never imported, so
mining CI and `[tool.*]` looked like the big lever. Precision collapses
67.0 → 45.0 for +0.4 recall; only 5 of 103 misses are recoverable that way, and
45 of them appear *nowhere in the repository at all*. Off by default.

**`sys.stdlib_module_names` describes the running interpreter, not the target.**
`tomllib` is stdlib on 3.11+ but not on the 3.10 running here, so it was being
looked up on PyPI. See `EXTRA_STDLIB` in `extract.py`.

## Next

Retrieval over the **large** subset (>120k tokens, avg 268 files for Python),
where All-In-One does not fit in context and published results are weakest —
18.0% executability at a **23.1% fake rate**. That is where retrieval is
necessary rather than decorative. Plan: `~/.claude/plans/`.

## Note

`get_metrics.py` was removed; `depinfer/evaluate.py` replaces it. The old scorer
read ground truth by scanning the patch for a `+dependencies = [` block, which
only exists in PEP 621 manifests — it extracted nothing for the 38 Poetry repos
(39% of the subset). It also matched with substring containment
(`any(pkg in name ...)`), so `requests` matched `requests-toolbelt`, and skipped
DI-Bench's `-`/`_` normalization. Numbers it produced are not comparable to the
table above.

`rag_analyzer.py` is the original single-file pipeline, superseded by `depinfer/`.
Its retrieval layer built a FAISS index over the ~15 PyPI documents it already
held in memory and then retrieved all of them against a fixed query — retrieval
over a corpus small enough to pass directly. It is kept for reference and can be
deleted.
