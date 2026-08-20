# Experimental results

DI-Bench regular Python subset, 98 repositories, 537 ground-truth dependencies
(mean 5.5 per repo, median 4, max 22). The paper reports 5.5 for this subset,
which is an independent check that ground-truth extraction is correct.

Reproduce with:

```bash
python cli.py --method deterministic     # ~4s
python cli.py --method llm               # ~31min, serialized
```

## Setup

| | |
|---|---|
| Dataset | `dataset-dibench-regular.jsonl` (DI-Bench v1.0), Python instances |
| Ground truth | instance patch applied with `git apply`, result parsed with tomlkit |
| Metric | micro-averaged P/R/F1 over dependency name sets |
| Normalization | `name.lower().replace("-", "_")`, matching the official harness |
| Model | qwen2.5-coder:7b via Ollama, temperature 0, structured JSON output |
| Versions | `uv pip compile`, never model-generated |

**What these numbers are not.** The paper's headline metric is *executability* —
whether the repository's CI test suite passes with the inferred dependencies.
That requires [Sysbox](https://github.com/nestybox/sysbox), which is Linux-only
and cannot run on macOS. Everything below is textual. The published P/R/F1
figures are directly comparable; the 42.9% executability figure is not.

## Headline

| Method | P | R | F1 | Fake rate | Perfect | Runtime |
|---|---|---|---|---|---|---|
| **`deterministic` (no LLM)** | **67.0** | **80.8** | **73.2** | 0.0 | **20/98** | **4s** |
| `llm` (qwen2.5-coder:7b) | 65.2 | 78.2 | 71.1 | 0.0 | 17/98 | 1889s |
| GPT-4o Imports-Only *(paper)* | 56.5 | 74.9 | 64.4 | 3.9 | — | — |
| GPT-4o All-In-One *(paper)* | 61.8 | 73.6 | 67.2 | 2.8 | — | — |

Counts: deterministic TP 434 / FP 214 / FN 103; llm TP 420 / FP 224 / FN 117.

Per-instance F1 spread (deterministic):

| F1 | repos |
|---|---|
| 1.00 | 20 |
| 0.75–0.99 | 37 |
| 0.50–0.74 | 33 |
| 0.01–0.49 | 6 |
| 0.00 | 2 |

**The fake rate of 0.0 is structural, not earned.** Predictions are constrained
to candidates that resolved to a real PyPI distribution, so an unresolvable name
cannot be emitted. It is not evidence of better judgment than the LLM baselines,
and should not be read as a win over their 2.8–3.9.

## Does the LLM help?

No. Adding a 7B model on top of the same candidate set costs 2.1 F1 and runs
200x slower.

| | |
|---|---|
| Instances where LLM scored higher | 9 |
| Instances where LLM scored lower | 22 |
| Instances identical | 67 |
| Packages added vs deterministic | 28 |
| Packages dropped vs deterministic | 32 |

On two thirds of repositories the model changes nothing. Where it does act it is
more than twice as likely to hurt as to help, and its net effect is to trade 14
true positives for 10 extra false positives.

**Scope.** This is one 7B model, textual metrics, the regular subset. A stronger
model may behave differently, and this says nothing about the large subset where
repositories exceed the context window and the task genuinely changes shape. What
it does show is that for context-sized repositories the work is being done by
static analysis, import-name resolution and the version resolver — not by
generation.

## Where the errors are

Top false negatives (missed): `pytest` 3, `ipykernel` 3, `pydantic` 2, `jaxlib` 2,
`werkzeug` 2, `pyarrow` 2, `setuptools` 2, `boto3` 2.

Top false positives (spurious): `numpy` 9, `requests` 6, `typing_extensions` 5,
`botocore` 4, `pillow` 4, `transformers` 4, `pydantic` 4, `boto3` 3.

### The recall ceiling is structural

**94 of 103 missed dependencies (91%) were never candidates at all** — they are
not imported anywhere in the repository, so no amount of better import analysis
can find them. These are build backends, test plugins activated by configuration
rather than import (`pytest-cov`), tools invoked only from CI, and optional
backends.

This bounds the approach: import analysis alone cannot exceed roughly 91% recall
on this subset regardless of how well the mapping and filtering work. Further
recall requires a different signal — CI workflow files, `[tool.*]` configuration
blocks, entry points — not a better parser.

Relevant here: DI-Bench masks only the *runtime* dependency section. Of ~660
restored lines, 651 land in `dependencies = [...]` or
`[tool.poetry.dependencies]` and 4 in `[project.optional-dependencies]`. Dev and
test groups stay unmasked in 73 of 98 repositories, and all 98 keep their
`.github/workflows/`. That surviving configuration is exactly where the invisible
9% of dependencies is described.

### Precision is dominated by transitive imports

False positives outnumber false negatives 214 to 103. The largest contributors
(`numpy`, `requests`, `botocore`) are packages a repository imports directly but
does not declare, because another declared dependency supplies them.

## Negative result: transitive filtering does not work

The obvious fix is to drop any predicted package that is required by another
predicted package. Measured before implementing:

| | Precision | Recall |
|---|---|---|
| Before | 67.0 | 79.5 |
| After transitive filtering | 67.8 | **54.6** |

It removes 71 false positives — and 134 true positives, because projects
routinely declare a dependency that is *also* transitive (a project depending on
both `pandas` and `numpy` is normal and correct). Precision barely moves while
recall collapses. **Not implemented, deliberately.**

## Version accuracy

Exact (version-sensitive) precision is 1.2% for both methods. `uv` emits
`==X.Y.Z` pins while ground truth mostly uses ranges (`>=1.9.0, <3.0.0`), so this
is largely a formatting mismatch rather than a correctness one. Comparing version
*ranges* for compatibility rather than string equality would be needed to make
this metric meaningful.

The paper's oracle-metadata ablation shows correct version metadata is worth
42.9% → 55.1% executability on Python, so this is real headroom — but it cannot
be measured with the textual metrics available here.

## What follows

The remaining recall is not reachable by import analysis, and the LLM is not
currently paying for itself. Two directions with measurable targets:

1. **Mine the unmasked configuration** — CI workflows and `[tool.*]` blocks — to
   reach the 91% of misses that are invisible to import analysis.
2. **The large subset**, where repositories exceed the context window and
   published results are weakest: 18.0% executability at a **23.1% fake rate**
   for the best method. That is where retrieval is necessary rather than
   decorative, and where hallucination is a real problem to attack rather than a
   number that is zero by construction.
