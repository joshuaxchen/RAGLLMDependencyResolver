#!/usr/bin/env python3
"""Run dependency inference over the DI-Bench regular Python subset and score it.

Replaces the interactive ``input()`` prompt in the original rag_analyzer.py so
runs are scriptable and repeatable.

    python cli.py --method deterministic
    python cli.py --method llm --model qwen2.5-coder:7b --limit 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from depinfer.evaluate import (
    Score, aggregate, load_dataset, oracle_dependencies, save_report, score_instance,
)
from depinfer.extract import find_repositories
from depinfer.installcheck import check_dependencies
from depinfer.generate import OllamaBackend
from depinfer.manifest import write_dependencies
from depinfer.pipeline import infer_repository
from depinfer.resolve import PyPIClient

ROOT = Path(__file__).resolve().parent


def default_dataset() -> Path:
    """Ground truth location. Tracked at the repo root; data/ is a local copy."""
    for candidate in (
        ROOT / "dataset-dibench-regular.jsonl",
        ROOT / "data" / "dataset-dibench-regular.jsonl",
    ):
        if candidate.exists():
            return candidate
    return ROOT / "dataset-dibench-regular.jsonl"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-dir", type=Path, default=ROOT / "repo-data" / "python")
    p.add_argument("--dataset", type=Path, default=default_dataset())
    p.add_argument("--method", choices=["deterministic", "llm"], default="deterministic")
    p.add_argument("--model", default="qwen2.5-coder:7b")
    p.add_argument("--ollama-url", default="http://localhost:11434")
    p.add_argument("--limit", type=int, default=None, help="only process the first N repositories")
    p.add_argument("--workers", type=int, default=None,
                   help="parallel repositories (default: 6 deterministic, 1 for llm). "
                        "Ollama serves generate requests serially; issuing several "
                        "concurrently has been observed to wedge it, with every worker "
                        "blocked on a response that never arrives.")
    p.add_argument("--out", type=Path, default=None, help="output dir (default results/<method>)")
    p.add_argument("--no-pin", action="store_true", help="emit unpinned names, skip uv resolution")
    p.add_argument("--pypi-fallback", choices=["off", "lexical", "dense"], default="off",
                   help="resolve import names that map to no package. 'lexical' searches the "
                        "full PyPI name list; 'dense' queries the prebuilt embedding index "
                        "(run build_index.py first).")
    p.add_argument("--config-mining", action="store_true",
                   help="also mine CI workflows, [tool.*] blocks and README for packages "
                        "that are declared but never imported")
    p.add_argument("--write-manifests", action="store_true",
                   help="write filled pyproject.toml copies into the output dir")
    p.add_argument("--force", action="store_true",
                   help="overwrite an existing report even if it covers more instances")
    p.add_argument("--install-check", choices=["off", "compile", "venv"], default="off",
                   help="verify the predicted set installs. 'compile' resolves with uv "
                        "(seconds); 'venv' does a real pip install + pip check (minutes). "
                        "This is a proxy for executability, not the paper's Exec metric.")
    return p.parse_args(argv)


def existing_report_size(out_dir: Path) -> int:
    """Instance count of a report already in out_dir, or 0."""
    path = out_dir / "score_report.json"
    if not path.exists():
        return 0
    try:
        return len(json.loads(path.read_text()).get("instances", []))
    except (json.JSONDecodeError, OSError):
        return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    # A partial run must never clobber a full one: a --limit 3 run once
    # overwrote the 98-instance report, and RESULTS.md was nearly written
    # from three instances.
    tag = args.method + (f"-limit{args.limit}" if args.limit else "")
    out_dir = args.out or (ROOT / "results" / tag)
    if args.workers is None:
        args.workers = 1 if args.method == "llm" else 6

    if not args.dataset.exists():
        print(f"error: dataset not found at {args.dataset}", file=sys.stderr)
        print("download dataset-dibench-regular.jsonl from the DI-Bench v1.0 release", file=sys.stderr)
        return 1
    if not args.repo_dir.exists():
        print(f"error: repo dir not found at {args.repo_dir}", file=sys.stderr)
        return 1

    dataset = load_dataset(args.dataset)
    repos = find_repositories(args.repo_dir)
    if args.limit:
        repos = repos[: args.limit]

    # Fail before doing the work, not after.
    prior = existing_report_size(out_dir)
    if prior > len(repos) and not args.force:
        print(
            f"error: {out_dir / 'score_report.json'} covers {prior} instances; "
            f"this run would write {len(repos)} and lose data.\n"
            f"       pass --force to overwrite, or --out to write elsewhere.",
            file=sys.stderr,
        )
        return 1

    backend = None
    if args.method == "llm":
        backend = OllamaBackend(model=args.model, url=args.ollama_url)

    print(f"method={args.method}" + (f" model={args.model}" if backend else ""))
    print(f"repositories: {len(repos)}   dataset instances: {len(dataset)}")
    print(f"output: {out_dir}\n")

    fallback = None
    if args.pypi_fallback == "lexical":
        from depinfer.pypi_index import LexicalPyPIMatcher
        fallback = LexicalPyPIMatcher()
    elif args.pypi_fallback == "dense":
        from depinfer.pypi_index import DensePyPIIndex
        fallback = DensePyPIIndex()
        if not fallback.exists():
            print("error: dense index not built; run build_index.py", file=sys.stderr)
            return 1
        # Load the embedding model on this thread. Lazy-loading it from several
        # worker threads at once races inside torch and fails with
        # "Cannot copy out of meta tensor".
        fallback.lookup("warmup")

    client = PyPIClient()
    started = time.time()

    def process(repo: Path) -> tuple[Score, dict]:
        iid = repo.name
        record = dataset.get(iid)
        if record is None:
            return Score(instance_id=iid, error="not in dataset"), {}

        result = infer_repository(
            repo, client, method=args.method, backend=backend,
            pin_versions=not args.no_pin, config_mining=args.config_mining,
            fallback=fallback,
        )
        if result.error:
            return Score(instance_id=iid, error=result.error), asdict(result)

        try:
            oracle = oracle_dependencies(record, repo)
        except Exception as exc:
            return Score(instance_id=iid, error=f"oracle failed: {exc}"), asdict(result)

        score = score_instance(
            iid, result.dependencies, oracle,
            client=client, local_modules=set(result.first_party),
        )

        if args.install_check != "off":
            outcome = check_dependencies(result.dependencies, mode=args.install_check)
            score.install_ok = outcome.ok
            score.install_detail = outcome.detail[:500]

        if args.write_manifests:
            src = repo / "pyproject.toml"
            if src.exists():
                dst = out_dir / "manifests" / iid / "pyproject.toml"
                dst.parent.mkdir(parents=True, exist_ok=True)
                try:
                    dst.write_text(write_dependencies(src.read_text(), result.dependencies))
                except Exception as exc:
                    print(f"  ! {iid}: manifest write failed: {exc}")

        return score, asdict(result)

    scores: list[Score] = []
    predictions: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process, r): r for r in repos}
        for i, fut in enumerate(as_completed(futures), 1):
            repo = futures[fut]
            try:
                score, detail = fut.result()
            except Exception as exc:
                score, detail = Score(instance_id=repo.name, error=str(exc)), {}
            scores.append(score)
            predictions[score.instance_id] = detail
            flag = "ERR" if score.error else f"P={score.precision:.2f} R={score.recall:.2f}"
            print(f"[{i:3d}/{len(repos)}] {score.instance_id:45s} {flag}")

    scores.sort(key=lambda s: s.instance_id)
    summary = aggregate(scores)
    summary["method"] = args.method + ("+config" if args.config_mining else "")
    summary["model"] = args.model if args.method == "llm" else None
    summary["elapsed_seconds"] = round(time.time() - started, 1)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "predictions.json").write_text(json.dumps(predictions, indent=2))
    report_path = save_report(scores, summary, out_dir)

    print("\n" + "=" * 62)
    print(f"  METHOD: {args.method}" + (f" ({args.model})" if args.method == "llm" else ""))
    print("=" * 62)
    n = summary["name_only"]
    print(f"  Name-only   P {n['precision']:5.1f}   R {n['recall']:5.1f}   F1 {n['f1']:5.1f}")
    v = summary["versions"]
    print(f"  Versions    {v['compatible_rate']:5.1f}% compatible "
          f"({v['compatible']} ok / {v['incompatible']} bad / {v['unconstrained']} unconstrained)")
    if summary.get("install_proxy"):
        ip = summary["install_proxy"]
        print(f"  Installs    {ip['rate']:5.1f}% ({ip['installable']}/{ip['checked']}) "
              f"[proxy, not the paper's Exec metric]")
    print(f"  Fake rate   {summary['fake_rate']:5.1f}")
    print(f"  Macro F1    {summary['macro_f1']:5.1f}")
    print(f"  Perfect     {summary['perfect_instances']}/{summary['instances_scored']}")
    if summary["instances_errored"]:
        print(f"  Errored     {summary['instances_errored']}")
    print(f"  Elapsed     {summary['elapsed_seconds']}s")
    print("=" * 62)
    print("\n  Paper reference (GPT-4o, Imports-Only, Python regular):")
    print("    P  56.5   R  74.9   F1  64.4   fake rate 3.9")
    print(f"\n  report: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
