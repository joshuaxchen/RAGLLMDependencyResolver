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
    p.add_argument("--write-manifests", action="store_true",
                   help="write filled pyproject.toml copies into the output dir")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    out_dir = args.out or (ROOT / "results" / args.method)
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

    backend = None
    if args.method == "llm":
        backend = OllamaBackend(model=args.model, url=args.ollama_url)

    print(f"method={args.method}" + (f" model={args.model}" if backend else ""))
    print(f"repositories: {len(repos)}   dataset instances: {len(dataset)}")
    print(f"output: {out_dir}\n")

    client = PyPIClient()
    started = time.time()

    def process(repo: Path) -> tuple[Score, dict]:
        iid = repo.name
        record = dataset.get(iid)
        if record is None:
            return Score(instance_id=iid, error="not in dataset"), {}

        result = infer_repository(
            repo, client, method=args.method, backend=backend,
            pin_versions=not args.no_pin,
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
    summary["method"] = args.method
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
    e = summary["exact_match"]
    print(f"  Exact       P {e['precision']:5.1f}   R {e['recall']:5.1f}")
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
