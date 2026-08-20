"""Score predicted dependencies against DI-Bench ground truth.

Mirrors the set arithmetic and name normalization used by the official harness
(``dibench/evaluate/evaluator.py``)::

    tp = len(model_deps_set.intersection(oracle_deps_set))
    fp = len(model_deps_set.difference(oracle_deps_set))
    fn = len(oracle_deps_set.difference(model_deps_set))

with names compared as ``name.lower().replace("-", "_")``.

Note this does NOT compute the paper's executability metric, which requires
running each repository's CI test suite under Sysbox (Linux-only). Numbers here
are textual plus an install proxy, and are not directly comparable to the
paper's 36.7%.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .manifest import Dependency, read_dependencies
from .resolve import PyPIClient


@dataclass
class Score:
    instance_id: str
    tp: int = 0
    fp: int = 0
    fn: int = 0
    exact_tp: int = 0
    fake: int = 0
    n_predicted: int = 0
    n_oracle: int = 0
    predicted: list[str] = field(default_factory=list)
    oracle: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    spurious: list[str] = field(default_factory=list)
    fake_names: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def load_dataset(jsonl_path: Path, language: str = "python") -> dict[str, dict]:
    records = {}
    with open(jsonl_path) as fh:
        for line in fh:
            rec = json.loads(line)
            if rec.get("language") == language:
                records[rec["instance_id"]] = rec
    return records


def oracle_dependencies(record: dict, repo_path: Path) -> list[Dependency]:
    """Ground truth = dependencies present after applying the instance patch.

    The patch is a unified diff restoring the masked runtime dependency
    section, so it is applied with `git apply` and the result parsed with a real
    TOML parser rather than scraped out of the diff text.
    """
    patch = record.get("patch") or ""
    if not patch.strip():
        return []

    with tempfile.TemporaryDirectory(prefix="depinfer_oracle_") as tmp:
        tmpdir = Path(tmp)
        for rel in record.get("build_files", []):
            src = repo_path / rel
            if src.exists():
                dst = tmpdir / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

        patch_file = tmpdir / "instance.patch"
        patch_file.write_text(patch if patch.endswith("\n") else patch + "\n")

        proc = subprocess.run(
            ["git", "apply", "--unsafe-paths", "-p1", str(patch_file)],
            cwd=tmpdir, capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"git apply failed: {(proc.stderr or proc.stdout).strip()[:200]}"
            )

        deps: list[Dependency] = []
        for rel in record.get("build_files", []):
            path = tmpdir / rel
            if path.exists() and path.name == "pyproject.toml":
                _, parsed = read_dependencies(path.read_text())
                deps.extend(parsed)
        return deps


def score_instance(
    instance_id: str,
    predicted: list[Dependency],
    oracle: list[Dependency],
    client: PyPIClient | None = None,
    local_modules: set[str] | None = None,
) -> Score:
    pred_by_key = {d.key: d for d in predicted}
    oracle_by_key = {d.key: d for d in oracle}
    pred_keys, oracle_keys = set(pred_by_key), set(oracle_by_key)

    s = Score(
        instance_id=instance_id,
        tp=len(pred_keys & oracle_keys),
        fp=len(pred_keys - oracle_keys),
        fn=len(oracle_keys - pred_keys),
        n_predicted=len(pred_keys),
        n_oracle=len(oracle_keys),
        predicted=sorted(d.to_pep508() for d in predicted),
        oracle=sorted(d.to_pep508() for d in oracle),
        missing=sorted(oracle_keys - pred_keys),
        spurious=sorted(pred_keys - oracle_keys),
    )

    # Exact match additionally requires the version constraint to agree.
    s.exact_tp = sum(
        1
        for k in pred_keys & oracle_keys
        if pred_by_key[k].version.replace(" ", "") == oracle_by_key[k].version.replace(" ", "")
    )

    if client is not None:
        local = local_modules or set()
        for key, dep in pred_by_key.items():
            if key in local:
                continue
            if not client.exists(dep.name):
                s.fake_names.append(dep.name)
        s.fake = len(s.fake_names)

    return s


def aggregate(scores: list[Score]) -> dict:
    """Micro-averaged metrics over all instances."""
    scored = [s for s in scores if s.error is None]
    tp = sum(s.tp for s in scored)
    fp = sum(s.fp for s in scored)
    fn = sum(s.fn for s in scored)
    exact_tp = sum(s.exact_tp for s in scored)
    n_pred = sum(s.n_predicted for s in scored)
    fake = sum(s.fake for s in scored)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "instances_scored": len(scored),
        "instances_errored": len(scores) - len(scored),
        "name_only": {
            "precision": round(precision * 100, 1),
            "recall": round(recall * 100, 1),
            "f1": round(f1 * 100, 1),
            "tp": tp, "fp": fp, "fn": fn,
        },
        "exact_match": {
            "precision": round(exact_tp / n_pred * 100, 1) if n_pred else 0.0,
            "recall": round(exact_tp / (tp + fn) * 100, 1) if (tp + fn) else 0.0,
            "tp": exact_tp,
        },
        "fake_rate": round(fake / n_pred * 100, 1) if n_pred else 0.0,
        "macro_f1": round(
            sum(s.f1 for s in scored) / len(scored) * 100, 1
        ) if scored else 0.0,
        "perfect_instances": sum(1 for s in scored if s.fn == 0 and s.fp == 0),
    }


def save_report(scores: list[Score], summary: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"summary": summary, "instances": [asdict(s) for s in scores]}
    path = out_dir / "score_report.json"
    path.write_text(json.dumps(report, indent=2))
    return path
