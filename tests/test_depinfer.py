"""Regression tests for the dependency inference pipeline.

Each test pins a bug that was present in the original rag_analyzer.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from depinfer.evaluate import (
    load_dataset, oracle_dependencies, score_instance,
)
from depinfer.extract import STDLIB, extract_imports, find_first_party
from depinfer.manifest import Dependency, parse_requirement, read_dependencies, write_dependencies
from depinfer.resolve import PyPIClient, normalize, resolve_distribution

ROOT = Path(__file__).resolve().parent.parent
REPO_DATA = ROOT / "repo-data" / "python"
DATASET = next(
    (
        p
        for p in (
            ROOT / "dataset-dibench-regular.jsonl",
            ROOT / "data" / "dataset-dibench-regular.jsonl",
        )
        if p.exists()
    ),
    ROOT / "dataset-dibench-regular.jsonl",
)

needs_data = pytest.mark.skipif(
    not REPO_DATA.exists() or not DATASET.exists(),
    reason="DI-Bench data not downloaded",
)
needs_network = pytest.mark.skipif(
    not (ROOT / ".cache" / "pypi").exists(),
    reason="no PyPI cache; run the pipeline once first",
)


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "mod.py").write_text(
        "import os, requests\n"
        "import numpy as np\n"
        "import a.b.c\n"
        "from yaml import safe_load\n"
        "from . import sibling\n"
        "from .relative import thing\n"
        "import mypkg.internal\n"
        "import tomllib\n"
        "if True:\n"
        "    import lazy_dep\n"
    )
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_x.py").write_text("import pytest\nimport requests\nimport testonly_pkg\n")
    vendored = tmp_path / ".venv" / "lib"
    vendored.mkdir(parents=True)
    (vendored / "junk.py").write_text("import should_not_appear\n")
    return tmp_path


# --- extraction -----------------------------------------------------------

def test_multiple_imports_on_one_line(sample_repo):
    """`import os, requests` used to yield only `os` and drop `requests`."""
    assert "requests" in extract_imports(sample_repo).runtime


def test_stdlib_and_relative_imports_excluded(sample_repo):
    modules = extract_imports(sample_repo).all_modules
    assert "os" not in modules
    assert "sibling" not in modules and "relative" not in modules


def test_version_skewed_stdlib_excluded(sample_repo):
    """tomllib is stdlib on 3.11+, but we may run on 3.10."""
    assert "tomllib" in STDLIB
    assert "tomllib" not in extract_imports(sample_repo).all_modules


def test_vendored_directories_skipped(sample_repo):
    assert "should_not_appear" not in extract_imports(sample_repo).all_modules


def test_conditional_imports_captured(sample_repo):
    assert "lazy_dep" in extract_imports(sample_repo).runtime


def test_runtime_and_test_imports_separated(sample_repo):
    scan = extract_imports(sample_repo)
    assert "testonly_pkg" in scan.test_only
    assert "pytest" in scan.test_only
    # imported by both tests and runtime code -> runtime wins
    assert "requests" in scan.runtime and "requests" not in scan.test_only


def test_first_party_excluded(sample_repo):
    assert "mypkg" in find_first_party(sample_repo)
    assert "mypkg" not in extract_imports(sample_repo).all_modules


@needs_data
def test_first_party_excluded_on_real_repo():
    repo = REPO_DATA / "Zuehlke_ConfZ"
    assert "confz" not in extract_imports(repo).all_modules


# --- name resolution ------------------------------------------------------

@needs_network
@pytest.mark.parametrize(
    "module,distribution",
    [
        ("yaml", "PyYAML"),
        ("sklearn", "scikit-learn"),
        ("cv2", "opencv-python"),
        ("PIL", "Pillow"),
        ("bs4", "beautifulsoup4"),
        ("requests", "requests"),
    ],
)
def test_import_name_maps_to_distribution(module, distribution):
    """Exact-name PyPI lookup used to 404 on all of these and drop them."""
    assert resolve_distribution(module, PyPIClient()).lower() == distribution.lower()


def test_normalize_matches_dibench():
    assert normalize("Foo-Bar") == "foo_bar"


# --- manifests ------------------------------------------------------------

def test_parse_requirement_strips_extras_and_markers():
    dep = parse_requirement("uvicorn[standard]>=0.20 ; python_version >= '3.8'")
    assert dep.name == "uvicorn" and dep.version == ">=0.20"


@needs_data
@pytest.mark.parametrize(
    "instance,fmt", [("Zuehlke_ConfZ", "poetry"), ("Tigge_openant", "pep621")]
)
def test_manifest_round_trip(instance, fmt):
    path = REPO_DATA / instance / "pyproject.toml"
    original = path.read_text()
    detected, _ = read_dependencies(original)
    assert detected == fmt

    updated = write_dependencies(original, [Dependency("pydantic", ">=1.9.0")])
    _, deps = read_dependencies(updated)
    names = {d.name for d in deps}
    assert "pydantic" in names
    assert "python" not in names  # poetry's interpreter constraint is not a dep


# --- scoring --------------------------------------------------------------

@needs_data
def test_oracle_matches_known_ground_truth():
    dataset = load_dataset(DATASET)
    deps = oracle_dependencies(dataset["Zuehlke_ConfZ"], REPO_DATA / "Zuehlke_ConfZ")
    assert {d.name.lower() for d in deps} == {
        "pydantic", "pyyaml", "python-dotenv", "toml",
    }


@needs_data
def test_all_oracles_extractable():
    dataset = load_dataset(DATASET)
    for iid, record in dataset.items():
        repo = REPO_DATA / iid
        if repo.exists():
            oracle_dependencies(record, repo)  # must not raise


def test_perfect_prediction_scores_one():
    deps = [Dependency("a"), Dependency("b")]
    score = score_instance("x", deps, deps)
    assert (score.precision, score.recall, score.f1) == (1.0, 1.0, 1.0)


def test_scorer_arithmetic():
    predicted = [Dependency("a"), Dependency("b"), Dependency("wrong")]
    oracle = [Dependency("a"), Dependency("b"), Dependency("c"), Dependency("d")]
    s = score_instance("x", predicted, oracle)
    assert (s.tp, s.fp, s.fn) == (2, 1, 2)
    assert s.precision == pytest.approx(2 / 3)
    assert s.recall == pytest.approx(0.5)


def test_scoring_is_name_normalized():
    s = score_instance("x", [Dependency("Foo-Bar")], [Dependency("foo_bar")])
    assert s.tp == 1


# --- serialization --------------------------------------------------------

@needs_data
def test_results_are_json_serializable(tmp_path):
    """The original stored a PosixPath and crashed in json.dump."""
    from depinfer.evaluate import save_report, aggregate
    scores = [score_instance("x", [Dependency("a")], [Dependency("a")])]
    path = save_report(scores, aggregate(scores), tmp_path)
    assert json.loads(path.read_text())["summary"]["instances_scored"] == 1


# --- version compatibility ------------------------------------------------

@pytest.mark.parametrize(
    "predicted,oracle,expected",
    [
        ("==6.0.3", ">=5.4.1, <7.0.0", True),
        ("==3.5.0", ">=1.9.0, <3.0.0", False),   # the ConfZ failure mode
        ("==0.10.2", "^0.10.2", True),           # poetry caret, 0.x
        ("==0.11.0", "^0.10.2", False),
        ("==1.9.0", "^1.2.3", True),             # poetry caret, 1.x
        ("==2.0.0", "^1.2.3", False),
        ("==1.2.9", "~1.2.3", True),             # poetry tilde
        ("==1.3.0", "~1.2.3", False),
        (">=1.0,<2.0", ">=1.5,<3.0", True),      # overlapping ranges
        (">=1.0,<1.4", ">=1.5,<3.0", False),     # disjoint ranges
    ],
)
def test_version_compatibility(predicted, oracle, expected):
    from depinfer.versions import is_compatible
    assert is_compatible(predicted, oracle) is expected


def test_unconstrained_oracle_is_unknown_not_agreement():
    """An oracle with no constraint must not be scored as a match."""
    from depinfer.versions import is_compatible
    assert is_compatible("==1.0", "") is None


def test_requires_python_extracted():
    from depinfer.manifest import read_requires_python
    assert read_requires_python('[project]\nrequires-python = ">=3.8"\n') == "3.8"
    assert read_requires_python(
        '[tool.poetry]\nname="x"\n[tool.poetry.dependencies]\npython = "^3.7.2"\n'
    ) == "3.7"


# --- install proxy --------------------------------------------------------

@needs_network
def test_install_proxy_accepts_valid_set():
    from depinfer.installcheck import check_dependencies
    assert check_dependencies([Dependency("requests"), Dependency("click")]).ok


@needs_network
def test_install_proxy_detects_conflicts():
    """requirementstest.txt exists in the repo with deliberate conflicts."""
    from depinfer.installcheck import check_dependencies
    deps = [d for d in (parse_requirement(l)
            for l in (ROOT / "requirementstest.txt").read_text().splitlines()) if d]
    assert not check_dependencies(deps).ok


# --- config mining --------------------------------------------------------

def test_config_mining_reads_ci_installs(tmp_path):
    from depinfer.config_mine import mine_repository
    wf = tmp_path / ".github" / "workflows"
    wf.mkdir(parents=True)
    (wf / "ci.yml").write_text("jobs:\n  t:\n    steps:\n      - run: pip install boto3 pytest-cov\n")
    packages = {p.lower() for p in mine_repository(tmp_path).packages}
    assert "boto3" in packages and "pytest-cov" in packages


def test_config_mining_infers_pytest_cov_from_addopts(tmp_path):
    from depinfer.config_mine import mine_repository
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname="x"\n[tool.pytest.ini_options]\naddopts = "--cov=x"\n'
    )
    assert "pytest-cov" in {p.lower() for p in mine_repository(tmp_path).packages}


# --- pypi fallback --------------------------------------------------------

@needs_network
@pytest.mark.parametrize(
    "module,distribution",
    [
        ("digitalocean", "python-digitalocean"),
        ("allauth", "django-allauth"),
        ("tagmatcher", "tag-matcher"),
        ("lightstreamer", "lightstreamer-client-lib"),
    ],
)
def test_lexical_fallback_recovers_known_cases(module, distribution):
    from depinfer.pypi_index import LexicalPyPIMatcher
    assert distribution in [m.lower() for m in LexicalPyPIMatcher().lookup(module)]


@needs_network
def test_fallback_cannot_invent_packages():
    """Every candidate is validated against PyPI before being accepted."""
    from depinfer.pypi_index import LexicalPyPIMatcher
    from depinfer.resolve import PyPIClient, resolve_distribution
    got = resolve_distribution(
        "zzz_not_a_real_module_xyz", PyPIClient(), fallback=LexicalPyPIMatcher()
    )
    assert got is None
