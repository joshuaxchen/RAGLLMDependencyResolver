"""Compare predicted version constraints against ground-truth constraints.

The scorer previously compared constraint *strings* literally, so `==6.0.3`
against `>=5.4.1, <7.0.0` counted as a mismatch. That made exact-match read
1.2% — an artifact of formatting, not a measure of correctness.

What matters is whether the predicted constraint is satisfiable within the
oracle's allowed range. `Zuehlke_ConfZ` scores a perfect 1.00 on names while
nothing checked that `pydantic==2.13.4` sits inside `>=1.9.0, <3.0.0`.
"""

from __future__ import annotations

import re

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

# Poetry constraint shorthands, which are not valid PEP 440.
_CARET = re.compile(r"^\^\s*([0-9][^,\s]*)$")
_TILDE = re.compile(r"^~\s*([0-9][^,\s]*)$")


def _bump_caret(version: str) -> str:
    """`^1.2.3` allows >=1.2.3,<2.0.0; `^0.2.3` allows >=0.2.3,<0.3.0."""
    parts = [int(p) for p in re.findall(r"\d+", version)[:3]]
    while len(parts) < 3:
        parts.append(0)
    major, minor, _ = parts
    if major > 0:
        return f">={version},<{major + 1}.0.0"
    if minor > 0:
        return f">={version},<0.{minor + 1}.0"
    return f">={version},<0.0.{parts[2] + 1}"


def _bump_tilde(version: str) -> str:
    """`~1.2.3` allows >=1.2.3,<1.3.0."""
    parts = [int(p) for p in re.findall(r"\d+", version)[:3]]
    while len(parts) < 3:
        parts.append(0)
    major, minor, _ = parts
    return f">={version},<{major}.{minor + 1}.0"


def normalize_constraint(raw: str) -> str:
    """Convert a constraint to PEP 440 form. Poetry carets/tildes are rewritten."""
    text = (raw or "").strip()
    if not text or text == "*":
        return ""
    caret = _CARET.match(text)
    if caret:
        return _bump_caret(caret.group(1))
    tilde = _TILDE.match(text)
    if tilde:
        return _bump_tilde(tilde.group(1))
    # A bare version ("1.2.3") means an exact pin in Poetry tables.
    if re.match(r"^\d[\w.\-+!]*$", text):
        return f"=={text}"
    return text


def parse(raw: str) -> SpecifierSet | None:
    try:
        return SpecifierSet(normalize_constraint(raw))
    except InvalidSpecifier:
        return None


def pinned_version(raw: str) -> Version | None:
    """The concrete version, when a constraint is a single `==` pin."""
    spec = parse(raw)
    if spec is None:
        return None
    pins = [s for s in spec if s.operator in ("==", "===")]
    if len(pins) != 1:
        return None
    try:
        return Version(pins[0].version.rstrip(".*"))
    except InvalidVersion:
        return None


def is_compatible(predicted: str, oracle: str) -> bool | None:
    """Is `predicted` allowed by `oracle`?

    Returns None when the comparison cannot be made (unparseable, or the oracle
    places no constraint), so 'unknown' is never silently counted as agreement.
    """
    oracle_spec = parse(oracle)
    if oracle_spec is None:
        return None
    if not str(oracle_spec):
        return None  # oracle allows anything; nothing to verify

    # Common case: we emit a pin, the oracle gives a range.
    pin = pinned_version(predicted)
    if pin is not None:
        return oracle_spec.contains(pin, prereleases=True)

    predicted_spec = parse(predicted)
    if predicted_spec is None or not str(predicted_spec):
        return None
    if str(predicted_spec) == str(oracle_spec):
        return True

    # Neither side is a single pin: probe the oracle's allowed versions for
    # overlap rather than comparing constraint text.
    return _ranges_overlap(predicted_spec, oracle_spec)


def _ranges_overlap(a: SpecifierSet, b: SpecifierSet) -> bool | None:
    """Approximate overlap test by probing candidate versions from both sides."""
    candidates: list[Version] = []
    for spec in list(a) + list(b):
        raw = spec.version.rstrip(".*")
        try:
            candidates.append(Version(raw))
        except InvalidVersion:
            continue
    if not candidates:
        return None
    probes = set()
    for version in candidates:
        probes.add(version)
        parts = list(version.release) + [0, 0]
        probes.add(Version(f"{parts[0]}.{parts[1]}.{parts[2] + 1}"))
        if parts[2] > 0:
            probes.add(Version(f"{parts[0]}.{parts[1]}.{parts[2] - 1}"))
        probes.add(Version(f"{parts[0]}.{parts[1] + 1}.0"))
    return any(
        a.contains(p, prereleases=True) and b.contains(p, prereleases=True)
        for p in probes
    )
