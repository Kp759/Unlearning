#!/usr/bin/env python3
"""Pinned RWKU data download, validation, and deterministic target splits.

The upstream RWKU benchmark is target-centric: one real-world person is
unlearned at a time.  This project uses seeds 0--9 as ten independent
single-target runs over the first ten targets in RWKU's published order.

Probe-derived objectives may use only the calibration probes.  The pinned
``positive.json`` file is a separately designated upstream training corpus,
not an evaluation source.  Headline direct-QA and paraphrase metrics are
computed on the complementary held-out probes.  The partition is deterministic
and recorded by content hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import urllib.parse
import urllib.request
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple


RWKU_REPOSITORY = "https://github.com/jinzhuoran/RWKU"
RWKU_CODE_REVISION = "b8a03b3ce34fb4a96001df545a56558d75a078a3"
RWKU_DATASET_REPOSITORY = "https://huggingface.co/datasets/jinzhuoran/RWKU"
RWKU_DATASET_REVISION = "d72f493d481d1b0a9bdc6e869d32baeffad8904f"
RWKU_RESOLVE_ROOT = (
    f"{RWKU_DATASET_REPOSITORY}/resolve/{RWKU_DATASET_REVISION}"
)

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "rwku"
PINNED_MANIFEST_PATH = DEFAULT_DATA_ROOT / "manifest.json"


@dataclass(frozen=True)
class TargetSpec:
    seed: int
    directory: str
    subject: str


TARGETS_BY_SEED: Tuple[TargetSpec, ...] = (
    TargetSpec(0, "1_Stephen_King", "Stephen King"),
    TargetSpec(1, "2_Confucius", "Confucius"),
    TargetSpec(2, "3_Bruce_Lee", "Bruce Lee"),
    TargetSpec(3, "4_Warren_Buffett", "Warren Buffett"),
    TargetSpec(4, "5_Christina_Aguilera", "Christina Aguilera"),
    TargetSpec(5, "6_Cindy_Crawford", "Cindy Crawford"),
    TargetSpec(6, "7_Marie_Osmond", "Marie Osmond"),
    TargetSpec(7, "8_Paris_Hilton", "Paris Hilton"),
    TargetSpec(8, "9_Justin_Bieber", "Justin Bieber"),
    TargetSpec(
        9,
        "10_Prince_Harry,_Duke_of_Sussex",
        "Prince Harry, Duke of Sussex",
    ),
)

REQUIRED_FILES: Tuple[str, ...] = (
    "intro.json",
    "forget_level1.json",
    "forget_level2.json",
    "forget_level3.json",
    "neighbor_level1.json",
    "neighbor_level2.json",
    "forget_mia.json",
    "retain_mia.json",
    "retain_mmlu.json",
    "retain_bbh.json",
    "truthful.json",
    "triviaqa.json",
    "fluency.json",
    "positive.json",
)

EXPECTED_KEYS: Mapping[str, frozenset[str]] = {
    "intro.json": frozenset({"target", "intro"}),
    "forget_level1.json": frozenset(
        {"query", "answer", "level", "type", "subject"}
    ),
    "forget_level2.json": frozenset(
        {"query", "answer", "level", "type", "subject"}
    ),
    "forget_level3.json": frozenset(
        {"query", "answer", "level", "type", "subject"}
    ),
    "neighbor_level1.json": frozenset(
        {"query", "answer", "level", "type", "subject", "neighbor"}
    ),
    "neighbor_level2.json": frozenset(
        {"query", "answer", "level", "type", "subject", "neighbor"}
    ),
    "forget_mia.json": frozenset({"text", "ngram", "subject"}),
    "retain_mia.json": frozenset({"text", "ngram", "subject"}),
    "retain_mmlu.json": frozenset(
        {"task", "question", "choices", "answer", "examples", "subject"}
    ),
    "retain_bbh.json": frozenset(
        {"task", "question", "answer", "cot", "subject"}
    ),
    "truthful.json": frozenset(
        {"question", "mc1_targets", "mc2_targets", "subject"}
    ),
    "triviaqa.json": frozenset({"question", "answers", "subject"}),
    "fluency.json": frozenset({"instruction", "subject"}),
    "positive.json": frozenset({"text", "subject"}),
}


def target_for_seed(seed: int) -> TargetSpec:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 9:
        raise ValueError(f"RWKU seed must be an integer from 0 through 9, got {seed!r}")
    target = TARGETS_BY_SEED[seed]
    if target.seed != seed:
        raise RuntimeError("RWKU target table is not indexed by seed")
    return target


def canonical_record(record: Mapping[str, Any]) -> str:
    return json.dumps(
        record,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def record_sha256(record: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_record(record).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_url(target: TargetSpec, filename: str) -> str:
    relative = f"Target/{target.directory}/{filename}"
    quoted = urllib.parse.quote(relative, safe="/")
    return f"{RWKU_RESOLVE_ROOT}/{quoted}?download=true"


def download_file(url: str, destination: Path) -> None:
    """Download atomically so an interrupted job cannot leave valid-looking data."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        with urllib.request.urlopen(url) as response, temporary.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_json_list(
    path: Path,
    *,
    allow_singleton_object: bool = False,
) -> List[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{path} is not JSON. A Git-LFS pointer was likely downloaded "
            "instead of the resolved RWKU object."
        ) from exc
    if allow_singleton_object and isinstance(value, dict):
        value = [value]
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path} must contain a non-empty JSON list")
    if not all(isinstance(row, dict) for row in value):
        raise ValueError(f"{path} contains a non-object row")
    return [dict(row) for row in value]


def validate_file(
    filename: str,
    rows: Sequence[Mapping[str, Any]],
    target: TargetSpec,
) -> None:
    expected = EXPECTED_KEYS[filename]
    missing = expected - set(rows[0])
    if missing:
        raise ValueError(f"{filename} is missing required keys: {sorted(missing)}")

    if filename == "intro.json":
        subjects = {str(row.get("target", "")) for row in rows}
    else:
        subjects = {
            str(row["subject"])
            for row in rows
            if isinstance(row.get("subject"), str)
        }
    if subjects and subjects != {target.subject}:
        raise ValueError(
            f"{filename} does not belong exclusively to {target.subject!r}: "
            f"{sorted(subjects)!r}"
        )

    if filename == "forget_level1.json" and any(
        str(row.get("level")) != "1" for row in rows
    ):
        raise ValueError("forget_level1.json contains a non-level-1 probe")
    if filename == "forget_level2.json" and any(
        str(row.get("level")) != "2" for row in rows
    ):
        raise ValueError("forget_level2.json contains a non-level-2 probe")
    if filename == "forget_level3.json" and any(
        str(row.get("level")) != "3" for row in rows
    ):
        raise ValueError("forget_level3.json contains a non-level-3 probe")


def pinned_target_manifest(seed: int) -> Mapping[str, Any]:
    """Return the committed hash/count contract for one target."""

    if not PINNED_MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"Missing committed RWKU manifest: {PINNED_MANIFEST_PATH}"
        )
    with PINNED_MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if (
        manifest.get("dataset_revision") != RWKU_DATASET_REVISION
        or manifest.get("upstream_code_revision") != RWKU_CODE_REVISION
    ):
        raise ValueError("Committed RWKU manifest revision is not the pinned revision")
    matches = [
        row
        for row in manifest.get("targets", [])
        if int(row.get("seed", -1)) == seed
    ]
    if len(matches) != 1:
        raise ValueError(f"Committed RWKU manifest has no unique seed {seed}")
    return matches[0]


def verify_pinned_file(
    *,
    seed: int,
    filename: str,
    rows: Sequence[Mapping[str, Any]],
    sha256: str,
) -> None:
    expected = pinned_target_manifest(seed)
    expected_count = int(expected["counts"][filename])
    expected_sha256 = str(expected["sha256"][filename])
    if len(rows) != expected_count or sha256 != expected_sha256:
        raise ValueError(
            f"{filename} for RWKU seed {seed} differs from the committed "
            "pinned manifest: "
            f"count={len(rows)} (expected {expected_count}), "
            f"sha256={sha256} (expected {expected_sha256})"
        )


def ensure_target_data(
    data_root: Path,
    seed: int,
    *,
    allow_download: bool = True,
) -> Tuple[TargetSpec, Dict[str, List[Dict[str, Any]]], Dict[str, str]]:
    target = target_for_seed(seed)
    target_root = Path(data_root) / "Target" / target.directory
    datasets: Dict[str, List[Dict[str, Any]]] = {}
    hashes: Dict[str, str] = {}
    for filename in REQUIRED_FILES:
        destination = target_root / filename
        if not destination.is_file():
            if not allow_download:
                raise FileNotFoundError(
                    f"Missing pinned RWKU file with downloads disabled: {destination}"
                )
            print(f"Downloading pinned RWKU {target.directory}/{filename}")
            download_file(_download_url(target, filename), destination)
        rows = load_json_list(
            destination,
            allow_singleton_object=filename == "intro.json",
        )
        validate_file(filename, rows, target)
        digest = file_sha256(destination)
        verify_pinned_file(
            seed=seed,
            filename=filename,
            rows=rows,
            sha256=digest,
        )
        datasets[filename] = rows
        hashes[filename] = digest
    return target, datasets, hashes


def ensure_fact_assignment_data(
    data_root: Path,
    seed: int,
    *,
    allow_download: bool = True,
) -> Tuple[TargetSpec, Dict[str, List[Dict[str, Any]]], Dict[str, str]]:
    """Load only Level 1/2 records permitted for probe-assisted preparation.

    This deliberately does not iterate over :data:`REQUIRED_FILES`.  Keeping a
    separate loader makes it mechanically impossible for the entity-fact
    builder to open Level 3, MIA, neighbor, utility, or fluency data by
    accident.
    """

    target = target_for_seed(seed)
    target_root = Path(data_root) / "Target" / target.directory
    datasets: Dict[str, List[Dict[str, Any]]] = {}
    hashes: Dict[str, str] = {}
    for filename in ("forget_level1.json", "forget_level2.json"):
        destination = target_root / filename
        if not destination.is_file():
            if not allow_download:
                raise FileNotFoundError(
                    f"Missing pinned RWKU file with downloads disabled: {destination}"
                )
            print(f"Downloading pinned RWKU {target.directory}/{filename}")
            download_file(_download_url(target, filename), destination)
        rows = load_json_list(destination)
        validate_file(filename, rows, target)
        digest = file_sha256(destination)
        verify_pinned_file(
            seed=seed,
            filename=filename,
            rows=rows,
            sha256=digest,
        )
        datasets[filename] = rows
        hashes[filename] = digest
    return target, datasets, hashes


def ensure_positive_training_data(
    data_root: Path,
    seed: int,
    *,
    allow_download: bool = True,
) -> Tuple[TargetSpec, List[Dict[str, Any]], str]:
    """Load only the pinned ``positive.json`` training corpus for a target."""

    target = target_for_seed(seed)
    destination = (
        Path(data_root) / "Target" / target.directory / "positive.json"
    )
    if not destination.is_file():
        if not allow_download:
            raise FileNotFoundError(
                f"Missing pinned RWKU file with downloads disabled: {destination}"
            )
        print(f"Downloading pinned RWKU {target.directory}/positive.json")
        download_file(_download_url(target, "positive.json"), destination)
    rows = load_json_list(destination)
    validate_file("positive.json", rows, target)
    digest = file_sha256(destination)
    verify_pinned_file(
        seed=seed,
        filename="positive.json",
        rows=rows,
        sha256=digest,
    )
    return target, rows, digest


def partition_records(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    calibration_fraction: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Content-hash partition with non-empty calibration and held-out sides."""

    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be strictly between 0 and 1")
    if len(records) < 2:
        raise ValueError("At least two records are required for a held-out split")
    # Some published target files contain exact duplicate probes.  Treat every
    # duplicate-content group as an indivisible unit; otherwise one copy could
    # calibrate the method while another identical copy appears in held-out
    # evaluation.
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(record_sha256(record), []).append(dict(record))
    if len(groups) < 2:
        raise ValueError(
            "At least two unique records are required for a held-out split"
        )
    keyed_groups = sorted(
        (
            hashlib.sha256(f"{seed}:{digest}".encode("utf-8")).hexdigest(),
            digest,
            group,
        )
        for digest, group in groups.items()
    )
    calibration_group_count = int(
        math.floor(len(keyed_groups) * calibration_fraction)
    )
    calibration_group_count = max(
        1,
        min(len(keyed_groups) - 1, calibration_group_count),
    )
    calibration = [
        record
        for _, _, group in keyed_groups[:calibration_group_count]
        for record in group
    ]
    held_out = [
        record
        for _, _, group in keyed_groups[calibration_group_count:]
        for record in group
    ]
    return calibration, held_out


def build_split_manifest(
    calibration: Sequence[Mapping[str, Any]],
    held_out: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    calibration_hashes = [record_sha256(row) for row in calibration]
    held_out_hashes = [record_sha256(row) for row in held_out]
    overlap = set(calibration_hashes) & set(held_out_hashes)
    if overlap:
        raise RuntimeError("RWKU calibration and held-out records overlap")
    return {
        "calibration_count": len(calibration_hashes),
        "held_out_count": len(held_out_hashes),
        "calibration_record_sha256": calibration_hashes,
        "held_out_record_sha256": held_out_hashes,
        "disjoint": True,
    }


def paraphrase_query(query: str) -> str:
    """Deterministic surface paraphrase for held-out direct questions."""

    stripped = query.strip()
    if not stripped:
        return stripped
    lower = stripped.lower()
    replacements = (
        ("what is the name of ", "Identify "),
        ("what was the name of ", "Identify "),
        ("what is ", "State "),
        ("what was ", "State "),
        ("who is ", "Name the person who is "),
        ("who was ", "Name the person who was "),
        ("who did ", "Name who "),
        ("when did ", "At what time did "),
        ("where did ", "At what place did "),
        ("which ", "Identify which "),
        ("how many ", "State the number of "),
    )
    for prefix, replacement in replacements:
        if lower.startswith(prefix):
            tail = stripped[len(prefix) :]
            if tail.endswith("?"):
                tail = tail[:-1]
            return f"{replacement}{tail}?"
    return f"In different words, provide the answer to this question: {stripped}"


def _ascii_surface(value: str) -> str:
    return (
        unicodedata.normalize("NFKD", value)
        .encode("ascii", "ignore")
        .decode("ascii")
    )


def answer_aliases(answer: str, *, subject: str) -> List[str]:
    """Generate auditable surface aliases without consulting evaluation labels.

    RWKU does not publish a dedicated answer-alias column.  The control
    therefore reports coverage and only uses deterministic surface variants:
    ASCII folding, punctuation/parenthetical removal, initials, and (for an
    answer equal to the target person) the final name component.
    """

    original = " ".join(str(answer).strip().split())
    if not original:
        return []
    candidates: List[str] = []
    ascii_value = _ascii_surface(original)
    if ascii_value and ascii_value != original:
        candidates.append(ascii_value)
    without_parenthetical = original.split("(", 1)[0].strip()
    if without_parenthetical and without_parenthetical != original:
        candidates.append(without_parenthetical)
    punctuation_folded = (
        original.replace(".", "")
        .replace(",", "")
        .replace("-", " ")
        .replace("’", "'")
    )
    punctuation_folded = " ".join(punctuation_folded.split())
    if punctuation_folded and punctuation_folded != original:
        candidates.append(punctuation_folded)

    words = [word.strip(".,'’\"()") for word in original.split()]
    words = [word for word in words if word]
    if len(words) >= 2:
        initials = "".join(word[0] for word in words if word[0].isalpha())
        if len(initials) >= 2:
            candidates.extend([initials, ".".join(initials) + "."])
    if original.casefold() == " ".join(subject.split()).casefold() and len(words) >= 2:
        candidates.append(words[-1])

    seen = {original.casefold()}
    aliases: List[str] = []
    for candidate in candidates:
        normalized = " ".join(candidate.strip().split())
        key = normalized.casefold()
        if normalized and key not in seen:
            seen.add(key)
            aliases.append(normalized)
    return aliases


def adversarial_type(value: str) -> str:
    normalized = " ".join(str(value).strip().lower().replace("-", " ").split())
    if normalized.startswith("cross lingual"):
        return "cross lingual"
    aliases = {
        "context hint": "background hint",
        "incontext learning": "in-context learning",
    }
    return aliases.get(normalized, normalized)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def validate_manifest_destination(
    seeds: Sequence[int],
    destination: Path,
) -> None:
    """Prevent a subset validation from replacing the full trust anchor."""

    if len(set(seeds)) != len(seeds):
        raise ValueError("--seeds contains duplicates")
    for seed in seeds:
        target_for_seed(seed)
    if (
        Path(destination).resolve() == PINNED_MANIFEST_PATH.resolve()
        and list(seeds) != list(range(10))
    ):
        raise ValueError(
            "Refusing to overwrite the committed full RWKU manifest with a "
            "subset; pass a different --manifest path"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--seeds",
        default="0,1,2,3,4,5,6,7,8,9",
        help="Comma-separated seed/target indices to download and validate.",
    )
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_DATA_ROOT / "manifest.json",
    )
    args = parser.parse_args()

    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        raise ValueError("--seeds must select at least one RWKU seed")
    validate_manifest_destination(seeds, args.manifest)
    manifest: MutableMapping[str, Any] = {
        "dataset": "RWKU",
        "upstream_repository": RWKU_REPOSITORY,
        "upstream_code_revision": RWKU_CODE_REVISION,
        "dataset_repository": RWKU_DATASET_REPOSITORY,
        "dataset_revision": RWKU_DATASET_REVISION,
        "targets": [],
    }
    for seed in seeds:
        target, datasets, hashes = ensure_target_data(
            args.data_root,
            seed,
            allow_download=not args.no_download,
        )
        manifest["targets"].append(
            {
                "seed": seed,
                "directory": target.directory,
                "subject": target.subject,
                "counts": {
                    filename: len(rows) for filename, rows in datasets.items()
                },
                "sha256": hashes,
            }
        )
    write_json(args.manifest, manifest)
    print(f"Validated pinned RWKU data; manifest written to {args.manifest}")


if __name__ == "__main__":
    main()
