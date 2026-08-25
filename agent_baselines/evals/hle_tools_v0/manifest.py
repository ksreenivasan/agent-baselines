import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .dataset import (
    HLE_DATASET_REPO,
    HLE_DATASET_REVISION,
    HLE_EVAL_CLASS,
    HLE_EXPECTED_COUNT,
    load_hle_verified_rows,
)

PILOT_SEED = "hle-tools-v0-hle-verified-gold-pilot-50"
SMOKE_SEED = "hle-tools-v0-hle-verified-gold-smoke-1"


def _rank(seed: str, sample_id: str) -> str:
    return hashlib.sha256(f"{seed}\0{sample_id}".encode()).hexdigest()


def _largest_remainder(counts: Counter[tuple[str, str]], size: int) -> dict[tuple[str, str], int]:
    total = sum(counts.values())
    exact = {cell: size * count / total for cell, count in counts.items()}
    quotas = {cell: int(value) for cell, value in exact.items()}
    remaining = size - sum(quotas.values())
    order = sorted(counts, key=lambda cell: (-(exact[cell] - quotas[cell]), cell))
    for cell in order[:remaining]:
        quotas[cell] += 1
    return quotas


def build_manifests(data_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    data_path = Path(data_path).expanduser().resolve()
    rows = load_hle_verified_rows(data_path)
    if len(rows) != HLE_EXPECTED_COUNT:
        raise ValueError(
            f"expected {HLE_EXPECTED_COUNT} evaluation rows, got {len(rows)}"
        )

    cells = Counter((str(row["category"]), str(row["answer_type"])) for row in rows)
    quotas = _largest_remainder(cells, 50)
    selected: list[dict[str, Any]] = []
    unselected_by_cell: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for cell, quota in sorted(quotas.items()):
        candidates = sorted(
            (row for row in rows if (str(row["category"]), str(row["answer_type"])) == cell),
            key=lambda row: _rank(PILOT_SEED, str(row["id"])),
        )
        selected.extend(candidates[:quota])
        unselected_by_cell[cell] = candidates[quota:]

    # Exercise multimodal plumbing while preserving category/answer-type quotas.
    while sum(bool(row.get("image")) for row in selected) < 8:
        swap = None
        for index, row in enumerate(selected):
            if row.get("image"):
                continue
            cell = (str(row["category"]), str(row["answer_type"]))
            replacement = next((candidate for candidate in unselected_by_cell[cell] if candidate.get("image")), None)
            if replacement is not None:
                swap = index, row, replacement, cell
                break
        if swap is None:
            raise ValueError("unable to select eight multimodal pilot rows")
        index, old, replacement, cell = swap
        selected[index] = replacement
        unselected_by_cell[cell].remove(replacement)
        unselected_by_cell[cell].append(old)

    selected = sorted(selected, key=lambda row: _rank(PILOT_SEED, str(row["id"])))
    pilot_ids = [str(row["id"]) for row in selected]
    pilot_set = set(pilot_ids)
    smoke_candidates = sorted(
        (row for row in rows if row.get("image") and str(row["id"]) not in pilot_set),
        key=lambda row: _rank(SMOKE_SEED, str(row["id"])),
    )
    if not smoke_candidates:
        raise ValueError("no multimodal smoke candidate outside pilot")
    smoke_id = str(smoke_candidates[0]["id"])

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    smoke = {
        "name": "smoke-1",
        "dataset_repo": HLE_DATASET_REPO,
        "dataset_revision": HLE_DATASET_REVISION,
        "dataset_class": HLE_EVAL_CLASS,
        "ids": [smoke_id],
    }
    pilot = {
        "name": "pilot-50",
        "dataset_repo": HLE_DATASET_REPO,
        "dataset_revision": HLE_DATASET_REVISION,
        "dataset_class": HLE_EVAL_CLASS,
        "ids": pilot_ids,
    }
    (output_dir / "smoke-1.json").write_text(json.dumps(smoke, indent=2) + "\n")
    (output_dir / "pilot-50.json").write_text(json.dumps(pilot, indent=2) + "\n")

    answer_counts = Counter(str(row["answer_type"]) for row in selected)
    category_counts = Counter(str(row["category"]) for row in selected)
    audit = {
        "dataset_repo": HLE_DATASET_REPO,
        "dataset_revision": HLE_DATASET_REVISION,
        "dataset_class": HLE_EVAL_CLASS,
        "evaluation_pool_rows": len(rows),
        "pilot_size": len(pilot_ids),
        "pilot_manifest_sha256": hashlib.sha256("\n".join(pilot_ids).encode()).hexdigest(),
        "smoke_manifest_sha256": hashlib.sha256(smoke_id.encode()).hexdigest(),
        "pilot_image_rows": sum(bool(row.get("image")) for row in selected),
        "pilot_answer_types": dict(sorted(answer_counts.items())),
        "pilot_categories": dict(sorted(category_counts.items())),
        "selection_seed": PILOT_SEED,
    }
    (output_dir / "balance.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    return audit
