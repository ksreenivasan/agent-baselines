import base64
import io
import json
import os
import re
from pathlib import Path
from typing import Any, Literal

from datasets import Dataset
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageUser, ContentImage, ContentText
from PIL import Image

DatasetVariant = Literal["standard", "verified"]

HLE_STANDARD_DATASET_REPO = "cais/hle"
HLE_STANDARD_RELEASE = "May 2025"
HLE_STANDARD_EXPECTED_TOTAL = 2500
HLE_STANDARD_EXPECTED_TEXT_ONLY = 2158

HLE_VERIFIED_DATASET_REPO = "skylenage-ai/HLE-Verified"
HLE_VERIFIED_DATASET_REVISION = "0bc83643672d4f68a5f89998617a639d85e7318b"
HLE_VERIFIED_EXPECTED_TOTAL = 2500
HLE_EVAL_CLASS = "Gold subset"
HLE_VERIFIED_EXPECTED_COUNT = 668

# Backward-compatible names from the original HLE-Verified-only adapter.
HLE_DATASET_REPO = HLE_VERIFIED_DATASET_REPO
HLE_DATASET_REVISION = HLE_VERIFIED_DATASET_REVISION
HLE_EXPECTED_TOTAL = HLE_VERIFIED_EXPECTED_TOTAL
HLE_EXPECTED_COUNT = HLE_VERIFIED_EXPECTED_COUNT


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if path.is_dir():
        files = sorted((path / "data").glob("*.parquet")) or sorted(
            path.glob("*.parquet")
        )
        if not files:
            raise ValueError(f"no Parquet files under dataset directory: {path}")
        return [row for file in files for row in Dataset.from_parquet(str(file))]
    if path.suffix == ".parquet":
        return list(Dataset.from_parquet(str(path)))
    if path.suffix in {".jsonl", ".json"}:
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in path.read_text().splitlines() if line]
        data = json.loads(path.read_text())
        return data if isinstance(data, list) else data["rows"]
    raise ValueError(f"unsupported dataset file: {path}")


def normalize_verified_row(row: dict[str, Any]) -> dict[str, Any]:
    required = {"id", "Verified_Classes", "question", "answer", "json"}
    missing = required - row.keys()
    if missing:
        raise ValueError(f"verified row missing fields: {sorted(missing)}")
    try:
        payload = json.loads(row["json"])
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"invalid verified JSON for sample {row.get('id', 'unknown')}"
        ) from error
    for field in ("id", "Verified_Classes", "question", "answer"):
        if str(payload.get(field)) != str(row[field]):
            raise ValueError(f"verified row disagrees with JSON field {field}")
    return payload


def load_hle_verified_rows(path: str | Path) -> list[dict[str, Any]]:
    raw_rows = _load_rows(Path(path).expanduser().resolve())
    if len(raw_rows) != HLE_VERIFIED_EXPECTED_TOTAL:
        raise ValueError(
            f"expected {HLE_VERIFIED_EXPECTED_TOTAL} rows for "
            f"{HLE_VERIFIED_DATASET_REVISION}, got {len(raw_rows)}"
        )
    if len({str(row.get("id")) for row in raw_rows}) != len(raw_rows):
        raise ValueError("verified dataset IDs are not unique")
    rows = [normalize_verified_row(row) for row in raw_rows]
    selected = [row for row in rows if row["Verified_Classes"] == HLE_EVAL_CLASS]
    if len(selected) != HLE_VERIFIED_EXPECTED_COUNT:
        raise ValueError(
            f"expected {HLE_VERIFIED_EXPECTED_COUNT} {HLE_EVAL_CLASS} rows, "
            f"got {len(selected)}"
        )
    return selected


def load_hle_standard_rows(path: str | Path) -> list[dict[str, Any]]:
    rows = _load_rows(Path(path).expanduser().resolve())
    if len(rows) != HLE_STANDARD_EXPECTED_TOTAL:
        raise ValueError(
            f"expected {HLE_STANDARD_EXPECTED_TOTAL} rows for the "
            f"{HLE_STANDARD_RELEASE} HLE release, got {len(rows)}"
        )
    if len({str(row.get("id")) for row in rows}) != len(rows):
        raise ValueError("standard HLE dataset IDs are not unique")
    selected = [row for row in rows if not row.get("image")]
    if len(selected) != HLE_STANDARD_EXPECTED_TEXT_ONLY:
        raise ValueError(
            f"expected {HLE_STANDARD_EXPECTED_TEXT_ONLY} text-only rows, "
            f"got {len(selected)}"
        )
    return selected


def _normalize_answer_type(value: str) -> str:
    normalized = value.replace("-", "_").lower()
    if normalized in {"multiplechoice", "multiple_choice", "mc"}:
        return "multiple_choice"
    if normalized in {"exactmatch", "exact_match", "short_answer"}:
        return "exact_match"
    raise ValueError(f"unknown answer_type: {value}")


def row_to_sample(
    row: dict[str, Any],
    *,
    dataset_variant: DatasetVariant = "standard",
    dataset_repo: str = HLE_STANDARD_DATASET_REPO,
    dataset_revision: str = "fixture",
) -> Sample:
    required = {"id", "question", "answer", "answer_type"}
    missing = required - row.keys()
    if missing:
        raise ValueError(f"row missing fields: {sorted(missing)}")

    content: list[ContentText | ContentImage] = [ContentText(text=str(row["question"]))]
    image = row.get("image")
    if image:
        if not isinstance(image, str) or not image.startswith(
            ("data:image/", "http://", "https://")
        ):
            raise ValueError(f"invalid image for sample {row['id']}")
        if image.startswith("data:image/"):
            try:
                encoded = image.split(",", 1)[1]
                with Image.open(
                    io.BytesIO(base64.b64decode(encoded, validate=True))
                ) as decoded:
                    decoded.verify()
            except Exception as error:
                raise ValueError(
                    f"invalid image data for sample {row['id']}"
                ) from error
        content.append(ContentImage(image=image))

    answer_type = _normalize_answer_type(str(row["answer_type"]))
    return Sample(
        id=str(row["id"]),
        input=[ChatMessageUser(content=content)],
        target=str(row["answer"]),
        metadata={
            "answer_type": answer_type,
            "category": str(row.get("category", "unknown")),
            "raw_subject": str(row.get("raw_subject", "unknown")),
            "image_present": bool(image),
            "verified_class": str(row.get("Verified_Classes", "fixture")),
            "dataset_variant": dataset_variant,
            "dataset_repo": dataset_repo,
            "dataset_revision": dataset_revision,
        },
    )


def load_hle_dataset(
    path: str | Path | None = None,
    *,
    fixture: bool = False,
    manifest_path: str | Path | None = None,
    limit: int | None = None,
    dataset_variant: DatasetVariant = "standard",
    dataset_revision: str | None = None,
) -> MemoryDataset:
    if dataset_variant not in {"standard", "verified"}:
        raise ValueError(f"unknown HLE dataset variant: {dataset_variant}")

    if fixture:
        repo = "fixture"
        revision = "fixture"
    elif dataset_variant == "standard":
        repo = HLE_STANDARD_DATASET_REPO
        revision = dataset_revision or os.environ.get("HLE_DATASET_REVISION", "")
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise RuntimeError(
                "HLE_DATASET_REVISION must be the immutable 40-character commit for "
                "the May 2025 cais/hle snapshot; mutable refs such as main are rejected."
            )
    else:
        repo = HLE_VERIFIED_DATASET_REPO
        revision = dataset_revision or HLE_VERIFIED_DATASET_REVISION
        if revision != HLE_VERIFIED_DATASET_REVISION:
            raise ValueError(
                "the verified variant is pinned to "
                f"{HLE_VERIFIED_DATASET_REVISION}, got {revision}"
            )

    if path is None:
        if fixture:
            path = Path(__file__).with_name("fixture.jsonl")
        else:
            env_name = (
                "HLE_DATA_PATH"
                if dataset_variant == "standard"
                else "HLE_VERIFIED_DATA_PATH"
            )
            configured = os.environ.get(env_name)
            if not configured:
                raise RuntimeError(
                    f"{env_name} is unset. The pinned {dataset_variant} HLE snapshot is "
                    "not available; use fixture=True only for offline plumbing checks."
                )
            path = configured

    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    if fixture:
        rows = _load_rows(path)
    elif dataset_variant == "standard":
        rows = load_hle_standard_rows(path)
    else:
        rows = load_hle_verified_rows(path)
    if manifest_path is not None:
        manifest = json.loads(Path(manifest_path).expanduser().read_text())
        ids = [str(sample_id) for sample_id in manifest["ids"]]
        by_id = {str(row["id"]): row for row in rows}
        missing = [sample_id for sample_id in ids if sample_id not in by_id]
        if missing:
            raise ValueError(f"manifest contains {len(missing)} unknown IDs")
        rows = [by_id[sample_id] for sample_id in ids]
    if limit is not None:
        rows = rows[:limit]
    samples = [
        row_to_sample(
            row,
            dataset_variant=dataset_variant,
            dataset_repo=repo,
            dataset_revision=revision,
        )
        for row in rows
    ]
    if len({sample.id for sample in samples}) != len(samples):
        raise ValueError("sample IDs are not unique")
    name = "hle-fixture" if fixture else f"hle-{dataset_variant}"
    return MemoryDataset(samples=samples, name=name)
