import base64
import io
import json
import os
from pathlib import Path
from typing import Any

from datasets import Dataset
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageUser, ContentImage, ContentText
from PIL import Image

HLE_DATASET_REPO = "skylenage-ai/HLE-Verified"
HLE_DATASET_REVISION = "0bc83643672d4f68a5f89998617a639d85e7318b"
HLE_EXPECTED_TOTAL = 2500
HLE_EVAL_CLASS = "Gold subset"
HLE_EXPECTED_COUNT = 668


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
    if len(raw_rows) != HLE_EXPECTED_TOTAL:
        raise ValueError(
            f"expected {HLE_EXPECTED_TOTAL} rows for {HLE_DATASET_REVISION}, got {len(raw_rows)}"
        )
    if len({str(row.get("id")) for row in raw_rows}) != len(raw_rows):
        raise ValueError("verified dataset IDs are not unique")
    rows = [normalize_verified_row(row) for row in raw_rows]
    selected = [row for row in rows if row["Verified_Classes"] == HLE_EVAL_CLASS]
    if len(selected) != HLE_EXPECTED_COUNT:
        raise ValueError(
            f"expected {HLE_EXPECTED_COUNT} {HLE_EVAL_CLASS} rows, got {len(selected)}"
        )
    return selected


def _normalize_answer_type(value: str) -> str:
    normalized = value.replace("-", "_").lower()
    if normalized in {"multiplechoice", "multiple_choice", "mc"}:
        return "multiple_choice"
    if normalized in {"exactmatch", "exact_match", "short_answer"}:
        return "exact_match"
    raise ValueError(f"unknown answer_type: {value}")


def row_to_sample(row: dict[str, Any]) -> Sample:
    required = {"id", "question", "answer", "answer_type"}
    missing = required - row.keys()
    if missing:
        raise ValueError(f"row missing fields: {sorted(missing)}")

    content: list[ContentText | ContentImage] = [ContentText(text=str(row["question"]))]
    image = row.get("image")
    if image:
        if not isinstance(image, str) or not image.startswith(("data:image/", "http://", "https://")):
            raise ValueError(f"invalid image for sample {row['id']}")
        if image.startswith("data:image/"):
            try:
                encoded = image.split(",", 1)[1]
                with Image.open(io.BytesIO(base64.b64decode(encoded, validate=True))) as decoded:
                    decoded.verify()
            except Exception as error:
                raise ValueError(f"invalid image data for sample {row['id']}") from error
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
        },
    )


def load_hle_dataset(
    path: str | Path | None = None,
    *,
    fixture: bool = False,
    manifest_path: str | Path | None = None,
    limit: int | None = None,
) -> MemoryDataset:
    if path is None:
        if fixture:
            path = Path(__file__).with_name("fixture.jsonl")
        else:
            configured = os.environ.get("HLE_VERIFIED_DATA_PATH")
            if not configured:
                raise RuntimeError(
                    "HLE_VERIFIED_DATA_PATH is unset. The pinned HLE-Verified snapshot is "
                    "not available; use fixture=True only for offline plumbing checks."
                )
            path = configured

    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    rows = _load_rows(path) if fixture else load_hle_verified_rows(path)
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
    samples = [row_to_sample(row) for row in rows]
    if len({sample.id for sample in samples}) != len(samples):
        raise ValueError("sample IDs are not unique")
    return MemoryDataset(samples=samples, name="hle-tools-v0-fixture" if fixture else "hle-tools-v0")
