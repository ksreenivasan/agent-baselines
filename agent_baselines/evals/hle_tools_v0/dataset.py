import json
import os
from pathlib import Path
from typing import Any

from datasets import Dataset
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageUser, ContentImage, ContentText

HLE_DATASET_REVISION = "5a81a4c7271a2a2a312b9a690f0c2fde837e4c29"
HLE_EXPECTED_COUNT = 2500


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".parquet":
        return list(Dataset.from_parquet(str(path)))
    if path.suffix in {".jsonl", ".json"}:
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in path.read_text().splitlines() if line]
        data = json.loads(path.read_text())
        return data if isinstance(data, list) else data["rows"]
    raise ValueError(f"unsupported dataset file: {path}")


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
        },
    )


def load_hle_dataset(
    path: str | Path | None = None,
    *,
    fixture: bool = False,
    limit: int | None = None,
) -> MemoryDataset:
    if path is None:
        if fixture:
            path = Path(__file__).with_name("fixture.jsonl")
        else:
            configured = os.environ.get("HLE_DATA_PATH")
            if not configured:
                raise RuntimeError(
                    "HLE_DATA_PATH is unset. The gated cais/hle file is not available; "
                    "use fixture=True only for offline plumbing checks."
                )
            path = configured

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = _load_rows(path)
    if not fixture and len(rows) != HLE_EXPECTED_COUNT:
        raise ValueError(
            f"expected {HLE_EXPECTED_COUNT} rows for {HLE_DATASET_REVISION}, got {len(rows)}"
        )
    if limit is not None:
        rows = rows[:limit]
    samples = [row_to_sample(row) for row in rows]
    if len({sample.id for sample in samples}) != len(samples):
        raise ValueError("sample IDs are not unique")
    return MemoryDataset(samples=samples, name="hle-tools-v0-fixture" if fixture else "hle-tools-v0")
