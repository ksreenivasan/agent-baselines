#!/usr/bin/env python3
"""Repair ledger-approved judge errors without generating solver answers."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

from inspect_ai._eval.score import score as score_log
from inspect_ai.log import read_eval_log, read_eval_log_samples, write_eval_log
from inspect_ai.scorer import Score, accuracy, scorer

from agent_baselines.evals.hle_tools_v0.recovery import (
    digest,
    disposition,
    file_digest,
    generation_digest,
    resolve,
    validate_samples,
)
from agent_baselines.evals.hle_tools_v0.scorer import hle_scorer

_STABLE_FIELDS = (
    "id",
    "epoch",
    "input",
    "target",
    "choices",
    "messages",
    "output",
    "metadata",
    "store",
)


def stable_digest(sample) -> str:
    value = sample.model_dump(mode="json")
    return digest(
        resolve(
            {key: value.get(key) for key in _STABLE_FIELDS},
            value.get("attachments", {}),
        )
    )


@scorer(metrics=[accuracy()])
def recording_judge(judge_model: str, judge_reasoning_effort: str):
    judge = hle_scorer(judge_model, judge_reasoning_effort)

    async def score(state, target):
        try:
            return await judge(state, target)
        except Exception as error:
            # Finish the Inspect scorer span so failed attempts remain serializable.
            # The caller treats this marker as failure, never as a benchmark score.
            return Score(
                value="I", metadata={"hle_judge_repair_failed": type(error).__name__}
            )

    return score


def write_archive(path: Path, log, expected_ids: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to replace an existing archive: {path}")
    fd, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".eval", dir=path.parent
    )
    os.close(fd)
    temporary = Path(name)
    try:
        write_eval_log(log, temporary, format="eval")
        restored = list(
            read_eval_log_samples(
                temporary, all_samples_required=False, resolve_attachments=True
            )
        )
        if [stable_digest(s) for s in restored] != [
            stable_digest(s) for s in log.samples
        ]:
            raise ValueError("saved generator fields changed during archive round trip")
        if expected_ids is not None:
            validation = validate_samples(
                (sample.model_dump(mode="json") for sample in restored), expected_ids
            )
            if not validation["valid"]:
                raise ValueError(
                    f"persisted judge repair failed validation: {validation}"
                )
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def write_result(path: Path, value: dict) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def repair(ledger_path: Path, output: Path) -> dict:
    ledger = json.loads(ledger_path.read_text())
    specification = ledger.get("judge_only_shard")
    if not specification:
        raise ValueError("ledger has no approved judge-only shard")
    source = Path(specification["path"])
    if file_digest(source) != specification["sha256"]:
        raise ValueError("judge-only source checksum differs from ledger")
    ids = list(map(str, specification["ids"]))
    if not 1 <= len(ids) <= 100 or len(ids) != len(set(ids)):
        raise ValueError("judge-only shard must contain 1 to 100 unique IDs")
    approved = {
        str(entry["id"]): entry
        for entry in ledger["samples"]
        if entry["disposition"] == "judge_only"
    }
    if set(ids) != set(approved):
        raise ValueError("judge-only shard membership differs from ledger")
    header = read_eval_log(source, header_only=True)
    if header.invalidated:
        raise ValueError("judge-only source is invalidated")
    samples = list(
        read_eval_log_samples(
            source, all_samples_required=False, resolve_attachments=True
        )
    )
    if [str(sample.id) for sample in samples] != ids:
        raise ValueError("judge-only archive IDs/order differ from ledger")
    for sample in samples:
        payload = sample.model_dump(mode="json", exclude_none=True)
        if (
            sample.epoch != 1
            or sample.output is None
            or disposition(payload)[0] != "judge_only"
        ):
            raise ValueError(f"sample {sample.id} is not an approved saved judge error")
        if generation_digest(payload) != approved[str(sample.id)]["generation_sha256"]:
            raise ValueError(f"saved generation checksum changed for {sample.id}")
    specs = [
        item
        for item in header.eval.scorers or []
        if item.name.split("/")[-1] == "hle_scorer"
    ]
    if len(specs) != 1:
        raise ValueError("source must declare exactly one HLE scorer")
    options = specs[0].options
    judge_model = options.get("judge_model", "openai/gpt-5.6-luna")
    effort = options.get("judge_reasoning_effort", "medium")
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "source": str(source),
        "source_sha256": specification["sha256"],
        "ledger": str(ledger_path.resolve()),
        "ledger_sha256": file_digest(ledger_path),
        "judge_model": judge_model,
        "judge_reasoning_effort": effort,
        "started_at": time.time(),
        "attempts": [],
        "complete": False,
    }
    repaired = []
    for index, original in enumerate(samples):
        mini = header.model_copy(
            deep=True,
            update={
                "samples": [original.model_copy(deep=True)],
                "results": None,
                "reductions": None,
            },
        )
        mini.eval.dataset.samples = 1
        mini.eval.dataset.sample_ids = [original.id]
        scored = score_log(
            mini,
            [recording_judge(judge_model, effort)],
            action="append",
            display="none",
        )
        candidate = scored.samples[0]
        if stable_digest(candidate) != stable_digest(original):
            raise ValueError("judge repair modified saved generator fields")
        if len(candidate.scores or {}) != 1:
            raise ValueError("judge repair produced an unexpected score set")
        judgment = next(iter(candidate.scores.values()))
        failed = (judgment.metadata or {}).get("hle_judge_repair_failed")
        candidate.scores = None if failed else {"hle_scorer": judgment}
        candidate.error = original.error if failed else None
        scored.status = "error" if failed else "success"
        scored.results = scored.reductions = None
        scored.error = None
        scored.eval.metadata = {
            **(scored.eval.metadata or {}),
            "hle_judge_repair": {
                "source_sha256": specification["sha256"],
                "ledger_sha256": result["ledger_sha256"],
                "generation_sha256": generation_digest(
                    original.model_dump(mode="json", exclude_none=True)
                ),
                "failure": failed,
            },
        }
        attempt = output / f"attempt-{index:03d}.eval"
        write_archive(attempt, scored)
        result["attempts"].append(
            {
                "id": str(original.id),
                "path": str(attempt),
                "sha256": file_digest(attempt),
                "failure": failed,
            }
        )
        if failed:
            result["finished_at"] = time.time()
            write_result(output / "result.json", result)
            return result
        repaired.append(candidate)
    final = header.model_copy(
        deep=True,
        update={
            "status": "success",
            "samples": repaired,
            "results": None,
            "reductions": None,
            "error": None,
        },
    )
    final.eval.metadata = {
        **(final.eval.metadata or {}),
        "hle_judge_repair": {
            "source_sha256": specification["sha256"],
            "ledger_sha256": result["ledger_sha256"],
            "judge_model": judge_model,
            "judge_reasoning_effort": effort,
        },
    }
    validation = validate_samples((s.model_dump(mode="json") for s in repaired), ids)
    if not validation["valid"]:
        raise ValueError(f"judge repair validation failed: {validation}")
    destination = output / "repaired.eval"
    write_archive(destination, final, expected_ids=ids)
    result.update(
        complete=True,
        archive=str(destination),
        archive_sha256=file_digest(destination),
        validation=validation,
        finished_at=time.time(),
    )
    write_result(output / "result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("run inside a Slurm allocation")
    os.umask(0o077)
    result = repair(args.ledger, args.output)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
