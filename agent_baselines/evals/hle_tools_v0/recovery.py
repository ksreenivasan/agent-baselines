"""Bounded recovery of HLE logs with explicit per-question provenance.

JSON sources are streamed one sample at a time. Retained records are written in
small native Inspect shards; the original logs are never rewritten.
"""

import argparse
import copy
import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json_header(path: Path) -> dict[str, Any]:
    import ijson

    builder = ijson.ObjectBuilder()
    with path.open("rb") as stream:
        events = iter(ijson.parse(stream, use_float=True))
        for prefix, event, value in events:
            if prefix == "" and event == "map_key" and value == "samples":
                builder.event("end_map", None)
                return builder.value
            builder.event(event, value)
    raise ValueError(f"source has no samples array: {path}")


def iter_samples(path: str | Path) -> Iterator[dict[str, Any]]:
    path = Path(path)
    if path.suffix == ".eval":
        from inspect_ai.log import read_eval_log_samples

        for sample in read_eval_log_samples(path, all_samples_required=False):
            yield sample.model_dump(mode="json", exclude_none=True)
    elif path.suffix == ".jsonl":
        with path.open() as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)
    else:
        import ijson

        with path.open("rb") as stream:
            yield from ijson.items(stream, "samples.item", use_float=True)


def resolve(value: Any, attachments: dict[str, Any]) -> Any:
    if isinstance(value, str) and value.startswith("attachment://"):
        key = value.removeprefix("attachment://")
        if key not in attachments:
            raise ValueError(f"missing sample attachment: {key}")
        return attachments[key]
    if isinstance(value, dict):
        return {key: resolve(item, attachments) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve(item, attachments) for item in value]
    return value


def generation_digest(sample: dict[str, Any]) -> str:
    return digest(
        resolve(
            {key: sample.get(key) for key in ("input", "target", "messages", "output")},
            sample.get("attachments", {}),
        )
    )


def infrastructure_reasons(sample: dict[str, Any]) -> list[str]:
    metadata = sample.get("metadata") or {}
    reasons = []
    if sample.get("invalidation") or sample.get("invalidated"):
        reasons.append("invalidated")
    if metadata.get("hle_tools_invalidated"):
        reasons.append("hle_tools_invalidated")
    if metadata.get("hle_tools_infrastructure_errors"):
        reasons.append("hle_tools_infrastructure_errors")
    for event in sample.get("events", []):
        if event.get("event") != "tool" or event.get("function") not in {
            "web_search",
            "fetch_url",
        }:
            continue
        result = resolve(event.get("result"), sample.get("attachments", {}))
        if isinstance(result, str) and result.startswith(("{", "[")):
            try:
                result = json.loads(result)
            except json.JSONDecodeError:
                pass
        # Individual URL failures and invalid model arguments are normal tool
        # outcomes; only search-service failures contaminate infrastructure.
        if event.get("function") == "web_search":
            if event.get("error"):
                reasons.append("web_search_execution_error")
            if isinstance(result, dict) and str(result.get("error", "")).startswith(
                "search_"
            ):
                reasons.append("web_backend_error")
        if isinstance(result, list) and any(
            isinstance(item, dict) and item.get("backend") in {"fixture", "exa"}
            for item in result
        ):
            reasons.append("incompatible_search_backend")
        if isinstance(result, str):
            text = result.lower()
            if any(
                marker in text
                for marker in (
                    "402 payment required",
                    "http 402",
                    "credit balance",
                    "quota exceeded",
                    "keenable search failed",
                    "exa search failed",
                    "search backend failed",
                )
            ):
                reasons.append("web_backend_error")
    return sorted(set(reasons))


def disposition(sample: dict[str, Any]) -> tuple[str, str]:
    bad = infrastructure_reasons(sample)
    if bad:
        return "generate", ",".join(bad)
    error = sample.get("error")
    if error:
        if sample.get("scores"):
            return "generate", "error_and_score"
        traceback = error.get("traceback", "")
        if "scorer.py" in traceback and "EqualityJudgment" in str(error):
            return "judge_only", "malformed_judge_result"
        text = str(error)
        if any(
            x in text for x in ("ServerError", "AttemptTimeoutError", "RateLimitError")
        ):
            return "generate", "provider_infrastructure_error"
        return "terminal_error", "unclassified_model_or_execution_error"
    if not sample.get("scores"):
        return "generate", "missing_score"
    return "retain_score", "compatible_completed_sample"


def _validate_samples(
    samples: Iterable[dict[str, Any]],
    expected_ids: Iterable[str],
    allow_errors: bool = False,
) -> dict[str, Any]:
    ids = list(map(str, expected_ids))
    if len(set(ids)) != len(ids):
        raise ValueError("expected manifest has duplicate IDs")
    expected = {(sample_id, 1) for sample_id in ids}
    seen: set[tuple[str, int]] = set()
    counts: Counter[str] = Counter()
    for sample in samples:
        key = (str(sample["id"]), sample.get("epoch", 1))
        if key not in expected:
            raise ValueError(f"unexpected sample ID or epoch: {key}")
        if key in seen:
            raise ValueError(f"duplicate sample ID and epoch: {key}")
        seen.add(key)
        kind, reason = disposition(sample)
        if infrastructure_reasons(sample) or reason == "error_and_score":
            raise ValueError(f"invalid sample {key}: {reason}")
        if kind != "retain_score" and not (allow_errors and sample.get("error")):
            raise ValueError(f"unaccepted sample {key}: {reason}")
        counts[kind] += 1
    if seen != expected:
        raise ValueError(f"missing {len(expected - seen)} expected samples")
    return {"valid": True, "samples": len(seen), "counts": dict(counts)}


def validate_samples(
    samples: Iterable[dict[str, Any]],
    expected_ids: Iterable[str],
    allow_errors: bool = False,
) -> dict[str, Any]:
    try:
        return _validate_samples(samples, expected_ids, allow_errors)
    except ValueError as error:
        return {"valid": False, "reason": str(error)}


def _save(path: Path, value: Any) -> None:
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"existing recovery artifact differs: {path}")
        return
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def write_shard(
    path: Path,
    header: dict[str, Any],
    samples: list[dict[str, Any]],
    source_sha256: str,
    *,
    judge_only: bool = False,
) -> dict[str, Any]:
    from inspect_ai.log import EvalLog, read_eval_log, write_eval_log

    shard = copy.deepcopy(header)
    ids = [str(sample["id"]) for sample in samples]
    shard.update(status="started" if judge_only else "success", samples=samples)
    for field in ("error", "results", "reductions"):
        shard.pop(field, None)
    shard["eval"]["dataset"].update(samples=len(ids), sample_ids=ids)
    shard["eval"]["config"].pop("limit", None)
    shard["eval"].setdefault("metadata", {})[
        "hle_recovery_source_sha256"
    ] = source_sha256
    if not path.exists():
        write_eval_log(EvalLog.model_validate(shard), path)
    restored = list(iter_samples(path))
    if [generation_digest(s) for s in restored] != [
        generation_digest(s) for s in samples
    ]:
        raise ValueError("generation records changed during Inspect round trip")
    if [s.get("scores") for s in restored] != [s.get("scores") for s in samples]:
        raise ValueError("score records changed during Inspect round trip")
    if not judge_only:
        _validate_samples(restored, ids)
        if read_eval_log(path, header_only=True).invalidated:
            raise ValueError("retained shard was invalidated")
    return {"path": str(path), "sha256": file_digest(path), "ids": ids}


def build_ledger(
    dataset_path: Path,
    revision: str,
    source: Path,
    priors: list[Path],
    output: Path,
    shard_size: int = 25,
) -> dict[str, Any]:
    from .dataset import load_hle_standard_rows

    if not 1 <= shard_size <= 100:
        raise ValueError("shard_size must be between 1 and 100")
    if any(
        "recovery-v2" in str(p) or "quarantine" in str(p) for p in [source, *priors]
    ):
        raise ValueError("quarantined and recovery-v2 sources are excluded")
    rows = load_hle_standard_rows(dataset_path)
    ids = [str(row["id"]) for row in rows]
    expected = set(ids)
    header = read_json_header(source)
    args = header["eval"]["task_args"]
    if (
        args.get("dataset_revision") != revision
        or args.get("dataset_variant") != "standard"
    ):
        raise ValueError("source dataset revision/variant mismatch")
    if header.get("invalidated"):
        raise ValueError("source log is invalidated")
    source_ids = set(map(str, header["eval"]["dataset"]["sample_ids"]))
    limits = header["eval"]["config"].get("limit")
    selected = ids[limits[0] : limits[1]] if isinstance(limits, list) else ids
    if source_ids != set(selected):
        raise ValueError(
            "source selected dataset membership differs from pinned text-only HLE"
        )
    output.mkdir(parents=True, exist_ok=True)
    if (output / "ledger.json").exists():
        raise FileExistsError(
            "completed ledger is immutable; choose a new output directory"
        )
    sources = []
    attempts: dict[str, list[dict[str, Any]]] = {sample_id: [] for sample_id in ids}
    entries: dict[str, dict[str, Any]] = {}
    retained: list[dict[str, Any]] = []
    judge: list[dict[str, Any]] = []
    shards = []
    config_counts: Counter[str] = Counter()
    for current in [*priors, source]:
        current_header = read_json_header(current)
        source_hash = file_digest(current)
        source_info = {
            "path": str(current),
            "sha256": source_hash,
            "bytes": current.stat().st_size,
            "eval": current_header["eval"],
            "plan": current_header.get("plan"),
            "source_role": (
                "selected" if current == source else "attempt_provenance_only"
            ),
        }
        sources.append(source_info)
        seen = set()
        for sample in iter_samples(current):
            sample_id = str(sample["id"])
            key = (sample_id, sample.get("epoch", 1))
            if sample_id not in expected or key[1] != 1 or key in seen:
                raise ValueError(f"unknown or duplicate source sample: {key}")
            if current == source and sample_id not in source_ids:
                raise ValueError(f"sample outside declared source selection: {key}")
            seen.add(key)
            kind, reason = disposition(sample)
            configs = sorted(
                {
                    json.dumps(event.get("config", {}), sort_keys=True)
                    for event in sample.get("events", [])
                    if event.get("event") == "model"
                    and event.get("model") == current_header["eval"]["model"]
                }
            )
            generation_hash = generation_digest(sample)
            attempt = {
                "source_sha256": source_hash,
                "uuid": sample.get("uuid"),
                "started_at": sample.get("started_at"),
                "disposition": kind,
                "reason": reason,
                "generation_sha256": generation_hash,
                "error_retries": sample.get("error_retries", []),
                "request_configs": [json.loads(config) for config in configs],
            }
            attempts[sample_id].append(attempt)
            if current != source:
                continue
            config_counts.update(configs)
            entries[sample_id] = {
                "id": sample_id,
                "epoch": 1,
                "disposition": kind,
                "reason": reason,
                "source_sha256": source_hash,
                "generation_sha256": generation_hash,
            }
            if kind == "retain_score":
                retained.append(sample)
                if len(retained) == shard_size:
                    shards.append(
                        write_shard(
                            output / f"retained-{len(shards):04d}.eval",
                            header,
                            retained,
                            source_hash,
                        )
                    )
                    retained = []
            elif kind == "judge_only":
                judge.append(sample)
        source_info["recorded_samples"] = len(seen)
    if retained:
        shards.append(
            write_shard(
                output / f"retained-{len(shards):04d}.eval",
                header,
                retained,
                source_hash,
            )
        )
    judge_shard = (
        write_shard(
            output / "judge-only.eval", header, judge, source_hash, judge_only=True
        )
        if judge
        else None
    )
    for sample_id in ids:
        if sample_id not in entries:
            entries[sample_id] = {
                "id": sample_id,
                "epoch": 1,
                "disposition": "generate",
                "reason": "missing_from_keenable_suffix",
            }
        entries[sample_id]["attempts"] = attempts[sample_id]
        entries[sample_id]["distinct_recorded_attempts"] = len(
            {(a["uuid"], a["generation_sha256"]) for a in attempts[sample_id]}
        )
    ordered = [entries[sample_id] for sample_id in ids]
    counts = dict(Counter(entry["disposition"] for entry in ordered))
    manifest_base = {"dataset_revision": revision, "dataset_variant": "standard"}
    for kind in ("generate", "judge_only", "retain_score", "terminal_error"):
        filename_kind = kind.replace("_", "-")
        _save(
            output / f"ids-{filename_kind}.json",
            {
                **manifest_base,
                "ids": [e["id"] for e in ordered if e["disposition"] == kind],
            },
        )
    _save(output / "ids-all.json", {**manifest_base, "ids": ids})
    ledger = {
        "version": 1,
        "dataset_revision": revision,
        "dataset_ids_sha256": digest(ids),
        "search_backend": "keenable",
        "backend_evidence": "explicitly selected Keenable restart sources; no Exa prefix accepted",
        "historical_retry_policy": "existing eval-retry attempts retained and enumerated; not a first-attempt-only estimate",
        "sources": sources,
        "counts": counts,
        "samples": ordered,
        "retained_shards": shards,
        "judge_only_shard": judge_shard,
        "actual_request_config_sample_counts": [
            {"config": json.loads(config), "samples": count}
            for config, count in sorted(config_counts.items())
        ],
    }
    _save(output / "ledger.json", ledger)
    return ledger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--dataset", type=Path, required=True)
    build.add_argument("--revision", required=True)
    build.add_argument("--source", type=Path, required=True)
    build.add_argument("--prior", action="append", type=Path, default=[])
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--shard-size", type=int, default=25)
    validate = commands.add_parser("validate")
    validate.add_argument("logs", type=Path, nargs="+")
    validate.add_argument("--expected-manifest", type=Path, required=True)
    validate.add_argument("--allow-errors", action="store_true")
    args = parser.parse_args()
    if args.command == "build":
        ledger = build_ledger(
            args.dataset,
            args.revision,
            args.source,
            args.prior,
            args.output,
            args.shard_size,
        )
        print(json.dumps({"counts": ledger["counts"], "output": str(args.output)}))
    else:
        from inspect_ai.log import read_eval_log

        for path in args.logs:
            if path.suffix == ".eval":
                header = read_eval_log(path, header_only=True)
                if header.invalidated:
                    raise ValueError(f"log is invalidated: {path}")
                if header.status != "success" and not args.allow_errors:
                    raise ValueError(f"log is not complete: {path} ({header.status})")
            elif path.suffix == ".json":
                header_data = read_json_header(path)
                if header_data.get("invalidated"):
                    raise ValueError(f"log is invalidated: {path}")
                if header_data.get("status") != "success" and not args.allow_errors:
                    raise ValueError(f"log is not complete: {path}")
        expected = json.loads(args.expected_manifest.read_text())["ids"]
        result = validate_samples(
            (sample for path in args.logs for sample in iter_samples(path)),
            expected,
            args.allow_errors,
        )
        print(json.dumps(result))
        if not result["valid"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
