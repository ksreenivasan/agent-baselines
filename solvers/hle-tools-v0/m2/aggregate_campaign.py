#!/usr/bin/env python3
"""Validate a completed HLE tools condition and index its native scored archives."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from inspect_ai.log import read_eval_log

from agent_baselines.evals.hle_tools_v0.recovery import (
    digest,
    disposition,
    file_digest,
    generation_digest,
    iter_samples,
    read_json_header,
)


def require(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def checked_file(path: str | Path, expected: str) -> Path:
    path = Path(path).resolve()
    require(file_digest(path) == expected, f"checksum mismatch: {path}")
    return path


def check_settings(actual: dict, config: dict, context: str) -> None:
    require(
        actual.get("max_connections") == config["concurrency"],
        f"{context}: concurrency differs from frozen configuration",
    )
    for key in (
        "reasoning_effort",
        "max_tokens",
        "temperature",
        "top_p",
        "reasoning_history",
        "max_retries",
        "timeout",
        "attempt_timeout",
    ):
        require(
            actual.get(key) == config.get(key),
            f"{context}: {key} differs from frozen configuration",
        )


def check_header(header: dict, config: dict, partial: bool) -> None:
    allowed = {"success", "started", "cancelled"} if partial else {"success"}
    require(header.get("status") in allowed, "archive status is not accepted")
    require(
        not header.get("invalidated") and not header.get("error"),
        "archive is invalidated or has a top-level error",
    )
    spec = header["eval"]
    require(spec["model"] == config["model"], "archive model mismatch")
    require(
        spec.get("model_base_url") == config.get("model_base_url"),
        "archive endpoint mismatch",
    )
    args = spec["task_args"]
    require(
        spec["task"].split("/")[-1] == "hle_tools_v0",
        "archive is not the production HLE tools task",
    )
    for key, expected in (
        ("dataset_revision", config["dataset_revision"]),
        ("dataset_variant", "standard"),
        ("judge_model", config["judge_model"]),
        ("judge_reasoning_effort", config.get("judge_reasoning_effort", "medium")),
        ("sandbox_backend", "m2-enroot"),
    ):
        require(args.get(key) == expected, f"archive task argument mismatch: {key}")
    require(args.get("fixture") is False, "fixture archive is not accepted")
    check_settings(
        spec.get("model_generate_config", {}), config, "archive generation settings"
    )


def judge_binding(ledger_path: Path, result_path: Path) -> dict:
    """Prove a completed repair still belongs to the active ledger's saved answers."""
    active = read_json(ledger_path)
    result = read_json(result_path)
    require(result.get("complete") is True, "judge repair is incomplete")
    original_path = Path(result.get("ledger", ledger_path))
    checked_file(original_path, result["ledger_sha256"])
    original = read_json(original_path)
    for key in ("dataset_revision", "dataset_ids_sha256", "search_backend"):
        require(active[key] == original[key], f"judge adoption changes {key}")
    original_sources = [
        source for source in original["sources"] if source["source_role"] == "selected"
    ]
    active_sources = [
        source for source in active["sources"] if source["source_role"] == "selected"
    ]
    require(
        len(original_sources) == len(active_sources) == 1,
        "judge adoption requires one original source",
    )
    require(
        original_sources[0]["sha256"] == active_sources[0]["sha256"],
        "judge adoption changes original source",
    )
    require(
        original_sources[0]["eval"] == active_sources[0]["eval"],
        "judge adoption changes original protocol",
    )
    generations = None
    for ledger in (original, active):
        specification = ledger["judge_only_shard"]
        require(specification is not None, "ledger has no approved judge-only shard")
        source = checked_file(specification["path"], specification["sha256"])
        approved = {
            str(row["id"]): row["generation_sha256"]
            for row in ledger["samples"]
            if row["disposition"] == "judge_only"
        }
        require(
            len(specification["ids"]) == len(set(specification["ids"]))
            and set(specification["ids"]) == set(approved),
            "judge adoption changes approved IDs",
        )
        actual = {}
        for row in iter_samples(source):
            sample_id = str(row["id"])
            require(
                sample_id not in actual
                and row.get("epoch", 1) == 1
                and disposition(row)[0] == "judge_only",
                "judge source contains an unapproved row",
            )
            actual[sample_id] = generation_digest(row)
        require(actual == approved, "judge adoption changes saved generation")
        require(
            generations is None or generations == actual,
            "judge adoption changes saved generation",
        )
        generations = actual
    require(
        result["source_sha256"] == original["judge_only_shard"]["sha256"],
        "repair original source hash mismatch",
    )
    repaired_path = checked_file(result["archive"], result["archive_sha256"])
    header = read_eval_log(repaired_path, header_only=True)
    require(
        header.status == "success" and not header.invalidated and not header.error,
        "repaired archive is not successful",
    )
    restored = {}
    for row in iter_samples(repaired_path):
        sample_id = str(row["id"])
        require(
            sample_id not in restored
            and row.get("epoch", 1) == 1
            and disposition(row)[0] == "retain_score",
            "repaired archive contains an unaccepted row",
        )
        restored[sample_id] = generation_digest(row)
    require(
        restored == generations,
        "repaired generation differs from approved saved answers",
    )
    return {
        "version": 1,
        "active_ledger": {
            "path": str(ledger_path.resolve()),
            "sha256": file_digest(ledger_path),
        },
        "original_ledger": {
            "path": str(original_path.resolve()),
            "sha256": result["ledger_sha256"],
        },
        "repair_result": {
            "path": str(result_path.resolve()),
            "sha256": file_digest(result_path),
        },
        "original_judge_source_sha256": original["judge_only_shard"]["sha256"],
        "active_judge_source_sha256": active["judge_only_shard"]["sha256"],
        "repaired_archive_sha256": result["archive_sha256"],
        "generation_sha256_by_id": generations,
    }


def write_judge_adoption(ledger: Path, result: Path, output: Path) -> dict:
    binding = judge_binding(ledger, result)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(binding, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return binding


def sources(
    config_path: Path,
    manifest: dict,
    ledger_path: Path | None,
    judge_result: Path | None,
    attempts: list[Path],
    allowed_commits: set[str],
    selections: dict[str, list[str]],
    judge_adoption: Path | None = None,
) -> tuple[list[dict], dict[str, dict], dict]:
    config = read_json(config_path)
    inputs = []
    approvals = {}
    provenance = {}
    if ledger_path:
        ledger = read_json(ledger_path)
        provenance["ledger"] = {
            "path": str(ledger_path.resolve()),
            "sha256": file_digest(ledger_path),
        }
        require(
            ledger["dataset_revision"] == config["dataset_revision"],
            "ledger revision mismatch",
        )
        require(
            ledger["dataset_ids_sha256"] == digest(manifest["ids"]),
            "ledger dataset membership/order mismatch",
        )
        require(
            ledger["search_backend"] == "keenable", "ledger search backend mismatch"
        )
        approvals = {str(row["id"]): row for row in ledger["samples"]}
        require(
            len(approvals) == len(ledger["samples"])
            and set(approvals) == set(map(str, manifest["ids"])),
            "ledger has missing/duplicate/unknown IDs",
        )
        selected = [
            row for row in ledger["sources"] if row["source_role"] == "selected"
        ]
        require(
            len(selected) == 1,
            "ledger must have exactly one selected historical source",
        )
        original = selected[0]
        original_path = checked_file(original["path"], original["sha256"])
        original_header = (
            read_eval_log(original_path, header_only=True).model_dump(
                mode="json", exclude_none=True
            )
            if original_path.suffix == ".eval"
            else read_json_header(original_path)
        )
        require(
            original_header["eval"] == original["eval"],
            "original source header differs from ledger",
        )
        require(
            original["eval"]["model"] == config["model"], "historical model mismatch"
        )
        provenance["historical_source"] = {
            key: original[key] for key in ("path", "sha256")
        }
        provenance["historical_revision"] = original["eval"].get("revision")
        for shard in ledger["retained_shards"]:
            inputs.append(
                {
                    **shard,
                    "kind": "retained",
                    "source_sha256": original["sha256"],
                    "revision": original["eval"].get("revision"),
                }
            )
        if judge_result:
            repaired = read_json(judge_result)
            require(repaired.get("complete") is True, "judge repair is incomplete")
            specification = ledger["judge_only_shard"]
            if repaired["ledger_sha256"] != provenance["ledger"]["sha256"]:
                if judge_adoption is None:
                    raise ValueError(
                        "judge repair ledger mismatch requires an explicit adoption record"
                    )
                binding = judge_binding(ledger_path, judge_result)
                require(
                    read_json(judge_adoption) == binding,
                    "judge adoption record mismatch",
                )
                provenance["judge_adoption"] = {
                    "path": str(judge_adoption.resolve()),
                    "sha256": file_digest(judge_adoption),
                    "binding": binding,
                }
            else:
                judge_binding(ledger_path, judge_result)
            require(
                repaired["judge_model"] == config["judge_model"],
                "repaired judge model mismatch",
            )
            require(
                repaired["judge_reasoning_effort"]
                == config.get("judge_reasoning_effort", "medium"),
                "repaired judge effort mismatch",
            )
            inputs.append(
                {
                    "path": repaired["archive"],
                    "sha256": repaired["archive_sha256"],
                    "ids": specification["ids"],
                    "kind": "judge_repair",
                    "source_sha256": original["sha256"],
                    "revision": original["eval"].get("revision"),
                    "repair_result": str(judge_result.resolve()),
                    "repair_result_sha256": file_digest(judge_result),
                }
            )
    else:
        require(judge_result is None, "judge repair requires its ledger")
    for result_path in attempts:
        result_path = result_path.resolve()
        result = read_json(result_path)
        launch_path = result_path.parent / "launch.json"
        launch = read_json(launch_path)
        selection = selections.get(str(result_path))
        require(
            result.get("complete")
            or result.get("validation", {}).get("continuable")
            or selection is not None,
            "failed attempts require explicit per-ID selection",
        )
        require(
            result.get("publisher_exit") == 0,
            "attempt has no successful final publication",
        )
        require(
            launch["config_sha256"] == file_digest(config_path)
            and launch["config"] == config,
            "launch configuration differs from frozen full configuration",
        )
        commit = launch["source_commit"]
        require(commit in allowed_commits, "launch source commit is not approved")
        # Verify the exact file bytes used by the launch, not a reserialized JSON hash.
        command_manifest = [
            arg.split("=", 1)[1]
            for arg in launch["command"]
            if arg.startswith("manifest_path=")
        ]
        require(len(command_manifest) == 1, "launch must record one manifest path")
        path = checked_file(command_manifest[0], launch["manifest_sha256"])
        require(
            read_json(path) == launch["manifest"], "embedded launch manifest mismatch"
        )
        ids = list(map(str, launch["manifest"]["ids"]))
        require(
            len(ids) == len(set(ids)) and set(ids) <= set(map(str, manifest["ids"])),
            "production manifest membership mismatch",
        )
        if selection is not None:
            require(
                selection
                and len(selection) == len(set(selection))
                and set(selection) <= set(ids),
                "invalid partial selection",
            )
        inputs.append(
            {
                "path": result["archive"],
                "sha256": result["archive_sha256"],
                "ids": ids,
                "selected_ids": selection,
                "kind": "production",
                "source_commit": commit,
                "result": str(result_path),
                "result_sha256": file_digest(result_path),
                "launch_sha256": file_digest(launch_path),
            }
        )
    require(
        set(selections) <= {str(path.resolve()) for path in attempts},
        "selection names an unprovided attempt",
    )
    return inputs, approvals, provenance


TOKEN_FIELDS = ("input_tokens", "output_tokens", "total_tokens", "reasoning_tokens")


def row_metrics(row: dict, config: dict | None = None) -> dict:
    config = config or {}
    events = row.get("events") or []
    tools = Counter(
        event.get("function", "unknown")
        for event in events
        if event.get("event") == "tool"
    )
    raw_usage = row.get("model_usage") or {}
    usage = {
        model: {
            key: value
            for key, value in fields.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        for model, fields in raw_usage.items()
        if isinstance(fields, dict)
    }
    solver = config.get("model") or row.get("output", {}).get("model")
    judge = config.get("judge_model")
    models = set(usage) | {model for model in (solver, judge) if model}
    models.update(
        event["model"]
        for event in events
        if event.get("event") == "model" and event.get("model")
    )
    recorded_events = {
        role: {"events": 0, "errors": 0} for role in ("solver", "judge", "other")
    }
    for event in events:
        if event.get("event") != "model":
            continue
        model = event.get("model")
        # Shared or unknown model identities do not establish a solver/judge role.
        role = (
            "solver"
            if model == solver and solver != judge
            else "judge" if model == judge and judge != solver else "other"
        )
        recorded_events[role]["events"] += 1
        recorded_events[role]["errors"] += bool(event.get("error"))
    tokens = {
        key: sum(fields.get(key, 0) for fields in usage.values())
        for key in TOKEN_FIELDS[:3]
    }
    elapsed = None
    if row.get("started_at") and row.get("completed_at"):
        elapsed = (
            datetime.fromisoformat(row["completed_at"].replace("Z", "+00:00"))
            - datetime.fromisoformat(row["started_at"].replace("Z", "+00:00"))
        ).total_seconds()
    return {
        "tool_calls": dict(tools),
        "tokens": tokens,
        "elapsed_seconds": elapsed,
        "usage_by_model": {model: usage.get(model, {}) for model in sorted(models)},
        "usage_models_recorded": sorted(usage),
        "events_available": isinstance(row.get("events"), list),
        "recorded_model_events": recorded_events,
        "recorded_tool_event_errors": sum(
            bool(event.get("error")) for event in events if event.get("event") == "tool"
        ),
        "limit": row.get("limit"),
    }


def summarize_metrics(outcomes: list[dict]) -> dict:
    count = len(outcomes)
    models = sorted({model for row in outcomes for model in row["usage_by_model"]})
    usage_by_model = {}
    for model in models:
        fields = sorted(
            set(TOKEN_FIELDS)
            | {
                field
                for row in outcomes
                for field in row["usage_by_model"].get(model, {})
            }
        )
        recorded = {
            field: sum(
                field in row["usage_by_model"].get(model, {}) for row in outcomes
            )
            for field in fields
        }
        with_usage = sum(model in row["usage_models_recorded"] for row in outcomes)
        usage_by_model[model] = {
            "totals": {
                field: (
                    sum(
                        row["usage_by_model"].get(model, {}).get(field, 0)
                        for row in outcomes
                    )
                    if recorded[field]
                    else None
                )
                for field in fields
            },
            "samples_with_usage": with_usage,
            "samples_without_usage": count - with_usage,
            "field_samples_recorded": recorded,
            "field_samples_missing": {
                field: count - n for field, n in recorded.items()
            },
        }
    tool_totals: Counter[str] = Counter()
    token_totals: Counter[str] = Counter()
    event_totals = {
        role: {"events": 0, "errors": 0} for role in ("solver", "judge", "other")
    }
    durations = []
    for row in outcomes:
        tool_totals.update(row["tool_calls"])
        token_totals.update(row["tokens"])
        if row["elapsed_seconds"] is not None:
            durations.append(row["elapsed_seconds"])
        for role, values in row["recorded_model_events"].items():
            for key, value in values.items():
                event_totals[role][key] += value
    with_usage = sum(bool(row["usage_models_recorded"]) for row in outcomes)
    with_events = sum(row["events_available"] for row in outcomes)
    limits = [row["limit"] for row in outcomes if row["limit"] is not None]
    return {
        "tool_call_totals": dict(tool_totals),
        "recorded_token_totals": dict(token_totals),
        "sum_sample_elapsed_seconds": sum(durations),
        "samples_with_elapsed_time": len(durations),
        "recorded_usage_by_model": usage_by_model,
        "metric_availability": {
            "samples_with_usage": with_usage,
            "samples_without_usage": count - with_usage,
            "samples_with_events": with_events,
            "samples_without_events": count - with_events,
            "samples_without_elapsed_time": count - len(durations),
        },
        "recorded_model_event_totals": event_totals,
        "recorded_tool_event_errors": sum(
            row["recorded_tool_event_errors"] for row in outcomes
        ),
        "samples_with_recorded_limit": len(limits),
        "limit_type_counts": dict(
            Counter(limit.get("type", "unknown") for limit in limits)
        ),
        "metrics_scope": (
            "Selected native outcomes only; not all paid attempts or billed cost. "
            "Model-event roles use distinct configured model identities; shared or unknown identities are other. "
            "Event errors count recorded failed events, not inferred transport retries. "
            "Usage availability means fields present in native samples; upstream defaults cannot be distinguished. "
            "Legacy token totals sum observed fields across models; consult availability counts. "
            "Native total_tokens is preserved; reasoning_tokens is not added to it."
        ),
        "duration_scope": (
            "Sum of selected native sample started_at-to-completed_at intervals, which may overlap. "
            "Not campaign wall time or total attempt latency. Saved-answer repairs may retain original sample timestamps. "
            "Recorded limits are reported separately; missing timestamps are excluded."
        ),
    }


def aggregate(
    config_path: Path,
    manifest_path: Path,
    protocol_path: Path,
    output: Path,
    *,
    ledger_path: Path | None = None,
    judge_result: Path | None = None,
    judge_adoption: Path | None = None,
    attempts: list[Path] | None = None,
    allowed_commits: set[str] | None = None,
    selections: dict[str, list[str]] | None = None,
) -> dict:
    config, manifest, protocol = map(
        read_json, (config_path, manifest_path, protocol_path)
    )
    ids = list(map(str, manifest["ids"]))
    require(
        ids and len(ids) == len(set(ids)), "expected IDs must be nonempty and unique"
    )
    require(
        config.get("run_phase") == "production",
        "diagnostic runs cannot enter final aggregation",
    )
    require(len(ids) == config["evaluation_count"], "expected dataset count mismatch")
    require(
        manifest["dataset_revision"] == config["dataset_revision"]
        and manifest["dataset_variant"] == "standard",
        "expected dataset protocol mismatch",
    )
    require(
        config["task"] == "hle_tools" and config["search_backend"] == "keenable",
        "only the frozen live HLE tools condition is accepted",
    )
    require(
        protocol["config_sha256"][config_path.name] == file_digest(config_path),
        "config differs from protocol record",
    )
    integrity_path = protocol_path.parent / protocol["dataset_manifest"]
    integrity = read_json(integrity_path)
    # The campaign integrity record uses default json.dumps separators; recovery
    # ledgers separately use compact canonical JSON for their ordered-ID hash.
    require(
        integrity["revision"] == config["dataset_revision"]
        and integrity["count"] == len(ids)
        and integrity["ordered_ids_sha256"]
        == hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        "expected IDs differ from frozen dataset integrity record",
    )
    for relative, checksum in protocol["file_sha256"].items():
        checked_file(Path(protocol["source_root"]) / relative, checksum)
    inputs, approvals, provenance = sources(
        config_path,
        manifest,
        ledger_path,
        judge_result,
        attempts or [],
        allowed_commits or set(),
        selections or {},
        judge_adoption,
    )
    outcomes = {}
    rejected_attempts = []
    index = []
    for entry in inputs:
        path = checked_file(entry["path"], entry["sha256"])
        explicit = entry.get("selected_ids")
        header = read_eval_log(path, header_only=True).model_dump(
            mode="json", exclude_none=True
        )
        check_header(header, config, explicit is not None)
        revision = header["eval"].get("revision")
        if entry["kind"] == "production":
            require(
                revision
                and 7 <= len(revision["commit"]) <= 40
                and not revision.get("dirty")
                and entry["source_commit"].startswith(revision["commit"]),
                "archive code revision differs from clean approved launch",
            )
        else:
            require(revision == entry["revision"], "historical source revision changed")
            require(
                header["eval"].get("metadata", {}).get("hle_recovery_source_sha256")
                == entry["source_sha256"],
                "historical source metadata mismatch",
            )
        seen, accepted = set(), []
        for row in iter_samples(path):
            sample_id = str(row["id"])
            require(
                sample_id in entry["ids"]
                and sample_id not in seen
                and row.get("epoch", 1) == 1,
                "archive has duplicate/unknown ID or epoch",
            )
            seen.add(sample_id)
            if explicit is not None and sample_id not in explicit:
                continue
            kind, reason = disposition(row)
            if kind != "retain_score":
                require(
                    entry["kind"] == "production"
                    and explicit is None
                    and reason
                    in {
                        "provider_infrastructure_error",
                        "malformed_judge_result",
                        "judge_provider_infrastructure_error",
                    },
                    f"unaccepted outcome {sample_id}: {reason}",
                )
                rejected_attempts.append(
                    {
                        "id": sample_id,
                        "archive_sha256": entry["sha256"],
                        "disposition": kind,
                        "reason": reason,
                    }
                )
                continue
            require(
                sample_id not in outcomes, f"overlapping accepted outcome: {sample_id}"
            )
            scores = row.get("scores", {})
            require(
                set(scores) == {"hle_scorer"}
                and scores["hle_scorer"]["value"] in {"C", "I"},
                "unexpected score name/value",
            )
            require(
                not scores["hle_scorer"]
                .get("metadata", {})
                .get("hle_judge_repair_failed"),
                "failed judge marker cannot count as incorrect",
            )
            generation = generation_digest(row)
            if entry["kind"] != "production":
                approved = approvals[sample_id]
                require(
                    approved["disposition"]
                    == (
                        "retain_score" if entry["kind"] == "retained" else "judge_only"
                    ),
                    "historical selection differs from ledger",
                )
                require(
                    generation == approved["generation_sha256"],
                    "historical generation changed",
                )
            events = [
                event
                for event in row.get("events", [])
                if event.get("event") == "model"
                and event.get("model") == config["model"]
            ]
            require(events, "accepted outcome has no recorded solver model request")
            for event in events:
                check_settings(
                    event.get("config", {}), config, f"actual request for {sample_id}"
                )
            outcomes[sample_id] = {
                "id": sample_id,
                "epoch": 1,
                "correct": scores["hle_scorer"]["value"] == "C",
                "archive": str(path),
                "archive_sha256": entry["sha256"],
                "kind": entry["kind"],
                "generation_sha256": generation,
                **row_metrics(row, config),
            }
            accepted.append(sample_id)
        required = set(explicit) if explicit is not None else set(entry["ids"])
        require(
            required <= seen if explicit is not None else required == seen,
            "archive is missing expected IDs",
        )
        require(
            set(accepted) == required if explicit is not None else True,
            "explicit selection includes unresolved IDs",
        )
        index.append(
            {
                **entry,
                "path": str(path),
                "archive_status": header["status"],
                "selected_ids": accepted,
            }
        )
    missing = set(ids) - set(outcomes)
    require(
        not missing and set(outcomes) == set(ids),
        f"condition is unresolved: {len(missing)} missing outcomes",
    )
    correct = sum(row["correct"] for row in outcomes.values())
    summary = {
        "version": 1,
        "complete": True,
        "configuration": config,
        **summarize_metrics(list(outcomes.values())),
        "model": config["model"],
        "dataset_revision": config["dataset_revision"],
        "dataset_ids_sha256": digest(ids),
        "dataset_integrity_sha256": file_digest(integrity_path),
        "samples": len(ids),
        "correct": correct,
        "incorrect": len(ids) - correct,
        "accuracy": correct / len(ids),
        "config_sha256": file_digest(config_path),
        "protocol_sha256": file_digest(protocol_path),
        "allowed_source_commits": sorted(allowed_commits or []),
        "source_counts": dict(Counter(row["kind"] for row in outcomes.values())),
        "resolved_failed_attempts": rejected_attempts,
        **provenance,
    }
    require(not output.exists(), "aggregation output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}-", dir=output.parent
    ) as temporary:
        staging = Path(temporary) / "result"
        staging.mkdir(mode=0o700)
        for name, value in (("summary.json", summary), ("shards.json", index)):
            (staging / name).write_text(
                json.dumps(value, indent=2, sort_keys=True) + "\n"
            )
        with (staging / "outcomes.jsonl").open("w") as stream:
            for sample_id in ids:
                stream.write(json.dumps(outcomes[sample_id], sort_keys=True) + "\n")
        for path in staging.iterdir():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
        os.rename(staging, output)
        fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    return summary


def main() -> None:
    os.umask(0o077)
    if sys.argv[1:2] == ["adopt-judge"]:
        parser = argparse.ArgumentParser(
            description="Bind an existing successful judge repair to a revised recovery ledger"
        )
        parser.add_argument("--ledger", type=Path, required=True)
        parser.add_argument("--judge-result", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        args = parser.parse_args(sys.argv[2:])
        binding = write_judge_adoption(args.ledger, args.judge_result, args.output)
        print(json.dumps(binding, sort_keys=True))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-commit", action="append", default=[])
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--judge-result", type=Path)
    parser.add_argument("--judge-adoption", type=Path)
    parser.add_argument("--attempt", type=Path, action="append", default=[])
    parser.add_argument(
        "--partial-selection",
        type=Path,
        help="JSON object mapping result.json absolute paths to explicitly selected IDs",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    commits = {
        subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "--verify", f"{commit}^{{commit}}"],
            text=True,
        ).strip()
        for commit in args.source_commit
    }
    summary = aggregate(
        args.config,
        args.expected_manifest,
        args.protocol,
        args.output,
        ledger_path=args.ledger,
        judge_result=args.judge_result,
        judge_adoption=args.judge_adoption,
        attempts=args.attempt,
        allowed_commits=commits,
        selections=(
            read_json(args.partial_selection) if args.partial_selection else None
        ),
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
