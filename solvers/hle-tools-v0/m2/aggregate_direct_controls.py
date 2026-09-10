#!/usr/bin/env python3
"""Validate the exact union of migration-owned and replacement direct controls."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from inspect_ai.log import read_eval_log, read_eval_log_sample_summaries

from aggregate_campaign import (
    checked_file,
    check_settings,
    read_json,
    require,
    row_metrics,
    sources,
)
from agent_baselines.evals.hle_tools_v0.recovery import (
    digest,
    disposition,
    file_digest,
    generation_digest,
    iter_samples,
)


def bound(specification: dict) -> Path:
    return checked_file(specification["path"], specification["sha256"])


def header(path: Path) -> dict:
    return read_eval_log(path, header_only=True).model_dump(
        mode="json", exclude_none=True
    )


def inventory(path: Path, allowed: list[str]) -> set[str]:
    summaries = read_eval_log_sample_summaries(path)
    ids = [str(sample.id) for sample in summaries]
    require(
        len(ids) == len(set(ids))
        and set(ids) <= set(allowed)
        and all(sample.epoch == 1 for sample in summaries),
        "native archive has duplicate/unknown sample or epoch",
    )
    return set(ids)


def membership(manifest: dict, protocol: dict) -> list[str]:
    ids = manifest["ids"]
    require(
        ids
        and all(isinstance(value, str) for value in ids)
        and len(ids) == len(set(ids)),
        "manifest IDs must be nonempty unique strings",
    )
    require(
        manifest["dataset_revision"] == protocol["dataset_revision"]
        and manifest["dataset_variant"] == protocol["dataset_variant"] == "standard",
        "manifest dataset differs from direct protocol",
    )
    return ids


def preparation(protocol: dict, lane_name: str) -> dict:
    lane = protocol["lanes"][lane_name]
    config_path = bound(lane["config"])
    config = read_json(config_path)
    require(
        config["task"] == "hle_direct"
        and config["run_phase"] == "production"
        and not config.get("search_backend"),
        "only production direct controls are accepted",
    )
    require(
        config["concurrency"] == protocol["replacement_concurrency"]
        and config["model"] == lane["model"]
        and config["model_base_url"] == lane["endpoint_url"]
        and config["native_context"] == lane["native_context"],
        "direct configuration differs from bound lane",
    )
    require(
        config["dataset_revision"] == protocol["dataset_revision"]
        and config["dataset_variant"] == protocol["dataset_variant"],
        "configuration dataset mismatch",
    )
    for key, value in protocol["sampling"].items():
        expected = value[lane_name] if key == "top_p" else value
        require(config.get(key) == expected, f"configuration sampling mismatch: {key}")
    all_ids = membership(read_json(bound(protocol["all_manifest"])), protocol)
    require(len(all_ids) == protocol["evaluation_count"], "dataset count mismatch")
    for name, checksum in protocol["source_file_sha256"].items():
        checked_file(Path(protocol["source_root"]) / name, checksum)
    for specification in lane["endpoint_proof_files"]:
        bound(specification)
    proofs = {Path(spec["path"]).name: bound(spec) for spec in lane["selection_proofs"]}
    require(len(proofs) == len(lane["selection_proofs"]), "duplicate selection proof")
    migration_ids = membership(
        read_json(proofs["ids-current-migration-owned.json"]), protocol
    )
    replacement_ids = membership(
        read_json(proofs["ids-generate-matched-current.json"]), protocol
    )
    require(
        len(migration_ids) == lane["migration_owned_count"]
        and len(replacement_ids) == lane["replacement_count"]
        and not set(migration_ids) & set(replacement_ids)
        and set(migration_ids) | set(replacement_ids) == set(all_ids),
        "ownership partitions must cover the exact dataset once",
    )
    manifests = {}
    shard_ids: list[str] = []
    for specification in lane["manifests"]:
        path = bound(specification)
        manifests[str(path)] = specification["sha256"]
        shard_ids.extend(membership(read_json(path), protocol))
    require(
        len(shard_ids) == len(set(shard_ids))
        and set(shard_ids) == set(replacement_ids),
        "replacement manifests overlap or differ from ownership",
    )
    lineage = read_json(proofs["lineage.json"])
    require(
        lineage["dataset_revision"] == protocol["dataset_revision"]
        and lineage["dataset_ids_sha256"] == digest(all_ids)
        and lineage["model"] == config["model"]
        and lineage["endpoint_url"] == config["model_base_url"]
        and lineage["migration_job_id"] == lane["migration_job"]
        and lineage["endpoint_job_id"] == lane["endpoint_job"]
        and lineage["native_context"] == lane["native_context"]
        and lineage["frozen_checkpoint"] == lane["migration_snapshot"]
        and set(lineage["old_excluded"]) == set(replacement_ids),
        "lineage differs from the frozen direct partition",
    )
    bound(lineage["old_source"])
    snapshot = bound(lane["migration_snapshot"])
    original = header(snapshot)
    inventory(snapshot, all_ids)
    require(
        original["eval"] == lineage["current_header"]["eval"],
        "snapshot identity mismatch",
    )
    # Preserve already valid migration answers even when the final source is newer.
    baseline = {}
    for row in iter_samples(snapshot):
        sample_id = str(row["id"])
        if sample_id not in lineage["current_scored_candidates"]:
            continue
        require(sample_id in migration_ids, "candidate outside migration ownership")
        require(
            generation_digest(row)
            == lineage["current_scored_candidates"][sample_id]["generation_sha256"],
            "candidate generation differs from lineage",
        )
        if disposition(row)[0] == "retain_score":
            baseline[sample_id] = {
                "generation": generation_digest(row),
                "scores": row.get("scores", {}),
            }
    return {
        "lane": lane,
        "config": config,
        "config_path": config_path,
        "all_ids": all_ids,
        "migration_ids": migration_ids,
        "replacement_ids": replacement_ids,
        "manifests": manifests,
        "lineage": lineage,
        "baseline": baseline,
    }


def check_direct_header(
    actual: dict,
    config: dict,
    *,
    migration: bool,
    partial: bool,
    expected_spec: dict | None = None,
) -> None:
    allowed = {"success", "started", "cancelled"} if partial else {"success"}
    require(
        actual["status"] in allowed, "archive status requires explicit clean selection"
    )
    require(
        not actual.get("error") and not actual.get("invalidated"), "invalid archive"
    )
    spec = actual["eval"]
    require(
        spec["task"].split("/")[-1] == "hle_direct"
        and spec["model"] == config["model"]
        and spec.get("model_base_url") == config["model_base_url"],
        "direct task/model/endpoint mismatch",
    )
    args = spec["task_args"]
    for key, expected in (
        ("dataset_revision", config["dataset_revision"]),
        ("dataset_variant", "standard"),
        ("data_path", config["dataset_path"]),
        ("fixture", False),
        ("judge_model", config["judge_model"]),
        ("judge_reasoning_effort", config["judge_reasoning_effort"]),
    ):
        require(args.get(key) == expected, f"direct task argument mismatch: {key}")
    require(
        spec.get("config", {}).get("max_samples") == config["concurrency"]
        and spec["config"].get("epochs", 1) == 1
        and spec["config"].get("time_limit") == 1800
        and spec["config"].get("message_limit") == 10,
        "direct task limits/concurrency mismatch",
    )
    header_config = dict(config)
    if migration:
        require(spec == expected_spec, "migration archive identity/protocol changed")
        require(
            args.get("manifest_path") is None, "migration unexpectedly has a manifest"
        )
        header_config["concurrency"] = 3
    check_settings(
        spec.get("model_generate_config", {}), header_config, "direct header"
    )
    revision = spec.get("revision", {})
    require(
        revision.get("commit") and not revision.get("dirty"), "unclean source revision"
    )


def accepted(row: dict, config: dict) -> dict:
    kind, reason = disposition(row)
    require(
        kind == "retain_score",
        f"unresolved direct outcome: {reason}; judge-only recovery requires explicit provenance",
    )
    scores = row.get("scores", {})
    require(
        set(scores) == {"hle_scorer"}
        and scores["hle_scorer"]["value"] in {"C", "I"}
        and not scores["hle_scorer"].get("metadata", {}).get("hle_judge_repair_failed"),
        "unexpected direct score schema",
    )
    require(
        not any(event.get("event") == "tool" for event in row.get("events", [])),
        "direct output contains tool calls",
    )
    requests = [
        event
        for event in row.get("events", [])
        if event.get("event") == "model" and event.get("model") == config["model"]
    ]
    require(requests, "direct row has no solver request")
    for event in requests:
        require(not event.get("tools"), "direct request exposes tools")
        check_settings(event.get("config", {}), config, "actual direct request")
    return {
        "id": str(row["id"]),
        "epoch": 1,
        "correct": scores["hle_scorer"]["value"] == "C",
        "generation_sha256": generation_digest(row),
        **row_metrics(row),
    }


def aggregate(
    protocol_path: Path,
    lane_name: str,
    output: Path,
    *,
    migration_sources: list[Path],
    replacement_attempts: list[Path],
    selections: dict[str, list[str]] | None = None,
) -> dict:
    protocol = read_json(protocol_path)
    prepared = preparation(protocol, lane_name)
    lane, config = prepared["lane"], prepared["config"]
    migration_config = {**config, "concurrency": protocol["migration_concurrency"]}
    expected_spec = prepared["lineage"]["current_header"]["eval"]
    entries: list[dict[str, Any]] = []
    for source_path in migration_sources:
        source = read_json(source_path)
        require(
            source.get("closed") is True
            and source.get("protocol_sha256") == file_digest(protocol_path)
            and source.get("lane") == lane_name
            and source.get("migration_job") == lane["migration_job"],
            "migration source requires bound closed-job attestation",
        )
        bound(source["closure_evidence"])
        path = bound(source["archive"])
        selected = source.get("selected_ids")
        if selected is not None:
            require(
                selected
                and len(selected) == len(set(selected))
                and set(selected) <= set(prepared["migration_ids"]),
                "invalid migration clean selection",
            )
        entries.append(
            {
                "path": str(path),
                "sha256": source["archive"]["sha256"],
                "kind": "migration",
                "ids": prepared["all_ids"],
                "selected_ids": selected or prepared["migration_ids"],
                "explicit": selected is not None,
                "attestation": str(source_path.resolve()),
                "attestation_sha256": file_digest(source_path),
            }
        )
    replacement_manifest = {
        "ids": prepared["replacement_ids"],
        "dataset_revision": protocol["dataset_revision"],
        "dataset_variant": protocol["dataset_variant"],
    }
    replacements, _, _ = sources(
        prepared["config_path"],
        replacement_manifest,
        None,
        None,
        replacement_attempts,
        {protocol["source_commit"]},
        selections or {},
    )
    for entry in replacements:
        result_path = Path(entry["result"]).resolve()
        launch = read_json(result_path.parent / "launch.json")
        manifests = [
            arg.split("=", 1)[1]
            for arg in launch["command"]
            if arg.startswith("manifest_path=")
        ]
        manifest_path = str(Path(manifests[0]).resolve())
        require(
            prepared["manifests"].get(manifest_path) == launch["manifest_sha256"],
            "replacement launch uses an unapproved manifest",
        )
        archive_path = Path(entry["path"]).resolve()
        require(
            archive_path.parent == result_path.parent / "logs"
            and list(archive_path.parent.glob("*.eval")) == [archive_path],
            "replacement result must bind its single durable archive",
        )
        entry.update(
            kind="replacement",
            manifest_path=manifest_path,
            explicit=entry["selected_ids"] is not None,
        )
        entries.append(entry)
    outcomes: dict[str, dict] = {}
    index = []
    for entry in entries:
        path = checked_file(entry["path"], entry["sha256"])
        actual = header(path)
        native_ids = inventory(path, entry["ids"])
        migration = entry["kind"] == "migration"
        row_config = migration_config if migration else config
        check_direct_header(
            actual,
            row_config,
            migration=migration,
            partial=entry["explicit"],
            expected_spec=expected_spec,
        )
        if not migration:
            revision = actual["eval"]["revision"]["commit"]
            require(
                7 <= len(revision) <= 40
                and protocol["source_commit"].startswith(revision),
                "replacement source revision differs from approved launch",
            )
            require(
                actual["eval"]["task_args"].get("manifest_path")
                == entry["manifest_path"],
                "replacement archive manifest differs from launch",
            )
        selected = entry["selected_ids"]
        required = set(selected if selected is not None else entry["ids"])
        seen = set()
        accepted_ids = []
        for row in iter_samples(path):
            sample_id = str(row["id"])
            require(
                sample_id in entry["ids"]
                and sample_id not in seen
                and row.get("epoch", 1) == 1,
                "duplicate/unknown sample or epoch",
            )
            seen.add(sample_id)
            if sample_id not in required:
                continue
            require(sample_id not in outcomes, "overlapping accepted outcome")
            outcome = accepted(row, row_config)
            if migration and sample_id in prepared["baseline"]:
                baseline = prepared["baseline"][sample_id]
                require(
                    outcome["generation_sha256"] == baseline["generation"]
                    and row.get("scores", {}) == baseline["scores"],
                    "previously valid migration answer or score changed",
                )
            outcome.update(
                archive=str(path),
                archive_sha256=entry["sha256"],
                stratum=entry["kind"],
                concurrency=row_config["concurrency"],
            )
            outcomes[sample_id] = outcome
            accepted_ids.append(sample_id)
        require(seen == native_ids, "native sample index hides stored rows")
        require(required <= seen, "archive lacks required owned IDs")
        if not entry["explicit"] and not migration:
            require(
                seen == set(entry["ids"]), "replacement archive membership mismatch"
            )
        index.append(
            {**entry, "archive_status": actual["status"], "selected_ids": accepted_ids}
        )
    require(
        set(outcomes) == set(prepared["all_ids"]),
        f"direct condition unresolved: {len(set(prepared['all_ids']) - set(outcomes))} missing IDs",
    )
    correct = sum(row["correct"] for row in outcomes.values())
    summary = {
        "version": 1,
        "complete": True,
        "lane": lane_name,
        "task": "hle_direct",
        "samples": len(outcomes),
        "correct": correct,
        "incorrect": len(outcomes) - correct,
        "accuracy": correct / len(outcomes),
        "protocol": str(protocol_path.resolve()),
        "protocol_sha256": file_digest(protocol_path),
        "config_sha256": file_digest(prepared["config_path"]),
        "dataset_revision": protocol["dataset_revision"],
        "dataset_ids_sha256": digest(prepared["all_ids"]),
        "source_counts": dict(Counter(row["stratum"] for row in outcomes.values())),
        "strata": {
            "migration": {
                "concurrency": migration_config["concurrency"],
                "source_revision": expected_spec["revision"],
                "inherited_header_max_connections": 3,
            },
            "replacement": {
                "concurrency": config["concurrency"],
                "source_commit": protocol["source_commit"],
            },
        },
        "qualification": (
            "Direct controls retain distinct migration and replacement provenance. "
            "Their concurrency differs from the tools condition at concurrency 3; "
            "these results are not an identical-load or first-attempt comparison."
        ),
    }
    require(not output.exists(), "output already exists")
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
            for sample_id in prepared["all_ids"]:
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--lane", choices=("glm-flash", "k2-horizon"), required=True)
    parser.add_argument("--migration-source", type=Path, action="append", default=[])
    parser.add_argument("--replacement-attempt", type=Path, action="append", default=[])
    parser.add_argument("--selections", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(
        args.protocol,
        args.lane,
        args.output,
        migration_sources=args.migration_source,
        replacement_attempts=args.replacement_attempt,
        selections=read_json(args.selections) if args.selections else None,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
