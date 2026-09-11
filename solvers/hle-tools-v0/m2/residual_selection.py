#!/usr/bin/env python3
"""Select bounded production residuals without dispatching model work.

Supply every launched attempt for the initial new-generation ID set, including
launch-only attempts. Every regeneration needs an explicit infrastructure
selection: {ID: {"attempt": "/absolute/attempt", "closed": true, "evidence": "..."}}
Closure must follow an actual stopped/finished job check, never elapsed time.
Partial selections map absolute failed attempt directories to clean sample IDs.
Rerun against current complete attempt records immediately before dispatch.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from inspect_ai.log import read_eval_log

from aggregate_campaign import (
    check_header,
    check_settings,
    checked_file,
    config_binding,
    launch_configuration,
    read_json,
    require,
    runtime_configurations,
)
from agent_baselines.evals.hle_tools_v0.recovery import (
    disposition,
    file_digest,
    generation_digest,
    iter_samples,
)


def binding(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": file_digest(path)}


def read_attempt(
    directory: Path,
    config_path: Path,
    config: dict,
    expected: set[str],
    allowed_commits: set[str],
    partial: dict,
    configurations: list[dict] | None = None,
) -> tuple[dict, dict]:
    directory = directory.resolve()
    launch_path = directory / "launch.json"
    launch = read_json(launch_path)
    runtime = launch_configuration(
        launch, configurations or runtime_configurations(config_path)
    )
    attempt_config = runtime["configuration"]
    require(
        launch["source_commit"] in allowed_commits,
        "attempt source commit is not approved",
    )
    require(
        isinstance(launch.get("started_at"), (int, float)),
        "attempt has no launch timestamp",
    )
    manifests = [
        arg.split("=", 1)[1]
        for arg in launch["command"]
        if arg.startswith("manifest_path=")
    ]
    require(len(manifests) == 1, "attempt must bind one manifest")
    manifest_path = checked_file(manifests[0], launch["manifest_sha256"])
    manifest = read_json(manifest_path)
    require(manifest == launch["manifest"], "embedded launch manifest differs")
    ids = list(map(str, manifest["ids"]))
    require(
        ids and len(ids) == len(set(ids)) and set(ids) <= expected,
        "unknown or duplicate manifest IDs",
    )
    # Earlier production shards contained only IDs; their native header must
    # supply the revision already bound by the config and expected manifest.
    legacy_manifest = "dataset_revision" not in manifest
    require(
        legacy_manifest or manifest["dataset_revision"] == config["dataset_revision"],
        "attempt manifest dataset revision differs",
    )
    require(
        manifest.get("dataset_variant", "standard") == "standard",
        "attempt manifest variant differs",
    )
    selection = partial.get(str(directory))
    if selection is not None:
        require(
            selection
            and len(selection) == len(set(selection))
            and set(selection) <= set(ids),
            "partial selection has unknown or duplicate IDs",
        )
    record = {
        "attempt": str(directory),
        "launch": binding(launch_path),
        "manifest": binding(manifest_path),
        "ids": ids,
        "started_at": launch["started_at"],
        "source_commit": launch["source_commit"],
        "retry_of": manifest.get("retry_of"),
        "selection_ledger": manifest.get("selection_ledger"),
        "generation_attempt": manifest.get("generation_attempt", 1),
    }
    if runtime["sha256"] != file_digest(config_path):
        record["runtime_config"] = config_binding(runtime)
    rows: dict[str, dict[str, Any]] = {}
    durable_archives = sorted((directory / "logs").glob("*.eval"))
    result_path = directory / "result.json"
    if not result_path.exists():
        require(not legacy_manifest, "legacy manifest requires bound native archive")
        if durable_archives:
            record["unfinalized_archives"] = [str(path) for path in durable_archives]
        require(
            selection is None,
            "launch-only attempt cannot contribute a partial selection",
        )
        return record, rows
    result = read_json(result_path)
    record["result"] = binding(result_path)
    record["complete"] = result.get("complete") is True
    if not result.get("archive"):
        require(not legacy_manifest, "legacy manifest requires bound native archive")
        if durable_archives:
            record["unfinalized_archives"] = [str(path) for path in durable_archives]
        require(selection is None, "attempt has no archive for partial selection")
        return record, rows
    archive = checked_file(result["archive"], result["archive_sha256"])
    require(
        archive.parent == (directory / "logs").resolve(),
        "result archive does not belong to its attempt",
    )
    require(
        len(durable_archives) == 1 and durable_archives[0].resolve() == archive,
        "attempt must contain exactly its bound durable archive",
    )
    record["archive"] = binding(archive)
    require(
        result.get("publisher_exit") == 0, "archive has no successful final publication"
    )
    header = read_eval_log(archive, header_only=True).model_dump(
        mode="json", exclude_none=True
    )
    check_header(header, attempt_config, partial=not record["complete"])
    require(
        header["eval"]["task_args"].get("manifest_path") == str(manifest_path),
        "archive manifest path differs from its launch",
    )
    revision = header["eval"].get("revision") or {}
    require(
        7 <= len(revision.get("commit", "")) <= 40
        and not revision.get("dirty")
        and launch["source_commit"].startswith(revision["commit"]),
        "archive revision differs from clean approved launch",
    )
    for row in iter_samples(archive):
        sample_id = str(row["id"])
        require(
            sample_id in ids and sample_id not in rows and row.get("epoch", 1) == 1,
            "archive has unknown or duplicate ID/epoch",
        )
        kind, reason = disposition(row)
        events = [
            event
            for event in row.get("events", [])
            if event.get("event") == "model" and event.get("model") == config["model"]
        ]
        for event in events:
            check_settings(
                event.get("config", {}),
                attempt_config,
                f"actual request for {sample_id}",
            )
        if kind in {"retain_score", "judge_only"}:
            require(events, "saved outcome has no recorded solver request")
        if kind == "retain_score":
            require(
                set(row["scores"]) == {"hle_scorer"}
                and not row["scores"]["hle_scorer"]
                .get("metadata", {})
                .get("hle_judge_repair_failed"),
                "invalid HLE score or failed judge marker",
            )
        rows[sample_id] = {
            "attempt": str(directory),
            "disposition": kind,
            "reason": reason,
            "generation_sha256": generation_digest(row),
            "archive": str(archive),
            "archive_sha256": record["archive"]["sha256"],
            "selected": kind == "retain_score"
            and (
                record["complete"] or (selection is not None and sample_id in selection)
            ),
        }
        if kind == "retain_score":
            rows[sample_id]["score"] = row["scores"]["hle_scorer"]["value"]
    if record["complete"]:
        require(
            set(rows) == set(ids)
            and all(r["disposition"] == "retain_score" for r in rows.values()),
            "completed attempt contains missing or unaccepted rows",
        )
    if selection is not None:
        require(
            all(
                sample_id in rows and rows[sample_id]["selected"]
                for sample_id in selection
            ),
            "partial selection includes missing or unaccepted rows",
        )
    return record, rows


def read_preflight_exclusion(
    path: Path | None, protocol: dict, attempts: dict, outcomes: dict
) -> dict | None:
    reference = protocol.get("preflight_exclusion")
    require(
        (path is None) == (reference is None),
        "preflight exclusion must be explicitly supplied and protocol-bound",
    )
    if path is None:
        return None
    require(reference == binding(path), "preflight exclusion protocol binding differs")
    proof = read_json(path)
    require(
        isinstance(proof, dict)
        and proof.get("version") == 1
        and proof.get("kind") == "closed_preflight_before_benchmark"
        and proof.get("benchmark_admissions") == 0
        and proof.get("max_replacement_admissions") == 1
        and proof.get("raw_assignment_retained") is True
        and proof.get("reviewed_pre_exec_failure") is True,
        "one closed preflight exclusion with one replacement admission is required",
    )
    directory = str(Path(proof["attempt"]).resolve())
    require(directory in attempts, "preflight exclusion names an unprovided attempt")
    record = attempts[directory]

    def evidence(key: str, expected: Path | None = None) -> Path:
        item = proof[key]
        result = checked_file(item["path"], item["sha256"])
        require(expected is None or result == expected, f"wrong preflight {key} path")
        return result

    root = Path(directory)
    launch_path = evidence("launch", root / "launch.json")
    manifest_path = evidence("manifest", Path(record["manifest"]["path"]))
    result_path = evidence("result", root / "result.json")
    log_path = evidence("eval_log", root / "eval.log")
    publisher_path = evidence("publisher_status", root / "checkpoint-status.json")
    launch, result, publisher = map(
        read_json, (launch_path, result_path, publisher_path)
    )
    require(
        proof["ids"] == record["ids"]
        and proof["source_commit"] == record["source_commit"]
        and proof["job_id"] == launch.get("job_id")
        and proof["launch"] == record["launch"]
        and proof["manifest"] == record["manifest"]
        and proof["result"] == record.get("result")
        and read_json(manifest_path) == launch["manifest"]
        and record["generation_attempt"] == 2,
        "preflight exclusion identity differs from its retry assignment",
    )
    require(
        not outcomes[directory]
        and not record.get("archive")
        and not record.get("unfinalized_archives")
        and not list((root / "logs").glob("*.eval"))
        and result.get("complete") is False
        and result.get("eval_exit") == 2
        and result.get("publisher_exit") == 1
        and not result.get("archive")
        and not result.get("archive_sha256"),
        "preflight exclusion cannot contain benchmark outcomes or archives",
    )
    require(
        all(
            publisher.get(key) == 0
            for key in ("expected", "published", "total_published")
        )
        and publisher.get("errors") == []
        and publisher.get("last_published_at") is None
        and publisher.get("last_error") is None,
        "preflight publisher observed benchmark archive activity",
    )
    wrapper, smoke = evidence("wrapper"), evidence("smoke")
    require(
        wrapper.name == "run_with_secrets.py"
        and smoke == wrapper.with_name("smoke.py"),
        "preflight reviewed source files differ from the wrapper entry point",
    )
    command = launch["command"]
    require(
        len(command) > 2
        and command[1] == str(wrapper)
        and "--skip-smoke-test" not in command
        and any(
            Path(token).name == "inspect" and command[index + 1] == "eval"
            for index, token in enumerate(command[:-1])
        ),
        "preflight wrapper was not the launch entry point",
    )
    text = log_path.read_text()
    marker = (
        f"{wrapper}:67: RuntimeWarning: HLE smoke test failed; evaluation stopped: "
        f"evaluated-model endpoint returned an empty response for {launch['config']['model']}"
    )
    require(
        text.count(marker) == 1 and "HLE smoke test passed:" not in text,
        "missing positive pre-exec model preflight failure evidence",
    )
    closure = read_json(evidence("closure"))
    require(
        closure.get("job_id") == proof["job_id"]
        and closure.get("state") == "FAILED"
        and closure.get("exit_code") == "2:0"
        and closure.get("closed") is True,
        "preflight job lacks exact terminal closure evidence",
    )
    return {"attempt": directory, "binding": reference}


def verify_retry(
    record: dict,
    sample_id: str,
    prior: dict,
    config_sha: str,
    exclusion: dict | None = None,
) -> None:
    require(record["generation_attempt"] == 2, "retry must be generation attempt 2")
    references = record.get("retry_of")
    require(
        isinstance(references, dict) and set(references) == set(record["ids"]),
        "retry manifest has incomplete prior bindings",
    )
    assert isinstance(references, dict)
    reference = references[sample_id]
    require(
        reference
        == {"attempt": prior["attempt"], "launch_sha256": prior["launch"]["sha256"]},
        "retry prior launch binding differs",
    )
    require(
        record["started_at"] > prior["started_at"], "retry predates its prior attempt"
    )
    ledger_ref = record.get("selection_ledger")
    require(isinstance(ledger_ref, dict), "retry has no immutable selection ledger")
    assert isinstance(ledger_ref, dict)
    ledger = read_json(checked_file(ledger_ref["path"], ledger_ref["sha256"]))
    require(
        ledger.get("config", {}).get("sha256") == config_sha,
        "retry ledger configuration differs",
    )
    entries = [row for row in ledger.get("samples", []) if row["id"] == sample_id]
    require(
        len(entries) == 1
        and entries[0].get("action") == "retry"
        and entries[0].get("launched_generation_attempts") == 1
        and entries[0].get("diagnosis", {}).get("closed") is True,
        "retry was not authorized by the selection ledger",
    )
    history = entries[0]["attempts"]
    skipped = [r for r in history if r.get("counts_toward_generation_budget") is False]
    if skipped:
        require(
            exclusion is not None
            and ledger.get("preflight_exclusion") == exclusion["binding"]
            and [r["attempt"] for r in skipped] == [exclusion["attempt"]],
            "retry ledger has an unapproved preflight exclusion",
        )
    require(
        [r["attempt"] for r in history if r not in skipped] == [prior["attempt"]],
        "retry ledger has different prior attempts",
    )
    prior_record = ledger["attempts"][prior["attempt"]]
    for key in ("launch", "manifest", "result", "archive"):
        if key in prior_record:
            require(
                prior_record[key] == prior.get(key),
                f"retry ledger prior {key} checksum differs",
            )


def verify_initial_selection(
    record: dict,
    attempts: dict,
    config: dict,
    expected_manifest: dict,
) -> None:
    reference = record["selection_ledger"]
    require(
        isinstance(reference, dict),
        "initial selection has no immutable ledger binding",
    )
    ledger = read_json(checked_file(reference["path"], reference["sha256"]))
    require(
        ledger.get("config") == config
        and ledger.get("expected_manifest") == expected_manifest
        and ledger.get("max_additional_generation_attempts") == 1,
        "initial selection ledger configuration, manifest or attempt cap differs",
    )
    for sample_id in record["ids"]:
        entries = [row for row in ledger.get("samples", []) if row["id"] == sample_id]
        require(
            len(entries) == 1
            and entries[0].get("action") == "unlaunched"
            and entries[0].get("raw_assigned_launches") == 0
            and entries[0].get("launched_generation_attempts") == 0
            and entries[0].get("attempts") == []
            and "selected" not in entries[0]
            and "diagnosis" not in entries[0],
            "initial selection ledger does not prove an unlaunched sample",
        )
    prior_attempts = ledger.get("attempts")
    require(isinstance(prior_attempts, dict), "initial selection ledger has no history")
    assert isinstance(prior_attempts, dict)
    for directory, prior in prior_attempts.items():
        actual = attempts.get(directory)
        require(actual is not None, "initial selection ledger prior attempt is missing")
        assert actual is not None
        require(
            actual["started_at"] < record["started_at"],
            "initial selection ledger contains a non-prior attempt",
        )
        for key in (
            "attempt",
            "launch",
            "manifest",
            "ids",
            "started_at",
            "source_commit",
            "generation_attempt",
            "retry_of",
            "selection_ledger",
        ):
            require(
                key in prior and prior[key] == actual.get(key),
                f"initial selection ledger prior {key} differs",
            )
        require(
            not set(prior["ids"]) & set(record["ids"]),
            "initial selection ledger contains a prior sample assignment",
        )
        for key in ("result", "archive", "runtime_config"):
            if key in prior:
                require(
                    prior[key] == actual.get(key),
                    f"initial selection ledger prior {key} checksum differs",
                )


def select(
    config_path: Path,
    expected_path: Path,
    protocol_path: Path,
    output: Path,
    *,
    attempt_dirs: list[Path],
    allowed_commits: set[str],
    partial: dict | None = None,
    infrastructure: dict | None = None,
    shard_size: int = 200,
    preflight_exclusion: Path | None = None,
    allowed_runtime_configs: dict[str, str] | None = None,
) -> dict:
    config, expected_manifest, protocol = map(
        read_json, (config_path, expected_path, protocol_path)
    )
    require(
        config.get("run_phase") == "production"
        and config.get("task") == "hle_tools"
        and config.get("search_backend") == "keenable",
        "diagnostic or incompatible configuration",
    )
    require(
        protocol["config_sha256"][config_path.name] == file_digest(config_path),
        "configuration differs from protocol",
    )
    require(
        protocol.get("generation_policy", {}).get(
            "max_new_infrastructure_generation_retries"
        )
        == 1,
        "selector requires the frozen one-additional-generation policy",
    )
    require(1 <= shard_size <= 200, "retry shard size must be between 1 and 200")
    for relative, checksum in protocol["file_sha256"].items():
        checked_file(Path(protocol["source_root"]) / relative, checksum)
    ids = list(map(str, expected_manifest["ids"]))
    require(
        ids and len(ids) == len(set(ids)), "expected IDs must be nonempty and unique"
    )
    require(
        expected_manifest.get("dataset_revision") == config["dataset_revision"]
        and expected_manifest.get("dataset_variant", "standard") == "standard",
        "expected manifest protocol differs",
    )
    configurations = runtime_configurations(config_path, allowed_runtime_configs)
    partial, infrastructure = partial or {}, infrastructure or {}
    require(set(infrastructure) <= set(ids), "infrastructure selection has unknown IDs")
    directories = [str(path.resolve()) for path in attempt_dirs]
    require(len(directories) == len(set(directories)), "duplicate attempt directory")
    require(
        set(partial) <= set(directories),
        "partial selection names an unprovided attempt",
    )
    attempts, outcomes = {}, {}
    for directory in directories:
        record, rows = read_attempt(
            Path(directory),
            config_path,
            config,
            set(ids),
            allowed_commits,
            partial,
            configurations,
        )
        attempts[directory], outcomes[directory] = record, rows
    exclusion = read_preflight_exclusion(
        preflight_exclusion, protocol, attempts, outcomes
    )
    for record in attempts.values():
        if record["generation_attempt"] == 1 and record["selection_ledger"] is not None:
            verify_initial_selection(
                record, attempts, binding(config_path), binding(expected_path)
            )
    decisions: list[dict[str, Any]] = []
    for sample_id in ids:
        assigned = sorted(
            (r for r in attempts.values() if sample_id in r["ids"]),
            key=lambda r: (r["started_at"], r["attempt"]),
        )
        launched = [
            record
            for record in assigned
            if exclusion is None or record["attempt"] != exclusion["attempt"]
        ]
        require(
            len(launched) <= 2,
            f"{sample_id}: more than one additional generation was launched",
        )
        if launched:
            require(
                not launched[0]["retry_of"] and launched[0]["generation_attempt"] == 1,
                "retry requires its initial attempt record",
            )
        for record in assigned[1:]:
            require(
                record["retry_of"] is not None,
                "overlapping initial shard manifests",
            )
            verify_retry(
                record, sample_id, assigned[0], file_digest(config_path), exclusion
            )
        require(
            not assigned or assigned[0] in launched,
            "preflight exclusion cannot replace an initial benchmark attempt",
        )
        history = []
        for record in assigned:
            row = outcomes[record["attempt"]].get(
                sample_id,
                {
                    "attempt": record["attempt"],
                    "disposition": "unrecorded",
                    "reason": (
                        "unfinalized_durable_archive"
                        if record.get("unfinalized_archives")
                        else "missing_result_or_sample"
                    ),
                    "selected": False,
                },
            )
            if exclusion is not None and record["attempt"] == exclusion["attempt"]:
                row = {
                    **row,
                    "disposition": "preflight_not_admitted",
                    "reason": "verified_closed_model_preflight_before_benchmark",
                    "counts_toward_generation_budget": False,
                }
            history.append(row)
        decision = {
            "id": sample_id,
            "raw_assigned_launches": len(assigned),
            "launched_generation_attempts": len(launched),
            "attempts": history,
        }
        diagnosis = infrastructure.get(sample_id)
        if diagnosis is not None:
            require(
                isinstance(diagnosis, dict)
                and diagnosis.get("closed") is True
                and isinstance(diagnosis.get("evidence"), str)
                and diagnosis["evidence"].strip(),
                "infrastructure selection requires closed:true and concrete evidence",
            )
            require(
                launched and diagnosis.get("attempt") == launched[-1]["attempt"],
                "infrastructure selection must name the latest supplied attempt",
            )
            require(
                not launched[-1].get("unfinalized_archives"),
                "unfinalized durable archives require review before regeneration",
            )
            decision["diagnosis"] = diagnosis
        preserved = [
            row
            for row in history
            if row["disposition"] in {"retain_score", "judge_only"}
        ]
        require(
            not any(row in preserved for row in history[:-1]),
            "a retry follows an existing valid score or saved judge-only answer",
        )
        if preserved:
            require(
                diagnosis is None,
                "cannot regenerate a valid score or saved judge-only answer",
            )
            selected = preserved[0]
            decision["selected"] = selected
            decision["action"] = (
                "judge_only"
                if selected["disposition"] == "judge_only"
                else "retain" if selected["selected"] else "partial_selection_required"
            )
        elif not launched:
            require(diagnosis is None, "unlaunched ID is not a retry")
            decision["action"] = "unlaunched"
        elif len(launched) == 2:
            decision["action"] = "exhausted"
        elif diagnosis is not None:
            decision["action"] = "retry"
        else:
            decision["action"] = "held"
        decisions.append(decision)
    ledger = {
        "version": 1,
        "config": binding(config_path),
        "expected_manifest": binding(expected_path),
        "protocol": binding(protocol_path),
        "allowed_source_commits": sorted(allowed_commits),
        "max_additional_generation_attempts": 1,
        "attempts": attempts,
        "samples": decisions,
        "counts": dict(Counter(row["action"] for row in decisions)),
        "partial_selection": partial,
        "closure_policy": "closed:true attests an actual stopped/finished job check; elapsed time is insufficient",
        "dispatch_policy": "no dispatch; supply all current launched attempts and revalidate immediately before dispatch",
    }
    if allowed_runtime_configs:
        ledger["allowed_runtime_configs"] = configurations[1:]
    if exclusion is not None:
        ledger["preflight_exclusion"] = exclusion["binding"]
        ledger["attempt_accounting"] = (
            "All raw assignments remain in attempts/history. Exactly the protocol-bound "
            "closed preflight is excluded from generation-budget counts; every other "
            "assignment remains counted, including ambiguous interrupted work."
        )
    require(not output.exists(), "selection output already exists")
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}-", dir=output.parent
    ) as temporary:
        stage = Path(temporary) / "selection"
        stage.mkdir(mode=0o700)

        def write(path: Path, value: object) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")

        write(stage / "ledger.json", ledger)
        ledger_ref = {
            "path": str(output / "ledger.json"),
            "sha256": file_digest(stage / "ledger.json"),
        }
        retry = [row for row in decisions if row["action"] == "retry"]
        for index, start in enumerate(range(0, len(retry), shard_size)):
            retry_rows = retry[start : start + shard_size]
            write(
                stage / "retry" / f"{index:03d}.json",
                {
                    "dataset_revision": config["dataset_revision"],
                    "dataset_variant": "standard",
                    "ids": [row["id"] for row in retry_rows],
                    "generation_attempt": 2,
                    "selection_ledger": ledger_ref,
                    "retry_of": {
                        row["id"]: {
                            "attempt": row["attempts"][0]["attempt"],
                            "launch_sha256": attempts[row["attempts"][0]["attempt"]][
                                "launch"
                            ]["sha256"],
                        }
                        for row in retry_rows
                    },
                },
            )
        write(
            stage / "judge-only.json",
            [row for row in decisions if row["action"] == "judge_only"],
        )
        write(
            stage / "retained-selections.json",
            {
                str(Path(directory) / "result.json"): [
                    row["id"]
                    for row in decisions
                    if row["action"] == "retain"
                    and row["selected"]["attempt"] == directory
                ]
                for directory in directories
                if any(
                    row["action"] == "retain"
                    and row["selected"]["attempt"] == directory
                    for row in decisions
                )
            },
        )
        for path in stage.rglob("*"):
            if path.is_file():
                with path.open("rb") as stream:
                    os.fsync(stream.fileno())
        os.rename(stage, output)
    return ledger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("config", "expected-manifest", "protocol", "output"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    parser.add_argument("--attempt", type=Path, action="append", default=[])
    parser.add_argument(
        "--runtime-config",
        nargs=2,
        action="append",
        default=[],
        metavar=("PATH", "SHA256"),
        help="Explicit hash-bound configuration allowed to differ only in concurrency",
    )
    parser.add_argument("--source-commit", action="append", default=[])
    parser.add_argument("--partial-selection", type=Path)
    parser.add_argument("--infrastructure-selection", type=Path)
    parser.add_argument("--shard-size", type=int, default=200)
    parser.add_argument("--preflight-exclusion", type=Path)
    args = parser.parse_args()
    ledger = select(
        args.config,
        args.expected_manifest,
        args.protocol,
        args.output,
        attempt_dirs=args.attempt,
        allowed_commits=set(args.source_commit),
        partial=read_json(args.partial_selection) if args.partial_selection else None,
        infrastructure=(
            read_json(args.infrastructure_selection)
            if args.infrastructure_selection
            else None
        ),
        shard_size=args.shard_size,
        preflight_exclusion=args.preflight_exclusion,
        allowed_runtime_configs=dict(args.runtime_config),
    )
    print(
        json.dumps(
            {"output": str(args.output), "counts": ledger["counts"]}, sort_keys=True
        )
    )


if __name__ == "__main__":
    os.umask(0o077)
    main()
