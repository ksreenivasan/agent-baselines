import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from inspect_ai.log import EvalLog, write_eval_log

from agent_baselines.evals.hle_tools_v0.recovery import (
    file_digest,
    generation_digest,
    iter_samples,
    digest,
)

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "solvers/hle-tools-v0/m2/aggregate_campaign.py"
)
SPEC = importlib.util.spec_from_file_location("aggregate_campaign", SCRIPT)
aggregator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(aggregator)
COMMIT = "a" * 40


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")
    return path


def configuration(count):
    return {
        "task": "hle_tools",
        "run_phase": "production",
        "model": "openai/test",
        "reasoning_effort": "high",
        "dataset_revision": "b" * 40,
        "dataset_variant": "standard",
        "judge_model": "openai/judge",
        "judge_reasoning_effort": "medium",
        "search_backend": "keenable",
        "evaluation_count": count,
        "concurrency": 3,
        "max_retries": 2,
        "timeout": 660,
        "attempt_timeout": 600,
        "max_tokens": None,
        "temperature": None,
        "top_p": None,
        "reasoning_history": None,
    }


def request_config(config):
    return {
        "max_connections": config["concurrency"],
        **{
            key: config[key]
            for key in ("reasoning_effort", "max_retries", "timeout", "attempt_timeout")
        },
    }


def row(sample_id, config, score="C", error=None):
    value = {
        "id": sample_id,
        "epoch": 1,
        "input": "question",
        "target": "answer",
        "output": {"model": config["model"], "completion": "saved answer"},
        "events": [
            {
                "event": "model",
                "model": config["model"],
                "input": [],
                "tools": [],
                "tool_choice": "auto",
                "config": request_config(config),
                "output": {},
            }
        ],
        "model_usage": {
            config["model"]: {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
            }
        },
    }
    if score is not None:
        value["scores"] = {"hle_scorer": {"value": score}}
    if error:
        value["error"] = {
            "message": error,
            "traceback": "scorer.py" if "EqualityJudgment" in error else "",
            "traceback_ansi": "",
        }
    return value


def write_archive(path, config, rows, *, status="success", metadata=None, dirty=False):
    log = EvalLog.model_validate(
        {
            "status": status,
            "eval": {
                "created": "2026-09-10T00:00:00Z",
                "task": "hle_tools_v0",
                "model": config["model"],
                "dataset": {
                    "samples": len(rows),
                    "sample_ids": [item["id"] for item in rows],
                },
                "config": {},
                "model_generate_config": request_config(config),
                "revision": {
                    "type": "git",
                    "origin": "local",
                    "commit": COMMIT[:7],
                    "dirty": dirty,
                },
                "metadata": metadata or {},
                "task_args": {
                    "dataset_revision": config["dataset_revision"],
                    "dataset_variant": "standard",
                    "fixture": False,
                    "judge_model": config["judge_model"],
                    "judge_reasoning_effort": "medium",
                    "sandbox_backend": "m2-enroot",
                },
            },
            "samples": rows,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    write_eval_log(log, path)
    return path


@pytest.fixture
def campaign(tmp_path):
    ids = ["a", "b"]
    config = configuration(len(ids))
    cfg = save(tmp_path / "full.json", config)
    manifest = save(
        tmp_path / "all.json",
        {
            "dataset_revision": config["dataset_revision"],
            "dataset_variant": "standard",
            "ids": ids,
        },
    )
    source = tmp_path / "source"
    source.mkdir()
    (source / "protocol.yaml").write_text("frozen protocol\n")
    protocol = save(
        tmp_path / "PROTOCOL.json",
        {
            "source_root": str(source),
            "file_sha256": {"protocol.yaml": file_digest(source / "protocol.yaml")},
            "config_sha256": {cfg.name: file_digest(cfg)},
        },
    )
    state = SimpleNamespace(
        root=tmp_path,
        config=config,
        cfg=cfg,
        manifest=manifest,
        protocol=protocol,
        output=tmp_path / "final",
        attempts=[],
    )

    def attempt(name, rows, *, manifest_ids=None, status="success", selection=False):
        root = tmp_path / name
        path = write_archive(root / "logs" / "result.eval", config, rows, status=status)
        input_manifest = save(
            root / "manifest.json",
            {
                "ids": manifest_ids or [item["id"] for item in rows],
                "dataset_revision": config["dataset_revision"],
            },
        )
        launch = {
            "config": config,
            "config_sha256": file_digest(cfg),
            "source_commit": COMMIT,
            "manifest": json.loads(input_manifest.read_text()),
            "manifest_sha256": file_digest(input_manifest),
            "command": [f"manifest_path={input_manifest}"],
        }
        save(root / "launch.json", launch)
        result = save(
            root / "result.json",
            {
                "complete": not selection
                and all(item.get("scores") and not item.get("error") for item in rows),
                "publisher_exit": 0,
                "archive": str(path),
                "archive_sha256": file_digest(path),
                "validation": {
                    "continuable": not selection
                    and any(item.get("error") for item in rows)
                },
            },
        )
        state.attempts.append(result)
        return result

    state.attempt = attempt
    return state


def aggregate(state, **kwargs):
    return aggregator.aggregate(
        state.cfg,
        state.manifest,
        state.protocol,
        state.output,
        attempts=state.attempts,
        allowed_commits={COMMIT},
        **kwargs,
    )


def test_complete_native_index_retains_incorrect_and_ordered_provenance(campaign):
    campaign.attempt("one", [row("b", campaign.config, "I"), row("a", campaign.config)])
    result = aggregate(campaign)
    assert result["complete"] and result["correct"] == result["incorrect"] == 1
    outcomes = [
        json.loads(line)
        for line in (campaign.output / "outcomes.jsonl").read_text().splitlines()
    ]
    assert [item["id"] for item in outcomes] == ["a", "b"]
    assert outcomes[1]["correct"] is False
    assert outcomes[0]["tokens"]["total_tokens"] == 15
    index = json.loads((campaign.output / "shards.json").read_text())
    assert index[0]["sha256"] == file_digest(Path(index[0]["path"]))
    assert not list(campaign.output.glob("*.eval"))


def test_partial_successful_archive_can_be_resolved_without_rerunning_good_answer(
    campaign,
):
    campaign.attempt(
        "partial",
        [
            row("a", campaign.config, "I"),
            row("b", campaign.config, None, "ServerError unavailable"),
        ],
    )
    campaign.attempt("retry", [row("b", campaign.config)])
    summary = aggregate(campaign)
    assert summary["correct"] == 1 and summary["incorrect"] == 1
    assert summary["resolved_failed_attempts"][0]["id"] == "b"


def test_cancelled_archive_requires_explicit_successful_id_selection(campaign):
    source = campaign.attempt(
        "cancelled",
        [row("a", campaign.config, "I")],
        manifest_ids=["a", "b"],
        status="cancelled",
        selection=True,
    )
    campaign.attempt("retry", [row("b", campaign.config)])
    with pytest.raises(ValueError, match="explicit per-ID"):
        aggregate(campaign)
    assert (
        aggregate(campaign, selections={str(source.resolve()): ["a"]})["incorrect"] == 1
    )


@pytest.mark.parametrize(
    "failure",
    [
        "overlap",
        "missing",
        "score_error",
        "tool_invalidation",
        "actual_settings",
        "header_model",
        "file_hash",
        "config_hash",
        "commit",
        "manifest_hash",
        "protocol_hash",
        "dirty",
        "judge_failure_marker",
    ],
)
def test_rejects_bad_membership_protocol_provenance_and_outcomes(campaign, failure):
    rows = [row("a", campaign.config), row("b", campaign.config)]
    if failure == "missing":
        rows.pop()
    if failure == "score_error":
        rows[0]["error"] = {
            "message": "ServerError",
            "traceback": "",
            "traceback_ansi": "",
        }
    if failure == "tool_invalidation":
        rows[0]["metadata"] = {"hle_tools_invalidated": True}
    if failure == "actual_settings":
        rows[0]["events"][0]["config"]["max_tokens"] = 99
    if failure == "judge_failure_marker":
        rows[0]["scores"]["hle_scorer"]["metadata"] = {
            "hle_judge_repair_failed": "ValidationError"
        }
    result_path = campaign.attempt("one", rows)
    if failure == "overlap":
        campaign.attempt("duplicate", [row("a", campaign.config)])
    result = json.loads(result_path.read_text())
    archive = Path(result["archive"])
    launch_path = result_path.parent / "launch.json"
    launch = json.loads(launch_path.read_text())
    if failure == "file_hash":
        result["archive_sha256"] = "0" * 64
    if failure in {"header_model", "dirty"}:
        cfg = dict(campaign.config)
        if failure == "header_model":
            cfg["model"] = "openai/wrong"
        write_archive(archive, cfg, rows, dirty=failure == "dirty")
        result["archive_sha256"] = file_digest(archive)
    if failure == "config_hash":
        launch["config_sha256"] = "0" * 64
    if failure == "commit":
        launch["source_commit"] = "0" * 40
    if failure == "manifest_hash":
        launch["manifest_sha256"] = "0" * 64
    if failure == "protocol_hash":
        (campaign.root / "source" / "protocol.yaml").write_text("changed")
    save(launch_path, launch)
    save(result_path, result)
    with pytest.raises(ValueError):
        aggregate(campaign)
    assert not campaign.output.exists()


@pytest.mark.parametrize(
    "tamper",
    [
        None,
        "generation",
        "repair_ledger",
        "source_hash",
        "adoption",
        "adoption_generation",
        "adoption_source_hash",
    ],
)
def test_historical_retention_and_judge_repair_preserve_generation(campaign, tamper):
    cfg = campaign.config
    original_rows = [
        row("a", cfg, "I"),
        row("b", cfg, None, "EqualityJudgment invalid JSON"),
    ]
    original = write_archive(
        campaign.root / "original.eval", cfg, original_rows, dirty=True
    )
    original_hash = file_digest(original)
    original_header = aggregator.read_eval_log(original, header_only=True).model_dump(
        mode="json", exclude_none=True
    )
    original_samples = list(iter_samples(original))
    metadata = {"hle_recovery_source_sha256": original_hash}
    retained = write_archive(
        campaign.root / "retained.eval",
        cfg,
        [original_rows[0]],
        metadata=metadata,
        dirty=True,
    )
    pending = write_archive(
        campaign.root / "pending.eval",
        cfg,
        [original_rows[1]],
        metadata=metadata,
        status="started",
        dirty=True,
    )
    repaired_row = dict(original_rows[1])
    repaired_row.pop("error")
    repaired_row["scores"] = {"hle_scorer": {"value": "C"}}
    repaired = write_archive(
        campaign.root / "repaired.eval",
        cfg,
        [repaired_row],
        metadata=metadata,
        dirty=True,
    )
    ledger = save(
        campaign.root / "ledger.json",
        {
            "dataset_revision": cfg["dataset_revision"],
            "dataset_ids_sha256": digest(["a", "b"]),
            "search_backend": "keenable",
            "sources": [
                {
                    "source_role": "selected",
                    "path": str(original),
                    "sha256": original_hash,
                    "eval": original_header["eval"],
                }
            ],
            "samples": [
                {
                    "id": item["id"],
                    "disposition": (
                        "retain_score" if item["id"] == "a" else "judge_only"
                    ),
                    "generation_sha256": generation_digest(item),
                }
                for item in original_samples
            ],
            "retained_shards": [
                {"path": str(retained), "sha256": file_digest(retained), "ids": ["a"]}
            ],
            "judge_only_shard": {
                "path": str(pending),
                "sha256": file_digest(pending),
                "ids": ["b"],
            },
        },
    )
    repair_result = save(
        campaign.root / "repair-result.json",
        {
            "complete": True,
            "ledger": str(ledger),
            "ledger_sha256": file_digest(ledger),
            "source_sha256": file_digest(pending),
            "archive": str(repaired),
            "archive_sha256": file_digest(repaired),
            "judge_model": cfg["judge_model"],
            "judge_reasoning_effort": "medium",
        },
    )
    if tamper and tamper.startswith("adoption"):
        active_pending = campaign.root / "pending-v2.eval"
        active_pending.write_bytes(pending.read_bytes() + b"\n")
        active_data = json.loads(ledger.read_text())
        active_data["audit_version"] = 2
        active_data["judge_only_shard"].update(
            path=str(active_pending), sha256=file_digest(active_pending)
        )
        active_ledger = save(campaign.root / "ledger-v2.json", active_data)
        if tamper == "adoption_generation":
            active_data["samples"][1]["generation_sha256"] = "0" * 64
            save(active_ledger, active_data)
        if tamper == "adoption_source_hash":
            active_pending.write_bytes(active_pending.read_bytes() + b"changed")
        adoption = campaign.root / "judge-adoption.json"
        if tamper != "adoption":
            with pytest.raises(ValueError):
                aggregator.write_judge_adoption(active_ledger, repair_result, adoption)
            assert not adoption.exists()
            return
        with pytest.raises(ValueError, match="explicit adoption"):
            aggregate(campaign, ledger_path=active_ledger, judge_result=repair_result)
        original_result_bytes = repair_result.read_bytes()
        binding = aggregator.write_judge_adoption(
            active_ledger, repair_result, adoption
        )
        assert (
            binding["original_judge_source_sha256"]
            != binding["active_judge_source_sha256"]
        )
        summary = aggregate(
            campaign,
            ledger_path=active_ledger,
            judge_result=repair_result,
            judge_adoption=adoption,
        )
        assert summary["complete"] and summary["judge_adoption"]["binding"] == binding
        assert repair_result.read_bytes() == original_result_bytes
        return
    if tamper == "generation":
        original_rows[0]["output"]["completion"] = "changed solver answer"
        write_archive(retained, cfg, [original_rows[0]], metadata=metadata, dirty=True)
        ledger_data = json.loads(ledger.read_text())
        ledger_data["retained_shards"][0]["sha256"] = file_digest(retained)
        save(ledger, ledger_data)
        repair_data = json.loads(repair_result.read_text())
        repair_data["ledger_sha256"] = file_digest(ledger)
        save(repair_result, repair_data)
    elif tamper == "repair_ledger":
        repair_data = json.loads(repair_result.read_text())
        repair_data["ledger_sha256"] = "0" * 64
        save(repair_result, repair_data)
    elif tamper == "source_hash":
        with original.open("ab") as stream:
            stream.write(b"changed source")
    if tamper:
        with pytest.raises(ValueError):
            aggregate(campaign, ledger_path=ledger, judge_result=repair_result)
        assert not campaign.output.exists()
        return
    summary = aggregate(campaign, ledger_path=ledger, judge_result=repair_result)
    assert summary["historical_revision"]["dirty"] is True
    assert summary["source_counts"] == {"retained": 1, "judge_repair": 1}
    assert summary["correct"] == summary["incorrect"] == 1


def test_diagnostic_phase_is_rejected_even_with_matching_protocol_hash(campaign):
    campaign.config["run_phase"] = "diagnostic"
    save(campaign.cfg, campaign.config)
    protocol = json.loads(campaign.protocol.read_text())
    protocol["config_sha256"][campaign.cfg.name] = file_digest(campaign.cfg)
    save(campaign.protocol, protocol)
    campaign.attempt("pilot", [row("a", campaign.config), row("b", campaign.config)])
    with pytest.raises(ValueError, match="diagnostic"):
        aggregate(campaign)
