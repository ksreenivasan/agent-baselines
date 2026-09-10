import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from inspect_ai.log import EvalLog, read_eval_log, write_eval_log

from agent_baselines.evals.hle_tools_v0.recovery import (
    digest,
    file_digest,
    generation_digest,
    iter_samples,
)
from test_aggregate_campaign import configuration, row, save

SCRIPT = (
    Path(__file__).parents[2] / "solvers/hle-tools-v0/m2/aggregate_direct_controls.py"
)
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("aggregate_direct_controls", SCRIPT)
direct = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(direct)
MIGRATION = "2" * 40
REPLACEMENT = "9" * 40


def binding(path):
    return {"path": str(path.resolve()), "sha256": file_digest(path)}


def native(path, config, rows, *, migration=False, manifest=None, status="success"):
    request = {
        "max_connections": config["concurrency"],
        **{
            key: config[key]
            for key in (
                "reasoning_effort",
                "reasoning_history",
                "top_p",
                "max_retries",
                "timeout",
                "attempt_timeout",
            )
        },
    }
    for item in rows:
        for event in item["events"]:
            if event.get("model") == config["model"]:
                event["config"] = deepcopy(request)
    log = EvalLog.model_validate(
        {
            "status": status,
            "eval": {
                "created": "2026-09-10T00:00:00Z",
                "task": "hle_direct",
                "task_id": "migration" if migration else "replacement",
                "model": config["model"],
                "model_base_url": config["model_base_url"],
                "dataset": {
                    "samples": 2 if migration else len(rows),
                    "sample_ids": [item["id"] for item in rows],
                },
                "config": {
                    "max_samples": config["concurrency"],
                    "epochs": 1,
                    "time_limit": 1800,
                    "message_limit": 10,
                },
                "model_generate_config": {
                    **request,
                    "max_connections": 3 if migration else config["concurrency"],
                },
                "revision": {
                    "type": "git",
                    "origin": "local",
                    "commit": (MIGRATION if migration else REPLACEMENT)[:7],
                    "dirty": False,
                },
                "task_args": {
                    "data_path": config["dataset_path"],
                    "dataset_revision": config["dataset_revision"],
                    "dataset_variant": "standard",
                    "fixture": False,
                    "judge_model": config["judge_model"],
                    "judge_reasoning_effort": "medium",
                    "manifest_path": str(manifest) if manifest else None,
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
    config = {
        **configuration(2),
        "task": "hle_direct",
        "search_backend": "",
        "model": "vllm/test",
        "model_base_url": "http://endpoint:8000/v1",
        "native_context": 1000000,
        "dataset_path": "/protected/hle",
        "concurrency": 12,
        "reasoning_history": "all",
        "top_p": 0.95,
        "timeout": 1650,
        "attempt_timeout": 1500,
    }
    cfg = save(tmp_path / "config.json", config)

    def manifest(name, ids):
        return save(
            tmp_path / name,
            {
                "ids": ids,
                "dataset_revision": config["dataset_revision"],
                "dataset_variant": "standard",
            },
        )

    all_manifest = manifest("all.json", ["m", "r"])
    owned = manifest("ids-current-migration-owned.json", ["m"])
    replaced = manifest("ids-generate-matched-current.json", ["r"])
    shard = manifest("manifests/000.json", ["r"])
    snapshot = native(
        tmp_path / "snapshot.eval",
        config,
        [row("m", config, "I"), row("r", config)],
        migration=True,
        status="started",
    )
    saved = list(iter_samples(snapshot))[0]
    snapshot_spec = {**binding(snapshot), "status": "started"}
    lineage = save(
        tmp_path / "lineage.json",
        {
            "dataset_revision": config["dataset_revision"],
            "dataset_ids_sha256": digest(["m", "r"]),
            "model": config["model"],
            "endpoint_url": config["model_base_url"],
            "migration_job_id": 123,
            "endpoint_job_id": 456,
            "native_context": config["native_context"],
            "frozen_checkpoint": snapshot_spec,
            "old_source": binding(snapshot),
            "current_header": direct.header(snapshot),
            "old_excluded": {"r": {}},
            "current_scored_candidates": {
                "m": {"generation_sha256": generation_digest(saved)}
            },
        },
    )
    source = save(tmp_path / "source/direct.json", {"code": "frozen"})
    endpoint = save(tmp_path / "endpoint-proof.json", {"job": 456})
    protocol_data = {
        "version": 2,
        "evaluation_count": 2,
        "all_manifest": binding(all_manifest),
        "dataset_revision": config["dataset_revision"],
        "dataset_variant": "standard",
        "replacement_concurrency": 12,
        "migration_concurrency": 12,
        "source_commit": REPLACEMENT,
        "source_root": str(source.parent),
        "source_file_sha256": {source.name: file_digest(source)},
        "sampling": {
            key: config[key]
            for key in (
                "reasoning_effort",
                "reasoning_history",
                "max_tokens",
                "temperature",
                "max_retries",
                "timeout",
                "attempt_timeout",
                "judge_model",
                "judge_reasoning_effort",
            )
        },
        "lanes": {
            "glm-flash": {
                "config": binding(cfg),
                "model": config["model"],
                "endpoint_url": config["model_base_url"],
                "endpoint_job": 456,
                "native_context": config["native_context"],
                "migration_job": 123,
                "migration_owned_count": 1,
                "replacement_count": 1,
                "selection_proofs": [
                    binding(owned),
                    binding(replaced),
                    binding(lineage),
                ],
                "manifests": [binding(shard)],
                "migration_snapshot": snapshot_spec,
                "endpoint_proof_files": [binding(endpoint)],
            }
        },
    }
    protocol_data["sampling"]["top_p"] = {"glm-flash": 0.95}
    protocol = save(tmp_path / "DIRECT_PROTOCOL.json", protocol_data)
    final = tmp_path / "migration-final.eval"
    log = read_eval_log(snapshot)
    log.status = "success"
    write_eval_log(log, final)
    evidence = save(tmp_path / "closure.json", {"job": 123, "state": "COMPLETED"})
    attestation = save(
        tmp_path / "migration-source.json",
        {
            "closed": True,
            "protocol_sha256": file_digest(protocol),
            "lane": "glm-flash",
            "migration_job": 123,
            "closure_evidence": binding(evidence),
            "archive": binding(final),
        },
    )
    archive = native(
        tmp_path / "attempt/logs/result.eval",
        config,
        [row("r", config)],
        manifest=shard,
    )
    launch = save(
        tmp_path / "attempt/launch.json",
        {
            "config": config,
            "config_sha256": file_digest(cfg),
            "source_commit": REPLACEMENT,
            "manifest": json.loads(shard.read_text()),
            "manifest_sha256": file_digest(shard),
            "command": [f"manifest_path={shard}"],
        },
    )
    result = save(
        tmp_path / "attempt/result.json",
        {
            "complete": True,
            "publisher_exit": 0,
            "archive": str(archive),
            "archive_sha256": file_digest(archive),
        },
    )
    return SimpleNamespace(
        root=tmp_path,
        config=config,
        cfg=cfg,
        protocol=protocol,
        lineage=lineage,
        snapshot=snapshot,
        final=final,
        evidence=evidence,
        attestation=attestation,
        archive=archive,
        launch=launch,
        result=result,
        shard=shard,
        owned=owned,
        output=tmp_path / "final",
    )


def run(campaign, **kwargs):
    return direct.aggregate(
        campaign.protocol,
        "glm-flash",
        campaign.output,
        migration_sources=kwargs.pop("migration_sources", [campaign.attestation]),
        replacement_attempts=kwargs.pop("replacement_attempts", [campaign.result]),
        **kwargs,
    )


def change_json(path, change):
    value = json.loads(path.read_text())
    change(value)
    save(path, value)


def change_archive(campaign, *, migration, change):
    path = campaign.final if migration else campaign.archive
    log = read_eval_log(path)
    change(log)
    write_eval_log(log, path)
    if migration:
        change_json(
            campaign.attestation, lambda value: value.update(archive=binding(path))
        )
    else:
        change_json(
            campaign.result,
            lambda value: value.update(archive_sha256=file_digest(path)),
        )


def test_complete_exact_union_preserves_incorrect(campaign):
    result = run(campaign)
    assert result["complete"]
    assert result["samples"] == 2 and result["correct"] == 1
    assert result["source_counts"] == {"migration": 1, "replacement": 1}
    assert result["strata"]["migration"]["inherited_header_max_connections"] == 3
    rows = [
        json.loads(line)
        for line in (campaign.output / "outcomes.jsonl").read_text().splitlines()
    ]
    assert [item["id"] for item in rows] == ["m", "r"]
    assert rows[0]["correct"] is False and all(
        item["concurrency"] == 12 for item in rows
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("closed", False),
        ("migration_job", 999),
        ("protocol_sha256", "bad"),
        ("lane", "k2-horizon"),
    ],
)
def test_closure_attestation_binding(campaign, field, value):
    change_json(campaign.attestation, lambda item: item.update({field: value}))
    with pytest.raises(ValueError, match="attestation"):
        run(campaign)
    assert not campaign.output.exists()


def test_closure_evidence_hash(campaign):
    campaign.evidence.write_text("changed")
    with pytest.raises(ValueError, match="checksum"):
        run(campaign)


def test_started_snapshot_not_accepted_from_counts(campaign):
    change_json(
        campaign.attestation,
        lambda item: item.update(archive=binding(campaign.snapshot)),
    )
    with pytest.raises(ValueError, match="explicit clean selection"):
        run(campaign)


def test_explicit_clean_closed_snapshot_can_supply_owned_row(campaign):
    change_json(
        campaign.attestation,
        lambda item: item.update(
            archive=binding(campaign.snapshot), selected_ids=["m"]
        ),
    )
    assert run(campaign)["complete"]


def test_missing_migration_ownership_prevents_publication(campaign):
    with pytest.raises(ValueError, match="1 missing"):
        run(campaign, migration_sources=[])
    assert not campaign.output.exists()


def test_old_copied_id_cannot_be_selected_as_migration(campaign):
    change_json(campaign.attestation, lambda item: item.update(selected_ids=["r"]))
    with pytest.raises(ValueError, match="invalid migration"):
        run(campaign)


def test_native_migration_identity_is_bound(campaign):
    change_archive(
        campaign,
        migration=True,
        change=lambda log: setattr(log.eval, "task_id", "other"),
    )
    with pytest.raises(ValueError, match="identity"):
        run(campaign)


@pytest.mark.parametrize("migration", [True, False])
def test_actual_request_concurrency_is_checked(campaign, migration):
    def change(log):
        log.samples[0].events[0].config.max_connections = 3

    change_archive(campaign, migration=migration, change=change)
    with pytest.raises(ValueError, match="concurrency"):
        run(campaign)


@pytest.mark.parametrize("value", ["invalid", None])
def test_malformed_or_missing_score_stays_unresolved(campaign, value):
    def change(log):
        if value is None:
            log.samples[0].scores = {}
        else:
            log.samples[0].scores["hle_scorer"].value = value

    change_archive(campaign, migration=False, change=change)
    with pytest.raises(ValueError, match="unresolved|score"):
        run(campaign)


def test_changed_valid_migration_score_rejected(campaign):
    change_archive(
        campaign,
        migration=True,
        change=lambda log: setattr(log.samples[0].scores["hle_scorer"], "value", "C"),
    )
    with pytest.raises(ValueError, match="previously valid"):
        run(campaign)


def test_unknown_epoch_rejected(campaign):
    change_archive(
        campaign,
        migration=False,
        change=lambda log: setattr(log.samples[0], "epoch", 2),
    )
    with pytest.raises(ValueError, match="epoch"):
        run(campaign)


def test_replacement_manifest_archive_binding(campaign):
    change_archive(
        campaign,
        migration=False,
        change=lambda log: log.eval.task_args.update(
            manifest_path="/other/manifest.json"
        ),
    )
    with pytest.raises(ValueError, match="manifest differs"):
        run(campaign)


def test_extra_durable_archive_rejected(campaign):
    (campaign.archive.parent / "other.eval").write_bytes(campaign.archive.read_bytes())
    with pytest.raises(ValueError, match="single durable"):
        run(campaign)


def test_replacement_source_commit_rejected(campaign):
    change_json(campaign.launch, lambda item: item.update(source_commit="a" * 40))
    with pytest.raises(ValueError, match="source commit"):
        run(campaign)


def test_overlapping_accepted_inputs_rejected(campaign):
    with pytest.raises(ValueError, match="overlapping"):
        run(campaign, migration_sources=[campaign.attestation, campaign.attestation])


def test_unknown_manifest_id_rejected(campaign):
    change_json(campaign.owned, lambda item: item.update(ids=["x"]))
    with pytest.raises(ValueError, match="checksum"):
        run(campaign)


def test_source_hash_mismatch_rejected(campaign):
    (campaign.root / "source/direct.json").write_text("changed")
    with pytest.raises(ValueError, match="checksum"):
        run(campaign)


def test_output_is_immutable(campaign):
    run(campaign)
    with pytest.raises(ValueError, match="already exists"):
        run(campaign)


def test_empty_time_limited_incorrect_is_retained(campaign):
    sample = list(iter_samples(campaign.final))[0]
    sample["output"]["choices"] = []
    sample["events"].append({"event": "sample_limit", "type": "time", "limit": 1800})
    assert direct.accepted(sample, campaign.config)["correct"] is False


def test_error_plus_score_never_accepted(campaign):
    sample = list(iter_samples(campaign.final))[0]
    sample["error"] = {"message": "failed", "traceback": "", "traceback_ansi": ""}
    with pytest.raises(ValueError, match="unresolved"):
        direct.accepted(sample, campaign.config)


def test_judge_provider_error_is_explicitly_unresolved(
    campaign, judge_provider_error_row
):
    sample = judge_provider_error_row()
    with pytest.raises(ValueError, match="judge_provider_infrastructure_error"):
        direct.accepted(sample, campaign.config)


def test_failed_judge_marker_never_counts_as_incorrect(campaign):
    sample = list(iter_samples(campaign.final))[0]
    sample["scores"]["hle_scorer"]["metadata"] = {"hle_judge_repair_failed": True}
    with pytest.raises(ValueError, match="unresolved|score"):
        direct.accepted(sample, campaign.config)


def test_tool_use_never_accepted_as_direct(campaign):
    sample = list(iter_samples(campaign.final))[0]
    sample["events"].append({"event": "tool", "function": "python"})
    with pytest.raises(ValueError, match="tool calls"):
        direct.accepted(sample, campaign.config)


def test_dynamic_replacement_header_and_event_concurrency(campaign):
    config = {**campaign.config, "concurrency": 3}
    actual = direct.header(campaign.archive)
    actual["eval"]["config"]["max_samples"] = 3
    actual["eval"]["model_generate_config"]["max_connections"] = 3
    direct.check_direct_header(actual, config, migration=False, partial=False)
    sample = list(iter_samples(campaign.archive))[0]
    sample["events"][0]["config"]["max_connections"] = 3
    assert direct.accepted(sample, config)["correct"] is True


def test_hidden_unknown_native_record_is_rejected(campaign):
    def change(log):
        sample = log.samples[0].model_copy(deep=True)
        sample.id = "hidden"
        log.samples.append(sample)

    change_archive(campaign, migration=False, change=change)
    with pytest.raises(ValueError, match="unknown"):
        run(campaign)


def test_existing_tools_gate_still_rejects_direct(campaign):
    from aggregate_campaign import check_header

    with pytest.raises(ValueError, match="not the production HLE tools task"):
        check_header(direct.header(campaign.archive), campaign.config, False)


def test_unexecuted_model_tool_call_does_not_override_incorrect(campaign):
    sample = list(iter_samples(campaign.final))[0]
    sample["messages"] = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "hallucinated", "function": "unavailable", "arguments": {}}
            ],
        }
    ]
    assert direct.accepted(sample, campaign.config)["correct"] is False


def test_actual_task_native_header_uses_new_parameter_names(tmp_path):
    from inspect_ai import eval as inspect_eval

    # Both solver and judge use Inspect's local mock provider; this verifies
    # the real task loader's serialized defaults without external model calls.
    logs = inspect_eval(
        str(SCRIPT.parents[3] / "agent_baselines/evals/hle_direct/task.py")
        + "@hle_direct",
        model="mockllm/model",
        task_args={
            "fixture": True,
            "dataset_revision": "021a3d71f516a7ac28ceb8d284969902edf1edeb",
            "judge_model": "mockllm/model",
        },
        limit=1,
        log_dir=str(tmp_path / "native-mock"),
        display="none",
    )
    args = logs[0].eval.task_args
    assert args["dataset_revision"] == "021a3d71f516a7ac28ceb8d284969902edf1edeb"
    assert args["dataset_variant"] == "standard"
    assert "revision" not in args and "variant" not in args
