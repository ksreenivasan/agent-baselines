import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_baselines.evals.hle_tools_v0.recovery import digest, generation_digest

SCRIPT = Path(__file__).parents[2] / "solvers/hle-tools-v0/m2/compare_conditions.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("compare_conditions", SCRIPT)
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def save_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.fixture
def campaign(tmp_path):
    ids = [f"synthetic-{i:04d}" for i in range(comparison.COUNT)]
    manifest = save(
        tmp_path / "expected.json",
        {
            "ids": ids,
            "dataset_revision": comparison.REVISION,
            "dataset_variant": "standard",
        },
    )
    model = "mockllm/solver"

    def condition(name, values):
        directory = tmp_path / name
        directory.mkdir()
        direct = name == "direct"
        config = {
            "model": model,
            "dataset_revision": comparison.REVISION,
            "task": "hle_direct" if direct else "hle_tools",
            "run_phase": "production",
            "search_backend": "keenable",
            "concurrency": 12 if direct else 3,
        }
        cfg = save(tmp_path / f"{name}-config.json", config)
        header = {
            "status": "success",
            "eval": {
                "model": model,
                "task": "hle_direct" if direct else "hle_tools_v0",
                "task_args": {
                    "dataset_revision": comparison.REVISION,
                    "dataset_variant": "standard",
                    "fixture": False,
                },
            },
        }
        samples = [
            {
                "id": sid,
                "epoch": 1,
                "input": "synthetic question",
                "target": "42",
                "messages": [],
                "output": {"completion": "" if i == 0 else "42"},
                "scores": {"hle_scorer": {"value": "C" if correct else "I"}},
                "events": [],
                # Scored limit outcomes and empty answers remain valid.
                "limit": {"type": "time", "limit": 1800} if i == 0 else None,
            }
            for i, (sid, correct) in enumerate(zip(ids, values))
        ]
        archive = save(tmp_path / f"{name}-native.json", {**header, "samples": samples})
        checksum = comparison.file_digest(archive)
        kind = "replacement" if direct else "production"
        rows = [
            {
                "id": sid,
                "epoch": 1,
                "correct": value,
                "generation_sha256": generation_digest(sample),
                "archive": str(archive),
                "archive_sha256": checksum,
                "stratum" if direct else "kind": kind,
            }
            for sid, value, sample in zip(ids, values, samples)
        ]
        save_rows(directory / "outcomes.jsonl", rows)
        save(
            directory / "shards.json",
            [{"path": str(archive), "sha256": checksum, "selected_ids": ids}],
        )
        summary = {
            "complete": True,
            "samples": len(ids),
            "dataset_revision": comparison.REVISION,
            "dataset_ids_sha256": digest(ids),
            "correct": sum(values),
            "incorrect": len(ids) - sum(values),
            "accuracy": sum(values) / len(ids),
            "source_counts": {kind: len(ids)},
        }
        if direct:
            protocol = save(
                tmp_path / "direct-protocol.json",
                {
                    "dataset_revision": comparison.REVISION,
                    "dataset_variant": "standard",
                    "lanes": {"synthetic": {"config": comparison.binding(cfg)}},
                },
            )
            summary.update(
                task="hle_direct",
                lane="synthetic",
                protocol=str(protocol),
                protocol_sha256=comparison.file_digest(protocol),
                config_sha256=comparison.file_digest(cfg),
                qualification="Synthetic direct provenance; distinct source and attempts.",
                strata={"replacement": {"concurrency": 12}},
            )
        else:
            summary.update(model=model, configuration=config)
        save(directory / "summary.json", summary)
        return directory

    direct_values = [False, True, False, True] + [False] * (len(ids) - 4)
    tools_values = [False, False, True, True] + [False] * (len(ids) - 4)
    return SimpleNamespace(
        root=tmp_path,
        ids=ids,
        expected=manifest,
        direct=condition("direct", direct_values),
        tools=condition("tools", tools_values),
        output=tmp_path / "comparison",
    )


def run(campaign, **kwargs):
    return comparison.compare(
        campaign.expected,
        campaign.tools,
        campaign.output,
        **({"direct_aggregate": campaign.direct} if not kwargs else kwargs),
    )


def rewrite_native(campaign, name, change):
    archive = campaign.root / f"{name}-native.json"
    value = comparison.read_json(archive)
    change(value)
    save(archive, value)
    checksum = comparison.file_digest(archive)
    directory = getattr(campaign, name)
    shards = comparison.read_json(directory / "shards.json")
    shards[0]["sha256"] = checksum
    save(directory / "shards.json", shards)
    rows = comparison.jsonl(directory / "outcomes.jsonl")
    for row in rows:
        row["archive_sha256"] = checksum
    save_rows(directory / "outcomes.jsonl", rows)


def historical_audit(campaign):
    archive = campaign.root / "direct-native.json"
    native = comparison.read_json(archive)
    records = []
    for sample in native["samples"]:
        selected = {
            "id": sample["id"],
            "epoch": 1,
            "eligible": True,
            "disposition": "retain_score",
            "problems": [],
            "infrastructure_reasons": [],
            "error_type": None,
            "tool_events": 0,
            "full_solver_request_input_matches": True,
            "canonical_judge_prompt_matches": True,
            "score": sample["scores"]["hle_scorer"]["value"],
            "generation_sha256": generation_digest(sample),
            "source_index": 0,
            # Recovered internal model errors do not invalidate a valid score.
            "solver_model_event_errors": 1,
        }
        records.append(
            {
                "id": sample["id"],
                "epoch": 1,
                "selected": selected,
                "observed_attempts": [selected],
            }
        )
    per_id = campaign.root / "historical.per-id.jsonl"
    save_rows(per_id, records)
    return save(
        campaign.root / "historical.json",
        {
            "status": "coverage_verified_with_lineage_and_opportunity_qualifications",
            "selected_count": len(records),
            "unresolved_ids": [],
            "selected_problem_counts": {},
            "selected_tool_events": 0,
            "selected_actual_solver_inputs_verified": len(records),
            "selected_canonical_judge_prompts_verified": len(records),
            "dataset": {
                "revision": comparison.REVISION,
                "variant": "standard",
                "count": len(records),
                "frozen_manifest": comparison.binding(campaign.expected),
            },
            "sources": [
                {
                    **comparison.binding(archive),
                    "parse_valid": True,
                    "header": {k: v for k, v in native.items() if k != "samples"},
                }
            ],
            "per_id_evidence": comparison.binding(per_id),
            "selected_score_counts": {"C": 2, "I": len(records) - 2},
            "selected_source_counts": {"0": len(records)},
            "qualifications": ["Historical repairs are full solver regenerations."],
            "repair_summary": {"changed_generation": 1},
        },
    )


def test_complete_pairing_discordance_and_provenance(campaign):
    # Different file order must not destroy pairing by ID.
    rows = comparison.jsonl(campaign.tools / "outcomes.jsonl")
    save_rows(campaign.tools / "outcomes.jsonl", list(reversed(rows)))
    result = run(campaign)
    assert result["samples"] == 2158 and result["complete"]
    assert result["direct"]["correct"] == result["tools"]["correct"] == 2
    assert result["tools_minus_direct_percentage_points"] == 0
    assert result["discordant"] == {
        "tools_correct_direct_incorrect": 1,
        "tools_incorrect_direct_correct": 1,
    }
    assert result["concordant"] == {"both_correct": 1, "both_incorrect": 2155}
    pairs = comparison.jsonl(campaign.output / "paired-outcomes.jsonl")
    assert [p["id"] for p in pairs] == campaign.ids
    assert pairs[0]["direct"]["correct"] is False  # Empty, scored time-limit answer.
    assert "first-attempt" in " ".join(result["qualifications"])
    assert (
        result["provenance"]["direct"]["summary"]["strata"]["replacement"][
            "concurrency"
        ]
        == 12
    )
    assert result["paired_outcomes_sha256"] == comparison.file_digest(
        campaign.output / "paired-outcomes.jsonl"
    )
    with pytest.raises(ValueError, match="already exists"):
        run(campaign)


def test_fixed_bootstrap_direction_pairing_and_reproducibility():
    direct = [False, True, False, True]
    tools = [False, False, True, True]
    result = comparison.paired_statistics(direct, tools)
    assert result == comparison.paired_statistics(direct[::-1], tools[::-1])
    assert result["bootstrap"]["seed"] == 20260910
    assert result["bootstrap"]["resamples"] == 10000
    assert result["bootstrap"]["tools_minus_direct_percentage_points_ci"] == [
        -75.0,
        75.0,
    ]
    for d, t, difference in [(False, True, 100), (True, False, -100), (True, True, 0)]:
        stats = comparison.paired_statistics([d] * 10, [t] * 10)
        assert stats["tools_minus_direct_percentage_points"] == difference
        assert stats["bootstrap"]["tools_minus_direct_percentage_points_ci"] == [
            difference,
            difference,
        ]


@pytest.mark.parametrize(
    "case",
    [
        "incomplete",
        "wrong_revision",
        "wrong_summary_total",
        "wrong_hash",
        "duplicate",
        "missing",
        "unknown",
        "wrong_epoch",
        "boolean_epoch",
        "numeric_correct",
        "row_error",
        "invalidated",
        "missing_source",
        "overlap",
    ],
)
def test_rejects_incomplete_or_invalid_aggregate(campaign, case):
    summary_path = campaign.tools / "summary.json"
    summary = comparison.read_json(summary_path)
    rows_path = campaign.tools / "outcomes.jsonl"
    rows = comparison.jsonl(rows_path)
    shards_path = campaign.tools / "shards.json"
    shards = comparison.read_json(shards_path)
    if case == "incomplete":
        summary["complete"] = False
    elif case == "wrong_revision":
        summary["dataset_revision"] = "wrong"
    elif case == "wrong_summary_total":
        summary["correct"] += 1
    elif case == "wrong_hash":
        rows[0]["archive_sha256"] = "0" * 64
    elif case == "duplicate":
        rows[-1] = copy.deepcopy(rows[0])
    elif case == "missing":
        rows.pop()
    elif case == "unknown":
        rows[0]["id"] = "foreign"
    elif case == "wrong_epoch":
        rows[0]["epoch"] = 2
    elif case == "boolean_epoch":
        rows[0]["epoch"] = True
    elif case == "numeric_correct":
        rows[0]["correct"] = 0
    elif case == "row_error":
        rows[0]["error"] = "infrastructure"
    elif case == "invalidated":
        rows[0]["invalidated"] = True
    elif case == "missing_source":
        shards[0]["selected_ids"].pop()
    elif case == "overlap":
        shards.append(copy.deepcopy(shards[0]))
    save(summary_path, summary)
    save_rows(rows_path, rows)
    save(shards_path, shards)
    with pytest.raises(ValueError):
        run(campaign)
    assert not campaign.output.exists()


@pytest.mark.parametrize("case", ["score", "error", "model", "revision", "tool"])
def test_rejects_changed_or_invalid_native_evidence(campaign, case):
    def change(native):
        if case == "score":
            native["samples"][0]["scores"]["hle_scorer"]["value"] = "C"
        elif case == "error":
            native["samples"][0]["error"] = {"message": "failure", "traceback": ""}
        elif case == "model":
            native["eval"]["model"] = "other/model"
        elif case == "revision":
            native["eval"]["task_args"]["dataset_revision"] = "wrong"
        else:
            native["samples"][0]["events"] = [{"event": "tool", "function": "python"}]

    rewrite_native(campaign, "direct", change)
    with pytest.raises(ValueError):
        run(campaign)
    assert not campaign.output.exists()


def test_rejects_wrong_size_expected_dataset(campaign):
    data = comparison.read_json(campaign.expected)
    data["ids"].pop()
    save(campaign.expected, data)
    with pytest.raises(ValueError, match="2158"):
        run(campaign)


def test_qualified_historical_evidence_preserves_attempt_history(campaign):
    audit = historical_audit(campaign)
    result = run(campaign, direct_audit=audit)
    assert result["direct"]["correct"] == 2
    assert (
        result["provenance"]["direct"]["audit"]["repair_summary"]["changed_generation"]
        == 1
    )
    pairs = comparison.jsonl(campaign.output / "paired-outcomes.jsonl")
    assert pairs[0]["direct"]["observed_attempts"][0]["solver_model_event_errors"] == 1


@pytest.mark.parametrize(
    "case", ["incomplete", "unresolved", "hash", "invalid_selected", "source"]
)
def test_rejects_unaccepted_historical_audit(campaign, case):
    path = historical_audit(campaign)
    audit = comparison.read_json(path)
    if case == "incomplete":
        audit["selected_count"] -= 1
    elif case == "unresolved":
        audit["unresolved_ids"] = [campaign.ids[0]]
    elif case == "hash":
        audit["per_id_evidence"]["sha256"] = "0" * 64
    elif case == "source":
        audit["sources"][0]["header"]["eval"]["task"] = "diagnostic"
    else:
        evidence = Path(audit["per_id_evidence"]["path"])
        rows = comparison.jsonl(evidence)
        rows[0]["selected"]["eligible"] = False
        save_rows(evidence, rows)
        audit["per_id_evidence"] = comparison.binding(evidence)
    save(path, audit)
    with pytest.raises(ValueError):
        run(campaign, direct_audit=path)
    assert not campaign.output.exists()


def test_model_alias_is_exact_and_does_not_normalize_other_providers():
    assert (
        comparison.model_identity("k2-vllm/IFM/K2-Horizon-375B-A23B")
        == "vllm/IFM/K2-Horizon-375B-A23B"
    )
    assert comparison.model_identity("k2-vllm/other") == "k2-vllm/other"
    assert comparison.model_identity("openai/model") != comparison.model_identity(
        "vllm/model"
    )


def test_native_eval_archive_round_trip(campaign):
    from test_aggregate_campaign import configuration, row, write_archive

    config = configuration(len(campaign.ids))
    config.update(model="mockllm/solver", dataset_revision=comparison.REVISION)
    outcomes = comparison.jsonl(campaign.tools / "outcomes.jsonl")
    samples = [
        row(item["id"], config, "C" if item["correct"] else "I") for item in outcomes
    ]
    archive = write_archive(campaign.root / "tools-native.eval", config, samples)
    restored = {item["id"]: item for item in comparison.iter_samples(archive)}
    checksum = comparison.file_digest(archive)
    for outcome in outcomes:
        outcome.update(
            archive=str(archive),
            archive_sha256=checksum,
            generation_sha256=generation_digest(restored[outcome["id"]]),
        )
    save_rows(campaign.tools / "outcomes.jsonl", outcomes)
    save(
        campaign.tools / "shards.json",
        [{"path": str(archive), "sha256": checksum, "selected_ids": campaign.ids}],
    )
    result = run(campaign)
    assert result["samples"] == 2158 and result["tools"]["correct"] == 2
    assert result["discordant"]["tools_incorrect_direct_correct"] == 1
