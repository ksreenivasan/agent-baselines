import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from inspect_ai.log import read_eval_log, write_eval_log

from agent_baselines.evals.hle_tools_v0.recovery import (
    file_digest,
    generation_digest,
    iter_samples,
)
from test_aggregate_campaign import COMMIT, configuration, row, save, write_archive

SCRIPT = Path(__file__).parents[2] / "solvers/hle-tools-v0/m2/residual_selection.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("residual_selection", SCRIPT)
selector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(selector)


@pytest.fixture
def campaign(tmp_path):
    ids = ["a", "b", "c", "d"]
    config = configuration(len(ids))
    cfg = save(tmp_path / "full.json", config)
    expected = save(
        tmp_path / "expected.json",
        {
            "ids": ids,
            "dataset_revision": config["dataset_revision"],
            "dataset_variant": "standard",
        },
    )
    source = tmp_path / "source"
    source.mkdir()
    (source / "protocol.yaml").write_text("frozen")
    protocol = save(
        tmp_path / "PROTOCOL.json",
        {
            "source_root": str(source),
            "file_sha256": {"protocol.yaml": file_digest(source / "protocol.yaml")},
            "config_sha256": {cfg.name: file_digest(cfg)},
            "generation_policy": {"max_new_infrastructure_generation_retries": 1},
        },
    )
    state = SimpleNamespace(
        root=tmp_path,
        config=config,
        cfg=cfg,
        expected=expected,
        protocol=protocol,
        attempts=[],
        counter=0,
    )

    def attempt(
        name, rows=None, *, ids=None, manifest=None, complete=False, status="success"
    ):
        directory = (tmp_path / name).resolve()
        state.counter += 1
        if manifest is None:
            manifest = {
                "ids": ids or ["a", "b", "c", "d"],
                "dataset_revision": config["dataset_revision"],
                "dataset_variant": "standard",
            }
        path = save(directory / "manifest.json", manifest)
        save(
            directory / "launch.json",
            {
                "config": config,
                "config_sha256": file_digest(cfg),
                "source_commit": COMMIT,
                "started_at": state.counter,
                "manifest": manifest,
                "manifest_sha256": file_digest(path),
                "command": [f"manifest_path={path}"],
            },
        )
        if rows is not None:
            archive = write_archive(
                directory / "logs/result.eval", config, rows, status=status
            )
            native = read_eval_log(archive)
            native.eval.task_args["manifest_path"] = str(path)
            write_eval_log(native, archive)
            save(
                directory / "result.json",
                {
                    "complete": complete,
                    "publisher_exit": 0,
                    "archive": str(archive),
                    "archive_sha256": file_digest(archive),
                },
            )
        state.attempts.append(directory)
        return directory

    state.attempt = attempt
    return state


def select(state, name="selection", **kwargs):
    return selector.select(
        state.cfg,
        state.expected,
        state.protocol,
        state.root / name,
        attempt_dirs=state.attempts,
        allowed_commits={COMMIT},
        **kwargs,
    )


def diagnose(attempt, *ids):
    return {
        sample_id: {
            "attempt": str(attempt),
            "closed": True,
            "evidence": "Supervisor verified job is finished; provider outage diagnosed.",
        }
        for sample_id in ids
    }


def test_preserves_incorrect_and_separates_saved_judge_from_explicit_retry(
    campaign, judge_provider_error_row
):
    judge = judge_provider_error_row(sample_id="b", completion="")
    saved = row("b", campaign.config, score=None)
    saved["output"]["completion"] = ""
    saved["error"] = judge["error"]
    saved["events"] += judge["events"]
    attempt = campaign.attempt(
        "first",
        [
            row("a", campaign.config, "I"),
            saved,
            row("c", campaign.config, None, "ServerError unavailable"),
            row("d", campaign.config, None, "unknown model error"),
        ],
    )
    native_saved = next(
        value
        for value in iter_samples(attempt / "logs/result.eval")
        if value["id"] == "b"
    )
    original_hash = generation_digest(native_saved)
    ledger = select(
        campaign, partial={str(attempt): ["a"]}, infrastructure=diagnose(attempt, "c")
    )
    assert ledger["counts"] == {"retain": 1, "judge_only": 1, "retry": 1, "held": 1}
    entries = {item["id"]: item for item in ledger["samples"]}
    assert entries["a"]["selected"]["score"] == "I"
    assert entries["b"]["selected"]["generation_sha256"] == original_hash
    assert entries["b"]["selected"]["disposition"] == "judge_only"
    manifest = json.loads((campaign.root / "selection/retry/000.json").read_text())
    assert manifest["ids"] == ["c"] and manifest["generation_attempt"] == 2
    assert manifest["selection_ledger"]["sha256"] == file_digest(
        campaign.root / "selection/ledger.json"
    )
    assert manifest["retry_of"]["c"]["launch_sha256"] == file_digest(
        attempt / "launch.json"
    )
    assert (
        json.loads((campaign.root / "selection/judge-only.json").read_text())[0]["id"]
        == "b"
    )
    with pytest.raises(ValueError, match="already exists"):
        select(
            campaign,
            partial={str(attempt): ["a"]},
            infrastructure=diagnose(attempt, "c"),
        )


def test_all_regeneration_requires_explicit_diagnosis(campaign):
    campaign.attempt(
        "first", [row("a", campaign.config, None, "ServerError")], ids=["a"]
    )
    ledger = select(campaign)
    assert ledger["counts"] == {"held": 1, "unlaunched": 3}
    assert not (campaign.root / "selection/retry").exists()


def test_launch_only_attempts_consume_budget_and_third_launch_is_rejected(campaign):
    first = campaign.attempt("first")
    held = select(campaign, name="held")
    assert held["counts"] == {"held": 4}
    ledger = select(campaign, infrastructure=diagnose(first, "b"))
    assert ledger["samples"][1]["launched_generation_attempts"] == 1
    manifest = json.loads((campaign.root / "selection/retry/000.json").read_text())
    campaign.attempt("second", manifest=manifest)
    second = select(campaign, name="second-selection")
    b = next(item for item in second["samples"] if item["id"] == "b")
    assert b["action"] == "exhausted" and b["launched_generation_attempts"] == 2
    assert not (campaign.root / "second-selection/retry").exists()
    campaign.attempt("third", manifest=manifest)
    with pytest.raises(ValueError, match="more than one additional"):
        select(campaign, name="third-selection")


@pytest.mark.parametrize(
    "case", ["closed_false", "empty_evidence", "wrong_attempt", "unknown_id"]
)
def test_diagnosis_requires_explicit_closure_and_matching_latest_attempt(
    campaign, case
):
    first = campaign.attempt("first")
    diagnosis = diagnose(first, "a")
    if case == "closed_false":
        diagnosis["a"]["closed"] = False
    elif case == "empty_evidence":
        diagnosis["a"]["evidence"] = ""
    elif case == "wrong_attempt":
        diagnosis["a"]["attempt"] = str(campaign.root / "not-provided")
    else:
        diagnosis["unknown"] = diagnosis.pop("a")
    with pytest.raises(ValueError):
        select(campaign, infrastructure=diagnosis)


def test_failed_shard_clean_rows_need_explicit_selection_and_cannot_be_regenerated(
    campaign,
):
    first = campaign.attempt(
        "first", [row("a", campaign.config, "I")], ids=["a"], status="cancelled"
    )
    ledger = select(campaign)
    assert ledger["samples"][0]["action"] == "partial_selection_required"
    with pytest.raises(ValueError, match="cannot regenerate"):
        select(campaign, name="bad-retry", infrastructure=diagnose(first, "a"))
    accepted = select(campaign, name="accepted", partial={str(first): ["a"]})
    assert accepted["samples"][0]["action"] == "retain"


def test_judge_only_answer_cannot_be_overridden_by_infrastructure_selection(
    campaign, judge_provider_error_row
):
    saved = row("a", campaign.config, None)
    native = judge_provider_error_row(sample_id="a")
    saved["error"], saved["events"] = (
        native["error"],
        saved["events"] + native["events"],
    )
    first = campaign.attempt("first", [saved], ids=["a"])
    with pytest.raises(ValueError, match="cannot regenerate"):
        select(campaign, infrastructure=diagnose(first, "a"))


def test_initial_manifests_must_be_disjoint(campaign):
    campaign.attempt("first", ids=["a"])
    campaign.attempt("overlap", ids=["a"])
    with pytest.raises(ValueError, match="overlapping initial"):
        select(campaign)


@pytest.mark.parametrize(
    "case",
    [
        "unknown_id",
        "manifest_hash",
        "archive_hash",
        "config_hash",
        "diagnostic",
        "unapproved_source",
        "partial_missing",
    ],
)
def test_invalid_records_are_rejected_before_output(campaign, case):
    first = campaign.attempt("first", [row("a", campaign.config)], ids=["a"])
    kwargs = {}
    if case == "unknown_id":
        campaign.attempt("unknown", ids=["z"])
    elif case == "manifest_hash":
        with (first / "manifest.json").open("a") as stream:
            stream.write(" ")
    elif case == "archive_hash":
        with (first / "logs/result.eval").open("ab") as stream:
            stream.write(b"changed")
    elif case == "config_hash":
        data = json.loads((first / "launch.json").read_text())
        data["config_sha256"] = "changed"
        save(first / "launch.json", data)
    elif case == "diagnostic":
        data = json.loads(campaign.cfg.read_text())
        data["run_phase"] = "diagnostic"
        save(campaign.cfg, data)
    elif case == "unapproved_source":
        data = json.loads((first / "launch.json").read_text())
        data["source_commit"] = "b" * 40
        save(first / "launch.json", data)
    else:
        kwargs["partial"] = {str(first): ["b"]}
    with pytest.raises(ValueError):
        select(campaign, **kwargs)
    assert not (campaign.root / "selection").exists()


def test_retry_requires_parent_record_and_unchanged_selection_ledger(campaign):
    first = campaign.attempt("first", ids=["a"])
    select(campaign, infrastructure=diagnose(first, "a"))
    manifest = json.loads((campaign.root / "selection/retry/000.json").read_text())
    second = campaign.attempt("second", manifest=manifest)
    campaign.attempts = [second]
    with pytest.raises(ValueError, match="initial attempt"):
        select(campaign, name="missing-parent")
    campaign.attempts = [first, second]
    with (campaign.root / "selection/ledger.json").open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="checksum"):
        select(campaign, name="changed-ledger")


def test_completed_incorrect_score_is_retained_without_partial_selection(campaign):
    campaign.attempt(
        "first", [row("a", campaign.config, "I")], ids=["a"], complete=True
    )
    ledger = select(campaign)
    assert ledger["samples"][0]["action"] == "retain"
    assert ledger["samples"][0]["selected"]["score"] == "I"


def test_retry_manifest_cannot_change_prior_launch_binding(campaign):
    first = campaign.attempt("first", ids=["a"])
    select(campaign, infrastructure=diagnose(first, "a"))
    manifest = json.loads((campaign.root / "selection/retry/000.json").read_text())
    manifest["retry_of"]["a"]["launch_sha256"] = "changed"
    campaign.attempt("second", manifest=manifest)
    with pytest.raises(ValueError, match="prior launch binding"):
        select(campaign, name="bad-binding")


def test_retry_manifest_cannot_authorize_more_ids_than_its_ledger(campaign):
    first = campaign.attempt("first", ids=["a", "b"])
    select(campaign, infrastructure=diagnose(first, "a"))
    manifest = json.loads((campaign.root / "selection/retry/000.json").read_text())
    manifest["ids"] = ["b"]
    manifest["retry_of"] = {"b": manifest["retry_of"]["a"]}
    campaign.attempt("second", manifest=manifest)
    with pytest.raises(ValueError, match="not authorized"):
        select(campaign, name="bad-ids")


def test_retry_shards_are_bounded_disjoint_and_preserve_expected_order(campaign):
    first = campaign.attempt("first")
    select(campaign, infrastructure=diagnose(first, "a", "b", "c", "d"), shard_size=2)
    manifests = [
        json.loads(path.read_text())
        for path in sorted((campaign.root / "selection/retry").glob("*.json"))
    ]
    assert [manifest["ids"] for manifest in manifests] == [["a", "b"], ["c", "d"]]
    with pytest.raises(ValueError, match="between 1 and 200"):
        select(campaign, name="too-large", shard_size=201)


def test_duplicate_attempt_directory_is_rejected(campaign):
    first = campaign.attempt("first")
    campaign.attempts.append(first)
    with pytest.raises(ValueError, match="duplicate attempt"):
        select(campaign)


@pytest.mark.parametrize("result_state", ["absent", "without_archive"])
def test_unfinalized_durable_scores_block_regeneration(campaign, result_state):
    first = campaign.attempt("first", [row("a", campaign.config, "I")], ids=["a"])
    if result_state == "absent":
        (first / "result.json").unlink()
    else:
        save(
            first / "result.json",
            {"complete": False, "reason": "publication interrupted"},
        )
    ledger = select(campaign)
    assert ledger["samples"][0]["action"] == "held"
    assert (
        ledger["samples"][0]["attempts"][0]["reason"] == "unfinalized_durable_archive"
    )
    with pytest.raises(ValueError, match="unfinalized durable"):
        select(campaign, name="bad-retry", infrastructure=diagnose(first, "a"))


def test_result_cannot_point_to_another_attempt_archive(campaign):
    first = campaign.attempt("first", [row("a", campaign.config)], ids=["a"])
    second = campaign.attempt("second", [row("b", campaign.config)], ids=["b"])
    changed = json.loads((first / "result.json").read_text())
    other = json.loads((second / "result.json").read_text())
    changed["archive"], changed["archive_sha256"] = (
        other["archive"],
        other["archive_sha256"],
    )
    save(first / "result.json", changed)
    with pytest.raises(ValueError, match="does not belong"):
        select(campaign)


def test_archive_native_manifest_binding_must_match_launch(campaign):
    first = campaign.attempt("first", [row("a", campaign.config)], ids=["a"])
    archive = first / "logs/result.eval"
    native = read_eval_log(archive)
    native.eval.task_args["manifest_path"] = "/other/manifest.json"
    write_eval_log(native, archive)
    result = json.loads((first / "result.json").read_text())
    result["archive_sha256"] = file_digest(archive)
    save(first / "result.json", result)
    with pytest.raises(ValueError, match="manifest path differs"):
        select(campaign)


def test_extra_unindexed_native_archive_is_not_ignored(campaign):
    first = campaign.attempt(
        "first", [row("a", campaign.config, None, "ServerError")], ids=["a"]
    )
    source = first / "logs/result.eval"
    (first / "logs/unindexed.eval").write_bytes(source.read_bytes())
    with pytest.raises(ValueError, match="exactly its bound"):
        select(campaign, infrastructure=diagnose(first, "a"))


def test_legacy_ids_only_manifest_preserves_bytes_and_bound_outcomes(campaign):
    first = campaign.attempt(
        "legacy", [row("a", campaign.config, "I")], manifest={"ids": ["a", "b"]}
    )
    path = first / "manifest.json"
    original = path.read_bytes()
    checksum = file_digest(path)
    ledger = select(
        campaign, partial={str(first): ["a"]}, infrastructure=diagnose(first, "b")
    )
    assert ledger["counts"] == {"retain": 1, "retry": 1, "unlaunched": 2}
    assert path.read_bytes() == original and file_digest(path) == checksum
    assert ledger["attempts"][str(first)]["manifest"]["sha256"] == checksum
    decisions = {record["id"]: record for record in ledger["samples"]}
    assert decisions["a"]["selected"]["score"] == "I"
    assert decisions["a"]["launched_generation_attempts"] == 1
    assert decisions["b"]["launched_generation_attempts"] == 1
    assert decisions["c"]["launched_generation_attempts"] == 0


@pytest.mark.parametrize("revision", [None, "wrong-revision"])
def test_explicit_invalid_manifest_revision_is_not_legacy(campaign, revision):
    campaign.attempt(
        "explicit",
        [row("a", campaign.config)],
        manifest={"ids": ["a"], "dataset_revision": revision},
    )
    with pytest.raises(ValueError, match="attempt manifest dataset revision differs"):
        select(campaign)


@pytest.mark.parametrize("case", ["native_revision", "expected_revision", "unknown_id"])
def test_legacy_manifest_still_requires_native_and_expected_bindings(campaign, case):
    sample_id = "foreign" if case == "unknown_id" else "a"
    first = campaign.attempt(
        "legacy", [row(sample_id, campaign.config)], manifest={"ids": [sample_id]}
    )
    if case == "native_revision":
        path = first / "logs/result.eval"
        native = read_eval_log(path)
        native.eval.task_args["dataset_revision"] = "wrong-revision"
        write_eval_log(native, path)
        result = json.loads((first / "result.json").read_text())
        result["archive_sha256"] = file_digest(path)
        save(first / "result.json", result)
    elif case == "expected_revision":
        expected = json.loads(campaign.expected.read_text())
        expected["dataset_revision"] = "wrong-revision"
        save(campaign.expected, expected)
    with pytest.raises(ValueError):
        select(campaign, partial={str(first): [sample_id]})


@pytest.mark.parametrize("result_present", [False, True])
def test_legacy_manifest_cannot_infer_revision_without_native_archive(
    campaign, result_present
):
    first = campaign.attempt("legacy", manifest={"ids": ["a"]})
    if result_present:
        save(first / "result.json", {"complete": False, "publisher_exit": 0})
    with pytest.raises(
        ValueError, match="legacy manifest requires bound native archive"
    ):
        select(campaign, infrastructure=diagnose(first, "a"))
