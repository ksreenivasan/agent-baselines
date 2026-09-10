import copy

import pytest

from agent_baselines.evals.hle_tools_v0.recovery import (
    disposition,
    generation_digest,
    read_json_header,
    validate_samples,
)


def sample():
    return {"id": "a", "epoch": 1, "scores": {"hle_scorer": {"value": "I"}}}


def test_exact_membership_and_epoch():
    assert validate_samples([sample()], ["a"])["valid"]
    assert not validate_samples([sample(), sample()], ["a"])["valid"]
    assert not validate_samples([sample()], ["a", "b"])["valid"]
    assert not validate_samples([{**sample(), "epoch": 2}], ["a"])["valid"]
    assert not validate_samples([sample()], ["b"])["valid"]


def test_error_plus_score_and_invalidations_are_never_accepted():
    for change in (
        {"error": {"message": "error"}},
        {"invalidation": {"reason": "bad"}},
        {"metadata": {"hle_tools_invalidated": True}},
        {"metadata": {"hle_tools_infrastructure_errors": ["search_http_402"]}},
    ):
        assert not validate_samples([{**sample(), **change}], ["a"], True)["valid"]


def test_judge_only_empty_response_is_preserved():
    value = {
        "id": "a",
        "epoch": 1,
        "output": {"completion": ""},
        "error": {"traceback": "scorer.py EqualityJudgment"},
    }
    assert disposition(value)[0] == "judge_only"
    assert not validate_samples([value], ["a"])["valid"]
    assert validate_samples([value], ["a"], True)["valid"]


def test_provider_failure_is_separate_from_terminal_model_error():
    value = {"id": "a", "epoch": 1, "error": {"message": "ServerError"}}
    assert disposition(value) == ("generate", "provider_infrastructure_error")
    value["error"]["message"] = "bad tool syntax"
    assert disposition(value)[0] == "terminal_error"


def test_search_outage_contaminates_but_fetch_and_argument_errors_do_not():
    value = sample()
    value["events"] = [
        {
            "event": "tool",
            "function": "web_search",
            "result": '{"error": "search_http_402"}',
        }
    ]
    assert not validate_samples([value], ["a"], True)["valid"]
    value["events"][0]["result"] = {"error": "invalid_query_length"}
    assert validate_samples([value], ["a"])["valid"]
    value["events"][0].update(function="fetch_url", result={"error": "fetch_http_404"})
    assert validate_samples([value], ["a"])["valid"]


def test_generation_hash_resolves_attachments_and_ignores_judge():
    value = {
        **sample(),
        "output": {"completion": "attachment://one"},
        "attachments": {"one": "original answer"},
    }
    restored = copy.deepcopy(value)
    restored["output"]["completion"] = "original answer"
    restored["scores"]["hle_scorer"]["value"] = "C"
    assert generation_digest(restored) == generation_digest(value)
    restored["output"]["completion"] = "changed answer"
    assert generation_digest(restored) != generation_digest(value)


def test_header_does_not_parse_sample_payload(tmp_path):
    path = tmp_path / "source.json"
    path.write_text('{"version": 2, "invalidated": false, "samples": [invalid JSON')
    assert read_json_header(path) == {"version": 2, "invalidated": False}


def test_native_repr_search_failures_and_backends_are_detected():
    for result in (
        repr({"backend": "keenable", "error": "search_timeout"}),
        repr({"backend": "exa", "error": "search_http_402"}),
        repr([{"url": "https://example.com", "backend": "exa"}]),
        repr([{"url": "https://example.com", "backend": "fixture"}]),
    ):
        value = sample()
        value["events"] = [
            {"event": "tool", "function": "web_search", "result": result}
        ]
        assert not validate_samples([value], ["a"])["valid"]


def test_native_repr_keenable_and_model_arguments_remain_ordinary():
    for result in (
        repr([{"url": "https://example.com", "backend": "keenable"}]),
        repr({"error": "invalid_query_length"}),
        "[]",
    ):
        value = sample()
        value["events"] = [
            {"event": "tool", "function": "web_search", "result": result}
        ]
        assert validate_samples([value], ["a"])["valid"]
    value["events"] = [
        {
            "event": "tool",
            "function": "fetch_url",
            "result": repr({"error": "fetch_http_404"}),
        }
    ]
    assert validate_samples([value], ["a"])["valid"]


def test_completion_requires_valid_hle_score_values():
    for scores in (
        {"hle_scorer": {"value": "invalid"}},
        {"hle_scorer": {"value": 1}},
        {"hle_scorer": {}},
        {"hle_scorer": "C"},
        {"trajectory": {"value": "C"}},
        {"hle_scorer": {"value": "C"}, "trajectory": {"value": "maybe"}},
    ):
        assert not validate_samples([{**sample(), "scores": scores}], ["a"])["valid"]
    for scores in (
        {"hle_scorer": {"value": "C"}},
        {"hle_scorer": {"value": "I"}},
        {"agent_baselines/hle_scorer": {"value": "C"}},
        {"hle_scorer": {"value": "C"}, "trajectory": {"value": "C"}},
    ):
        assert validate_samples([{**sample(), "scores": scores}], ["a"])["valid"]


@pytest.mark.parametrize(
    "error_type", ["RateLimitError", "ServerError", "AttemptTimeoutError"]
)
@pytest.mark.parametrize("completion", ["saved response", ""])
def test_judge_provider_error_requires_explicit_stage_and_preserves_generation(
    judge_provider_error_row, error_type, completion
):
    value = judge_provider_error_row(error_type=error_type, completion=completion)
    before = copy.deepcopy(value)
    original_hash = generation_digest(value)
    assert disposition(value) == ("judge_only", "judge_provider_infrastructure_error")
    assert value == before
    assert generation_digest(value) == original_hash


@pytest.mark.parametrize(
    "change",
    [
        "missing_traceback",
        "other_scorer_path",
        "missing_events",
        "missing_model_error",
        "solver_span",
        "wrong_scorer",
        "missing_parent",
        "cycle",
        "duplicate_span",
        "later_success",
    ],
)
def test_ambiguous_stage_never_becomes_judge_only(judge_provider_error_row, change):
    value = judge_provider_error_row(completion="nonempty is not stage evidence")
    if change == "missing_traceback":
        value["error"]["traceback"] = ""
    elif change == "other_scorer_path":
        value["error"]["traceback"] = value["error"]["traceback"].replace(
            "hle_tools_v0", "other"
        )
    elif change == "missing_events":
        value["events"] = []
    elif change == "missing_model_error":
        value["events"][2].pop("error")
    elif change == "solver_span":
        value["events"][1]["type"] = "solver"
    elif change == "wrong_scorer":
        value["events"][1]["name"] = "other_scorer"
    elif change == "missing_parent":
        value["events"][1]["parent_id"] = "missing"
    elif change == "cycle":
        value["events"][1]["parent_id"] = "hle"
    elif change == "duplicate_span":
        value["events"].append(copy.deepcopy(value["events"][1]))
    elif change == "later_success":
        event = copy.deepcopy(value["events"][2])
        event.pop("error")
        value["events"].append(event)
    assert disposition(value) == ("generate", "provider_infrastructure_error")


def test_stage_detection_never_overrides_scores_or_search_invalidation(
    judge_provider_error_row,
):
    value = judge_provider_error_row()
    value["scores"] = {"hle_scorer": {"value": "I"}}
    assert disposition(value) == ("generate", "error_and_score")
    value.pop("error")
    assert disposition(value)[0] == "retain_score"
    value = judge_provider_error_row()
    value["metadata"] = {"hle_tools_invalidated": True}
    assert disposition(value) == ("generate", "hle_tools_invalidated")


def test_nested_judge_request_uses_span_ancestry(judge_provider_error_row):
    value = judge_provider_error_row()
    value["events"][1]["name"] = "agent_baselines/hle_scorer"
    value["events"][2]["span_id"] = "nested-request"
    value["events"].insert(
        2,
        {
            "event": "span_begin",
            "id": "nested-request",
            "parent_id": "hle",
            "type": "request",
            "name": "request",
        },
    )
    assert disposition(value) == ("judge_only", "judge_provider_infrastructure_error")
