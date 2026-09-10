import copy

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
