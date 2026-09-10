import asyncio
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from inspect_ai.model import ModelOutput, ModelUsage
from inspect_ai.scorer import Target

from agent_baselines.evals.hle_tools_v0 import scorer as scorer_module

from agent_baselines.evals.hle_tools_v0.prompts import (
    EQUALITY_CHECKER_PROMPT,
    EXACT_ANSWER_SYSTEM_PROMPT,
    MULTIPLE_CHOICE_SYSTEM_PROMPT,
    system_prompt_for,
)
from agent_baselines.evals.hle_tools_v0.scorer import parse_judgment


def test_judge_prompt_passes_complete_response():
    response = "Explanation: work\nExact Answer: 1\nConfidence: 75%"
    prompt = EQUALITY_CHECKER_PROMPT.format(
        question="q", response=response, correct_answer="1"
    )
    assert f"[response]: {response}" in prompt
    assert "extracted_final_answer:" in prompt
    assert "within a small margin of error" in prompt


def test_answer_type_selects_canonical_hle_response_prompt():
    assert system_prompt_for("exact_match") == EXACT_ANSWER_SYSTEM_PROMPT
    assert system_prompt_for("multiple_choice") == MULTIPLE_CHOICE_SYSTEM_PROMPT
    assert "Exact Answer:" in EXACT_ANSWER_SYSTEM_PROMPT
    assert "\nAnswer:" in MULTIPLE_CHOICE_SYSTEM_PROMPT


def test_parse_judgment():
    parsed = parse_judgment(
        '{"extracted_final_answer":"1","reasoning":"matches",'
        '"correct":"yes","confidence":75}'
    )
    assert parsed.extracted_final_answer == "1"
    assert parsed.confidence == 75


@pytest.mark.parametrize(
    "payload",
    [
        '{"extracted_final_answer":"1","reasoning":"x","correct":"maybe","confidence":75}',
        '{"extracted_final_answer":"1","reasoning":"x","correct":"yes","confidence":101}',
        '{"extracted_final_answer":"1","reasoning":"x","correct":"yes","confidence":75,"extra":true}',
    ],
)
def test_invalid_judgment(payload):
    with pytest.raises(ValidationError):
        parse_judgment(payload)


_VALID_JUDGMENT = (
    '{"extracted_final_answer":"1","reasoning":"matches",'
    '"correct":"yes","confidence":75}'
)


def run_judge(monkeypatch, outputs):
    calls, events = [], []

    class Judge:
        name = "test-judge"

        async def generate(self, **kwargs):
            calls.append(kwargs)
            return outputs[len(calls) - 1]

    monkeypatch.setattr(scorer_module, "get_model", lambda model: Judge())
    monkeypatch.setattr(
        scorer_module,
        "transcript",
        lambda: SimpleNamespace(info=lambda data, **kwargs: events.append(data)),
    )
    state = SimpleNamespace(
        input_text="question",
        output=SimpleNamespace(completion="saved generator response"),
        metadata={},
    )
    return scorer_module.hle_scorer(), state, calls, events


def judge_output(text, *, stop_reason="stop", tokens=25):
    output = ModelOutput.from_content("test-judge", text)
    output.choices[0].stop_reason = stop_reason
    output.usage = ModelUsage(
        input_tokens=10, output_tokens=tokens, total_tokens=tokens + 10
    )
    return output


def test_judge_repair_preserves_prompt_and_records_raw_attempts(monkeypatch):
    active, state, calls, events = run_judge(
        monkeypatch, [judge_output("not json"), judge_output(_VALID_JUDGMENT)]
    )
    result = asyncio.run(active(state, Target("1")))
    assert result.value == "C"
    assert len(calls) == len(events) == 2
    assert calls[0] == calls[1]
    assert "saved generator response" in calls[0]["input"]
    assert events[0]["output"]["completion"] == "not json"
    assert result.metadata["judge_attempts"] == 2


@pytest.mark.parametrize(
    "stop_reason,tokens,text,expected_cap",
    [
        ("max_tokens", 4096, '{"extracted_final_answer":', 16384),
        ("stop", 4096, '{"extracted_final_answer":', 4096),
        ("max_tokens", 4095, '{"extracted_final_answer":', 4096),
        ("max_tokens", 4096, "not json", 4096),
    ],
)
def test_cap_increase_requires_actual_truncation_evidence(
    monkeypatch, stop_reason, tokens, text, expected_cap
):
    active, state, calls, events = run_judge(
        monkeypatch,
        [
            judge_output(text, stop_reason=stop_reason, tokens=tokens),
            judge_output(_VALID_JUDGMENT),
        ],
    )
    asyncio.run(active(state, Target("1")))
    assert calls[0]["config"].max_tokens == 4096
    assert calls[1]["config"].max_tokens == expected_cap
    assert calls[0]["input"] == calls[1]["input"]
    assert calls[0]["config"].response_schema == calls[1]["config"].response_schema
    assert calls[0]["config"].reasoning_effort == calls[1]["config"].reasoning_effort


def test_judge_repair_is_bounded_and_failure_preserves_attempts(monkeypatch):
    active, state, calls, events = run_judge(monkeypatch, [judge_output("invalid")] * 3)
    with pytest.raises(ValidationError):
        asyncio.run(active(state, Target("1")))
    assert len(calls) == len(events) == 3
