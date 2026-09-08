import pytest
from pydantic import ValidationError

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
