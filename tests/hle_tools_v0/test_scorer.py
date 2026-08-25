import pytest

from agent_baselines.evals.hle_tools_v0.scorer import JUDGE_PROMPT, parse_submission


def test_judge_prompt_formats_json_schema():
    prompt = JUDGE_PROMPT.format(question="q", answer="a", target="t")
    assert '{"correct": "yes" or "no"' in prompt


def test_parse_submission():
    parsed = parse_submission('{"answer":"1","confidence":0.75,"explanation":"counted"}')
    assert parsed["answer"] == "1"
    assert parsed["confidence"] == 0.75


@pytest.mark.parametrize(
    "payload",
    [
        '{"answer":"","confidence":0.5}',
        '{"answer":"1","confidence":1.1}',
        '{"answer":"1","confidence":0.5,"extra":true}',
    ],
)
def test_invalid_submission(payload):
    with pytest.raises(ValueError):
        parse_submission(payload)
