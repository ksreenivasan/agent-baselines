from inspect_ai import eval
from inspect_ai.model import ContentImage, ModelOutput, get_model

from agent_baselines.evals.hle_direct.task import hle_direct


def test_direct_condition_is_one_tool_free_multimodal_generation():
    calls = 0

    def output(messages, tools, tool_choice, config):
        nonlocal calls
        calls += 1
        assert tools == []
        assert "Exact Answer: {your succinct, final answer}" in messages[0].text
        assert any(
            isinstance(item, ContentImage)
            for message in messages
            for item in (message.content if isinstance(message.content, list) else [])
        )
        return ModelOutput.from_content(
            model="mockllm/model",
            content="Explanation: counted\nExact Answer: 1\nConfidence: 90%",
        )

    def judge_output(messages, tools, tool_choice, config):
        assert config.reasoning_effort == "medium"
        assert config.response_schema.name == "hle_equality_judgment"
        assert "Exact Answer: 1" in messages[-1].text
        return ModelOutput.from_content(
            model="mockllm/judge",
            content=(
                '{"extracted_final_answer":"1","reasoning":"matches",'
                '"correct":"yes","confidence":90,"strict":true}'
            ),
        )

    judge = get_model("mockllm/judge", custom_outputs=judge_output, memoize=False)

    logs = eval(
        hle_direct(fixture=True, judge_model=judge),
        model=get_model("mockllm/model", custom_outputs=output, memoize=False),
        display="none",
    )

    assert calls == 1
    assert logs[0].status == "success"
    assert str(logs[0].samples[0].scores["hle_scorer"].value) == "C"
