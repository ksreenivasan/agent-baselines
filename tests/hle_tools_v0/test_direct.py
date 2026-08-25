from inspect_ai import eval
from inspect_ai.model import ContentImage, ModelOutput, get_model

from agent_baselines.evals.hle_verified_direct.task import hle_verified_direct


def test_direct_condition_is_one_tool_free_multimodal_generation():
    calls = 0

    def output(messages, tools, tool_choice, config):
        nonlocal calls
        calls += 1
        assert tools == []
        assert "one response without using tools" in messages[0].text
        assert any(
            isinstance(item, ContentImage)
            for message in messages
            for item in (message.content if isinstance(message.content, list) else [])
        )
        return ModelOutput.from_content(
            model="mockllm/model",
            content='{"answer":"1","confidence":0.9,"explanation":"fixture"}',
        )

    logs = eval(
        hle_verified_direct(fixture=True),
        model=get_model("mockllm/model", custom_outputs=output, memoize=False),
        display="none",
    )

    assert calls == 1
    assert logs[0].status == "success"
    assert str(logs[0].samples[0].scores["hle_scorer"].value) == "C"
