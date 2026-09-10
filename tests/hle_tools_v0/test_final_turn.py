from inspect_ai import Task, eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessageAssistant,
    ModelOutput,
    get_model,
)
from inspect_ai.scorer import match
from inspect_ai.tool import ToolCall, tool

from agent_baselines.solvers.hle_tools_v0.solver import hle_tools_agent
from agent_baselines.solvers.react.basic_agent import basic_agent


@tool
def dummy_tool():
    async def execute():
        """A tool that must not be offered on the reserved final turn."""
        return "unused"

    return execute


def test_final_turn_exposes_only_submit():
    observed_tools = []

    def output(messages, tools, tool_choice, config):
        observed_tools.append([tool.name for tool in tools])
        if len(observed_tools) < 15:
            return ModelOutput.from_content(
                model="mockllm/model", content="still working"
            )
        return ModelOutput(
            model="mockllm/model",
            choices=[
                ChatCompletionChoice(
                    message=ChatMessageAssistant(
                        content="",
                        model="mockllm/model",
                        tool_calls=[
                            ToolCall(
                                id="submit-final",
                                function="submit",
                                arguments={"submission": "done"},
                            )
                        ],
                    ),
                    stop_reason="tool_calls",
                )
            ],
        )

    model = get_model("mockllm/model", custom_outputs=output, memoize=False)
    logs = eval(
        Task(
            dataset=MemoryDataset([Sample(input="test", target="done")]),
            solver=basic_agent(
                tools=[dummy_tool()], max_steps=15, final_step_submit_only=True
            ),
            scorer=match(),
        ),
        model=model,
        display="none",
        sandbox="local",
    )
    assert logs[0].status == "success"
    assert len(observed_tools) == 15
    assert "dummy_tool" in observed_tools[0]
    assert observed_tools[-1] == ["submit"]


def test_hle_prompt_and_submit_tool_use_plain_hle_response(monkeypatch):
    monkeypatch.setenv("HLE_SEARCH_BACKEND", "fixture")
    submission = "Explanation: counted\nExact Answer: 1\nConfidence: 100%"

    def output(messages, tools, tool_choice, config):
        assert "Exact Answer: {your succinct, final answer}" in messages[0].text
        assert "Do not JSON-encode" in messages[1].text
        assert [tool.name for tool in tools] == ["submit"]
        assert tools[0].parameters.required == ["submission"]
        return ModelOutput(
            model="mockllm/model",
            choices=[
                ChatCompletionChoice(
                    message=ChatMessageAssistant(
                        content="",
                        model="mockllm/model",
                        tool_calls=[
                            ToolCall(
                                id="submit-shaped",
                                function="submit",
                                arguments={"submission": submission},
                            )
                        ],
                    ),
                    stop_reason="tool_calls",
                )
            ],
        )

    logs = eval(
        Task(
            dataset=MemoryDataset([Sample(input="test", target=submission)]),
            solver=hle_tools_agent(max_steps=1),
            scorer=match(),
        ),
        model=get_model("mockllm/model", custom_outputs=output, memoize=False),
        display="none",
        sandbox="local",
    )
    assert logs[0].status == "success"
