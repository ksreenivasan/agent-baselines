import asyncio
from copy import deepcopy

import pytest
from inspect_ai._util.content import ContentReasoning, ContentText
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageTool,
    ChatMessageUser,
    GenerateConfig,
    get_model,
)
from inspect_ai.model._providers.vllm import VLLMAPI
from inspect_ai.tool import ToolCall

from agent_baselines.evals.hle_tools_v0.k2_vllm import (
    K2_MODEL,
    _ensure_thinking_fields,
)


def test_replay_empty_native_tool_turn_preserves_real_converter_schema():
    # Same shape as the saved K2 canary's second Python assistant turn.
    messages = [
        ChatMessageUser(content="Continue the check."),
        ChatMessageAssistant(
            content=[
                ContentReasoning(
                    reasoning="Keep the Python state.", internal="reasoning_content"
                ),
                ContentText(text=""),
            ],
            tool_calls=[
                ToolCall(
                    id="assign",
                    function="python_session",
                    arguments={"code": "probe_value = 17"},
                )
            ],
        ),
        ChatMessageTool(content="", tool_call_id="assign", function="python_session"),
        ChatMessageAssistant(
            content="",
            tool_calls=[
                ToolCall(
                    id="calculate",
                    function="python_session",
                    arguments={"code": "print(probe_value + 25)"},
                )
            ],
        ),
        ChatMessageTool(
            content="42\n", tool_call_id="calculate", function="python_session"
        ),
    ]
    original = deepcopy(messages)
    config = GenerateConfig(reasoning_effort="high", reasoning_history="all", top_p=1)
    model = get_model(
        f"k2-vllm/{K2_MODEL}",
        base_url="http://example.invalid/v1",
        api_key="dummy",
        config=config,
        memoize=False,
    )
    standard = VLLMAPI(
        K2_MODEL, base_url="http://example.invalid/v1", api_key="dummy", config=config
    )
    expected = asyncio.run(standard.messages_to_openai(messages))
    assert "reasoning_content" not in expected[3]
    expected[3]["reasoning_content"] = ""
    actual = asyncio.run(model.api.messages_to_openai(messages))
    assert actual == expected
    assert actual[1]["reasoning_content"] == "Keep the Python state."
    assert (
        actual[3]["tool_calls"][0]["function"]["arguments"]
        == '{"code": "print(probe_value + 25)"}'
    )
    assert messages == original
    assert model.api.service_model_name() == K2_MODEL
    assert model.api._init_base_url == "http://example.invalid/v1"
    assert model.api._init_config == config


@pytest.mark.parametrize(
    "field", ["think", "reasoning", "reasoning_content", "think_fast", "think_faster"]
)
@pytest.mark.parametrize("value", ["", "Existing reasoning"])
def test_accepted_thinking_fields_are_preserved(field, value):
    messages = [{"role": "assistant", "content": "", field: value}]
    expected = deepcopy(messages)
    assert _ensure_thinking_fields(messages) == expected


def test_other_roles_and_existing_fields_are_untouched():
    messages = [
        {"role": "system", "content": "System"},
        {"role": "user", "content": "Question"},
        {"role": "tool", "content": "42", "tool_call_id": "calculate"},
        {"role": "assistant", "content": "Answer", "reasoning_content": None},
    ]
    expected = deepcopy(messages)
    assert _ensure_thinking_fields(messages) == expected


def test_exact_model_guard_and_standard_provider_isolation():
    with pytest.raises(ValueError, match="only supports"):
        get_model(
            "k2-vllm/other-model", base_url="http://example.invalid/v1", memoize=False
        )
    standard = get_model(
        f"vllm/{K2_MODEL}", base_url="http://example.invalid/v1", memoize=False
    )
    message = ChatMessageAssistant(
        content="",
        tool_calls=[
            ToolCall(id="t", function="submit", arguments={"submission": "42"})
        ],
    )
    serialized = asyncio.run(standard.api.messages_to_openai([message]))
    assert "reasoning_content" not in serialized[0]


def test_fresh_process_bootstrap_registers_before_cli_model_resolution():
    import subprocess
    import sys
    from pathlib import Path

    script = """
import importlib
import runpy
import sys
from inspect_ai.model import GenerateConfig, get_model

cli_eval = importlib.import_module("inspect_ai._cli.eval")
def resolve_without_generation(**params):
    name = params["model"]
    if isinstance(name, list):
        name = name[0]
    model = get_model(name, base_url=params["model_base_url"], api_key="dummy", memoize=False)
    assert model.api.service_model_name() == "IFM/K2-Horizon-375B-A23B"
    assert not model.api._server_resolved
    importlib.import_module("agent_baselines.evals.hle_tools_v0.task")
    second = get_model(name, base_url=params["model_base_url"], api_key="dummy", memoize=False)
    assert type(second.api) is type(model.api)
    print("fresh-k2-registration-ok")
cli_eval.eval = resolve_without_generation
sys.argv = ["inspect", "eval", "agent_baselines/evals/hle_tools_v0/task.py@hle_tools_v0", "--model", "k2-vllm/IFM/K2-Horizon-375B-A23B", "--model-base-url", "http://example.invalid/v1"]
runpy.run_module("agent_baselines.evals.hle_tools_v0.k2_vllm_cli", run_name="__main__")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[2],
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stderr
    assert "fresh-k2-registration-ok" in result.stdout
