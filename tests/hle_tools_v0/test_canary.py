import importlib.util
import json
from pathlib import Path

import pytest
from inspect_ai.model import ChatMessageAssistant, ChatMessageTool
from inspect_ai.tool import ToolCall, ToolCallError

SCRIPT = Path(__file__).resolve().parents[2] / "solvers/hle-tools-v0/m2/canary.py"
spec = importlib.util.spec_from_file_location("hle_canary", SCRIPT)
canary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(canary)

ANSWER = "Explanation: computed\nExact Answer: 42\nConfidence: 100%"


def messages():
    calls = [
        (
            "web_search",
            {"query": "Python documentation"},
            json.dumps([{"url": "https://docs.python.org/", "backend": "keenable"}]),
        ),
        (
            "fetch_url",
            {"url": "https://docs.python.org/"},
            json.dumps(
                {
                    "url": "https://docs.python.org/",
                    "status": 200,
                    "text": "Python docs",
                }
            ),
        ),
        ("python_session", {"code": "probe_value = 17"}, ""),
        ("python_session", {"code": "print(probe_value + 25)"}, "42\n"),
        ("submit", {"submission": ANSWER}, "Submission recorded"),
    ]
    result = []
    for index, (name, arguments, response) in enumerate(calls):
        result += [
            ChatMessageAssistant(
                content="",
                tool_calls=[
                    ToolCall(id=f"call-{index}", function=name, arguments=arguments)
                ],
            ),
            ChatMessageTool(
                content=response, tool_call_id=f"call-{index}", function=name
            ),
        ]
    return result


def test_successful_sequential_stateful_trajectory():
    assert canary.check_trajectory(messages(), ANSWER, False)[0]


@pytest.mark.parametrize("failed_step", [1, 3, 5, 7, 9])
def test_tool_execution_error_rejects_canary(failed_step):
    items = messages()
    items[failed_step].error = ToolCallError("unknown", "failed")
    assert not canary.check_trajectory(items, ANSWER, False)[0]


def test_fetch_http_error_rejects_canary():
    items = messages()
    items[3].content = json.dumps(
        {"url": "https://docs.python.org/", "error": "fetch_http_404"}
    )
    assert not canary.check_trajectory(items, ANSWER, False)[0]


def test_two_python_calls_in_same_model_turn_rejects_canary():
    items = messages()
    items[4].tool_calls += items[6].tool_calls
    del items[6]
    assert not canary.check_trajectory(items, ANSWER, False)[0]


def test_printing_constant_or_reassigning_does_not_prove_state():
    for code in ("print(42)", "probe_value = 17; print(probe_value + 25)"):
        items = messages()
        items[6].tool_calls[0].arguments["code"] = code
        assert not canary.check_trajectory(items, ANSWER, False)[0]


def test_python_text_error_or_wrong_result_rejects_canary():
    for response in ("Error/Traceback: NameError: probe_value", "142"):
        items = messages()
        items[7].content = response
        assert not canary.check_trajectory(items, ANSWER, False)[0]


def test_fixture_search_and_invalidated_sample_reject_canary():
    items = messages()
    items[1].content = json.dumps(
        [{"url": "https://docs.python.org/", "backend": "fixture"}]
    )
    assert not canary.check_trajectory(items, ANSWER, False)[0]
    assert not canary.check_trajectory(messages(), ANSWER, True)[0]


def test_native_inspect_python_repr_payloads():
    items = messages()
    items[1].content = repr(
        [{"rank": 1, "url": "https://docs.python.org/", "backend": "keenable"}]
    )
    items[3].content = repr(
        {
            "url": "https://docs.python.org/",
            "status": 200,
            "text": "Python docs",
            "truncated": False,
        }
    )
    assert canary.check_trajectory(items, ANSWER, False)[0]


def test_native_redirect_does_not_prove_content_retrieval():
    items = messages()
    items[1].content = repr(
        [{"url": "https://docs.python.org/", "backend": "keenable"}]
    )
    items[3].content = repr(
        {
            "url": "https://docs.python.org/",
            "status": 302,
            "text": "<html><h1>302 Found</h1></html>",
            "truncated": False,
        }
    )
    valid, reason = canary.check_trajectory(items, ANSWER, False)
    assert not valid
    assert reason == "Fetch did not retrieve a searched URL successfully"


def test_tool_payload_is_never_executed():
    items = messages()
    items[1].content = "__import__('builtins').print('not a literal')"
    assert not canary.check_trajectory(items, ANSWER, False)[0]
