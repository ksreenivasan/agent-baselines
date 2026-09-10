"""Synthetic model-generated multistep tool compatibility check."""

import ast
import json
import re

from inspect_ai import Task, task
from inspect_ai.model import ChatMessageAssistant, ChatMessageTool
from inspect_ai.dataset import Sample
from inspect_ai.event import ToolEvent
from inspect_ai.log import transcript
from inspect_ai.scorer import CORRECT, INCORRECT, Score, accuracy, scorer
from inspect_ai.solver import TaskState
from inspect_ai.scorer import Target

from agent_baselines.evals.hle_tools_v0.task import _sandbox
from agent_baselines.evals.hle_tools_v0.scorer import hle_scorer
from agent_baselines.solvers.hle_tools_v0.solver import hle_tools_agent


def _matches_code(code: str, expected: str) -> bool:
    try:
        return ast.dump(ast.parse(code)) == ast.dump(ast.parse(expected))
    except (SyntaxError, TypeError):
        return False


def _literal_payload(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return ast.literal_eval(text)


def _tool_payload(reply, call, events):
    try:
        return _literal_payload(reply.text)
    except (ValueError, SyntaxError):
        pass
    event = next(
        (
            event
            for event in events
            if isinstance(event, ToolEvent)
            and event.id == call.id
            and event.function == call.function
            and event.message_id in {None, reply.id}
        ),
        None,
    )
    if event is None or event.error or event.failed:
        raise ValueError("no matching successful tool event")
    if not isinstance(event.result, str):
        return event.result
    raw = event.result
    prefix = (
        f"\nThe output of your call to {call.function} was too long to be displayed.\n"
        "Here is a truncated version:\n<START_TOOL_OUTPUT>\n"
    )
    suffix = "\n<END_TOOL_OUTPUT>\n"
    if raw.startswith(prefix):
        if not event.truncated or not raw.endswith(suffix):
            raise ValueError("unverified truncation wrapper")
        # Inspect truncates both the message and ToolEvent. Validate only the
        # retained literal; never infer syntax or claim the full body survived.
        raw = raw[len(prefix) : -len(suffix)]
    return _literal_payload(raw)


def check_trajectory(
    messages, completion: str, invalidated: bool, events=()
) -> tuple[bool, str]:
    """Require successful replies between each model-generated canary step."""
    if invalidated:
        return False, "Search infrastructure invalidated the sample"
    replies = {
        message.tool_call_id: (index, message)
        for index, message in enumerate(messages)
        if isinstance(message, ChatMessageTool)
    }
    expected = ["web_search", "fetch_url", "python_session", "python_session", "submit"]
    step = 0
    previous_reply = -1
    search_urls = set()
    for turn, message in enumerate(messages):
        if not isinstance(message, ChatMessageAssistant):
            continue
        for call in message.tool_calls or []:
            if step == len(expected):
                break
            if call.function != expected[step]:
                continue
            if turn <= previous_reply:
                return (
                    False,
                    "Required tools must run in separate sequential model turns",
                )
            pair = replies.get(call.id)
            if pair is None or pair[0] <= turn:
                return False, f"Missing reply for {call.function}"
            reply_index, reply = pair
            if reply.error:
                return False, f"{call.function} returned a tool execution error"
            if step in {0, 1}:
                try:
                    payload = _tool_payload(reply, call, events)
                except (ValueError, SyntaxError):
                    return False, f"{call.function} returned an invalid payload"
                if step == 0:
                    if not isinstance(payload, list) or not payload:
                        return False, "Search returned no usable results"
                    if any(
                        not isinstance(item, dict) or item.get("backend") == "fixture"
                        for item in payload
                    ):
                        return False, "Search returned fixture or invalid results"
                    search_urls = {
                        item.get("url") for item in payload if item.get("url")
                    }
                    if not search_urls:
                        return False, "Search returned no URLs"
                elif (
                    not isinstance(payload, dict)
                    or payload.get("error")
                    or payload.get("status") != 200
                    or not payload.get("text")
                    or call.arguments.get("url") not in search_urls
                    or payload.get("url") != call.arguments.get("url")
                ):
                    return False, "Fetch did not retrieve a searched URL successfully"
            elif step == 2:
                if not _matches_code(call.arguments.get("code"), "probe_value = 17"):
                    return False, "First Python call did not assign the requested state"
                if reply.text.strip():
                    return (
                        False,
                        "Python assignment returned unexpected output or error",
                    )
            elif step == 3:
                if not _matches_code(
                    call.arguments.get("code"), "print(probe_value + 25)"
                ):
                    return False, "Second Python call did not consume prior state"
                if reply.text.strip() != "42":
                    return False, "Stateful Python calculation did not return 42"
            previous_reply = reply_index
            step += 1
    if step != len(expected):
        return False, f"Missing successful canary step: {expected[step]}"
    if not re.search(r"(?m)^Exact Answer:\s*42\s*$", completion):
        return False, "Submission did not contain Exact Answer: 42"
    return (
        True,
        "Successful search, fetch, separate stateful Python rounds, and submission",
    )


@scorer(metrics=[accuracy()])
def trajectory():
    async def score(state: TaskState, target: Target) -> Score:
        valid, explanation = check_trajectory(
            state.messages,
            state.output.completion,
            bool(state.metadata.get("hle_tools_invalidated")),
            events=transcript().events,
        )
        return Score(value=CORRECT if valid else INCORRECT, explanation=explanation)

    return score


@task
def hle_tools_canary():
    return Task(
        dataset=[
            Sample(
                id="hle-tools-canary",
                input=(
                    "This is an integration check, not a research question. "
                    "First call web_search with the query "
                    "'site:docs.python.org/3/library/math.html Python math documentation'. "
                    "Fetch a specific documentation page URL returned by that search "
                    "(prefer https://docs.python.org/3/library/math.html, avoid the "
                    "redirecting documentation root). Then call python_session "
                    "with exactly the code probe_value = 17 (no print). In a separate later python_session call, "
                    "execute exactly print(probe_value + 25), without assigning it again. Finally call submit with an HLE-formatted "
                    "plain-text response giving that calculated number as Exact Answer. "
                    "Invoke submit using the same tool-call mechanism as python_session, "
                    "with one string argument named submission. A plain response or JSON "
                    "list describing submit does not complete this check. "
                    "Carry out every step even if you already know the answer."
                ),
                target="42",
                metadata={"answer_type": "exact_match"},
            )
        ],
        solver=hle_tools_agent(),
        scorer=[hle_scorer(), trajectory()],
        sandbox=_sandbox("m2-enroot"),
        time_limit=1800,
        message_limit=80,
    )
