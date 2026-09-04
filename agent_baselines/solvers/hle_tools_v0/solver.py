from astabench.tools.stateful_python import python_session
from inspect_ai.model import ChatMessageSystem
from inspect_ai.solver import Generate, Solver, TaskState, solver

from agent_baselines.evals.hle_tools_v0.prompts import system_prompt_for
from agent_baselines.solvers.hle_tools_v0.web_tools import (
    hle_web_tools,
    reset_web_state,
)
from agent_baselines.solvers.react.basic_agent import basic_agent

TOOLS_SYSTEM_MESSAGE = """You may use web_search, fetch_url, and the stateful python_session tool when useful. Search and fetch are common external tools, not provider-native browsing.

You have at most 15 model turns. On or before the final turn, call submit exactly once.
The submit tool accepts exactly one top-level string argument named submission. Put your
complete response, formatted exactly as required by the HLE response-format system prompt,
directly in that string. Do not JSON-encode the response inside the submission string. Never
attempt to access benchmark answers, credentials, or host files."""


@solver
def _initialize_sample() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        reset_web_state()
        answer_type = str(state.metadata.get("answer_type", "exact_match"))
        state.messages.insert(
            0, ChatMessageSystem(content=system_prompt_for(answer_type))
        )
        state.messages.insert(1, ChatMessageSystem(content=TOOLS_SYSTEM_MESSAGE))
        return state

    return solve


@solver
def hle_tools_agent(max_steps: int = 15) -> Solver:
    """HLE tool-use extension using the canonical HLE response format."""
    return basic_agent(
        init=[_initialize_sample()],
        tools=[*hle_web_tools(), python_session()],
        max_steps=max_steps,
        max_tool_output=20_000,
        tool_call_format="native",
        final_step_submit_only=True,
        final_step_message=(
            "You have reached the final model turn. Call submit now with your "
            "complete response in the required HLE format; no other tools are available."
        ),
    )
