from astabench.tools.stateful_python import python_session
from inspect_ai.solver import Generate, Solver, TaskState, solver, system_message

from agent_baselines.solvers.hle_tools_v0.web_tools import hle_web_tools, reset_web_state
from agent_baselines.solvers.react.basic_agent import basic_agent

SYSTEM_MESSAGE = """You are solving one Humanity's Last Exam question under the hle-tools-v0 protocol.
You may use web_search, fetch_url, and the stateful python_session tool. Search and fetch are
common external tools, not provider-native browsing. Use tools only when useful.

You have at most 15 model turns. On or before the final turn, call submit exactly once. The
submit argument must be a JSON string with this schema:
{"answer": "succinct final answer or multiple-choice letter", "confidence": 0.0,
 "explanation": "optional concise explanation"}
Confidence is your probability from 0 through 1 that the answer is correct. Do not place any
other keys in the JSON. Never attempt to access benchmark answers, credentials, or host files.
"""


@solver
def _reset_sample_state() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        reset_web_state()
        return state

    return solve


@solver
def hle_tools_agent(max_steps: int = 15) -> Solver:
    """Thin HLE configuration of the repository's Inspect/Asta ReAct agent."""
    return basic_agent(
        init=[_reset_sample_state(), system_message(SYSTEM_MESSAGE)],
        tools=[*hle_web_tools(), python_session()],
        max_steps=max_steps,
        max_tool_output=20_000,
        tool_call_format="native",
        final_step_submit_only=True,
    )
