from inspect_ai.model import ChatMessageSystem
from inspect_ai.solver import Generate, Solver, TaskState, chain, generate, solver

from agent_baselines.evals.hle_tools_v0.prompts import system_prompt_for


@solver
def _response_format_prompt() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        answer_type = str(state.metadata.get("answer_type", "exact_match"))
        state.messages.insert(
            0, ChatMessageSystem(content=system_prompt_for(answer_type))
        )
        return state

    return solve


@solver
def hle_direct_agent() -> Solver:
    """One canonical HLE generation with no tools or agent loop."""
    return chain([_response_format_prompt(), generate()])
