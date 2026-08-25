from inspect_ai.solver import Solver, chain, generate, solver, system_message

SYSTEM_MESSAGE = """You are solving one HLE-Verified question under the direct protocol.
Answer in one response without using tools or requesting external actions.
Return only a JSON-encoded object with exactly three keys: answer (a succinct final answer
or multiple-choice letter), confidence (your probability from 0 through 1 that the answer
is correct), and explanation (an optional concise explanation).
"""


@solver
def hle_verified_direct_agent() -> Solver:
    """One model generation with no tools or agent loop."""
    return chain([system_message(SYSTEM_MESSAGE), generate()])
