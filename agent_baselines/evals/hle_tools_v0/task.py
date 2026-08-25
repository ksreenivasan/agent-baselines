from pathlib import Path

from inspect_ai import Task, task

from agent_baselines.evals.hle_tools_v0.dataset import load_hle_dataset
from agent_baselines.evals.hle_tools_v0.scorer import hle_scorer
from agent_baselines.solvers.hle_tools_v0.solver import hle_tools_agent

_SANDBOX = Path(__file__).parents[3] / "solvers" / "hle-tools-v0" / "sandbox" / "compose.yaml"


@task
def hle_tools_v0(
    fixture: bool = False,
    data_path: str | None = None,
    limit: int | None = None,
    judge_model: str = "openai/o3-mini-2025-01-31",
) -> Task:
    return Task(
        dataset=load_hle_dataset(data_path, fixture=fixture, limit=limit),
        solver=hle_tools_agent(max_steps=15),
        scorer=hle_scorer(judge_model=judge_model),
        sandbox=("docker", str(_SANDBOX)),
        time_limit=1800,
        token_limit=250_000,
        message_limit=80,
    )
