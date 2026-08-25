from inspect_ai import Task, task

from agent_baselines.evals.hle_tools_v0.dataset import load_hle_dataset
from agent_baselines.evals.hle_tools_v0.scorer import hle_scorer
from agent_baselines.solvers.hle_verified_direct.solver import hle_verified_direct_agent


@task
def hle_verified_direct(
    fixture: bool = False,
    data_path: str | None = None,
    manifest_path: str | None = None,
    limit: int | None = None,
    judge_model: str = "openai/o3-mini-2025-01-31",
) -> Task:
    """Pinned HLE-Verified Gold evaluation with one tool-free model generation."""
    return Task(
        dataset=load_hle_dataset(
            data_path, fixture=fixture, manifest_path=manifest_path, limit=limit
        ),
        solver=hle_verified_direct_agent(),
        scorer=hle_scorer(judge_model=judge_model),
        time_limit=1800,
        message_limit=10,
    )
