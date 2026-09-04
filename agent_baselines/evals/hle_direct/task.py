from inspect_ai import Task, task
from inspect_ai.model import Model

from agent_baselines.evals.hle_tools_v0.dataset import load_hle_dataset
from agent_baselines.evals.hle_tools_v0.scorer import DEFAULT_JUDGE_MODEL, hle_scorer
from agent_baselines.solvers.hle_direct.solver import hle_direct_agent


@task
def hle_direct(
    fixture: bool = False,
    data_path: str | None = None,
    manifest_path: str | None = None,
    limit: int | None = None,
    dataset_variant: str = "standard",
    dataset_revision: str | None = None,
    judge_model: str | Model = DEFAULT_JUDGE_MODEL,
    judge_reasoning_effort: str = "medium",
) -> Task:
    """AA-compatible, tool-free HLE evaluation; standard text-only HLE by default."""
    return Task(
        dataset=load_hle_dataset(
            data_path,
            fixture=fixture,
            manifest_path=manifest_path,
            limit=limit,
            dataset_variant=dataset_variant,
            dataset_revision=dataset_revision,
        ),
        solver=hle_direct_agent(),
        scorer=hle_scorer(
            judge_model=judge_model,
            judge_reasoning_effort=judge_reasoning_effort,
        ),
        time_limit=1800,
        message_limit=10,
    )
