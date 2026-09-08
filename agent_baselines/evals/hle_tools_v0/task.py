from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.model import Model

from agent_baselines.evals.hle_tools_v0.dataset import load_hle_dataset
from agent_baselines.evals.hle_tools_v0.scorer import DEFAULT_JUDGE_MODEL, hle_scorer
from agent_baselines.solvers.hle_tools_v0.solver import hle_tools_agent

_SANDBOX = (
    Path(__file__).parents[3] / "solvers" / "hle-tools-v0" / "sandbox" / "compose.yaml"
)


def _sandbox(backend: str):
    if backend == "docker":
        return ("docker", str(_SANDBOX))
    if backend == "m2-enroot":
        # Importing the module registers the M2-specific Inspect sandbox.
        from agent_baselines.evals.hle_tools_v0 import m2_enroot_sandbox  # noqa: F401

        return "m2-enroot"
    raise ValueError("sandbox_backend must be one of: docker, m2-enroot")


@task
def hle_tools_v0(
    fixture: bool = False,
    data_path: str | None = None,
    manifest_path: str | None = None,
    limit: int | None = None,
    dataset_variant: str = "standard",
    dataset_revision: str | None = None,
    judge_model: str | Model = DEFAULT_JUDGE_MODEL,
    judge_reasoning_effort: str = "medium",
    sandbox_backend: str = "docker",
) -> Task:
    return Task(
        dataset=load_hle_dataset(
            data_path,
            fixture=fixture,
            manifest_path=manifest_path,
            limit=limit,
            dataset_variant=dataset_variant,
            dataset_revision=dataset_revision,
        ),
        solver=hle_tools_agent(max_steps=15),
        scorer=hle_scorer(
            judge_model=judge_model,
            judge_reasoning_effort=judge_reasoning_effort,
        ),
        sandbox=_sandbox(sandbox_backend),
        time_limit=1800,
        message_limit=80,
    )
