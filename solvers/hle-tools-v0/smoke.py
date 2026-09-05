from __future__ import annotations

import asyncio
import json
import os
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from astabench.tools.stateful_python import exec_python_session
from astabench.tools.submission import get_submission_manager, submit_tool
from inspect_ai import Task, eval
from inspect_ai._cli.util import parse_cli_config
from inspect_ai._eval.loader import load_tasks
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig, ResponseSchema, get_model
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import json_schema

from agent_baselines.evals.hle_tools_v0.prompts import EQUALITY_CHECKER_PROMPT
from agent_baselines.evals.hle_tools_v0.scorer import (
    DEFAULT_JUDGE_MODEL,
    EqualityJudgment,
    parse_judgment,
)
from agent_baselines.solvers.hle_tools_v0.web_tools import (
    fetch_url,
    reset_web_state,
    web_search,
)


class SmokeTestError(RuntimeError):
    pass


@dataclass(frozen=True)
class EvalLaunch:
    task_spec: str
    task_args: dict[str, Any]
    model: str
    model_base_url: str | None
    model_args: dict[str, Any]
    reasoning_effort: str | None
    reasoning_history: str | None
    judge_model: str
    uses_tools: bool
    fixture: bool


def _option_values(command: list[str], *names: str) -> list[str]:
    values: list[str] = []
    index = 0
    while index < len(command):
        token = command[index]
        matched = False
        for name in names:
            if token == name:
                if index + 1 >= len(command):
                    raise SmokeTestError(f"{name} has no value")
                values.append(command[index + 1])
                index += 2
                matched = True
                break
            if token.startswith(f"{name}="):
                values.append(token.split("=", 1)[1])
                index += 1
                matched = True
                break
        if not matched:
            index += 1
    return values


def _single_option(command: list[str], *names: str) -> str | None:
    values = _option_values(command, *names)
    return values[-1] if values else None


def _inspect_eval_index(command: list[str]) -> int | None:
    for index, token in enumerate(command[:-1]):
        if Path(token).name == "inspect" and command[index + 1] == "eval":
            return index + 1
    return None


def is_inspect_eval(command: list[str]) -> bool:
    return _inspect_eval_index(command) is not None


def parse_eval_launch(command: list[str]) -> EvalLaunch:
    eval_index = _inspect_eval_index(command)
    if eval_index is None:
        raise SmokeTestError("command is not an inspect eval launch")

    task_spec = next(
        (
            token
            for token in command[eval_index + 1 :]
            if not token.startswith("-") and "@" in token
        ),
        None,
    )
    if task_spec is None:
        raise SmokeTestError("unable to identify a single task.py@task eval target")

    model = _single_option(command, "--model")
    if not model:
        raise SmokeTestError(
            "inspect eval must specify --model for endpoint smoke testing"
        )

    task_args = parse_cli_config(
        _option_values(command, "-T"), _single_option(command, "--task-config")
    )
    model_args = parse_cli_config(
        _option_values(command, "-M"), _single_option(command, "--model-config")
    )
    judge_model = str(task_args.get("judge_model", DEFAULT_JUDGE_MODEL))
    task_name = task_spec.rsplit("@", 1)[-1]
    model_base_url = _single_option(command, "--model-base-url")
    reasoning_history = _single_option(command, "--reasoning-history")
    if model_base_url and not reasoning_history:
        raise SmokeTestError(
            "endpoint evaluations must specify --reasoning-history explicitly"
        )
    return EvalLaunch(
        task_spec=task_spec,
        task_args=task_args,
        model=model,
        model_base_url=model_base_url,
        model_args=model_args,
        reasoning_effort=_single_option(command, "--reasoning-effort"),
        reasoning_history=reasoning_history,
        judge_model=judge_model,
        uses_tools="hle_tools" in task_name,
        fixture=bool(task_args.get("fixture", False)),
    )


def _probe_model_catalog(launch: EvalLaunch) -> bool:
    provider, _, model_id = launch.model.partition("/")
    if not model_id:
        raise SmokeTestError("Inspect model must include an explicit provider/model ID")
    if launch.model_base_url:
        if provider == "vllm":
            key = os.environ.get("VLLM_API_KEY", "").strip()
            variable = "VLLM_API_KEY"
        elif provider == "openai":
            key = os.environ.get("OPENAI_API_KEY", "").strip()
            variable = "OPENAI_API_KEY"
        else:
            raise SmokeTestError(
                f"model catalog canary does not support {provider!r} with --model-base-url"
            )
        if not key:
            raise SmokeTestError(
                f"endpoint evaluation requires explicit {variable} (real key or dummy token)"
            )
        url = f"{launch.model_base_url.rstrip('/')}/models"
        headers = {"Authorization": f"Bearer {key}"}
    elif provider == "google":
        key = os.environ.get("GOOGLE_API_KEY", "").strip()
        if not key:
            raise SmokeTestError("Google model catalog requires GOOGLE_API_KEY")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}"
        headers = {"x-goog-api-key": key}
    else:
        return False

    try:
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.load(response)
    except (urllib.error.URLError, TimeoutError, ValueError) as error:
        raise SmokeTestError(
            f"model catalog canary failed for {launch.model}: {type(error).__name__}"
        ) from error

    if launch.model_base_url:
        ids = {item.get("id") for item in data.get("data", [])}
        available = model_id in ids
    else:
        available = str(data.get("name", "")).removeprefix("models/") == model_id
    if not available:
        raise SmokeTestError(
            f"exact model {model_id!r} is absent from the provider model catalog"
        )
    return True


async def _probe_model_endpoint(launch: EvalLaunch) -> None:
    try:
        output = await get_model(
            launch.model,
            base_url=launch.model_base_url,
            memoize=False,
            **launch.model_args,
        ).generate(
            input="Endpoint health check. Reply with OK.",
            config=GenerateConfig(
                max_tokens=256,
                max_retries=0,
                timeout=120,
                attempt_timeout=120,
                reasoning_effort=launch.reasoning_effort,
                reasoning_history=launch.reasoning_history,
            ),
        )
    except Exception as error:
        raise SmokeTestError(
            f"evaluated-model endpoint failed for {launch.model}: {type(error).__name__}"
        ) from error
    if not output.completion.strip():
        raise SmokeTestError(
            f"evaluated-model endpoint returned an empty response for {launch.model}"
        )


async def _probe_judge_endpoint(launch: EvalLaunch) -> None:
    prompt = EQUALITY_CHECKER_PROMPT.format(
        question="What is 1 + 1?",
        response="Explanation: arithmetic\nExact Answer: 2\nConfidence: 100%",
        correct_answer="2",
    )
    try:
        output = await get_model(
            launch.judge_model,
            base_url=os.environ.get("HLE_JUDGE_BASE_URL"),
            memoize=False,
        ).generate(
            input=prompt,
            config=GenerateConfig(
                max_tokens=4096,
                max_retries=0,
                timeout=120,
                attempt_timeout=120,
                reasoning_effort="medium",
                response_schema=ResponseSchema(
                    name="hle_equality_judgment",
                    json_schema=json_schema(EqualityJudgment),
                    strict=True,
                ),
            ),
        )
        judgment = parse_judgment(output.completion)
    except Exception as error:
        raise SmokeTestError(
            f"judge endpoint failed for {launch.judge_model}: {type(error).__name__}"
        ) from error
    if judgment.correct != "yes" or judgment.extracted_final_answer.strip() != "2":
        raise SmokeTestError(
            f"judge endpoint returned an invalid canary judgment for {launch.judge_model}"
        )


async def _probe_web_tools(*, fixture: bool) -> None:
    backend = os.environ.get("HLE_SEARCH_BACKEND", "fixture")
    if backend == "fixture" and not fixture:
        raise SmokeTestError(
            "HLE_SEARCH_BACKEND resolved to fixture for a live tools evaluation"
        )
    reset_web_state()
    results = await web_search()(
        query="site:nist.gov Planck constant exact value",
        max_results=5,
    )
    if not isinstance(results, list) or not results:
        raise SmokeTestError(f"web_search failed for backend {backend}: {results!r}")
    if any(item.get("backend") != backend for item in results):
        raise SmokeTestError(f"web_search did not use the configured {backend} backend")

    failures: list[str] = []
    for item in results:
        url = str(item.get("url", ""))
        fetched = await fetch_url()(url=url)
        if (
            isinstance(fetched, dict)
            and fetched.get("status") == 200
            and str(fetched.get("text", "")).strip()
        ):
            return
        failures.append(str(fetched.get("error", fetched.get("status", "unknown"))))
    raise SmokeTestError(
        f"fetch_url could not fetch any of {len(results)} search results: {failures}"
    )


@solver
def _tool_smoke_solver() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        first = await exec_python_session("hle_smoke_value = 6 * 7; hle_smoke_value")
        second = await exec_python_session("hle_smoke_value + 1")
        if "42" not in str(first) or "43" not in str(second):
            raise SmokeTestError("stateful python_session returned unexpected results")

        manager = get_submission_manager()
        manager.clear_submission()
        result = await submit_tool()(submission="hle-smoke-ok")
        if not manager.has_submission() or manager.get_submission() != "hle-smoke-ok":
            raise SmokeTestError("submit tool did not preserve the submitted response")
        if "successful" not in str(result).lower():
            raise SmokeTestError("submit tool did not report success")
        state.output.completion = "hle-smoke-ok"
        return state

    return solve


def _probe_sandbox_tools(launch: EvalLaunch) -> None:
    try:
        tasks = load_tasks([launch.task_spec], launch.task_args)
        if len(tasks) != 1:
            raise SmokeTestError("smoke test requires exactly one loaded task")
        sandbox = tasks[0].sandbox
        if sandbox is None:
            raise SmokeTestError("tools evaluation has no sandbox configured")
        with tempfile.TemporaryDirectory(prefix="hle-smoke-") as log_dir:
            logs = eval(
                Task(
                    dataset=[Sample(id="hle-tool-smoke", input="offline tool smoke")],
                    solver=_tool_smoke_solver(),
                    sandbox=sandbox,
                    time_limit=120,
                ),
                model="mockllm/model",
                display="none",
                log_dir=log_dir,
            )
    except SmokeTestError:
        raise
    except Exception as error:
        raise SmokeTestError(
            f"python_session/submit smoke failed: {type(error).__name__}"
        ) from error
    if len(logs) != 1 or logs[0].status != "success":
        detail = str(logs[0].error) if logs and logs[0].error else "unknown error"
        raise SmokeTestError(f"python_session/submit smoke failed: {detail}")


def run_smoke_test(command: list[str]) -> list[str]:
    launch = parse_eval_launch(command)
    checks: list[str] = []
    if launch.uses_tools:
        asyncio.run(_probe_web_tools(fixture=launch.fixture))
        _probe_sandbox_tools(launch)
        checks.extend(["web_search", "fetch_url", "python_session", "submit"])
    if _probe_model_catalog(launch):
        checks.append("model_catalog")
    asyncio.run(_probe_model_endpoint(launch))
    asyncio.run(_probe_judge_endpoint(launch))
    checks.extend(["model_endpoint", "judge_endpoint"])
    return checks
