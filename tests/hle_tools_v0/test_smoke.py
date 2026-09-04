import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest

SMOKE_DIR = Path(__file__).parents[2] / "solvers" / "hle-tools-v0"
sys.path.insert(0, str(SMOKE_DIR))

import smoke


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "hle_run_with_secrets", SMOKE_DIR / "run_with_secrets.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_eval_launch_recovers_endpoints_and_task_configuration(tmp_path):
    model_config = tmp_path / "model.yaml"
    model_config.write_text("extra_body:\n  provider: test\n")
    launch = smoke.parse_eval_launch(
        [
            "/venv/bin/inspect",
            "eval",
            "agent_baselines/evals/hle_tools_v0/task.py@hle_tools_v0",
            "--model",
            "vllm/checkpoint",
            "--model-base-url=http://model.example/v1",
            "--model-config",
            str(model_config),
            "-T",
            "fixture=true",
            "-T",
            "judge_model=openai/gpt-5.6-luna",
        ]
    )

    assert launch.model == "vllm/checkpoint"
    assert launch.model_base_url == "http://model.example/v1"
    assert launch.model_args == {"extra_body": {"provider": "test"}}
    assert launch.judge_model == "openai/gpt-5.6-luna"
    assert launch.uses_tools is True
    assert launch.fixture is True


def test_live_tools_eval_rejects_fixture_search_backend(monkeypatch):
    monkeypatch.delenv("HLE_SEARCH_BACKEND", raising=False)
    with pytest.raises(smoke.SmokeTestError, match="resolved to fixture"):
        asyncio.run(smoke._probe_web_tools(fixture=False))


def test_fixture_web_search_and_fetch_smoke(monkeypatch):
    monkeypatch.setenv("HLE_SEARCH_BACKEND", "fixture")
    asyncio.run(smoke._probe_web_tools(fixture=True))


def test_skip_override_warns_and_does_not_run_smoke(monkeypatch):
    runner = _load_runner()
    command = [
        "/venv/bin/inspect",
        "eval",
        "agent_baselines/evals/hle_direct/task.py@hle_direct",
        "--model",
        "mockllm/model",
    ]
    with pytest.warns(RuntimeWarning, match="explicitly skipped"):
        runner._smoke_or_stop(
            command,
            skip=True,
            smoke_runner=lambda command: pytest.fail(
                "smoke test should have been skipped"
            ),
        )


def test_failed_smoke_warns_and_stops(monkeypatch):
    runner = _load_runner()

    def fail(command):
        raise smoke.SmokeTestError("endpoint unavailable")

    command = [
        "/venv/bin/inspect",
        "eval",
        "agent_baselines/evals/hle_direct/task.py@hle_direct",
        "--model",
        "mockllm/model",
    ]
    with pytest.warns(RuntimeWarning, match="evaluation stopped"):
        with pytest.raises(SystemExit) as error:
            runner._smoke_or_stop(command, skip=False, smoke_runner=fail)
    assert error.value.code == 2
