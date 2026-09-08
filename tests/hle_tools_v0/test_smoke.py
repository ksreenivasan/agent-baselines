import asyncio
import importlib.util
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

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
            "--reasoning-history",
            "all",
            "--reasoning-effort",
            "high",
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
    assert launch.reasoning_history == "all"
    assert launch.reasoning_effort == "high"
    assert launch.judge_model == "openai/gpt-5.6-luna"
    assert launch.uses_tools is True
    assert launch.fixture is True


def test_endpoint_requires_explicit_reasoning_history():
    with pytest.raises(smoke.SmokeTestError, match="reasoning-history"):
        smoke.parse_eval_launch(
            [
                "inspect",
                "eval",
                "agent_baselines/evals/hle_direct/task.py@hle_direct",
                "--model",
                "vllm/served-id",
                "--model-base-url",
                "http://model.example/v1",
            ]
        )


def test_endpoint_catalog_requires_explicit_token_and_exact_model(monkeypatch):
    launch = smoke.parse_eval_launch(
        [
            "inspect",
            "eval",
            "agent_baselines/evals/hle_direct/task.py@hle_direct",
            "--model",
            "vllm/served-id",
            "--model-base-url",
            "http://model.example/v1",
            "--reasoning-history",
            "all",
        ]
    )
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    with pytest.raises(smoke.SmokeTestError, match="explicit VLLM_API_KEY"):
        smoke._probe_model_catalog(launch)

    monkeypatch.setenv("VLLM_API_KEY", "dummy-token")
    response = io.BytesIO(json.dumps({"data": [{"id": "served-id"}]}).encode())
    monkeypatch.setattr(smoke.urllib.request, "urlopen", lambda *args, **kwargs: response)
    assert smoke._probe_model_catalog(launch) is True


def test_endpoint_inference_requires_exact_response_model(monkeypatch):
    launch = smoke.parse_eval_launch(
        [
            "inspect",
            "eval",
            "agent_baselines/evals/hle_direct/task.py@hle_direct",
            "--model",
            "vllm/served-id",
            "--model-base-url",
            "http://model.example/v1",
            "--reasoning-history",
            "all",
        ]
    )

    class Model:
        async def generate(self, **kwargs):
            return SimpleNamespace(completion="OK", model="unexpected-alias")

    monkeypatch.setattr(smoke, "get_model", lambda *args, **kwargs: Model())
    with pytest.raises(smoke.SmokeTestError, match="expected exact ID"):
        asyncio.run(smoke._probe_model_endpoint(launch))


def test_live_tools_eval_rejects_fixture_search_backend(monkeypatch):
    monkeypatch.delenv("HLE_SEARCH_BACKEND", raising=False)
    with pytest.raises(smoke.SmokeTestError, match="resolved to fixture"):
        asyncio.run(smoke._probe_web_tools(fixture=False))


def test_keenable_secret_provider_mapping():
    runner = _load_runner()
    assert runner.KEYS["keenable"] == ("keenable.key", "KEENABLE_API_KEY")


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
