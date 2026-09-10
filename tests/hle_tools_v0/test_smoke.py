import asyncio
import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest
from inspect_ai._util.content import ContentReasoning
from inspect_ai.model import ModelOutput, ModelUsage

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


@pytest.mark.parametrize("provider", ["vllm", "k2-vllm"])
def test_endpoint_catalog_requires_explicit_token_and_exact_model(
    monkeypatch, provider
):
    launch = smoke.parse_eval_launch(
        [
            "inspect",
            "eval",
            "agent_baselines/evals/hle_direct/task.py@hle_direct",
            "--model",
            f"{provider}/served-id",
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
    monkeypatch.setattr(
        smoke.urllib.request, "urlopen", lambda *args, **kwargs: response
    )
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
            return ModelOutput.from_content("unexpected-alias", "OK")

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


def test_k2_bootstrap_runs_all_smoke_checks(monkeypatch):
    runner = _load_runner()
    command = [
        sys.executable,
        "-m",
        "agent_baselines.evals.hle_tools_v0.k2_vllm_cli",
        "eval",
        "solvers/hle-tools-v0/m2/canary.py@hle_tools_canary",
        "--model",
        "k2-vllm/IFM/K2-Horizon-375B-A23B",
        "--model-base-url",
        "http://model.example/v1",
        "--reasoning-history",
        "all",
    ]
    calls = []

    async def probe_web_tools(**kwargs):
        calls.append("web")

    async def probe_model_endpoint(launch):
        assert launch.model == "k2-vllm/IFM/K2-Horizon-375B-A23B"
        calls.append("model")

    async def probe_judge_endpoint(launch):
        calls.append("judge")

    monkeypatch.setattr(smoke, "_probe_web_tools", probe_web_tools)
    monkeypatch.setattr(
        smoke, "_probe_sandbox_tools", lambda launch: calls.append("sandbox")
    )
    monkeypatch.setattr(
        smoke, "_probe_model_catalog", lambda launch: calls.append("catalog") or True
    )
    monkeypatch.setattr(smoke, "_probe_model_endpoint", probe_model_endpoint)
    monkeypatch.setattr(smoke, "_probe_judge_endpoint", probe_judge_endpoint)
    runner._smoke_or_stop(command, skip=False, smoke_runner=smoke.run_smoke_test)
    assert calls == ["web", "sandbox", "catalog", "model", "judge"]


def test_unknown_module_is_not_accepted_as_inspect_bootstrap():
    command = [sys.executable, "-m", "other_module", "eval"]
    assert not smoke.is_inspect_eval(command)
    assert not _load_runner()._is_inspect_eval(command)


def _endpoint_launch():
    return smoke.parse_eval_launch(
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
            "--reasoning-effort",
            "high",
            "--timeout",
            "1650",
            "--attempt-timeout",
            "1500",
        ]
    )


def test_endpoint_smoke_has_reasoning_headroom_and_reports_native_usage(
    monkeypatch, capsys
):
    launch = _endpoint_launch()
    launch.model_args["extra_body"] = {"provider_option": "unchanged"}
    calls = []
    resolutions = []
    output = ModelOutput.from_content("served-id", "OK")
    output.usage = ModelUsage(input_tokens=20, output_tokens=300, total_tokens=320)

    class Model:
        async def generate(self, **kwargs):
            calls.append(kwargs)
            return output

    def resolve(*args, **kwargs):
        resolutions.append((args, kwargs))
        return Model()

    monkeypatch.setattr(smoke, "get_model", resolve)
    asyncio.run(smoke._probe_model_endpoint(launch))

    assert len(calls) == 1
    assert calls[0]["input"] == "Endpoint health check. Reply with OK."
    assert calls[0]["config"].model_dump(exclude_none=True) == {
        "max_tokens": 4096,
        "max_retries": 0,
        "timeout": 120,
        "attempt_timeout": 120,
        "reasoning_effort": "high",
        "reasoning_history": "all",
    }
    assert resolutions == [
        (
            ("vllm/served-id",),
            {
                "base_url": "http://model.example/v1",
                "memoize": False,
                "extra_body": {"provider_option": "unchanged"},
            },
        )
    ]
    assert launch.model_args == {"extra_body": {"provider_option": "unchanged"}}
    diagnostic = json.loads(capsys.readouterr().out.split(": ", 1)[1])
    assert diagnostic["max_tokens"] == 4096
    assert diagnostic["stop_reasons"] == ["stop"]
    assert diagnostic["usage"] == {
        "input_tokens": 20,
        "output_tokens": 300,
        "total_tokens": 320,
    }


@pytest.mark.parametrize("stop_reason", ["max_tokens", "stop"])
def test_endpoint_reasoning_only_remains_failure_with_sanitized_diagnostics(
    monkeypatch, capsys, stop_reason
):
    output = ModelOutput.from_content("served-id", "")
    output.choices[0].message.content = [
        ContentReasoning(reasoning="PRIVATE REASONING", internal="reasoning_content")
    ]
    output.choices[0].stop_reason = stop_reason
    output.usage = ModelUsage(input_tokens=20, output_tokens=4096, total_tokens=4116)
    calls = []

    class Model:
        async def generate(self, **kwargs):
            calls.append(kwargs)
            return output

    monkeypatch.setattr(smoke, "get_model", lambda *args, **kwargs: Model())
    with pytest.raises(smoke.SmokeTestError, match="empty response") as error:
        asyncio.run(smoke._probe_model_endpoint(_endpoint_launch()))

    assert len(calls) == 1
    assert stop_reason in str(error.value) and "4096" in str(error.value)
    captured = capsys.readouterr().out
    assert "PRIVATE REASONING" not in captured + str(error.value)
    diagnostic = json.loads(captured.split(": ", 1)[1])
    assert diagnostic["completion_characters"] == 0
    assert diagnostic["stop_reasons"] == [stop_reason]


def test_endpoint_missing_usage_is_unavailable_not_zero(monkeypatch, capsys):
    class Model:
        async def generate(self, **kwargs):
            return ModelOutput.from_content("served-id", "OK")

    monkeypatch.setattr(smoke, "get_model", lambda *args, **kwargs: Model())
    asyncio.run(smoke._probe_model_endpoint(_endpoint_launch()))
    assert json.loads(capsys.readouterr().out.split(": ", 1)[1])["usage"] is None


@pytest.mark.parametrize("failure", [TimeoutError, RuntimeError])
def test_endpoint_smoke_failures_do_not_retry_or_skip(monkeypatch, failure):
    calls = []

    class Model:
        async def generate(self, **kwargs):
            calls.append(kwargs)
            raise failure("unavailable")

    monkeypatch.setattr(smoke, "get_model", lambda *args, **kwargs: Model())
    with pytest.raises(smoke.SmokeTestError, match=failure.__name__):
        asyncio.run(smoke._probe_model_endpoint(_endpoint_launch()))
    assert len(calls) == 1


def test_endpoint_smoke_cancellation_propagates(monkeypatch):
    class Model:
        async def generate(self, **kwargs):
            raise asyncio.CancelledError()

    monkeypatch.setattr(smoke, "get_model", lambda *args, **kwargs: Model())
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(smoke._probe_model_endpoint(_endpoint_launch()))
