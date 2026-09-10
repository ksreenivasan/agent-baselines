"""Sample-owned cleanup, including detached native Jupyter descendants."""

import asyncio
import json
import os
import time
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from agent_baselines.evals.hle_tools_v0.m2_enroot_sandbox import M2EnrootSandbox


@pytest.fixture
def sandbox(monkeypatch, tmp_path):
    monkeypatch.setenv("M2_ENROOT_CONTAINER", "test")
    monkeypatch.setenv("ENROOT_DATA_PATH", str(tmp_path))
    value = M2EnrootSandbox()
    yield value
    value.directory.cleanup()
    M2EnrootSandbox._active.discard(value)


def test_close_escalates_and_is_bounded(sandbox):
    process = Mock(returncode=None)
    process.wait = AsyncMock(side_effect=[asyncio.TimeoutError, 0])
    sandbox._namespace_process = process
    asyncio.run(sandbox.close())
    process.terminate.assert_called_once()
    process.kill.assert_called_once()
    assert process.wait.await_count == 2
    with pytest.raises(RuntimeError, match="closed"):
        asyncio.run(sandbox._ensure_namespace())


def test_unreaped_owner_is_reported(sandbox):
    process = Mock(returncode=None)
    process.wait = AsyncMock(side_effect=asyncio.TimeoutError)
    sandbox._namespace_process = process
    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(
            M2EnrootSandbox.sample_cleanup("test", None, {"default": sandbox}, False)
        )
    assert sandbox in M2EnrootSandbox._active
    assert sandbox.root.exists()


def test_close_tolerates_already_exited_process(sandbox):
    process = Mock(returncode=None)
    process.terminate.side_effect = ProcessLookupError
    process.wait = AsyncMock(return_value=0)
    sandbox._namespace_process = process
    asyncio.run(sandbox.close())
    process.wait.assert_awaited_once()
    process.kill.assert_not_called()


def test_task_cleanup_fallback_and_opt_out(sandbox):
    sandbox.close = AsyncMock()
    asyncio.run(M2EnrootSandbox.task_cleanup("test", None, False))
    sandbox.close.assert_not_called()
    assert sandbox.root.exists()
    asyncio.run(M2EnrootSandbox.task_cleanup("test", None, True))
    sandbox.close.assert_awaited_once()
    assert not sandbox.root.exists()
    assert sandbox not in M2EnrootSandbox._active


def test_task_cleanup_attempts_other_owned_samples_after_failure(sandbox, monkeypatch):
    other = M2EnrootSandbox()
    sandbox.close = AsyncMock(side_effect=RuntimeError("owned process not reaped"))
    other.close = AsyncMock()
    try:
        with pytest.raises(ExceptionGroup, match="M2 sandbox cleanup failed"):
            asyncio.run(M2EnrootSandbox.task_cleanup("test", None, True))
        other.close.assert_awaited_once()
        assert not other.root.exists()
        assert sandbox in M2EnrootSandbox._active
    finally:
        other.directory.cleanup()
        M2EnrootSandbox._active.discard(other)


def namespace_processes(namespace):
    result = []
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            if os.readlink(path / "ns/pid") == namespace:
                result.append(int(path.name))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return result


def current_anon():
    relative = Path("/proc/self/cgroup").read_text().strip().split(":", 2)[-1]
    values = dict(
        line.split()
        for line in (Path("/sys/fs/cgroup") / relative.lstrip("/") / "memory.stat")
        .read_text()
        .splitlines()
    )
    return int(values["anon"])


@pytest.mark.skipif(
    os.environ.get("HLE_NATIVE_LIFECYCLE") != "1",
    reason="requires prepared M2 Enroot allocation",
)
def test_native_completed_samples_reap_kernels_and_preserve_active_sample(
    monkeypatch, tmp_path
):
    from inspect_ai import Task, eval_async
    from inspect_ai.dataset import Sample
    from inspect_ai.solver import solver
    from inspect_ai.util import sandbox as current_sandbox
    from astabench.tools.stateful_python import exec_python_session

    async def exercise():
        observations = []
        namespaces = {}
        first_peer_cleaned = asyncio.Event()
        original_cleanup = M2EnrootSandbox.sample_cleanup
        baseline = current_anon()

        async def cleanup(cls, task_name, config, environments, interrupted):
            owned = [environment for environment in environments.values()]
            await original_cleanup(task_name, config, environments, interrupted)
            for environment in owned:
                namespace = namespaces.get(environment)
                if namespace:
                    deadline = time.monotonic() + 5
                    while (
                        namespace_processes(namespace) and time.monotonic() < deadline
                    ):
                        await asyncio.sleep(0.05)
                    assert not namespace_processes(
                        namespace
                    ), "sample-owned detached processes survived"
                    observations.append(
                        {
                            "namespace": namespace,
                            "remaining_processes": 0,
                            "anon_bytes": current_anon(),
                        }
                    )
            first_peer_cleaned.set()

        monkeypatch.setattr(M2EnrootSandbox, "sample_cleanup", classmethod(cleanup))

        @solver
        def run_python():
            async def solve(state, generate):
                environment = current_sandbox().as_type(M2EnrootSandbox)
                probe = await environment.exec(
                    [
                        "python",
                        "-c",
                        "import os; print(os.readlink('/proc/self/ns/pid'))",
                    ],
                    timeout=30,
                )
                assert probe.success
                namespace = probe.stdout.strip()
                namespaces[environment] = namespace
                await exec_python_session(
                    "payload = bytearray(32 * 1024 * 1024)\nvalue = 41\nprint(value)"
                )
                # A detached descendant must be removed even though it changed session.
                result = await environment.exec(
                    [
                        "python",
                        "-c",
                        "import subprocess; subprocess.Popen(['sleep','300'],start_new_session=True,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)",
                    ],
                    timeout=30,
                )
                assert result.success
                assert namespace_processes(namespace)
                if str(state.sample_id) == "0":
                    await asyncio.wait_for(first_peer_cleaned.wait(), timeout=90)
                    result = await exec_python_session(
                        "assert value == 41\nprint(value + 1)"
                    )
                    assert "42" in str(
                        result
                    ), "cleanup of another sample damaged persistent state"
                state.completed = True
                return state

            return solve

        task = Task(
            dataset=[Sample(id=str(i), input="offline lifecycle") for i in range(6)],
            solver=run_python(),
            sandbox="m2-enroot",
        )
        logs = await eval_async(
            task,
            model="mockllm/model",
            max_samples=2,
            max_sandboxes=2,
            log_dir=str(tmp_path / "logs"),
            display="none",
        )
        assert logs[0].status == "success"
        assert not any(sample.error for sample in logs[0].samples or [])
        assert len(observations) == 6
        assert all(not namespace_processes(ns) for ns in namespaces.values())
        # Logs/import caches can grow modestly; six retained kernels would exceed this.
        after = current_anon()
        assert after - baseline < 96 * 1024 * 1024

        # A sample interrupted before its normal cleanup still belongs to task cleanup.
        leftover = M2EnrootSandbox()
        probe = await leftover.exec(
            [
                "python",
                "-c",
                "import os,subprocess; subprocess.Popen(['sleep','300'],start_new_session=True,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); print(os.readlink('/proc/self/ns/pid'))",
            ],
            timeout=30,
        )
        assert probe.success
        leftover_ns = probe.stdout.strip()
        assert namespace_processes(leftover_ns)
        # Inject a lost TERM, exercising the real bounded KILL fallback.
        monkeypatch.setattr(leftover._namespace_process, "terminate", lambda: None)
        monkeypatch.setattr(leftover, "_close_timeout", 0.1)
        await M2EnrootSandbox.task_cleanup("fallback", None, True)
        assert not namespace_processes(leftover_ns)
        assert not leftover.root.exists()
        report = {
            "samples": observations,
            "baseline_anon_bytes": baseline,
            "after_anon_bytes": after,
            "active_peer_survived": True,
            "task_cleanup_reaped_detached_process": True,
        }
        if os.environ.get("HLE_LIFECYCLE_REPORT"):
            with Path(os.environ["HLE_LIFECYCLE_REPORT"]).open("x") as stream:
                json.dump(report, stream, indent=2)
        print(json.dumps(report))

    asyncio.run(exercise())
