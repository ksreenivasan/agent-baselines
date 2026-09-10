import importlib.util
import json
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

MODULE = Path(__file__).parents[2] / "solvers/hle-tools-v0/m2/run_shard.py"
SPEC = importlib.util.spec_from_file_location("hle_run_shard", MODULE)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def config():
    return {
        "model": "vllm/test",
        "reasoning_effort": "high",
        "concurrency": 3,
        "max_retries": 2,
        "timeout": 660,
        "attempt_timeout": 600,
        "dataset_path": "/data",
        "dataset_revision": "a" * 40,
        "judge_model": "openai/judge",
        "search_backend": "keenable",
        "model_base_url": "http://endpoint/v1",
        "reasoning_history": "all",
        "max_tokens": None,
        "top_p": 1.0,
    }


def test_command_preserves_uncapped_baseline_and_exact_manifest(tmp_path):
    command = runner.make_command(config(), tmp_path / "ids.json", tmp_path, tmp_path)
    assert "--max-tokens" not in command
    assert command[command.index("--top-p") + 1] == "1.0"
    assert command[command.index("--reasoning-history") + 1] == "all"
    assert f"manifest_path={tmp_path / 'ids.json'}" in command
    assert command[command.index("--log-format") + 1] == "eval"
    assert "--skip-smoke-test" not in command


def test_provider_specific_arguments_do_not_leak_to_gemini(tmp_path):
    cfg = config()
    cfg.update(model="google/gemini-test", model_base_url=None, top_p=None)
    command = runner.make_command(cfg, tmp_path / "ids.json", tmp_path, tmp_path)
    assert "--top-p" not in command
    assert "--temperature" not in command
    assert "--model-base-url" not in command
    assert "google" in command


def test_no_completed_archive_does_not_mistake_long_generation_for_lost_checkpoint():
    state = {"expected": 0, "errors": [], "checked_at": 900}
    assert runner.publication_problem(state, 0, 1000, 300) is None


def test_failed_durability_and_hung_publisher_are_visible():
    state = {
        "expected": 1,
        "errors": ["ENOSPC"],
        "checked_at": 990,
        "last_published_at": 650,
    }
    assert runner.publication_problem(state, 0, 1000, 300)
    state = {"expected": 1, "errors": [], "checked_at": 650}
    assert runner.publication_problem(state, 0, 1000, 300)
    state = {"expected": 1, "errors": [], "incomplete": 1, "checked_at": 990}
    assert runner.publication_problem(state, 0, 1000, 300)


def test_atomic_status_replacement_keeps_valid_json(tmp_path):
    import json

    path = tmp_path / "state.json"
    runner.atomic_json(path, {"complete": False})
    runner.atomic_json(path, {"complete": True})
    assert json.loads(path.read_text()) == {"complete": True}
    assert not list(tmp_path.glob("*.tmp"))


def test_directory_io_failure_is_not_hidden_as_no_completed_samples():
    state = {"expected": 0, "errors": ["directory ENOSPC"], "checked_at": 990}
    assert runner.publication_problem(state, 0, 1000, 300)


# Exercise the supervisor itself without spawning workers or contacting providers.
def write_archive(
    path, *, status="success", invalidated=False, error=None, samples=None
):
    from inspect_ai.log import (
        EvalConfig,
        EvalDataset,
        EvalLog,
        EvalSample,
        EvalSpec,
        write_eval_log,
    )
    from inspect_ai.scorer import Score

    log = EvalLog(
        status=status,
        invalidated=invalidated,
        error=error,
        eval=EvalSpec(
            created="2026-09-10T00:00:00Z",
            task="offline-supervisor-test",
            dataset=EvalDataset(samples=1),
            model="offline/test",
            config=EvalConfig(),
        ),
        samples=(
            [EvalSample(**row) for row in samples]
            if samples is not None
            else [
                EvalSample(
                    id="a",
                    epoch=1,
                    input="question",
                    target="answer",
                    scores={"hle_scorer": Score(value="C")},
                )
            ]
        ),
    )
    write_eval_log(log, str(path))


@pytest.mark.parametrize(
    "header",
    [
        {"status": "started"},
        {"status": "cancelled"},
        {"status": "error"},
        {"invalidated": True},
        {
            "error": {
                "message": "failure",
                "traceback": "failure",
                "traceback_ansi": "failure",
            }
        },
    ],
)
def test_archive_header_must_be_successful_and_valid(tmp_path, header):
    archive = tmp_path / "one.eval"
    write_archive(archive, **header)
    try:
        result = runner.validate_archive(archive, ["a"])
    except ValueError:
        return
    assert not result["valid"], header


def test_successful_archive_with_exact_scored_ids_is_accepted(tmp_path):
    archive = tmp_path / "one.eval"
    write_archive(archive)
    assert runner.validate_archive(archive, ["a"])["valid"]


@pytest.fixture
def supervisor(monkeypatch, tmp_path):
    config_path = tmp_path / "config.json"
    manifest_path = tmp_path / "ids.json"
    output = tmp_path / "unit" / "attempt-1"
    config_path.write_text(json.dumps(config()))
    manifest_path.write_text(json.dumps({"ids": ["a"]}))
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    proc, cgroup = memory_tree(tmp_path / "memory")
    monkeypatch.setattr(
        runner, "job_memory_cgroup", lambda *_args: (cgroup, "/jobs/job_12345")
    )
    monkeypatch.setenv("SLURM_TMPDIR", str(tmp_path / "scratch"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(MODULE),
            "--config",
            str(config_path),
            "--manifest",
            str(manifest_path),
            "--output",
            str(output),
            "--max-durable-lag",
            "10",
        ],
    )
    clock = SimpleNamespace(now=1000.0)
    monkeypatch.setattr(
        runner,
        "time",
        SimpleNamespace(
            time=lambda: clock.now,
            sleep=lambda seconds: setattr(clock, "now", clock.now + seconds),
        ),
    )
    monkeypatch.setattr(
        runner,
        "signal",
        SimpleNamespace(
            SIGTERM=signal.SIGTERM, SIGINT=signal.SIGINT, signal=lambda *args: None
        ),
    )
    monkeypatch.setattr(runner, "make_command", lambda *args: ["offline-evaluator"])
    state = SimpleNamespace(
        output=output,
        manifest_path=manifest_path,
        processes=[],
        evaluator_running=False,
        publisher_failed=False,
        final_timeout=False,
        final_exit=0,
        header={},
        clock=clock,
        circuit_breaker=False,
        final_calls=0,
        memory_cgroup=cgroup,
        memory_on_start=None,
    )

    class Process:
        def __init__(self, command, **kwargs):
            self.publisher = command[0] != "offline-evaluator"
            self.returncode = None if self.publisher or state.evaluator_running else 0
            if self.publisher and state.publisher_failed:
                self.returncode = 1
            self.signals = []
            self.killed = False
            state.processes.append(self)
            if not self.publisher and state.memory_on_start:
                state.memory_on_start()
            if not self.publisher and state.circuit_breaker:
                guard = Path(kwargs["env"]["HLE_TOOL_GUARD_DIR"])
                guard.mkdir()
                (guard / "fatal.json").write_text('{"fatal":true}')

        def poll(self):
            return self.returncode

        def send_signal(self, signum):
            self.signals.append(signum)
            self.returncode = -signum

        def wait(self, timeout=None):
            assert (
                self.returncode is not None
            ), "supervisor must stop a running worker before waiting"
            return self.returncode

        def kill(self):
            self.killed = True
            self.returncode = -9

    def final_publication(command, **kwargs):
        state.final_calls += 1
        assert command[-1] == "--once"
        assert all(process.poll() is not None for process in state.processes)
        if state.final_timeout:
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        write_archive(state.output / "logs" / "one.eval", **state.header)
        return SimpleNamespace(returncode=state.final_exit)

    monkeypatch.setattr(
        runner,
        "subprocess",
        SimpleNamespace(
            Popen=Process,
            run=final_publication,
            check_output=lambda *args, **kwargs: "offline-revision\n",
            STDOUT=subprocess.STDOUT,
            TimeoutExpired=subprocess.TimeoutExpired,
        ),
    )
    return state


def result(state):
    durable = json.loads((state.output / "result.json").read_text())
    local = json.loads((Path(durable["local_root"]) / "result.json").read_text())
    assert local == durable
    return durable


def test_supervisor_success_requires_durable_validated_output(supervisor):
    assert runner.main() == 0
    saved = result(supervisor)
    assert saved["complete"] and saved["validation"]["valid"]
    from agent_baselines.evals.hle_tools_v0.recovery import file_digest

    assert saved["archive_sha256"] == file_digest(Path(saved["archive"]))
    assert supervisor.final_calls == 1


def test_missing_startup_heartbeat_stops_evaluator(supervisor):
    supervisor.evaluator_running = True
    assert runner.main() == 2
    saved = result(supervisor)
    assert not saved["complete"]
    assert saved["reason"] == "checkpoint publisher never emitted a heartbeat"
    assert supervisor.clock.now > 1010
    assert supervisor.processes[1].signals == [signal.SIGTERM]


def test_failed_publisher_stops_evaluator(supervisor):
    supervisor.evaluator_running = True
    supervisor.publisher_failed = True
    assert runner.main() == 2
    saved = result(supervisor)
    assert saved["reason"] == "checkpoint publisher exited"
    assert not saved["complete"]
    assert supervisor.processes[1].signals == [signal.SIGTERM]


def test_final_publication_timeout_preserves_failed_status(supervisor):
    supervisor.final_timeout = True
    assert runner.main() == 2
    saved = result(supervisor)
    assert not saved["complete"]
    assert "TimeoutExpired" in saved["reason"]
    assert all(process.poll() is not None for process in supervisor.processes)


def test_failed_final_publication_rejects_even_existing_valid_archive(supervisor):
    supervisor.final_exit = 1
    assert runner.main() == 2
    saved = result(supervisor)
    assert saved["publisher_exit"] == 1 and not saved["complete"]


def test_supervisor_cannot_complete_from_non_success_archive(supervisor):
    supervisor.header = {"status": "cancelled"}
    assert runner.main() == 2
    assert not result(supervisor)["complete"]


def test_tool_circuit_breaker_stops_evaluator(supervisor):
    supervisor.evaluator_running = True
    supervisor.circuit_breaker = True
    assert runner.main() == 2
    saved = result(supervisor)
    assert saved["reason"] == "tool infrastructure circuit breaker"
    assert not saved["complete"]
    assert supervisor.processes[1].signals == [signal.SIGTERM]


def scored_row(sample_id="a"):
    return {
        "id": sample_id,
        "epoch": 1,
        "input": "question",
        "target": "answer",
        "scores": {"hle_scorer": {"value": "C"}},
    }


def unresolved_row(reason="provider", sample_id="b"):
    row = {"id": sample_id, "epoch": 1, "input": "question", "target": "answer"}
    errors = {
        "provider": {
            "message": "ServerError unavailable",
            "traceback": "",
            "traceback_ansi": "",
        },
        "judge": {
            "message": "EqualityJudgment invalid JSON",
            "traceback": "scorer.py",
            "traceback_ansi": "",
        },
        "unknown": {
            "message": "unexpected model error",
            "traceback": "",
            "traceback_ansi": "",
        },
    }
    row["error"] = errors[reason]
    if reason == "judge":
        row["output"] = {"completion": "unchanged saved solver answer"}
    return row


@pytest.mark.parametrize(
    "reason,disposition", [("provider", "generate"), ("judge", "judge_only")]
)
def test_scored_archive_with_known_residual_is_continuable(
    tmp_path, reason, disposition
):
    archive = tmp_path / "partial.eval"
    write_archive(archive, samples=[scored_row(), unresolved_row(reason)])
    validation = runner.validate_archive(archive, ["a", "b"])
    assert not validation["valid"]
    assert validation["continuable"]
    assert validation["residual"] == [
        {
            "id": "b",
            "disposition": disposition,
            "reason": (
                "provider_infrastructure_error"
                if reason == "provider"
                else "malformed_judge_result"
            ),
        }
    ]


@pytest.mark.parametrize("reason", ["provider", "judge"])
def test_supervisor_returns_residual_exit_without_claiming_completion(
    supervisor, reason
):
    supervisor.manifest_path.write_text(json.dumps({"ids": ["a", "b"]}))
    supervisor.header = {"samples": [scored_row(), unresolved_row(reason)]}
    assert runner.main() == 3
    saved = result(supervisor)
    assert not saved["complete"]
    assert saved["validation"]["continuable"]
    assert [row["id"] for row in saved["validation"]["residual"]] == ["b"]
    assert saved["publisher_exit"] == saved["eval_exit"] == 0


def invalid_residual_rows(case):
    rows = [scored_row(), unresolved_row()]
    if case == "missing_id":
        rows.pop()
    elif case == "unknown_error":
        rows[1] = unresolved_row("unknown")
    elif case == "tool_invalidation":
        rows[1]["metadata"] = {"hle_tools_invalidated": True}
    elif case == "score_and_error":
        rows[1]["scores"] = {"hle_scorer": {"value": "I"}}
    elif case == "all_provider_failures":
        rows[0] = unresolved_row(sample_id="a")
    elif case == "unexpected_id":
        rows[1]["id"] = "c"
    elif case == "unexpected_epoch":
        rows[1]["epoch"] = 2
    elif case == "missing_score_without_error":
        rows[1].pop("error")
    else:
        raise AssertionError(case)
    return rows


@pytest.mark.parametrize(
    "case",
    [
        "missing_id",
        "unknown_error",
        "tool_invalidation",
        "score_and_error",
        "all_provider_failures",
        "unexpected_id",
        "unexpected_epoch",
        "missing_score_without_error",
    ],
)
def test_supervisor_rejects_unsafe_or_ambiguous_residuals(supervisor, case):
    supervisor.manifest_path.write_text(json.dumps({"ids": ["a", "b"]}))
    supervisor.header = {"samples": invalid_residual_rows(case)}
    assert runner.main() == 2
    saved = result(supervisor)
    assert not saved["complete"]
    assert not saved["validation"].get("continuable", False)


@pytest.mark.parametrize(
    "codes,expected_exit,executed",
    [([0, 0], 0, 2), ([3, 0], 3, 2), ([0, 3, 0], 3, 3), ([0, 2, 0], 2, 2)],
)
def test_lane_continues_residuals_but_stops_infrastructure_failures(
    tmp_path, codes, expected_exit, executed
):
    import os

    stub_repo = tmp_path / "stub-repo"
    scripts = stub_repo / "solvers/hle-tools-v0/m2"
    scripts.mkdir(parents=True)
    (scripts / "prepare-enroot.sh").write_text("cleanup_m2_enroot() { :; }\n")
    (scripts / "selftest.py").write_text("pass\n")
    (scripts / "run_shard.py").write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "manifest = Path(sys.argv[sys.argv.index('--manifest') + 1])\n"
        "with Path(os.environ['HLE_OUTPUT']).joinpath('executed').open('a') as stream:\n"
        "    stream.write(manifest.name + '\\n')\n"
        "raise SystemExit(json.loads(manifest.read_text())['exit'])\n"
    )
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    for index, code in enumerate(codes):
        (manifests / f"{index:04}.json").write_text(json.dumps({"exit": code}))
    output = tmp_path / "output"
    env = {
        **os.environ,
        "HLE_REPO": str(stub_repo),
        "HLE_PYTHON": sys.executable,
        "HLE_CONFIG": str(tmp_path / "unused-config.json"),
        "HLE_MANIFEST_DIR": str(manifests),
        "HLE_OUTPUT": str(output),
        "SLURM_TMPDIR": str(tmp_path / "scratch"),
        "SLURM_JOB_ID": "offline-lane-test",
    }
    invocation = subprocess.run(
        ["bash", str(MODULE.with_name("run_lane.sbatch"))],
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert invocation.returncode == expected_exit, invocation.stderr
    assert (output / "executed").read_text().splitlines() == [
        f"{index:04}.json" for index in range(executed)
    ]
    assert (output / "driver-offline-lane-test.log").exists()


def test_distinct_phase_outputs_cannot_share_node_local_scratch(supervisor, tmp_path):
    local_roots = []
    for phase in ("canary", "pilot"):
        supervisor.output = tmp_path / phase / "same-lane" / "same-attempt"
        sys.argv[sys.argv.index("--output") + 1] = str(supervisor.output)
        assert runner.main() == 0
        local_roots.append(result(supervisor)["local_root"])
    assert local_roots[0] != local_roots[1]
    assert all(Path(root).exists() for root in local_roots)


def test_k2_bootstrap_is_explicit_and_other_model_commands_are_unchanged(tmp_path):
    cfg = config()
    baseline = runner.make_command(cfg, tmp_path / "ids.json", tmp_path, tmp_path)
    assert "-m" not in baseline
    cfg["model"] = "k2-vllm/IFM/K2-Horizon-375B-A23B"
    command = runner.make_command(cfg, tmp_path / "ids.json", tmp_path, tmp_path)
    module_index = command.index("-m")
    assert command[module_index - 1 : module_index + 3] == [
        sys.executable,
        "-m",
        "agent_baselines.evals.hle_tools_v0.k2_vllm_cli",
        "eval",
    ]
    # The provider prefix is the sole change to the actual Inspect arguments.
    baseline_args = baseline[baseline.index("eval") :]
    expected = [
        cfg["model"] if value == "vllm/test" else value for value in baseline_args
    ]
    assert command[command.index("eval") :] == expected


def test_judge_provider_residual_preserves_good_rows_and_is_continuable(
    tmp_path, judge_provider_error_row
):
    archive = tmp_path / "judge-provider.eval"
    write_archive(archive, samples=[scored_row(), judge_provider_error_row()])
    result = runner.validate_archive(archive, ["a", "b"])
    assert not result["valid"] and result["continuable"]
    assert result["residual"] == [
        {
            "id": "b",
            "disposition": "judge_only",
            "reason": "judge_provider_infrastructure_error",
        }
    ]


def memory_tree(root):
    proc = root / "proc"
    (proc / "self").mkdir(parents=True)
    mount = root / "cgroup"
    cgroup = mount / "jobs/job_12345"
    cgroup.mkdir(parents=True)
    (proc / "self/cgroup").write_text("0::/jobs/job_12345/step_batch/user/task_0\n")
    (proc / "self/mountinfo").write_text(f"10 1 0:1 / {mount} rw - cgroup2 cgroup rw\n")
    (cgroup / "memory.max").write_text(str(64 * 1024**3))
    (cgroup / "memory.current").write_text(str(4 * 1024**3))
    (cgroup / "memory.stat").write_text(
        f"anon {3 * 1024**3}\ninactive_file {1024**3}\n"
    )
    (cgroup / "memory.events").write_text("oom 0\noom_kill 0\nmax 0\n")
    (cgroup / "cgroup.procs").write_text("")
    return proc, cgroup


def fake_process(proc, cgroup, pid, *, job="/jobs/job_12345", rss=100, ppid=1):
    directory = proc / str(pid)
    directory.mkdir()
    # Fields following comm begin at process state (field3); starttime is22.
    fields = ["S", str(ppid)] + ["0"] * 17 + [str(pid * 10)]
    (directory / "stat").write_text(f"{pid} (worker name) " + " ".join(fields))
    (directory / "cgroup").write_text(f"0::{job}/step_batch/user/task_0\n")
    (directory / "status").write_text(
        f"Name:	worker name\nVmRSS:	{rss} kB\nRssAnon:	{rss - 1} kB\n"
        f"VmHWM:	{rss + 5} kB\nVmPeak:	{rss + 10} kB\n"
    )
    with (cgroup / "cgroup.procs").open("a") as stream:
        stream.write(f"{pid}\n")


def test_resolves_whole_job_through_nested_cgroup_and_mount_root(tmp_path):
    proc, cgroup = memory_tree(tmp_path)
    assert runner.job_memory_cgroup("12345", proc) == (cgroup, "/jobs/job_12345")
    # A delegated mount can expose /jobs at a different mountpoint.
    (proc / "self/mountinfo").write_text(
        f"10 1 0:1 /jobs {cgroup.parent} rw - cgroup2 cgroup rw\n"
    )
    assert runner.job_memory_cgroup("12345", proc)[0] == cgroup
    (proc / "self/mountinfo").write_text(
        f"10 1 0:1 /jobs/job_12345/step_batch {cgroup} rw - cgroup2 cgroup rw\n"
    )
    with pytest.raises(ValueError, match="outside"):
        runner.job_memory_cgroup("12345", proc)


@pytest.mark.parametrize("job", ["1234", "2345", "../12345", "offline", ""])
def test_memory_guard_rejects_other_or_invalid_jobs(tmp_path, job):
    proc, _ = memory_tree(tmp_path)
    with pytest.raises(ValueError):
        runner.job_memory_cgroup(job, proc)


@pytest.mark.parametrize("limit", ["max", "0", "-1", "bad"])
def test_memory_guard_requires_a_finite_positive_whole_job_limit(tmp_path, limit):
    proc, cgroup = memory_tree(tmp_path)
    (cgroup / "memory.max").write_text(limit)
    with pytest.raises(ValueError, match="finite positive"):
        runner.memory_snapshot(cgroup, "/jobs/job_12345", proc)


def test_memory_pressure_uses_whole_job_working_set_not_page_cache_alone(tmp_path):
    proc, cgroup = memory_tree(tmp_path)
    (cgroup / "memory.current").write_text(str(60 * 1024**3))
    (cgroup / "memory.stat").write_text(
        f"anon {3 * 1024**3}\ninactive_file {56 * 1024**3}\n"
    )
    snapshot = runner.memory_snapshot(cgroup, "/jobs/job_12345", proc)
    assert snapshot["threshold_bytes"] == 48 * 1024**3
    assert snapshot["working_set_bytes"] == 4 * 1024**3
    assert not snapshot["pressure"]
    (cgroup / "memory.stat").write_text(
        f"anon {48 * 1024**3}\ninactive_file {12 * 1024**3}\n"
    )
    assert runner.memory_snapshot(cgroup, "/jobs/job_12345", proc)["pressure"]


def test_process_inventory_includes_orphans_all_steps_but_not_peer_jobs(tmp_path):
    proc, cgroup = memory_tree(tmp_path)
    step = cgroup / "step_other/user/task_0"
    step.mkdir(parents=True)
    (step / "cgroup.procs").write_text("")
    fake_process(proc, cgroup, 11, ppid=1)
    fake_process(proc, step, 12, rss=200)
    fake_process(proc, cgroup, 13, job="/jobs/job_999", rss=99999)
    with (step / "cgroup.procs").open("a") as stream:
        stream.write("999999\n")  # exited before /proc inspection
    snapshot = runner.memory_snapshot(cgroup, "/jobs/job_12345", proc)
    assert [p["pid"] for p in snapshot["processes"]] == [12, 11]
    assert snapshot["pids_unavailable"] == 2
    assert snapshot["processes"][1]["ppid"] == 1
    assert snapshot["processes"][0]["start_ticks"] == 120
    assert snapshot["processes"][0]["comm"] == "worker_name"
    assert "argv" not in json.dumps(snapshot)


def test_memory_inventory_and_rotated_telemetry_are_bounded(tmp_path, monkeypatch):
    proc, cgroup = memory_tree(tmp_path)
    for pid in range(1, 5):
        fake_process(proc, cgroup, pid, rss=pid * 100)
    monkeypatch.setattr(runner, "_MEMORY_PID_LIMIT", 2)
    local = tmp_path / "local"
    local.mkdir()
    monitor = runner.MemoryMonitor("12345", local, proc)
    first = monitor.sample()
    assert [p["pid"] for p in first["processes"]] == [4, 3]
    assert first["process_records_omitted"] == 2
    size = (local / "memory.jsonl").stat().st_size
    monkeypatch.setattr(runner, "_MEMORY_LOG_BYTES", size + 10)
    for _ in range(5):
        monitor.sample()
    assert (local / "memory.previous.jsonl").stat().st_size <= size + 10
    assert (local / "memory.jsonl").stat().st_size <= size + 10
    assert len(list(local.glob("memory*.jsonl"))) == 2
    assert len(json.loads((local / "memory-latest.json").read_text())["processes"]) == 2


def test_memory_pressure_stops_owned_evaluator_and_preserves_partial_native(supervisor):
    supervisor.evaluator_running = True
    supervisor.header = {"status": "cancelled"}
    supervisor.memory_on_start = lambda: (
        supervisor.memory_cgroup / "memory.current"
    ).write_text(str(52 * 1024**3))
    assert runner.main() == 2
    saved = result(supervisor)
    assert saved["reason"].startswith("memory infrastructure guard:")
    assert not saved["complete"]
    assert saved["memory_guard"]["latest"]["pressure"]
    assert supervisor.processes[1].signals == [signal.SIGTERM]
    assert supervisor.final_calls == 1 and Path(saved["archive"]).exists()
    assert (supervisor.output / "memory-latest.json").exists()
    assert (supervisor.output / "memory.jsonl").exists()


def test_memory_read_failure_stops_without_disabling_guard(supervisor):
    supervisor.evaluator_running = True
    supervisor.memory_on_start = lambda: (
        supervisor.memory_cgroup / "memory.stat"
    ).unlink()
    assert runner.main() == 2
    saved = result(supervisor)
    assert (
        saved["reason"] == "memory infrastructure guard unavailable: FileNotFoundError"
    )
    assert saved["memory_guard"]["failure"] == "FileNotFoundError"
    assert supervisor.processes[1].signals == [signal.SIGTERM]
    assert supervisor.final_calls == 1


def test_memory_admission_failure_prevents_worker_dispatch(supervisor):
    (supervisor.memory_cgroup / "memory.max").write_text("max")
    with pytest.raises(RuntimeError, match="memory infrastructure guard unavailable"):
        runner.main()
    assert not supervisor.processes
    assert not (supervisor.output / "launch.json").exists()


def test_cgroup_mountpoint_octal_escapes_are_decoded(tmp_path):
    proc, cgroup = memory_tree(tmp_path / "with space")
    mount = str(cgroup.parents[1]).replace(" ", chr(92) + "040")
    (proc / "self/mountinfo").write_text(f"10 1 0:1 / {mount} rw - cgroup2 cgroup rw\n")
    assert runner.job_memory_cgroup("12345", proc)[0] == cgroup
