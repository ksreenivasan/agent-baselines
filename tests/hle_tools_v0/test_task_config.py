import pytest

from agent_baselines.evals.hle_tools_v0.task import hle_tools_v0


def test_task_uses_turn_and_time_bounds_without_cumulative_token_limit(monkeypatch):
    monkeypatch.setenv("HLE_SEARCH_BACKEND", "fixture")
    task = hle_tools_v0(fixture=True)
    assert task.token_limit is None
    assert task.message_limit == 80
    assert task.time_limit == 1800
    assert task.dataset[0].metadata["dataset_variant"] == "standard"


def test_tools_task_keeps_verified_as_an_explicit_option(monkeypatch):
    monkeypatch.setenv("HLE_SEARCH_BACKEND", "fixture")
    task = hle_tools_v0(fixture=True, dataset_variant="verified")
    assert task.dataset[0].metadata["dataset_variant"] == "verified"


def test_tools_task_can_select_the_m2_enroot_backend(monkeypatch):
    monkeypatch.setenv("HLE_SEARCH_BACKEND", "fixture")
    task = hle_tools_v0(fixture=True, sandbox_backend="m2-enroot")
    assert task.sandbox is not None
    assert task.sandbox.type == "m2-enroot"


def test_tools_task_rejects_unknown_sandbox_backend(monkeypatch):
    monkeypatch.setenv("HLE_SEARCH_BACKEND", "fixture")
    with pytest.raises(ValueError, match="sandbox_backend"):
        hle_tools_v0(fixture=True, sandbox_backend="unknown")
