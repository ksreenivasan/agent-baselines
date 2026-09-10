import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from agent_baselines.solvers.hle_tools_v0 import web_tools as module
from agent_baselines.solvers.hle_tools_v0.web_tools import (
    _blocked_hle_url,
    fetch_url,
    reset_web_state,
    web_search,
)


def test_fixture_search_then_fetch(monkeypatch):
    monkeypatch.setenv("HLE_SEARCH_BACKEND", "fixture")
    reset_web_state()
    results = asyncio.run(web_search()(query="synthetic blue square", max_results=3))
    assert results[0]["backend"] == "fixture"
    fetched = asyncio.run(fetch_url()(url=results[0]["url"]))
    assert fetched["status"] == 200
    assert "one blue square" in fetched["text"]


def test_hle_dataset_and_official_result_urls_are_blocked():
    assert _blocked_hle_url("https://huggingface.co/datasets/cais/hle")
    assert _blocked_hle_url("https://huggingface.co/datasets/skylenage-ai/HLE-Verified")
    assert _blocked_hle_url("https://www.lastexam.ai/results")
    assert not _blocked_hle_url("https://example.com/science")


def test_search_timeout_is_returned_to_model(monkeypatch):
    async def timeout(query, max_results):
        raise httpx.ReadTimeout("timed out")

    monkeypatch.setenv("HLE_SEARCH_BACKEND", "exa")
    monkeypatch.setattr(module, "_exa_search", timeout)
    result = asyncio.run(web_search()(query="safe synthetic query"))
    assert result == {"backend": "exa", "error": "search_timeout"}


def test_keenable_search_is_selectable(monkeypatch):
    async def search(query, max_results):
        assert query == "safe synthetic query"
        assert max_results == 3
        return [
            {
                "rank": 1,
                "title": "Synthetic result",
                "url": "https://example.com/result",
                "snippet": "Synthetic snippet",
                "backend": "keenable",
            }
        ]

    monkeypatch.setenv("HLE_SEARCH_BACKEND", "keenable")
    monkeypatch.setattr(module, "_keenable_search", search)
    reset_web_state()
    result = asyncio.run(web_search()(query="safe synthetic query", max_results=3))
    assert result[0]["backend"] == "keenable"
    assert result[0]["url"] in module._urls()


def test_fetch_timeout_is_returned_to_model(monkeypatch):
    class TimeoutStream:
        async def __aenter__(self):
            raise httpx.ReadTimeout("timed out")

        async def __aexit__(self, *args):
            return False

    class TimeoutClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def stream(self, *args, **kwargs):
            return TimeoutStream()

    reset_web_state()
    module._urls().add("https://example.com/slow")
    monkeypatch.setattr(module, "_public_http_url", lambda url: True)
    monkeypatch.setattr(module.httpx, "AsyncClient", TimeoutClient)
    result = asyncio.run(fetch_url()(url="https://example.com/slow"))
    assert result == {"url": "https://example.com/slow", "error": "fetch_timeout"}


def test_fetch_requires_prior_search(monkeypatch):
    monkeypatch.setenv("HLE_SEARCH_BACKEND", "fixture")
    reset_web_state()
    result = asyncio.run(fetch_url()(url="https://example.com/not-returned"))
    assert result == {
        "url": "https://example.com/not-returned",
        "error": "url_not_from_search",
    }


@pytest.fixture(autouse=True)
def clean_search_health(monkeypatch):
    module._search_failures.clear()
    monkeypatch.delenv("HLE_TOOL_GUARD_DIR", raising=False)


def test_backend_must_be_explicit(monkeypatch, tmp_path):
    monkeypatch.delenv("HLE_SEARCH_BACKEND", raising=False)
    monkeypatch.setenv("HLE_TOOL_GUARD_DIR", str(tmp_path))
    with pytest.raises(RuntimeError, match="Set HLE_SEARCH_BACKEND"):
        module.hle_web_tools()
    assert json.loads((tmp_path / "fatal.json").read_text())["fatal"]


@pytest.mark.parametrize("status", [401, 402, 403])
def test_auth_and_credit_failure_invalidate_sample_and_halt(
    monkeypatch, tmp_path, status
):
    async def failure(query, max_results):
        response = httpx.Response(
            status, request=httpx.Request("POST", "https://example.com")
        )
        response.raise_for_status()

    state = SimpleNamespace(sample_id="question-17", epoch=1, metadata={})
    monkeypatch.setattr(module, "sample_state", lambda: state)
    monkeypatch.setattr(module, "_keenable_search", failure)
    monkeypatch.setenv("HLE_SEARCH_BACKEND", "keenable")
    monkeypatch.setenv("HLE_TOOL_GUARD_DIR", str(tmp_path))
    result = asyncio.run(web_search()(query="do not retain this query"))
    assert result["error"] == f"search_http_{status}"
    assert state.metadata["hle_tools_invalidated"] is True
    event = json.loads((tmp_path / "fatal.json").read_text())
    assert event["sample_id"] == "question-17"
    assert event["epoch"] == 1
    assert event["fatal"] is True
    assert "do not retain" not in (tmp_path / "events.jsonl").read_text()
    # A tripped lane never calls the backend again.
    result = asyncio.run(web_search()(query="safe query"))
    assert result["error"] == "search_backend_halted"


def test_sustained_outage_trips_after_three_failures(monkeypatch, tmp_path):
    async def timeout(query, max_results):
        raise httpx.ReadTimeout("timed out")

    monkeypatch.setattr(module, "_keenable_search", timeout)
    monkeypatch.setenv("HLE_SEARCH_BACKEND", "keenable")
    monkeypatch.setenv("HLE_TOOL_GUARD_DIR", str(tmp_path))
    for _ in range(2):
        asyncio.run(web_search()(query="safe query"))
    assert not (tmp_path / "fatal.json").exists()
    asyncio.run(web_search()(query="safe query"))
    assert (
        json.loads((tmp_path / "fatal.json").read_text())["consecutive_failures"] == 3
    )


def test_empty_search_resets_outage_counter(monkeypatch, tmp_path):
    calls = 0

    async def intermittent(query, max_results):
        nonlocal calls
        calls += 1
        if calls == 3:
            return []
        raise httpx.ReadTimeout("timed out")

    monkeypatch.setattr(module, "_keenable_search", intermittent)
    monkeypatch.setenv("HLE_SEARCH_BACKEND", "keenable")
    monkeypatch.setenv("HLE_TOOL_GUARD_DIR", str(tmp_path))
    results = [asyncio.run(web_search()(query="safe query")) for _ in range(5)]
    assert results[2] == []
    assert not (tmp_path / "fatal.json").exists()


def test_isolated_fetch_404_does_not_trip_search_guard(monkeypatch, tmp_path):
    class Stream:
        async def __aenter__(self):
            return httpx.Response(404)

        async def __aexit__(self, *args):
            return False

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def stream(self, *args, **kwargs):
            return Stream()

    monkeypatch.setenv("HLE_TOOL_GUARD_DIR", str(tmp_path))
    monkeypatch.setattr(module, "_public_http_url", lambda url: True)
    monkeypatch.setattr(module.httpx, "AsyncClient", Client)
    reset_web_state()
    module._urls().add("https://example.com/missing")
    result = asyncio.run(fetch_url()(url="https://example.com/missing"))
    assert result["error"] == "fetch_http_404"
    assert not (tmp_path / "events.jsonl").exists()


def test_real_inspect_log_retains_sample_invalidation(monkeypatch, tmp_path):
    from inspect_ai import Task, eval
    from inspect_ai.dataset import Sample
    from inspect_ai.model import ModelOutput
    from inspect_ai.scorer import match
    from inspect_ai.solver import solver

    async def failure(query, max_results):
        response = httpx.Response(
            402, request=httpx.Request("POST", "https://example.com")
        )
        response.raise_for_status()

    @solver
    def exercise_search():
        async def solve(state, generate):
            await web_search()(query="safe query")
            state.output = ModelOutput.from_content("mockllm/model", "answer")
            return state

        return solve

    monkeypatch.setattr(module, "_keenable_search", failure)
    monkeypatch.setenv("HLE_SEARCH_BACKEND", "keenable")
    monkeypatch.setenv("HLE_TOOL_GUARD_DIR", str(tmp_path / "guard"))
    logs = eval(
        Task(
            dataset=[Sample(id="sample-42", input="question", target="answer")],
            solver=exercise_search(),
            scorer=match(),
        ),
        model="mockllm/model",
        display="none",
        log_dir=str(tmp_path / "logs"),
    )
    sample = logs[0].samples[0]
    assert sample.metadata["hle_tools_invalidated"] is True
    assert (
        sample.metadata["hle_tools_infrastructure_errors"][0]["sample_id"]
        == "sample-42"
    )
    assert (
        sample.scores
    )  # Acceptance must reject invalidation even when a score exists.
