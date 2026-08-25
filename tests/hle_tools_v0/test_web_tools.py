import asyncio

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
    assert _blocked_hle_url("https://www.lastexam.ai/results")
    assert not _blocked_hle_url("https://example.com/science")


def test_search_timeout_is_returned_to_model(monkeypatch):
    async def timeout(query, max_results):
        raise httpx.ReadTimeout("timed out")

    monkeypatch.setenv("HLE_SEARCH_BACKEND", "exa")
    monkeypatch.setattr(module, "_exa_search", timeout)
    result = asyncio.run(web_search()(query="safe synthetic query"))
    assert result == {"backend": "exa", "error": "search_timeout"}


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
    with pytest.raises(ValueError, match="returned by web_search"):
        asyncio.run(fetch_url()(url="https://example.com/not-returned"))
