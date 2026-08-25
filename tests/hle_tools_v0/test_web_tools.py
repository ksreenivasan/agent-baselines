import asyncio

import pytest

from agent_baselines.solvers.hle_tools_v0.web_tools import fetch_url, reset_web_state, web_search


def test_fixture_search_then_fetch(monkeypatch):
    monkeypatch.setenv("HLE_SEARCH_BACKEND", "fixture")
    reset_web_state()
    results = asyncio.run(web_search()(query="synthetic blue square", max_results=3))
    assert results[0]["backend"] == "fixture"
    fetched = asyncio.run(fetch_url()(url=results[0]["url"]))
    assert fetched["status"] == 200
    assert "one blue square" in fetched["text"]


def test_fetch_requires_prior_search(monkeypatch):
    monkeypatch.setenv("HLE_SEARCH_BACKEND", "fixture")
    reset_web_state()
    with pytest.raises(ValueError, match="returned by web_search"):
        asyncio.run(fetch_url()(url="https://example.com/not-returned"))
