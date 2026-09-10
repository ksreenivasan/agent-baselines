import asyncio
import hashlib
import json
from types import SimpleNamespace

import httpx
import pytest

from agent_baselines.evals.hle_tools_v0.recovery import disposition, iter_samples
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


_CONTENT_ACCESS_DENIED = {
    "error": "Upstream forbidden",
    "message": "The target server denied access to this URL",
}
_CONTENT_NOT_FOUND = {
    "error": "Not found",
    "message": "The requested URL could not be found",
}
_CONTENT_NOT_EXTRACTABLE = {
    "error": "Unprocessable entity",
    "message": "The page was reached but content could not be extracted",
}


@pytest.mark.parametrize(
    "status,body,expected_error",
    [
        (403, _CONTENT_ACCESS_DENIED, "content_access_denied"),
        (404, _CONTENT_NOT_FOUND, "content_not_found"),
        (422, _CONTENT_NOT_EXTRACTABLE, "content_not_extractable"),
    ],
)
def test_keenable_content_failure_is_normal_and_resets_health(
    monkeypatch, tmp_path, status, body, expected_error
):
    async def failure(query, max_results):
        response = httpx.Response(
            status, json=body, request=httpx.Request("POST", "https://example.com")
        )
        response.raise_for_status()

    state = SimpleNamespace(sample_id="sample-42", epoch=1, metadata={})
    monkeypatch.setattr(module, "sample_state", lambda: state)
    monkeypatch.setattr(module, "_keenable_search", failure)
    monkeypatch.setenv("HLE_SEARCH_BACKEND", "keenable")
    monkeypatch.setenv("HLE_TOOL_GUARD_DIR", str(tmp_path))
    module._search_failures["keenable"] = 2

    result = asyncio.run(web_search()(query="safe synthetic query"))
    assert result == {
        "backend": "keenable",
        "error": expected_error,
        "status": status,
        "message": body["message"],
    }
    assert module._search_failures["keenable"] == 0
    assert state.metadata == {}
    assert not (tmp_path / "events.jsonl").exists()
    assert not (tmp_path / "fatal.json").exists()


@pytest.mark.parametrize(
    "backend,status,body",
    [
        (
            "keenable",
            404,
            {"error": "Not found", "message": "Upstream service missing"},
        ),
        ("keenable", 404, {"error": "Not found"}),
        ("keenable", 404, {**_CONTENT_NOT_FOUND, "detail": "unrecognized"}),
        ("keenable", 422, _CONTENT_NOT_FOUND),
        ("keenable", 404, "not JSON"),
        ("keenable", 422, []),
        ("exa", 404, _CONTENT_NOT_FOUND),
        ("exa", 422, _CONTENT_NOT_EXTRACTABLE),
        ("keenable", 401, _CONTENT_NOT_FOUND),
        ("keenable", 402, _CONTENT_NOT_FOUND),
        ("keenable", 403, _CONTENT_NOT_FOUND),
        ("keenable", 401, _CONTENT_ACCESS_DENIED),
        ("keenable", 402, _CONTENT_ACCESS_DENIED),
        ("keenable", 404, _CONTENT_ACCESS_DENIED),
        ("keenable", 403, {"error": "Forbidden", "message": "Invalid API key"}),
        ("keenable", 403, {"error": "Upstream forbidden"}),
        ("keenable", 403, {**_CONTENT_ACCESS_DENIED, "detail": "unrecognized"}),
        ("keenable", 403, {**_CONTENT_ACCESS_DENIED, "message": "Access denied"}),
        ("keenable", 403, "not JSON"),
        ("keenable", 403, []),
        ("exa", 403, _CONTENT_ACCESS_DENIED),
        ("keenable", 429, _CONTENT_NOT_EXTRACTABLE),
        ("keenable", 500, _CONTENT_NOT_EXTRACTABLE),
        ("keenable", 503, _CONTENT_NOT_FOUND),
    ],
)
def test_other_http_failures_still_invalidate(
    monkeypatch, tmp_path, backend, status, body
):
    async def failure(query, max_results):
        content = body if isinstance(body, str) else json.dumps(body)
        response = httpx.Response(
            status,
            content=content,
            request=httpx.Request("POST", "https://example.com"),
        )
        response.raise_for_status()

    state = SimpleNamespace(sample_id="sample-42", epoch=1, metadata={})
    monkeypatch.setattr(module, "sample_state", lambda: state)
    monkeypatch.setattr(module, "_" + backend + "_search", failure)
    monkeypatch.setenv("HLE_SEARCH_BACKEND", backend)
    monkeypatch.setenv("HLE_TOOL_GUARD_DIR", str(tmp_path))

    result = asyncio.run(web_search()(query="safe synthetic query"))
    assert result == {"backend": backend, "error": f"search_http_{status}"}
    assert state.metadata["hle_tools_invalidated"] is True
    event = json.loads((tmp_path / "events.jsonl").read_text())
    assert event["error"] == f"search_http_{status}"
    assert event["fatal"] is (status in {401, 402, 403})


@pytest.mark.parametrize(
    "status,body,expected_error",
    [
        (403, _CONTENT_ACCESS_DENIED, "content_access_denied"),
        (404, _CONTENT_NOT_FOUND, "content_not_found"),
        (422, _CONTENT_NOT_EXTRACTABLE, "content_not_extractable"),
    ],
)
@pytest.mark.parametrize("score_value", ["C", "I"])
def test_native_scored_content_failure_remains_retainable(
    monkeypatch, tmp_path, status, body, expected_error, score_value
):
    from inspect_ai import Task, eval
    from inspect_ai.dataset import Sample
    from inspect_ai.model import ChatMessageAssistant, ModelOutput, execute_tools
    from inspect_ai.scorer import Score, scorer
    from inspect_ai.solver import solver
    from inspect_ai.tool import ToolCall

    async def failure(query, max_results):
        response = httpx.Response(
            status, json=body, request=httpx.Request("POST", "https://example.com")
        )
        response.raise_for_status()

    @solver
    def exercise_search():
        async def solve(state, generate):
            await execute_tools(
                [
                    ChatMessageAssistant(
                        content="",
                        tool_calls=[
                            ToolCall(
                                id="search-1",
                                function="web_search",
                                arguments={"query": "safe query"},
                            )
                        ],
                    )
                ],
                [web_search()],
            )
            state.output = ModelOutput.from_content("mockllm/model", "answer")
            return state

        return solve

    @scorer(metrics=[])
    def hle_scorer():
        async def score(state, target):
            return Score(value=score_value)

        return score

    monkeypatch.setattr(module, "_keenable_search", failure)
    monkeypatch.setenv("HLE_SEARCH_BACKEND", "keenable")
    monkeypatch.setenv("HLE_TOOL_GUARD_DIR", str(tmp_path / "guard"))
    eval(
        Task(
            dataset=[Sample(id="sample-42", input="question", target="answer")],
            solver=exercise_search(),
            scorer=hle_scorer(),
        ),
        model="mockllm/model",
        display="none",
        log_dir=str(tmp_path / "logs"),
    )
    row = next(iter_samples(next((tmp_path / "logs").glob("*.eval"))))
    tool_events = [event for event in row["events"] if event["event"] == "tool"]
    assert len(tool_events) == 1
    assert expected_error in str(tool_events[0]["result"])
    assert row["scores"]["hle_scorer"]["value"] == score_value
    assert disposition(row) == ("retain_score", "compatible_completed_sample")
    assert not (tmp_path / "guard" / "events.jsonl").exists()


@pytest.mark.parametrize("backend", ["keenable", "exa"])
def test_unknown_http_diagnostics_redact_key_and_do_not_change_model_error(
    monkeypatch, tmp_path, backend
):
    key = "synthetic-backend-key"
    body = {
        "error": "unknown " + key,
        "message": key + " " + "x" * 1000,
        "headers": {"X-API-Key": key},
    }
    response = httpx.Response(
        422,
        json=body,
        headers={"x-request-id": "request-42"},
        request=httpx.Request("POST", "https://example.com"),
    )

    async def failure(query, max_results):
        response.raise_for_status()

    state = SimpleNamespace(sample_id="sample-42", epoch=1, metadata={})
    monkeypatch.setattr(module, "sample_state", lambda: state)
    monkeypatch.setattr(module, "_" + backend + "_search", failure)
    monkeypatch.setenv("HLE_SEARCH_BACKEND", backend)
    monkeypatch.setenv(
        "KEENABLE_API_KEY" if backend == "keenable" else "EXA_API_KEY", key
    )
    monkeypatch.setenv("HLE_TOOL_GUARD_DIR", str(tmp_path))

    result = asyncio.run(web_search()(query="safe synthetic query"))
    assert result == {"backend": backend, "error": "search_http_422"}
    event = state.metadata["hle_tools_infrastructure_errors"][0]
    diagnostics = event["http_diagnostics"]
    assert diagnostics == {
        "status": 422,
        "body_sha256": hashlib.sha256(response.content).hexdigest(),
        "error": "unknown [REDACTED]",
        "message": ("[REDACTED] " + "x" * 1000)[:512],
        "request_id": "request-42",
    }
    assert key not in json.dumps(event)
    assert key not in (tmp_path / "events.jsonl").read_text()


@pytest.mark.parametrize(
    "body",
    [b"not JSON", b"[]", b'{"error":7,"message":null}', b"x" * 65_537],
)
@pytest.mark.parametrize("request_id", ["synthetic-backend-key", "unsafe request id"])
def test_http_diagnostics_fallback_is_bounded_without_raw_body(
    monkeypatch, body, request_id
):
    monkeypatch.setenv("KEENABLE_API_KEY", "synthetic-backend-key")
    response = httpx.Response(404, content=body, headers={"x-request-id": request_id})
    diagnostics = module._http_error_diagnostics("keenable", response)
    assert diagnostics == {
        "status": 404,
        "body_sha256": hashlib.sha256(body).hexdigest(),
    }


def test_content_denial_does_not_erase_existing_invalidation(monkeypatch, tmp_path):
    async def failure(query, max_results):
        response = httpx.Response(
            403,
            json=_CONTENT_ACCESS_DENIED,
            request=httpx.Request("POST", "https://example.com"),
        )
        response.raise_for_status()

    prior_error = {
        "backend": "keenable",
        "error": "search_http_403",
        "sample_id": "sample-42",
        "epoch": 1,
        "invalidated": True,
        "http_diagnostics": {
            "status": 403,
            "body_sha256": "4e7ce1cdd15337c3728b96a62cd2bbfddff569b43cdd0e1b43fd8734e8d68041",
            **_CONTENT_ACCESS_DENIED,
        },
    }
    metadata = {
        "hle_tools_invalidated": True,
        "hle_tools_infrastructure_errors": [prior_error],
    }
    before = json.loads(json.dumps(metadata))
    state = SimpleNamespace(sample_id="sample-42", epoch=1, metadata=metadata)
    monkeypatch.setattr(module, "sample_state", lambda: state)
    monkeypatch.setattr(module, "_keenable_search", failure)
    monkeypatch.setenv("HLE_SEARCH_BACKEND", "keenable")
    monkeypatch.setenv("HLE_TOOL_GUARD_DIR", str(tmp_path))

    assert (
        asyncio.run(web_search()(query="safe query"))["error"]
        == "content_access_denied"
    )
    assert metadata == before
    historical = {
        "id": "sample-42",
        "epoch": 1,
        "metadata": metadata,
        "scores": {"hle_scorer": {"value": "I"}},
    }
    assert disposition(historical)[0] == "generate"


@pytest.mark.parametrize(
    "payload",
    [
        {"error": "search_http_403"},
        {
            "backend": "keenable",
            "error": "search_http_403",
            "http_diagnostics": {"status": 403, **_CONTENT_ACCESS_DENIED},
        },
    ],
)
def test_historical_403_search_error_is_not_reinterpreted(payload):
    historical = {
        "id": "sample-42",
        "epoch": 1,
        "scores": {"hle_scorer": {"value": "I"}},
        "events": [
            {
                "event": "tool",
                "function": "web_search",
                "result": repr(payload),
            }
        ],
    }
    assert disposition(historical) == ("generate", "web_backend_error")


@pytest.fixture
def fake_keenable_http(monkeypatch, tmp_path):
    def configure(outcomes, *, patch_state=True, close_cancel=False):
        requests, delays = [], []
        pending = list(outcomes)
        state = SimpleNamespace(sample_id="sample-retry", epoch=1, metadata={})
        original_sleep = asyncio.sleep

        class Client:
            def __init__(self, **kwargs):
                assert kwargs == {"timeout": 30}

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                if close_cancel:
                    raise asyncio.CancelledError()
                return False

            async def post(self, url, **kwargs):
                requests.append({"url": url, **kwargs})
                assert pending, "more than the planned HTTP attempts"
                outcome = pending.pop(0)
                if isinstance(outcome, BaseException):
                    raise outcome
                if outcome == "wait":
                    await asyncio.Event().wait()
                response = httpx.Response(
                    outcome,
                    json=(
                        {"results": []}
                        if outcome == 200
                        else {"error": "synthetic-key", "message": "private-query"}
                    ),
                    headers={"x-request-id": "request-safe"},
                    request=httpx.Request("POST", url),
                )
                return response

        async def sleep(delay):
            assert delay == 2
            delays.append(delay)
            await original_sleep(0)

        monkeypatch.setenv("HLE_SEARCH_BACKEND", "keenable")
        monkeypatch.setenv("KEENABLE_API_KEY", "synthetic-key")
        monkeypatch.setenv("HLE_TOOL_GUARD_DIR", str(tmp_path / "guard"))
        monkeypatch.setattr(module.httpx, "AsyncClient", Client)
        monkeypatch.setattr(module.asyncio, "sleep", sleep)
        if patch_state:
            monkeypatch.setattr(module, "sample_state", lambda: state)
        return SimpleNamespace(
            state=state, requests=requests, delays=delays, guard=tmp_path / "guard"
        )

    return configure


@pytest.mark.parametrize("initial", [500, 429])
def test_transient_retry_success_uses_identical_request_without_invalidation(
    fake_keenable_http, initial
):
    fixture = fake_keenable_http([initial, 200])
    module._search_failures["keenable"] = 2
    assert asyncio.run(web_search()(query="private-query", max_results=7)) == []
    assert (
        fixture.requests
        == [
            {
                "url": "https://api.keenable.ai/v1/search",
                "headers": {"X-API-Key": "synthetic-key"},
                "json": {"query": "private-query", "max_results": 7, "mode": "pro"},
            }
        ]
        * 2
    )
    assert fixture.delays == [2]
    assert module._search_failures["keenable"] == 0
    assert "hle_tools_invalidated" not in fixture.state.metadata
    assert not (fixture.guard / "events.jsonl").exists()
    assert not (fixture.guard / "fatal.json").exists()
    events = fixture.state.metadata["hle_tools_search_transport_attempts"]
    assert [event["attempt"] for event in events] == [1, 2]
    assert [event["terminal"] for event in events] == [False, True]
    assert [event["http_diagnostics"]["status"] for event in events] == [initial, 200]
    assert events[0]["retry_id"] == events[1]["retry_id"]
    sidecar = (fixture.guard / "transport-attempts.jsonl").read_text()
    assert [json.loads(line) for line in sidecar.splitlines()] == events
    assert "synthetic-key" not in sidecar and "private-query" not in sidecar
    assert set(events[0]["http_diagnostics"]) == {"status", "body_sha256", "request_id"}


@pytest.mark.parametrize(
    "initial,final", [(500, 500), (500, 429), (429, 500), (429, 429)]
)
def test_retry_exhaustion_invalidates_once_and_never_makes_a_third_request(
    fake_keenable_http, initial, final
):
    fixture = fake_keenable_http([initial, final])
    result = asyncio.run(web_search()(query="safe query"))
    assert result == {"backend": "keenable", "error": f"search_http_{final}"}
    assert len(fixture.requests) == 2 and fixture.delays == [2]
    assert fixture.state.metadata["hle_tools_invalidated"] is True
    assert len(fixture.state.metadata["hle_tools_infrastructure_errors"]) == 1
    assert module._search_failures["keenable"] == 1
    assert not (fixture.guard / "fatal.json").exists()
    assert len(fixture.state.metadata["hle_tools_search_transport_attempts"]) == 2


@pytest.mark.parametrize("status", [401, 402, 403, 404, 422, 502, 503, 504])
def test_no_new_retry_for_auth_content_or_other_5xx(fake_keenable_http, status):
    fixture = fake_keenable_http([status])
    result = asyncio.run(web_search()(query="safe query"))
    assert result["error"] == f"search_http_{status}"
    assert len(fixture.requests) == 1 and not fixture.delays
    assert "hle_tools_search_transport_attempts" not in fixture.state.metadata
    assert fixture.state.metadata["hle_tools_invalidated"] is True
    assert (fixture.guard / "fatal.json").exists() is (status in {401, 402, 403})


@pytest.mark.parametrize(
    "error,expected",
    [
        (httpx.ReadTimeout("private-query synthetic-key"), "search_timeout"),
        (httpx.ConnectError("private-query synthetic-key"), "search_transport_error"),
    ],
)
def test_retry_final_transport_exception_keeps_evidence_and_existing_error(
    fake_keenable_http, error, expected
):
    fixture = fake_keenable_http([500, error])
    assert asyncio.run(web_search()(query="safe query"))["error"] == expected
    assert len(fixture.requests) == 2
    events = fixture.state.metadata["hle_tools_search_transport_attempts"]
    assert events[-1]["exception_type"] == type(error).__name__
    assert events[-1]["terminal"] is True
    assert fixture.state.metadata["hle_tools_invalidated"] is True
    sidecar = (fixture.guard / "transport-attempts.jsonl").read_text()
    assert "private-query" not in sidecar and "synthetic-key" not in sidecar


@pytest.mark.parametrize("initial", [500, 429])
@pytest.mark.parametrize("phase", ["backoff", "request"])
def test_cancellation_is_reraised_and_only_unrecovered_500_invalidates(
    monkeypatch, fake_keenable_http, initial, phase
):
    fixture = fake_keenable_http([initial, asyncio.CancelledError()])
    if phase == "backoff":

        async def cancel(delay):
            assert delay == 2
            raise asyncio.CancelledError()

        monkeypatch.setattr(module.asyncio, "sleep", cancel)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(web_search()(query="safe query"))
    assert len(fixture.requests) == (1 if phase == "backoff" else 2)
    events = fixture.state.metadata["hle_tools_search_transport_attempts"]
    assert events[-1]["exception_type"] == "CancelledError"
    assert events[-1]["phase"] == phase
    assert events[-1]["terminal"] is True
    assert bool(fixture.state.metadata.get("hle_tools_invalidated")) is (initial == 500)
    if initial == 500:
        invalidation = fixture.state.metadata["hle_tools_infrastructure_errors"][0]
        assert invalidation["error"] == "search_retry_cancelled"
        assert invalidation["http_diagnostics"] == events[0]["http_diagnostics"]
        assert invalidation["http_diagnostics"]["status"] == 500
        assert json.loads((fixture.guard / "events.jsonl").read_text()) == invalidation
    else:
        assert not (fixture.guard / "events.jsonl").exists()


def test_first_request_cancellation_is_unchanged(fake_keenable_http):
    fixture = fake_keenable_http([asyncio.CancelledError()])
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(web_search()(query="safe query"))
    assert len(fixture.requests) == 1 and fixture.state.metadata == {}
    assert not fixture.guard.exists()


def test_existing_deadline_cancels_retry_without_a_new_timeout_policy(
    fake_keenable_http,
):
    fixture = fake_keenable_http([500, "wait"])

    async def deadline():
        await asyncio.wait_for(web_search()(query="safe query"), timeout=0.02)

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(deadline())
    assert len(fixture.requests) == 2
    assert fixture.state.metadata["hle_tools_invalidated"] is True
    assert (
        fixture.state.metadata["hle_tools_search_transport_attempts"][-1][
            "exception_type"
        ]
        == "CancelledError"
    )


def test_recovered_500_preserves_prior_invalidation(fake_keenable_http):
    fixture = fake_keenable_http([500, 200])
    prior = {"error": "search_http_500"}
    fixture.state.metadata.update(
        hle_tools_invalidated=True, hle_tools_infrastructure_errors=[prior]
    )
    assert asyncio.run(web_search()(query="safe query")) == []
    assert fixture.state.metadata["hle_tools_invalidated"] is True
    assert fixture.state.metadata["hle_tools_infrastructure_errors"] == [prior]


@pytest.mark.parametrize("final,expected", [(200, "retain_score"), (500, "generate")])
def test_native_retry_evidence_and_acceptance_round_trip(
    monkeypatch, fake_keenable_http, tmp_path, final, expected
):
    from inspect_ai import Task, eval
    from inspect_ai.dataset import Sample
    from inspect_ai.model import ChatMessageAssistant, ModelOutput, execute_tools
    from inspect_ai.scorer import Score, scorer
    from inspect_ai.solver import solver
    from inspect_ai.tool import ToolCall

    # Keep Inspect's own asyncio scheduling intact; two real seconds is bounded.
    original_sleep = asyncio.sleep
    fixture = fake_keenable_http([500, final], patch_state=False)
    monkeypatch.setattr(module.asyncio, "sleep", original_sleep)

    @solver
    def exercise_retry():
        async def solve(state, generate):
            await execute_tools(
                [
                    ChatMessageAssistant(
                        content="",
                        tool_calls=[
                            ToolCall(
                                id="retry-search",
                                function="web_search",
                                arguments={"query": "safe query"},
                            )
                        ],
                    )
                ],
                [web_search()],
            )
            state.output = ModelOutput.from_content("mockllm/model", "answer")
            return state

        return solve

    @scorer(metrics=[])
    def hle_scorer():
        async def score(state, target):
            return Score(value="I")

        return score

    eval(
        Task(
            dataset=[Sample(id="native-retry", input="question", target="answer")],
            solver=exercise_retry(),
            scorer=hle_scorer(),
        ),
        model="mockllm/model",
        display="none",
        log_dir=str(tmp_path / "logs"),
    )
    row = next(iter_samples(next((tmp_path / "logs").glob("*.eval"))))
    assert disposition(row)[0] == expected
    assert row["scores"]["hle_scorer"]["value"] == "I"
    events = row["metadata"]["hle_tools_search_transport_attempts"]
    assert [e["http_diagnostics"]["status"] for e in events] == [500, final]
    assert all(e["sample_id"] == "native-retry" for e in events)
    assert events == [
        json.loads(line)
        for line in (fixture.guard / "transport-attempts.jsonl")
        .read_text()
        .splitlines()
    ]
    assert len([e for e in row["events"] if e["event"] == "tool"]) == 1


@pytest.mark.parametrize(
    "outcomes,invalidated",
    [([500, 500], True), ([500, 200], False), ([429, 429], False), ([200], False)],
)
def test_client_close_cancellation_after_unrecovered_500_is_not_lost(
    fake_keenable_http, outcomes, invalidated
):
    fixture = fake_keenable_http(outcomes, close_cancel=True)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(web_search()(query="safe query"))
    assert len(fixture.requests) == len(outcomes)
    assert bool(fixture.state.metadata.get("hle_tools_invalidated")) is invalidated
    if len(outcomes) == 2:
        events = fixture.state.metadata["hle_tools_search_transport_attempts"]
        assert events[-1]["phase"] == "client_close"
        assert events[-1]["exception_type"] == "CancelledError"
        if invalidated:
            invalidation = fixture.state.metadata["hle_tools_infrastructure_errors"][0]
            assert invalidation["http_diagnostics"] == events[-2]["http_diagnostics"]
            assert invalidation["http_diagnostics"]["status"] == 500
            assert (
                json.loads((fixture.guard / "events.jsonl").read_text()) == invalidation
            )
