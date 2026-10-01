import asyncio
import hashlib
import ipaddress
import json
import math
import os
import random
import re
import socket
import time
from contextvars import ContextVar
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

import httpx
from inspect_ai.solver._task_state import sample_state
from inspect_ai.tool import Tool, tool

_allowed_urls: ContextVar[set[str] | None] = ContextVar(
    "hle_allowed_urls", default=None
)
_FIXTURE_URL = "https://example.com/hle-tools-v0-fixture"
_SEARCH_FAILURE_LIMIT = 3
_KEENABLE_MAX_ATTEMPTS = 3
_KEENABLE_MAX_ELAPSED = 60.0
_KEENABLE_BACKOFF_BASE = 2.0
_search_failures: dict[str, int] = {}
_keenable_gate_loop: asyncio.AbstractEventLoop | None = None
_keenable_gate_lock: asyncio.Lock | None = None
_keenable_next_start = 0.0


def _keenable_request_rate() -> float:
    name = "HLE_KEENABLE_REQUESTS_PER_SECOND"
    try:
        rate = float(os.environ.get(name, "4"))
    except ValueError:
        raise RuntimeError(f"{name} must be a positive finite number") from None
    if not math.isfinite(rate) or rate <= 0:
        raise RuntimeError(f"{name} must be a positive finite number")
    return rate


async def _wait_for_keenable_request() -> None:
    """Space Keenable API request starts across this process's active event loop."""
    global _keenable_gate_loop, _keenable_gate_lock, _keenable_next_start
    interval = 1.0 / _keenable_request_rate()
    loop = asyncio.get_running_loop()
    if loop is not _keenable_gate_loop:
        _keenable_gate_loop = loop
        _keenable_gate_lock = asyncio.Lock()
        _keenable_next_start = 0.0
    assert _keenable_gate_lock is not None
    async with _keenable_gate_lock:
        while (delay := _keenable_next_start - time.monotonic()) > 0:
            await asyncio.sleep(delay)
        _keenable_next_start = time.monotonic() + interval


def _keenable_retry_delay(attempt: int, response: httpx.Response | None) -> float:
    """Honor server minimum waits without synchronizing callers on that minimum."""
    retry_after = 0.0
    value = response.headers.get("retry-after", "").strip() if response else ""
    if value and len(value) <= 128:
        if re.fullmatch(r"[0-9]+", value):
            retry_after = float(value)
        else:
            try:
                date = parsedate_to_datetime(value)
                if date.tzinfo is not None:
                    retry_after = max(0.0, date.timestamp() - time.time())
            except (ValueError, TypeError, OverflowError):
                pass  # Malformed server hints fall back to local backoff.
    base = _KEENABLE_BACKOFF_BASE * 2 ** (attempt - 1)
    return max(base, retry_after) + random.uniform(0.0, base)


def _search_health(
    backend: str,
    error: str | None = None,
    *,
    fatal: bool = False,
    http_diagnostics: dict[str, str | int] | None = None,
) -> None:
    """Invalidate affected samples and expose backend failures to the supervisor."""
    _search_failures[backend] = _search_failures.get(backend, 0) + 1 if error else 0
    if error is None:
        return
    fatal = fatal or _search_failures[backend] >= _SEARCH_FAILURE_LIMIT
    state = sample_state()
    event = {
        "time": time.time(),
        "backend": backend,
        "error": error,
        "sample_id": state.sample_id if state else None,
        "epoch": state.epoch if state else None,
        "invalidated": True,
        "fatal": fatal,
        "consecutive_failures": _search_failures[backend],
    }
    if http_diagnostics is not None:
        event["http_diagnostics"] = http_diagnostics
    if state is not None:
        state.metadata["hle_tools_invalidated"] = True
        state.metadata.setdefault("hle_tools_infrastructure_errors", []).append(event)
    guard_dir = os.environ.get("HLE_TOOL_GUARD_DIR")
    if guard_dir:
        directory = Path(guard_dir)
        directory.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(event, sort_keys=True) + "\n"
        # No await in this section: concurrent tool coroutines cannot interleave writes.
        with (directory / "events.jsonl").open("a") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if fatal:
            temporary = directory / f".fatal.{os.getpid()}.tmp"
            with temporary.open("w") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(directory / "fatal.json")


def _search_backend() -> str:
    backend = os.environ.get("HLE_SEARCH_BACKEND", "")
    if backend not in {"fixture", "exa", "keenable"}:
        _search_health(backend, "unsupported_or_missing_search_backend", fatal=True)
        raise RuntimeError(
            "Set HLE_SEARCH_BACKEND explicitly to exa, keenable, or fixture"
        )
    return backend


def _search_error(
    backend: str,
    error: str,
    *,
    fatal: bool = False,
    http_diagnostics: dict[str, str | int] | None = None,
) -> dict[str, str]:
    _search_health(backend, error, fatal=fatal, http_diagnostics=http_diagnostics)
    return {"backend": backend, "error": error}


def _http_error_diagnostics(
    backend: str, response: httpx.Response
) -> dict[str, str | int]:
    details: dict[str, str | int] = {
        "status": response.status_code,
        "body_sha256": hashlib.sha256(response.content).hexdigest(),
    }
    key_name = {"keenable": "KEENABLE_API_KEY", "exa": "EXA_API_KEY"}.get(backend)
    key = os.environ.get(key_name, "") if key_name else ""
    # Keep only bounded structured fields; never retain request headers or raw bodies.
    if len(response.content) <= 65_536:
        try:
            body = response.json()
        except ValueError:
            body = None
        if isinstance(body, dict):
            for field in ("error", "message"):
                value = body.get(field)
                if isinstance(value, str):
                    if key:
                        value = value.replace(key, "[REDACTED]")
                    details[field] = value[:512]
    request_id = response.headers.get("x-request-id")
    if (
        request_id
        and not (key and key in request_id)
        and re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", request_id)
    ):
        details["request_id"] = request_id
    return details


def _search_transport_event(
    retry_id: str,
    attempt: int,
    *,
    terminal: bool,
    phase: str = "request",
    response: httpx.Response | None = None,
    exception: BaseException | None = None,
    elapsed_seconds: float | None = None,
    retry_delay_seconds: float | None = None,
    stop_reason: str | None = None,
) -> None:
    """Record bounded retry evidence separately from sample invalidation."""
    state = sample_state()
    event: dict[str, object] = {
        "time": time.time(),
        "backend": "keenable",
        "retry_id": retry_id,
        "attempt": attempt,
        "max_attempts": _KEENABLE_MAX_ATTEMPTS,
        "max_elapsed_seconds": _KEENABLE_MAX_ELAPSED,
        "phase": phase,
        "terminal": terminal,
        "sample_id": state.sample_id if state else None,
        "epoch": state.epoch if state else None,
    }
    if response is not None:
        diagnostics = _http_error_diagnostics("keenable", response)
        event["http_diagnostics"] = {
            key: diagnostics[key]
            for key in ("status", "body_sha256", "request_id")
            if key in diagnostics
        }
    if exception is not None:
        # Exception messages can contain headers or queries; retain only the type.
        event["exception_type"] = type(exception).__name__
        causes: list[str] = []
        seen = {id(exception)}
        cause = exception.__cause__ or exception.__context__
        while cause is not None and id(cause) not in seen and len(causes) < 4:
            seen.add(id(cause))
            causes.append(type(cause).__name__)
            cause = cause.__cause__ or cause.__context__
        event["exception_cause_types"] = causes
    if elapsed_seconds is not None:
        event["elapsed_seconds"] = round(elapsed_seconds, 6)
    if retry_delay_seconds is not None:
        event["retry_delay_seconds"] = round(retry_delay_seconds, 6)
    if stop_reason is not None:
        event["stop_reason"] = stop_reason
    if state is not None:
        state.metadata.setdefault("hle_tools_search_transport_attempts", []).append(
            event
        )
    guard_dir = os.environ.get("HLE_TOOL_GUARD_DIR")
    if guard_dir:
        directory = Path(guard_dir)
        directory.mkdir(parents=True, exist_ok=True)
        # No await: concurrent coroutines cannot interleave these bounded writes.
        with (directory / "transport-attempts.jsonl").open("a") as stream:
            stream.write(json.dumps(event, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())


def _keenable_content_error(response: httpx.Response) -> dict[str, str | int] | None:
    known = {
        403: (
            {
                "error": "Upstream forbidden",
                "message": "The target server denied access to this URL",
            },
            "content_access_denied",
        ),
        404: (
            {"error": "Not found", "message": "The requested URL could not be found"},
            "content_not_found",
        ),
        422: (
            {
                "error": "Unprocessable entity",
                "message": "The page was reached but content could not be extracted",
            },
            "content_not_extractable",
        ),
    }
    expected = known.get(response.status_code)
    if expected is None:
        return None
    try:
        body = response.json()
    except ValueError:
        return None
    if body != expected[0]:
        return None
    return {
        "backend": "keenable",
        "error": expected[1],
        "status": response.status_code,
        "message": body["message"],
    }


def _urls() -> set[str]:
    urls = _allowed_urls.get()
    if urls is None:
        urls = set()
        _allowed_urls.set(urls)
    return urls


def _public_http_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False
    if parsed.hostname.lower() in {"localhost", "metadata.google.internal"}:
        return False
    try:
        addresses = socket.getaddrinfo(parsed.hostname, parsed.port or 443)
    except OSError:
        # gaierror is a subclass; the resolver can also raise a bare OSError (e.g. errno 0)
        # for some hostnames. Treat any resolution failure as unresolvable, not a sample error.
        return False
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if not ip.is_global:
            return False
    return True


def _blocked_hle_url(url: str) -> bool:
    lowered = url.lower()
    return any(
        blocked in lowered
        for blocked in (
            "huggingface.co/datasets/cais/hle",
            "huggingface.co/datasets/skylenage-ai/hle-verified",
            "lastexam.ai",
        )
    )


async def _exa_search(query: str, max_results: int) -> list[dict[str, str | int]]:
    key = os.environ.get("EXA_API_KEY")
    if not key:
        raise RuntimeError("EXA_API_KEY is unavailable; live search is blocked")
    async with httpx.AsyncClient(timeout=30) as client:
        for attempt in range(2):
            response = await client.post(
                "https://api.exa.ai/search",
                headers={"x-api-key": key},
                json={"query": query, "numResults": max_results, "type": "auto"},
            )
            if response.status_code != 429 or attempt == 1:
                break
            await asyncio.sleep(2)
        response.raise_for_status()
    results = []
    for rank, item in enumerate(response.json().get("results", []), start=1):
        url = str(item.get("url", ""))
        if not _public_http_url(url) or _blocked_hle_url(url):
            continue
        results.append(
            {
                "rank": rank,
                "title": str(item.get("title", "")),
                "url": url,
                "snippet": str(item.get("text", ""))[:2000],
                "backend": "exa",
            }
        )
    return results


async def _keenable_search(query: str, max_results: int) -> list[dict[str, str | int]]:
    key = os.environ.get("KEENABLE_API_KEY")
    if not key:
        raise RuntimeError("KEENABLE_API_KEY is unavailable; live search is blocked")
    retry_id = None
    unrecovered_infrastructure_error = False
    last_server_error_diagnostics: dict[str, str | int] | None = None
    recorded_error: httpx.HTTPError | None = None
    attempt_number = 0
    search_started = attempt_started = time.monotonic()
    deadline = search_started + _KEENABLE_MAX_ELAPSED
    phase = "client_open"
    try:
        # HTTPX's 30s timeout is per I/O phase; this bounds the whole network
        # operation, including admission, all attempts, waits and client cleanup.
        async with asyncio.timeout(_KEENABLE_MAX_ELAPSED):
            async with httpx.AsyncClient(timeout=30) as client:
                for attempt_number in range(1, _KEENABLE_MAX_ATTEMPTS + 1):
                    phase = "rate_limit"
                    await _wait_for_keenable_request()
                    attempt_started = time.monotonic()
                    if attempt_started >= deadline:
                        raise TimeoutError
                    phase = "request"
                    response = None
                    request_error = None
                    try:
                        response = await client.post(
                            "https://api.keenable.ai/v1/search",
                            headers={"X-API-Key": key},
                            json={
                                "query": query,
                                "max_results": max_results,
                                "mode": "pro",
                            },
                        )
                    except httpx.HTTPError as error:
                        unrecovered_infrastructure_error = True
                        recorded_error = request_error = error
                        retryable = isinstance(
                            error,
                            (httpx.NetworkError, httpx.RemoteProtocolError),
                        )
                    else:
                        retryable = response.status_code in {429, 500, 502, 504}
                        if response.status_code in {500, 502, 504}:
                            unrecovered_infrastructure_error = True
                            diagnostics = _http_error_diagnostics("keenable", response)
                            last_server_error_diagnostics = {
                                key: diagnostics[key]
                                for key in (
                                    "status",
                                    "body_sha256",
                                    "request_id",
                                )
                                if key in diagnostics
                            }
                        elif response.is_success:
                            unrecovered_infrastructure_error = False
                    delay = None
                    stop_reason = None
                    if not retryable:
                        stop_reason = (
                            "success"
                            if response is not None and response.is_success
                            else "not_retryable"
                        )
                    elif attempt_number == _KEENABLE_MAX_ATTEMPTS:
                        stop_reason = "attempt_limit"
                    else:
                        delay = _keenable_retry_delay(attempt_number, response)
                        if delay >= deadline - time.monotonic():
                            # Do not truncate Retry-After and retry prematurely.
                            stop_reason = "elapsed_budget"
                    terminal = stop_reason is not None
                    if retryable or request_error is not None or retry_id is not None:
                        retry_id = retry_id or uuid4().hex
                        _search_transport_event(
                            retry_id,
                            attempt_number,
                            terminal=terminal,
                            response=response,
                            exception=request_error,
                            elapsed_seconds=time.monotonic() - attempt_started,
                            retry_delay_seconds=delay,
                            stop_reason=stop_reason,
                        )
                    if terminal:
                        phase = "client_close"
                        if request_error is not None:
                            raise request_error
                        assert response is not None
                        response.raise_for_status()
                        break
                    assert delay is not None
                    phase = "backoff"
                    await asyncio.sleep(delay)
                phase = "client_close"
    except TimeoutError as error:
        _search_transport_event(
            retry_id or uuid4().hex,
            attempt_number,
            terminal=True,
            phase=phase,
            exception=error,
            elapsed_seconds=time.monotonic() - search_started,
            stop_reason="elapsed_budget",
        )
        # Keep the existing public timeout result and strict sample invalidation.
        raise httpx.ReadTimeout("Keenable search elapsed budget exhausted") from error
    except httpx.HTTPError as error:
        if error is not recorded_error and not isinstance(error, httpx.HTTPStatusError):
            _search_transport_event(
                retry_id or uuid4().hex,
                attempt_number,
                terminal=True,
                phase=phase,
                exception=error,
                elapsed_seconds=time.monotonic() - attempt_started,
            )
        raise
    except asyncio.CancelledError as error:
        if retry_id is not None:
            _search_transport_event(
                retry_id,
                attempt_number,
                terminal=True,
                phase=phase,
                exception=error,
                elapsed_seconds=time.monotonic() - attempt_started,
            )
        if unrecovered_infrastructure_error:
            _search_health(
                "keenable",
                "search_retry_cancelled",
                http_diagnostics=last_server_error_diagnostics,
            )
        raise
    assert response is not None
    results = []
    for rank, item in enumerate(response.json().get("results", []), start=1):
        url = str(item.get("url", ""))
        if not _public_http_url(url) or _blocked_hle_url(url):
            continue
        results.append(
            {
                "rank": rank,
                "title": str(item.get("title", "")),
                "url": url,
                "snippet": str(item.get("snippet") or item.get("description", ""))[
                    :2000
                ],
                "backend": "keenable",
            }
        )
    return results


@tool
def web_search() -> Tool:
    async def execute(query: str, max_results: int = 5):
        """Search the public web. Returns titles, URLs, and snippets.

        Args:
            query: Specific search query, at most 512 characters.
            max_results: Number of results from 1 through 10.
        """
        if not query or len(query) > 512:
            return {"error": "invalid_query_length"}
        max_results = max(1, min(int(max_results), 10))
        backend = _search_backend()
        guard_dir = os.environ.get("HLE_TOOL_GUARD_DIR")
        if guard_dir and (Path(guard_dir) / "fatal.json").exists():
            return _search_error(backend, "search_backend_halted", fatal=True)
        if backend == "fixture":
            results = [
                {
                    "rank": 1,
                    "title": "HLE tools fixture",
                    "url": _FIXTURE_URL,
                    "snippet": "Synthetic offline search result: one blue square.",
                    "backend": "fixture",
                }
            ]
        else:
            search = _exa_search if backend == "exa" else _keenable_search
            try:
                results = await search(query, max_results)
            except httpx.TimeoutException:
                return _search_error(backend, "search_timeout")
            except httpx.HTTPStatusError as error:
                status = error.response.status_code
                if backend == "keenable":
                    content_error = _keenable_content_error(error.response)
                    if content_error is not None:
                        _search_health(backend)
                        return content_error
                return _search_error(
                    backend,
                    f"search_http_{status}",
                    fatal=status in {401, 402, 403},
                    http_diagnostics=_http_error_diagnostics(backend, error.response),
                )
            except httpx.HTTPError:
                return _search_error(backend, "search_transport_error")
            except RuntimeError:
                _search_health(backend, "search_configuration_error", fatal=True)
                raise
        _search_health(backend)
        _urls().update(str(result["url"]) for result in results)
        return results

    return execute


@tool
def fetch_url() -> Tool:
    async def execute(url: str):
        """Fetch text from a URL returned by web_search in this sample.

        Args:
            url: Exact HTTP(S) URL returned by an earlier web_search call.
        """
        if url not in _urls():
            return {"url": url, "error": "url_not_from_search"}
        if url == _FIXTURE_URL:
            return {
                "url": url,
                "status": 200,
                "text": "Synthetic offline fixture: the diagram contains one blue square.",
                "truncated": False,
            }
        if not _public_http_url(url):
            return {"url": url, "error": "non_public_url"}
        try:
            async with httpx.AsyncClient(timeout=15, follow_redirects=False) as client:
                async with client.stream(
                    "GET", url, headers={"user-agent": "hle-tools-v0/0.1"}
                ) as response:
                    if response.is_error:
                        return {
                            "url": url,
                            "error": f"fetch_http_{response.status_code}",
                        }
                    content_type = response.headers.get("content-type", "")
                    if not any(
                        kind in content_type
                        for kind in ("text/", "application/json", "application/xml")
                    ):
                        return {"url": url, "error": "unsupported_content_type"}
                    body = bytearray()
                    async for chunk in response.aiter_bytes():
                        body.extend(chunk)
                        if len(body) > 2_000_000:
                            return {"url": url, "error": "response_too_large"}
        except httpx.TimeoutException:
            return {"url": url, "error": "fetch_timeout"}
        except httpx.HTTPError:
            return {"url": url, "error": "fetch_transport_error"}
        text = bytes(body).decode(response.encoding or "utf-8", errors="replace")
        truncated = len(text) > 20_000
        return {
            "url": url,
            "status": response.status_code,
            "text": text[:20_000],
            "truncated": truncated,
        }

    return execute


def reset_web_state() -> None:
    _allowed_urls.set(set())


def hle_web_tools() -> list[Tool]:
    _search_backend()
    return [web_search(), fetch_url()]
