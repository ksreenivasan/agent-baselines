import asyncio
import ipaddress
import json
import os
import socket
import time
from contextvars import ContextVar
from pathlib import Path
from urllib.parse import urlparse

import httpx
from inspect_ai.solver._task_state import sample_state
from inspect_ai.tool import Tool, tool

_allowed_urls: ContextVar[set[str] | None] = ContextVar(
    "hle_allowed_urls", default=None
)
_FIXTURE_URL = "https://example.com/hle-tools-v0-fixture"
_SEARCH_FAILURE_LIMIT = 3
_search_failures: dict[str, int] = {}


def _search_health(
    backend: str, error: str | None = None, *, fatal: bool = False
) -> None:
    """Invalidate affected samples and expose lane outages without logging queries."""
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


def _search_error(backend: str, error: str, *, fatal: bool = False) -> dict[str, str]:
    _search_health(backend, error, fatal=fatal)
    return {"backend": backend, "error": error}


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
    except socket.gaierror:
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
    async with httpx.AsyncClient(timeout=30) as client:
        for attempt in range(2):
            response = await client.post(
                "https://api.keenable.ai/v1/search",
                headers={"X-API-Key": key},
                json={"query": query, "max_results": max_results, "mode": "pro"},
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
                return _search_error(
                    backend, f"search_http_{status}", fatal=status in {401, 402, 403}
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
