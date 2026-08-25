import asyncio
import ipaddress
import os
import socket
from contextvars import ContextVar
from urllib.parse import urlparse

import httpx
from inspect_ai.tool import Tool, tool

_allowed_urls: ContextVar[set[str] | None] = ContextVar("hle_allowed_urls", default=None)
_FIXTURE_URL = "https://example.com/hle-tools-v0-fixture"


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
    return "huggingface.co/datasets/cais/hle" in lowered or "lastexam.ai" in lowered


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


@tool
def web_search() -> Tool:
    async def execute(query: str, max_results: int = 5):
        """Search the public web. Returns titles, URLs, and snippets.

        Args:
            query: Specific search query, at most 512 characters.
            max_results: Number of results from 1 through 10.
        """
        if not query or len(query) > 512:
            raise ValueError("query must contain 1-512 characters")
        max_results = max(1, min(int(max_results), 10))
        backend = os.environ.get("HLE_SEARCH_BACKEND", "fixture")
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
        elif backend == "exa":
            results = await _exa_search(query, max_results)
        else:
            raise RuntimeError(f"unsupported search backend: {backend}")
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
            raise ValueError("fetch_url only accepts URLs returned by web_search")
        if url == _FIXTURE_URL:
            return {
                "url": url,
                "status": 200,
                "text": "Synthetic offline fixture: the diagram contains one blue square.",
                "truncated": False,
            }
        if not _public_http_url(url):
            raise ValueError("URL does not resolve to a public HTTP(S) address")
        async with httpx.AsyncClient(timeout=15, follow_redirects=False) as client:
            async with client.stream("GET", url, headers={"user-agent": "hle-tools-v0/0.1"}) as response:
                response.raise_for_status()
                content_type = response.headers.get("content-type", "")
                if not any(kind in content_type for kind in ("text/", "application/json", "application/xml")):
                    raise ValueError(f"unsupported content type: {content_type}")
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > 2_000_000:
                        raise ValueError("response exceeds 2 MB")
        text = bytes(body).decode(response.encoding or "utf-8", errors="replace")
        truncated = len(text) > 20_000
        return {"url": url, "status": response.status_code, "text": text[:20_000], "truncated": truncated}

    return execute


def reset_web_state() -> None:
    _allowed_urls.set(set())


def hle_web_tools() -> list[Tool]:
    return [web_search(), fetch_url()]
