"""Shared HTTP helpers with simple on-disk response caching."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

import requests


class HttpRequestError(RuntimeError):
    """Raised when an upstream provider request fails."""


class JsonHttpClient:
    def __init__(
        self,
        *,
        base_url: str,
        cache_dir: Path,
        timeout_seconds: int = 30,
        default_headers: dict[str, str] | None = None,
        auth_query_param: str | None = None,
        auth_token: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/") + "/"
        self.cache_dir = cache_dir
        self.timeout_seconds = timeout_seconds
        self.default_headers = default_headers or {}
        self.auth_query_param = auth_query_param
        self.auth_token = auth_token
        self.session = requests.Session()
        self.session.headers.update(self.default_headers)

    def get_json(
        self,
        path_or_url: str,
        *,
        params: dict[str, Any] | None = None,
        cache_namespace: str,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        url = self._resolve_url(path_or_url)
        request_params = self._with_auth(params or {})
        cache_path = self._cache_path(cache_namespace, url, request_params)
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))

        response = self.session.get(url, params=request_params, headers=headers, timeout=self.timeout_seconds)
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise HttpRequestError(f"Request failed for {response.url}: {response.status_code} {response.text[:200]}") from exc

        payload = response.json()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        return payload

    def get_paginated_results(
        self,
        path_or_url: str,
        *,
        params: dict[str, Any] | None = None,
        cache_namespace: str,
        results_key: str = "results",
        next_url_key: str = "next_url",
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        next_ref = path_or_url
        next_params = params or {}

        while next_ref:
            payload = self.get_json(next_ref, params=next_params, cache_namespace=cache_namespace)
            page_results = payload.get(results_key, [])
            if isinstance(page_results, list):
                results.extend(item for item in page_results if isinstance(item, dict))
            next_ref = payload.get(next_url_key) if isinstance(payload, dict) else None
            next_params = {}

        return results

    def _resolve_url(self, path_or_url: str) -> str:
        parsed = urlparse(path_or_url)
        if parsed.scheme and parsed.netloc:
            return path_or_url
        return urljoin(self.base_url, path_or_url.lstrip("/"))

    def _with_auth(self, params: dict[str, Any]) -> dict[str, Any]:
        enriched = dict(params)
        if self.auth_query_param and self.auth_token and self.auth_query_param not in enriched:
            enriched[self.auth_query_param] = self.auth_token
        return enriched

    def _cache_path(self, cache_namespace: str, url: str, params: dict[str, Any]) -> Path:
        normalized_url = self._normalize_url(url, params)
        digest = hashlib.sha256(normalized_url.encode("utf-8")).hexdigest()
        return self.cache_dir / cache_namespace / f"{digest}.json"

    @staticmethod
    def _normalize_url(url: str, params: dict[str, Any]) -> str:
        parsed = urlparse(url)
        query_items = parse_qsl(parsed.query, keep_blank_values=True)
        query_items.extend((key, str(value)) for key, value in sorted(params.items()) if value is not None)
        query = urlencode(query_items, doseq=True)
        return urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, query, parsed.fragment))
