"""Python client for the Bio-ARN associative memory REST API."""

from __future__ import annotations

import json
from typing import Any
from urllib import error, request


class BioARNMemoryClient:
    """Small JSON client for the built-in Bio-ARN memory server."""

    def __init__(self, base_url: str = "http://localhost:8765", *, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)

    def store(
        self,
        content: list[float],
        metadata: dict[str, Any] | None = None,
        importance: float = 1.0,
    ) -> str:
        response = self._request(
            "POST",
            "/store",
            {
                "content": content,
                "metadata": metadata or {},
                "importance": float(importance),
            },
        )
        return str(response["memory_id"])

    def query(
        self,
        probe: list[float],
        *,
        top_k: int = 5,
        threshold: float = 0.3,
    ) -> list[dict[str, Any]]:
        response = self._request(
            "POST",
            "/query",
            {
                "probe": probe,
                "top_k": int(top_k),
                "threshold": float(threshold),
            },
        )
        return list(response.get("results", []))

    def recall(self, memory_id: str) -> dict[str, Any]:
        return self._request("GET", f"/recall/{memory_id}")

    def associate(self, memory_id_a: str, memory_id_b: str, *, strength: float = 1.0) -> dict[str, Any]:
        return self._request(
            "POST",
            "/associate",
            {
                "memory_id_a": memory_id_a,
                "memory_id_b": memory_id_b,
                "strength": float(strength),
            },
        )

    def forget(self, memory_id: str) -> bool:
        response = self._request("DELETE", f"/forget/{memory_id}")
        return bool(response.get("forgotten", False))

    def consolidate(self) -> dict[str, Any]:
        return self._request("POST", "/consolidate", {})

    def stats(self) -> dict[str, Any]:
        return self._request("GET", "/stats")

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        headers = {"Accept": "application/json"}
        body: bytes | None = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            body = json.dumps(payload).encode("utf-8")

        req = request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers=headers,
            method=method.upper(),
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                charset = response.headers.get_content_charset() or "utf-8"
                raw = response.read().decode(charset)
        except error.HTTPError as exc:
            charset = exc.headers.get_content_charset() if exc.headers is not None else None
            raw = exc.read().decode(charset or "utf-8", errors="replace")
            message = raw
            try:
                parsed = json.loads(raw) if raw else {}
                if isinstance(parsed, dict):
                    message = str(parsed.get("error") or parsed.get("message") or raw)
            except json.JSONDecodeError:
                pass
            raise RuntimeError(f"Bio-ARN memory API request failed ({exc.code}): {message}") from exc

        if not raw:
            return {}
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise RuntimeError("Bio-ARN memory API returned a non-object JSON response.")
        return parsed
