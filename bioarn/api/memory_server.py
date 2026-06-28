"""Thread-safe REST API wrapper for Bio-ARN associative memory."""

from __future__ import annotations

import argparse
import json
import threading
from copy import deepcopy
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Sequence
from urllib.parse import unquote, urlparse

import torch

from bioarn import AssociativeMemoryEngine
from bioarn.config import AssociativeMemoryConfig


def _tensor_from_payload(values: Sequence[float] | torch.Tensor, *, field_name: str) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values.detach().clone().to(torch.float32).reshape(-1)
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise ValueError(f"Field {field_name!r} must be a JSON array of numbers.")
    try:
        return torch.tensor([float(value) for value in values], dtype=torch.float32).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Field {field_name!r} must contain only numeric values.") from exc


def _json_ready(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return float(value.item())
        return [_json_ready(item) for item in value.detach().cpu().tolist()]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class MemoryAPI:
    """Concurrency-safe API surface over an associative memory engine."""

    def __init__(self, engine: AssociativeMemoryEngine | None = None):
        self.engine = engine or AssociativeMemoryEngine(AssociativeMemoryConfig())
        self._lock = threading.RLock()

    def store(
        self,
        content: Sequence[float] | torch.Tensor,
        *,
        metadata: dict[str, Any] | None = None,
        importance: float = 1.0,
    ) -> dict[str, Any]:
        payload = metadata or {}
        if not isinstance(payload, dict):
            raise ValueError("Field 'metadata' must be a JSON object.")
        vector = _tensor_from_payload(content, field_name="content")
        with self._lock:
            memory_id = self.engine.store(vector, metadata=payload, importance=float(importance))
        return {"memory_id": memory_id, "status": "stored"}

    def query(
        self,
        probe: Sequence[float] | torch.Tensor,
        *,
        top_k: int = 5,
        threshold: float = 0.3,
    ) -> dict[str, Any]:
        vector = _tensor_from_payload(probe, field_name="probe")
        with self._lock:
            results = self.engine.query(vector, top_k=int(top_k), threshold=float(threshold))
        return {
            "results": [
                {
                    "memory_id": result.memory_id,
                    "confidence": float(result.confidence),
                    "metadata": deepcopy(result.metadata),
                    "importance": float(result.importance),
                    "age": int(result.age),
                }
                for result in results
            ]
        }

    def recall(
        self,
        memory_id: str,
        *,
        partial_cue: Sequence[float] | torch.Tensor | None = None,
    ) -> dict[str, Any]:
        cue = _tensor_from_payload(partial_cue, field_name="partial_cue") if partial_cue is not None else None
        with self._lock:
            content = self.engine.reconstruct(memory_id, partial_cue=cue)
            record = self._record_for(memory_id)
        return {
            "memory_id": memory_id,
            "content": _json_ready(content),
            "metadata": deepcopy(record.metadata),
            "importance": float(record.importance),
        }

    def associate(self, memory_id_a: str, memory_id_b: str, *, strength: float = 1.0) -> dict[str, Any]:
        with self._lock:
            self.engine.associate(memory_id_a, memory_id_b, strength=float(strength))
        return {
            "memory_id_a": memory_id_a,
            "memory_id_b": memory_id_b,
            "strength": float(strength),
            "status": "associated",
        }

    def forget(self, memory_id: str) -> dict[str, Any]:
        with self._lock:
            forgotten = self.engine.forget(memory_id)
        if not forgotten:
            raise KeyError(f"Unknown memory id: {memory_id}")
        return {"memory_id": memory_id, "forgotten": True, "status": "forgotten"}

    def consolidate(self) -> dict[str, Any]:
        with self._lock:
            changed = int(self.engine.consolidate())
        return {"status": "consolidated", "updated": changed}

    def stats(self) -> dict[str, Any]:
        with self._lock:
            stats = deepcopy(self.engine.stats)
        return {
            "capacity": int(stats["capacity"]),
            "stored": int(stats["active_memories"]),
            "active_memories": int(stats["active_memories"]),
            "free_slots": int(stats["free_slots"]),
            "locked": int(stats["locked_memories"]),
            "locked_memories": int(stats["locked_memories"]),
            "active_memory_ids": list(stats["active_memory_ids"]),
            "stores": int(stats["stores"]),
            "queries": int(stats["queries"]),
            "reconstructions": int(stats["reconstructions"]),
            "associations": int(stats["associations"]),
            "auto_consolidations": int(stats["auto_consolidations"]),
            "precision": float(stats["precision"]),
            "mean_importance": float(stats["mean_importance"]),
            "workspace_occupancy": float(stats["workspace_occupancy"]),
            "workspace": _json_ready(stats["workspace"]),
            "pool": _json_ready(stats["pool"]),
            "sdm": _json_ready(stats["sdm"]),
        }

    def health(self) -> dict[str, Any]:
        stats = self.stats()
        return {
            "status": "ok",
            "capacity": stats["capacity"],
            "stored": stats["stored"],
            "locked": stats["locked"],
        }

    def clear(self) -> dict[str, Any]:
        with self._lock:
            memory_ids = list(self.engine.stats["active_memory_ids"])
            for memory_id in memory_ids:
                self.engine.forget(memory_id)
        return {"status": "cleared", "forgotten": len(memory_ids)}

    def _record_for(self, memory_id: str):
        index = self.engine._index_from_memory_id(memory_id)
        return self.engine._records[index]


class MemoryHTTPServer(ThreadingHTTPServer):
    """Threaded HTTP server carrying a shared MemoryAPI instance."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler],
        api: MemoryAPI,
    ):
        self.api = api
        super().__init__(server_address, handler_class)


class MemoryRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler exposing the associative-memory API."""

    server: MemoryHTTPServer

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/health":
            self._send_json(HTTPStatus.OK, self.server.api.health())
            return
        if path == "/stats":
            self._send_json(HTTPStatus.OK, self.server.api.stats())
            return
        if path.startswith("/recall/"):
            memory_id = unquote(path.rsplit("/", maxsplit=1)[-1])
            self._dispatch(lambda: self.server.api.recall(memory_id))
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown route: {path}"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            payload = self._read_json_body()
        except ValueError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        if path == "/store":
            try:
                self._require_fields(payload, "content")
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            self._dispatch(
                lambda: self.server.api.store(
                    payload["content"],
                    metadata=payload.get("metadata"),
                    importance=float(payload.get("importance", 1.0)),
                )
            )
            return
        if path == "/query":
            try:
                self._require_fields(payload, "probe")
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            self._dispatch(
                lambda: self.server.api.query(
                    payload["probe"],
                    top_k=int(payload.get("top_k", 5)),
                    threshold=float(payload.get("threshold", 0.3)),
                )
            )
            return
        if path == "/associate":
            try:
                self._require_fields(payload, "memory_id_a", "memory_id_b")
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            self._dispatch(
                lambda: self.server.api.associate(
                    str(payload["memory_id_a"]),
                    str(payload["memory_id_b"]),
                    strength=float(payload.get("strength", 1.0)),
                )
            )
            return
        if path == "/consolidate":
            self._dispatch(self.server.api.consolidate)
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown route: {path}"})

    def do_DELETE(self) -> None:
        path = urlparse(self.path).path
        if path.startswith("/forget/"):
            memory_id = unquote(path.rsplit("/", maxsplit=1)[-1])
            self._dispatch(lambda: self.server.api.forget(memory_id))
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown route: {path}"})

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _dispatch(self, func) -> None:
        try:
            response = func()
        except KeyError as exc:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
        except (ValueError, TypeError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except Exception as exc:  # pragma: no cover - defensive server boundary
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
        else:
            self._send_json(HTTPStatus.OK, _json_ready(response))

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("Request body must be valid JSON.") from exc
        if payload is None:
            return {}
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object.")
        return payload

    def _require_fields(self, payload: dict[str, Any], *names: str) -> None:
        missing = [name for name in names if name not in payload]
        if missing:
            raise ValueError(f"Missing required field(s): {', '.join(missing)}")

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(_json_ready(payload)).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def create_memory_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    engine: AssociativeMemoryEngine | None = None,
    config: AssociativeMemoryConfig | None = None,
) -> MemoryHTTPServer:
    """Create a threaded HTTP server for associative-memory operations."""

    if engine is None:
        engine = AssociativeMemoryEngine(config or AssociativeMemoryConfig())
    api = MemoryAPI(engine)
    return MemoryHTTPServer((host, int(port)), MemoryRequestHandler, api)


def serve(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    engine: AssociativeMemoryEngine | None = None,
    config: AssociativeMemoryConfig | None = None,
) -> None:
    """Start serving the associative-memory API until interrupted."""

    server = create_memory_server(host=host, port=port, engine=engine, config=config)
    bound_host, bound_port = server.server_address[:2]
    print(f"Bio-ARN memory API listening on http://{bound_host}:{bound_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> None:
    defaults = AssociativeMemoryConfig()
    parser = argparse.ArgumentParser(description="Serve the Bio-ARN associative memory REST API.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind.")
    parser.add_argument("--port", default=8765, type=int, help="TCP port to bind.")
    parser.add_argument("--capacity", default=defaults.capacity, type=int, help="Memory capacity.")
    parser.add_argument("--input-dim", default=defaults.input_dim, type=int, help="Input vector dimension.")
    parser.add_argument("--concept-dim", default=defaults.concept_dim, type=int, help="Concept-space dimension.")
    parser.add_argument(
        "--top-k-retrieval",
        default=defaults.top_k_retrieval,
        type=int,
        help="Default retrieval width for the engine workspace.",
    )
    args = parser.parse_args()

    config = AssociativeMemoryConfig(
        capacity=args.capacity,
        input_dim=args.input_dim,
        concept_dim=args.concept_dim,
        top_k_retrieval=args.top_k_retrieval,
    )
    serve(host=args.host, port=args.port, config=config)


if __name__ == "__main__":
    main()
