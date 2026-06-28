"""LangChain-style memory wrapper backed by Bio-ARN associative memory."""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

import torch

from bioarn import AssociativeMemoryEngine
from bioarn.config import AssociativeMemoryConfig

from .client import BioARNMemoryClient


class BioARNMemory:
    """LangChain-compatible memory backend using Bio-ARN's associative engine."""

    def __init__(
        self,
        engine: AssociativeMemoryEngine | None = None,
        base_url: str | None = None,
        *,
        memory_key: str = "history",
        input_key: str | None = None,
        output_key: str | None = None,
        top_k: int = 5,
        threshold: float = 0.3,
        embedding_dim: int | None = None,
    ):
        if engine is None and base_url is None:
            engine = AssociativeMemoryEngine(AssociativeMemoryConfig())

        self.engine = engine
        self.client = None if engine is not None else BioARNMemoryClient(base_url or "http://localhost:8765")
        self.memory_key = memory_key
        self.input_key = input_key
        self.output_key = output_key
        self.top_k = int(max(1, top_k))
        self.threshold = float(min(max(threshold, 0.0), 1.0))
        if embedding_dim is not None:
            self.embedding_dim = int(max(1, embedding_dim))
        elif self.engine is not None:
            self.embedding_dim = int(self.engine.config.input_dim)
        else:
            self.embedding_dim = 768

    @property
    def memory_variables(self) -> list[str]:
        return [self.memory_key]

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, Any]) -> None:
        """Store one conversational turn as an associative memory."""

        input_text = self._extract_text(inputs, preferred_key=self.input_key)
        output_text = self._extract_text(outputs, preferred_key=self.output_key)
        turn_text = "\n".join(
            part
            for part in (
                f"Human: {input_text}" if input_text else "",
                f"Assistant: {output_text}" if output_text else "",
            )
            if part
        )
        vector = self._embed_text(turn_text or json.dumps({"inputs": inputs, "outputs": outputs}, sort_keys=True))
        metadata = {
            "kind": "conversation_turn",
            "inputs": inputs,
            "outputs": outputs,
            "text": turn_text,
        }
        importance = self._importance_for_text(turn_text)
        self._store(vector, metadata=metadata, importance=importance)

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, str]:
        """Retrieve relevant conversational context for the current prompt."""

        query_text = self._extract_text(inputs, preferred_key=self.input_key)
        if not query_text:
            return {self.memory_key: ""}

        vector = self._embed_text(query_text)
        results = self._query(vector)
        snippets: list[str] = []
        for result in results:
            metadata = result.get("metadata", {})
            text = metadata.get("text") if isinstance(metadata, dict) else None
            if text:
                snippets.append(str(text))
        return {self.memory_key: "\n\n".join(snippets)}

    def clear(self) -> None:
        """Forget all currently stored memories."""

        if self.engine is not None:
            memory_ids = list(self.engine.stats["active_memory_ids"])
            for memory_id in memory_ids:
                self.engine.forget(memory_id)
            return

        assert self.client is not None
        memory_ids = list(self.client.stats().get("active_memory_ids", []))
        for memory_id in memory_ids:
            self.client.forget(str(memory_id))

    def _store(self, vector: list[float], *, metadata: dict[str, Any], importance: float) -> None:
        if self.engine is not None:
            self.engine.store(torch.tensor(vector, dtype=torch.float32), metadata=metadata, importance=importance)
            return
        assert self.client is not None
        self.client.store(vector, metadata=metadata, importance=importance)

    def _query(self, vector: list[float]) -> list[dict[str, Any]]:
        if self.engine is not None:
            results = self.engine.query(
                torch.tensor(vector, dtype=torch.float32),
                top_k=self.top_k,
                threshold=self.threshold,
            )
            return [
                {
                    "memory_id": result.memory_id,
                    "confidence": float(result.confidence),
                    "metadata": result.metadata,
                    "importance": float(result.importance),
                    "age": int(result.age),
                }
                for result in results
            ]

        assert self.client is not None
        return self.client.query(vector, top_k=self.top_k, threshold=self.threshold)

    def _embed_text(self, text: str) -> list[float]:
        vector = [0.0] * self.embedding_dim
        tokens = re.findall(r"[a-z0-9_]+", text.lower())
        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for offset in range(0, 16, 4):
                chunk = int.from_bytes(digest[offset : offset + 4], "big", signed=False)
                index = chunk % self.embedding_dim
                sign = -1.0 if chunk & 1 else 1.0
                weight = 1.0 / (1 + (offset // 4))
                vector[index] += sign * weight

        norm = math.sqrt(sum(value * value for value in vector))
        if norm > 0.0:
            vector = [value / norm for value in vector]
        return vector

    def _extract_text(self, payload: dict[str, Any], *, preferred_key: str | None = None) -> str:
        if preferred_key and preferred_key in payload:
            return self._stringify(payload[preferred_key])
        if len(payload) == 1:
            return self._stringify(next(iter(payload.values())))
        return "\n".join(f"{key}: {self._stringify(value)}" for key, value in payload.items())

    def _importance_for_text(self, text: str) -> float:
        token_count = len(re.findall(r"[a-z0-9_]+", text.lower()))
        return float(min(1.0, 0.35 + (0.03 * token_count)))

    def _stringify(self, value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float, bool)) or value is None:
            return str(value)
        return json.dumps(value, sort_keys=True)
