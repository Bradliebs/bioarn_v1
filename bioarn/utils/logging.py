"""Structured logging utilities for Bio-ARN."""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

METRIC_LEVEL = 25
logging.addLevelName(METRIC_LEVEL, "METRIC")


def _coerce_level(level: str | int | None) -> int:
    if level is None:
        level = os.getenv("BIOARN_LOG_LEVEL", "INFO")
    if isinstance(level, int):
        return level
    normalized = str(level).strip().upper()
    if normalized == "METRIC":
        return METRIC_LEVEL
    return int(getattr(logging, normalized, logging.INFO))


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # pragma: no cover - defensive conversion
            pass
    return str(value)


class _StructuredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "component": getattr(record, "component", record.name),
            "event": getattr(record, "event", "log"),
            "data": getattr(record, "data", {}),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=_json_default)


class BioARNLogger:
    """Structured logger for Bio-ARN training and inference."""

    def __init__(
        self,
        component: str = "bioarn",
        *,
        level: str | int | None = None,
        log_dir: str | os.PathLike[str] | None = None,
        session_name: str | None = None,
    ) -> None:
        self.component = component
        self.level = _coerce_level(level)
        self.log_dir = Path(log_dir or os.getenv("BIOARN_LOG_DIR", "logs"))
        self.log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        suffix = session_name or component.replace(".", "_")
        self.log_path = self.log_dir / f"{suffix}-{timestamp}.jsonl"

        logger_name = f"bioarn.{component}.{timestamp}"
        self._logger = logging.getLogger(logger_name)
        self._logger.setLevel(self.level)
        self._logger.propagate = False
        self._logger.handlers.clear()

        formatter = _StructuredFormatter()

        file_handler = logging.FileHandler(self.log_path, encoding="utf-8")
        file_handler.setLevel(self.level)
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.level)
        console_handler.setFormatter(formatter)

        self._logger.addHandler(file_handler)
        self._logger.addHandler(console_handler)

    def _emit(
        self,
        level: int,
        component: str,
        event: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        payload = data or {}
        self._logger.log(
            level,
            event,
            extra={
                "component": component,
                "event": event,
                "data": payload,
            },
        )

    def log_metric(
        self,
        component: str,
        metric_name: str,
        value: float | int,
        step: int,
        **data: Any,
    ) -> None:
        payload = {
            "metric_name": metric_name,
            "value": value,
            "step": int(step),
            **data,
        }
        self._emit(METRIC_LEVEL, component, "metric", payload)

    def log_event(
        self,
        component: str,
        event: str,
        details: dict[str, Any] | None = None,
        *,
        level: str | int = "INFO",
    ) -> None:
        self._emit(_coerce_level(level), component, event, details)

    def debug(self, component: str, event: str, details: dict[str, Any] | None = None) -> None:
        self.log_event(component, event, details, level="DEBUG")

    def warning(self, component: str, event: str, details: dict[str, Any] | None = None) -> None:
        self.log_event(component, event, details, level="WARNING")

    def error(self, component: str, event: str, details: dict[str, Any] | None = None) -> None:
        self.log_event(component, event, details, level="ERROR")

    def flush(self) -> None:
        for handler in self._logger.handlers:
            handler.flush()


__all__ = ["BioARNLogger", "METRIC_LEVEL"]
