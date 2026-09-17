"""Structured logging and stage timing for the /chat service.

Cloud Run ships stdout to Cloud Logging. A plain ``print`` lands there as an
undifferentiated text blob, so it cannot be filtered by severity or correlated
across a request. Emitting one JSON object per line with the field names Cloud
Logging understands (``severity``, ``message``) turns the same output into
queryable structured logs at no cost.

Per-stage timings are produced by ``RAGPipeline`` itself (see
``RAGResult.timings_ms``) and logged from here, so the breakdown is available to
anything driving the pipeline, not only to HTTP callers.
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
import uuid
from typing import Any

#: Correlates every log line emitted while handling one request.
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


def new_request_id() -> str:
    """A short id that is cheap to read back from a bug report."""
    return uuid.uuid4().hex[:12]


class CloudLoggingFormatter(logging.Formatter):
    """Renders a record as one JSON line using Cloud Logging's field names."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "severity": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "request_id": request_id_var.get(),
        }
        # Anything passed via logger.info(..., extra={"x": 1}) rides along.
        for key, value in getattr(record, "context", {}).items():
            payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Install the JSON formatter on the root handler and return our logger."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(CloudLoggingFormatter())

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
    return logging.getLogger("kmp.api")


def log(logger: logging.Logger, level: int, message: str, **context: Any) -> None:
    """Log ``message`` with arbitrary structured fields attached."""
    logger.log(level, message, extra={"context": context})
