"""REST and client helpers for Bio-ARN associative memory."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .client import BioARNMemoryClient
    from .langchain_memory import BioARNMemory
    from .memory_server import MemoryAPI, MemoryHTTPServer, create_memory_server, serve

__all__ = [
    "BioARNMemory",
    "BioARNMemoryClient",
    "MemoryAPI",
    "MemoryHTTPServer",
    "create_memory_server",
    "serve",
]


def __getattr__(name: str):
    if name == "BioARNMemoryClient":
        from .client import BioARNMemoryClient

        return BioARNMemoryClient
    if name == "BioARNMemory":
        from .langchain_memory import BioARNMemory

        return BioARNMemory
    if name in {"MemoryAPI", "MemoryHTTPServer", "create_memory_server", "serve"}:
        from .memory_server import MemoryAPI, MemoryHTTPServer, create_memory_server, serve

        exports = {
            "MemoryAPI": MemoryAPI,
            "MemoryHTTPServer": MemoryHTTPServer,
            "create_memory_server": create_memory_server,
            "serve": serve,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
