"""MCP stdio server so Pi and OpenCode share one Maxi home.

The tool names are ``friend_context`` and ``friend_remember``, the pair
AI Friend 1.8.3 publishes (MIT). The write path is not that pair.
``friend_context`` commits the user sentence itself, then reads it
back, because a model that only answers never calls ``friend_remember``.
``source_agent`` is the MCP ``clientInfo.name`` (``pi``, ``opencode``),
not a hardcoded ``mcp``.

Stdout is protocol only.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from hyper_ambient_maxi import __version__
from hyper_ambient_maxi.loop import run_turn
from hyper_ambient_maxi.memory.store import MemoryStore, MemoryWriteError

_PROTOCOL = "2024-11-05"
_TOOLS = [
    {
        "name": "friend_context",
        "description": (
            "Before answering, fetch what Maxi remembers for this turn. "
            "The server also saves durable facts from the message, so a "
            "skipped friend_remember does not drop them. Returns a short "
            "context block with no scores."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "message": {"type": "string"},
                "conversation_id": {"type": "string"},
            },
            "required": ["message"],
            "additionalProperties": False,
        },
    },
    {
        "name": "friend_remember",
        "description": (
            "After answering, store the user message again. Repeating a "
            "fact that friend_context already saved is idempotent. A "
            "failed read-back is an error, not 'stored 0'."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "user_message": {"type": "string"},
                "assistant_message": {"type": "string"},
                "conversation_id": {"type": "string"},
            },
            "required": ["user_message"],
            "additionalProperties": False,
        },
    },
]


class McpServer:
    def __init__(self, home: Path | str):
        self.store = MemoryStore(home)
        self.source_agent = "mcp"
        self._traces: dict[str, str] = {}

    def handle(self, raw: str | dict) -> dict | None:
        if isinstance(raw, str):
            try:
                request = json.loads(raw)
            except json.JSONDecodeError:
                return _error(None, -32700, "parse error")
        else:
            request = raw
        if not isinstance(request, dict) or not isinstance(request.get("method"), str):
            return _error(request.get("id") if isinstance(request, dict) else None, -32600, "invalid request")
        method = request["method"]
        is_notification = request.get("id") is None and "id" not in request or request.get("id") is None
        # A missing id is a notification. An explicit null id is too.
        if "id" not in request or request.get("id") is None:
            is_notification = True
        handler = {
            "initialize": self._initialize,
            "tools/list": self._tools,
            "tools/call": self._call,
            "ping": self._ping,
        }.get(method)
        if handler is None:
            if is_notification or method.startswith("notifications/"):
                return None
            return _error(request.get("id"), -32601, f"method not found: {method}")
        if is_notification or method.startswith("notifications/"):
            return None
        try:
            return handler(request)
        except MemoryWriteError as exc:
            return _error(request.get("id"), -32603, str(exc))
        except Exception as exc:  # keep the server up for the next line
            return _error(request.get("id"), -32603, f"internal error: {exc}")

    def _initialize(self, request: dict) -> dict:
        params = request.get("params") if isinstance(request.get("params"), dict) else {}
        info = params.get("clientInfo") if isinstance(params.get("clientInfo"), dict) else {}
        name = info.get("name") if isinstance(info.get("name"), str) else ""
        cleaned = re.sub(r"[^a-z0-9_-]", "", name.lower())[:32]
        self.source_agent = cleaned or "mcp"
        return _result(
            request.get("id"),
            {
                "protocolVersion": _PROTOCOL,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "hyper-ambient-maxi", "version": __version__},
                "instructions": (
                    "Maxi holds this user's durable facts. Call friend_context "
                    "with the user message before answering. The server writes "
                    "those facts itself and reads them back; do not claim a "
                    "memory the block does not contain. Call friend_remember "
                    "afterwards with the same conversation_id. Never repeat "
                    "importance or relevance scores; there are none in the block."
                ),
            },
        )

    def _tools(self, request: dict) -> dict:
        return _result(request.get("id"), {"tools": _TOOLS})

    def _ping(self, request: dict) -> dict:
        return _result(request.get("id"), {})

    def _call(self, request: dict) -> dict:
        params = request.get("params") if isinstance(request.get("params"), dict) else {}
        name = params.get("name")
        args = params.get("arguments") if isinstance(params.get("arguments"), dict) else {}
        if name == "friend_context":
            return _result(request.get("id"), self._context(args))
        if name == "friend_remember":
            return _result(request.get("id"), self._remember(args))
        return _error(request.get("id"), -32602, f"unknown tool: {name}")

    def _context(self, args: dict) -> dict:
        message = args.get("message")
        if not isinstance(message, str) or not message.strip():
            return _text("friend_context needs a non-empty message", error=True)
        conversation = _conversation(args)
        result = run_turn(
            self.store,
            _Brain(),
            message,
            conversation_id=conversation,
            source_agent=self.source_agent,
        )
        self._traces[conversation] = "ok" if result.memory_ok else "failed"
        if not result.memory_ok:
            return _text(result.reply, error=True)
        body = result.context or "(nothing relevant remembered for this turn)"
        return _text(body)

    def _remember(self, args: dict) -> dict:
        message = args.get("user_message")
        if not isinstance(message, str) or not message.strip():
            return _text("friend_remember needs a non-empty user_message", error=True)
        conversation = _conversation(args)
        result = run_turn(
            self.store,
            _Brain(),
            message,
            conversation_id=conversation,
            source_agent=self.source_agent,
        )
        self._traces.pop(conversation, None)
        if not result.memory_ok:
            return _text(result.reply, error=True)
        stored = 0 if result.receipt is None else result.receipt.stored
        threads = 0 if result.receipt is None else len(result.receipt.threads_opened) + len(result.receipt.threads_closed)
        return _text(f"stored {stored} memory update(s) and {threads} thread update(s)")


class _Brain:
    """friend_remember does not need a model. The reply is unused."""

    name = "scripted"

    def reply(self, user, context, receipt, memories, notes=None, warning=None, **_extra):
        return ""


def serve_stdio(home: Path | str) -> int:
    server = McpServer(home)
    for line in sys.stdin:
        if not line.strip():
            continue
        response = server.handle(line)
        if response is None:
            continue
        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        sys.stdout.flush()
    return 0


def _conversation(args: dict) -> str:
    value = args.get("conversation_id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return "default"


def _result(request_id, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _error(request_id, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def _text(text: str, error: bool = False) -> dict:
    body = {"content": [{"type": "text", "text": text}]}
    if error:
        body["isError"] = True
    return body
