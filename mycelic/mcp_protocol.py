"""Dependency-free JSON-RPC 2.0 / MCP server core and the Streamable HTTP transport.

Implements the MCP server methods Claude clients use (``initialize``, ``notifications/initialized``, ``ping``,
``tools/list``, ``tools/call``, and empty ``resources/*`` and ``prompts/*`` answers). Protocol version negotiation
follows the spec: the server answers with the client's version when it is supported, otherwise with its latest.

Derived from ``NeuralGraph.chat_memory.mcp_server`` at anovruzov/NeuralGraph@e7ef73ea when Mycelic moved to its own
repository, with the chat-memory tool set, resources, prompts and stdio server removed. Mycelic owns this copy now.
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections import OrderedDict
from typing import Any

logger = logging.getLogger(__name__)

SUPPORTED_PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
MAX_HTTP_SESSIONS = 10_000
LATEST_PROTOCOL_VERSION = SUPPORTED_PROTOCOL_VERSIONS[0]

JSONRPC_PARSE_ERROR = -32700
JSONRPC_INVALID_REQUEST = -32600
JSONRPC_METHOD_NOT_FOUND = -32601
JSONRPC_INVALID_PARAMS = -32602
JSONRPC_INTERNAL_ERROR = -32603


class ToolError(Exception):
    """Raised by a tool to return an ``isError`` result (the model sees the message)."""


class MCPProtocol:
    """JSON-RPC/MCP core. ``tool_defs`` is the ``tools/list`` payload and ``tools`` an object with
    ``async call(name, args)``; resources and prompts answer empty."""

    def __init__(self, *, instructions: str, tool_defs: list[dict[str, Any]], tools: Any, server_info: dict[str, str]) -> None:
        self.tools = tools
        self.tool_defs = tool_defs
        self.server_info = server_info
        self.instructions = instructions

    # -- helpers
    @staticmethod
    def _result(id_: Any, result: Any) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": id_, "result": result}

    @staticmethod
    def _error(id_: Any, code: int, message: str, data: Any = None) -> dict[str, Any]:
        err: dict[str, Any] = {"code": code, "message": message}
        if data is not None:
            err["data"] = data
        return {"jsonrpc": "2.0", "id": id_, "error": err}

    async def handle(self, message: Any) -> dict[str, Any] | None:
        """Process one JSON-RPC message (request or notification). Returns a response dict or None."""
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0" or not isinstance(message.get("method"), str):
            return self._error(message.get("id") if isinstance(message, dict) else None, JSONRPC_INVALID_REQUEST, "invalid request")
        method = message["method"]
        params = message.get("params")
        if params is None:
            params = {}
        if not isinstance(params, dict):
            return self._error(message.get("id"), JSONRPC_INVALID_PARAMS, "params must be an object")
        id_ = message.get("id")
        if id_ is not None and not isinstance(id_, (str, int)):
            return self._error(None, JSONRPC_INVALID_REQUEST, "id must be a string or number")
        is_notification = "id" not in message
        try:
            if method == "initialize":
                result = self._initialize(params)
            elif method == "notifications/initialized" or method.startswith("notifications/"):
                return None
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": self.tool_defs}
            elif method == "tools/call":
                result = await self._tools_call(params)
            elif method == "resources/list":
                result = {"resources": []}
            elif method == "resources/read":
                raise ToolError(f"unknown resource: {params.get('uri') or ''}")
            elif method == "resources/templates/list":
                result = {"resourceTemplates": []}
            elif method == "prompts/list":
                result = {"prompts": []}
            elif method == "prompts/get":
                raise ToolError("no prompts")
            elif method == "logging/setLevel":
                result = {}
            else:
                if is_notification:
                    return None
                return self._error(id_, JSONRPC_METHOD_NOT_FOUND, f"method not found: {method}")
        except ToolError as exc:
            return self._error(id_, JSONRPC_INVALID_PARAMS, str(exc))
        except TypeError as exc:  # bad/missing tool arguments
            return self._error(id_, JSONRPC_INVALID_PARAMS, f"invalid params: {exc}")
        except Exception as exc:
            logger.exception("MCP %s failed", method)
            return self._error(id_, JSONRPC_INTERNAL_ERROR, f"{type(exc).__name__}: {exc}")
        if is_notification:
            return None
        return self._result(id_, result)

    def _initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        requested = str(params.get("protocolVersion") or "")
        version = requested if requested in SUPPORTED_PROTOCOL_VERSIONS else LATEST_PROTOCOL_VERSION
        return {
            "protocolVersion": version,
            "capabilities": {"tools": {"listChanged": False}, "resources": {"subscribe": False, "listChanged": False},
                             "prompts": {"listChanged": False}, "logging": {}},
            "serverInfo": self.server_info,
            "instructions": self.instructions,
        }

    async def _tools_call(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        args = params.get("arguments") or {}
        if not isinstance(name, str) or not any(t["name"] == name for t in self.tool_defs):
            raise ToolError(f"unknown tool: {name}")
        if not isinstance(args, dict):
            raise ToolError("arguments must be an object")
        try:
            out = await self.tools.call(name, args)
        except ToolError as exc:
            return {"content": [{"type": "text", "text": str(exc)}], "isError": True}
        except TypeError as exc:
            return {"content": [{"type": "text", "text": f"invalid arguments: {exc}"}], "isError": True}
        except Exception as exc:
            logger.exception("tool %s failed", name)
            return {"content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}], "isError": True}
        text = json.dumps(out, ensure_ascii=False, indent=1, default=str)
        return {"content": [{"type": "text", "text": text}], "structuredContent": out, "isError": False}

    async def handle_payload(self, payload: Any) -> Any:
        """Handle a decoded JSON payload that may be a single message or a batch. Returns the response payload or None."""
        if isinstance(payload, list):
            if not payload:
                return self._error(None, JSONRPC_INVALID_REQUEST, "empty batch")
            responses = [r for r in await asyncio.gather(*(self.handle(m) for m in payload)) if r is not None]
            return responses or None
        return await self.handle(payload)


class StreamableHTTPTransport:
    """MCP Streamable HTTP (2025-03-26+): one endpoint, POST for client->server messages.

    * Responses are returned as ``application/json`` (single response) or, when the client only accepts
      ``text/event-stream``, as one SSE ``message`` event followed by stream end.
    * A session id is issued on ``initialize`` (``Mcp-Session-Id``); sessions are lightweight here (no
      server-initiated messages), so an unknown/missing session is tolerated for stateless clients.
    * ``GET`` opens an SSE stream that only carries keep-alives (the server never pushes requests);
      ``DELETE`` ends a session.
    * ``Origin`` is validated against ``allowed_origins`` when set (DNS-rebinding protection). Authentication is the
      caller's job (Mycelic's API middleware authenticates before the request reaches the transport).
    """

    def __init__(self, protocol: MCPProtocol, *, allowed_origins: list[str] | None = None) -> None:
        self.proto = protocol
        self.allowed_origins = allowed_origins
        self.sessions: "OrderedDict[str, dict[str, Any]]" = OrderedDict()

    def _origin_ok(self, request) -> bool:
        origin = request.headers.get("Origin")
        if not origin or self.allowed_origins is None:
            return True
        return origin in self.allowed_origins

    async def handle(self, request):
        from aiohttp import web

        if not self._origin_ok(request):
            return web.json_response({"error": "origin not allowed"}, status=403)
        if request.method == "GET":
            return await self._handle_get(request)
        if request.method == "DELETE":
            sid = request.headers.get("Mcp-Session-Id")
            self.sessions.pop(sid or "", None)
            return web.Response(status=204)
        if request.method != "POST":
            return web.json_response({"error": "method not allowed"}, status=405, headers={"Allow": "GET, POST, DELETE"})
        pv = request.headers.get("MCP-Protocol-Version")
        if pv and pv not in SUPPORTED_PROTOCOL_VERSIONS:
            return web.json_response({"error": f"unsupported MCP-Protocol-Version {pv}", "supported": list(SUPPORTED_PROTOCOL_VERSIONS)}, status=400)
        try:
            payload = await request.json()
        except Exception:
            return web.json_response(MCPProtocol._error(None, JSONRPC_PARSE_ERROR, "parse error"), status=400)
        first = payload[0] if isinstance(payload, list) and payload else payload
        is_init = isinstance(first, dict) and first.get("method") == "initialize"
        resp = await self.proto.handle_payload(payload)
        headers = {"MCP-Protocol-Version": LATEST_PROTOCOL_VERSION}
        if is_init:
            sid = uuid.uuid4().hex
            self.sessions[sid] = {"created": asyncio.get_running_loop().time()}
            while len(self.sessions) > MAX_HTTP_SESSIONS:
                self.sessions.popitem(last=False)
            headers["Mcp-Session-Id"] = sid
            if isinstance(resp, dict) and "result" in resp:
                headers["MCP-Protocol-Version"] = resp["result"]["protocolVersion"]
        elif request.headers.get("Mcp-Session-Id"):
            headers["Mcp-Session-Id"] = request.headers["Mcp-Session-Id"]
        if resp is None:  # notifications / responses only
            return web.Response(status=202, headers=headers)
        accept = request.headers.get("Accept", "")
        wants_sse = "text/event-stream" in accept and "application/json" not in accept
        if wants_sse:
            body = "event: message\ndata: " + json.dumps(resp, ensure_ascii=False) + "\n\n"
            return web.Response(text=body, content_type="text/event-stream", headers={**headers, "Cache-Control": "no-cache"})
        return web.json_response(resp, headers=headers)

    async def _handle_get(self, request):
        from aiohttp import web

        accept = request.headers.get("Accept", "")
        if "text/event-stream" not in accept:
            return web.json_response({"error": "GET requires Accept: text/event-stream"}, status=405, headers={"Allow": "POST"})
        resp = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache",
                                                       "MCP-Protocol-Version": LATEST_PROTOCOL_VERSION})
        await resp.prepare(request)
        try:
            while True:
                await resp.write(b": keep-alive\n\n")
                await asyncio.sleep(15)
        except (asyncio.CancelledError, ConnectionResetError):
            pass
        return resp
