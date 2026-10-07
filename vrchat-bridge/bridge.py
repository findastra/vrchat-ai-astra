"""Small, local-only adapter between AI Astra's HTTP backend and VRChat chatbox.

``ask`` calls AI Astra's ``generate_response`` method. AI Astra may update conversation
state while generating a response. It never forwards that response to VRChat;
the operator must review it and explicitly run ``queue`` with approved text.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import socket
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Any


DEFAULT_ENDPOINT = "http://127.0.0.1:8765"
TOKEN_ENVIRONMENT_VARIABLE = "AI_ASTRA_BACKEND_AUTH_TOKEN"
HTTP_TIMEOUT_SECONDS = 5
MAX_HTTP_RESPONSE_BYTES = 1024 * 1024
MAX_ASK_CHARS = 4000
OSC_HOST = "127.0.0.1"
OSC_PORT = 9000
OSC_ADDRESS = "/chatbox/input"
OSC_TYPE_TAGS = ",sFF"
MAX_CHAT_UTF16_UNITS = 144
MAX_CHAT_LINES = 9


class BridgeError(Exception):
    """A safe, user-facing bridge failure."""


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Refuse redirects so the local auth header cannot leave loopback."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


def normalize_endpoint(endpoint: str) -> str:
    """Validate and normalize a loopback-only HTTP base URL."""
    if not isinstance(endpoint, str) or not endpoint:
        raise BridgeError("Endpoint must be an HTTP loopback URL.")
    if "?" in endpoint or "#" in endpoint:
        raise BridgeError("Endpoint must not contain a query or fragment.")

    try:
        parsed = urllib.parse.urlsplit(endpoint)
        port = parsed.port
    except ValueError as exc:
        raise BridgeError("Endpoint has an invalid port or URL format.") from exc

    if parsed.scheme.lower() != "http":
        raise BridgeError("Endpoint must use plain HTTP on loopback.")
    if parsed.username is not None or parsed.password is not None or "@" in parsed.netloc:
        raise BridgeError("Endpoint must not contain credentials.")
    if parsed.hostname is None or parsed.hostname.lower() not in {"localhost", "127.0.0.1"}:
        raise BridgeError("Endpoint host must be localhost or 127.0.0.1.")
    if parsed.path not in {"", "/"}:
        raise BridgeError("Endpoint must be a base URL without a path.")

    normalized_port = port if port is not None else 80
    if not 1 <= normalized_port <= 65535:
        raise BridgeError("Endpoint port must be between 1 and 65535.")
    return f"http://127.0.0.1:{normalized_port}"


def _request_json(
    endpoint: str,
    path: str,
    method: str,
    payload: dict[str, Any] | None = None,
    *,
    timeout: float = HTTP_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Make one bounded JSON request to a validated local endpoint."""
    base_url = normalize_endpoint(endpoint)
    if path not in {"/health", "/api"}:
        raise BridgeError("Only /health and /api are available through this bridge.")
    if (path, method) not in {("/health", "GET"), ("/api", "POST")}:
        raise BridgeError("Unsupported local API operation.")

    headers = {"Accept": "application/json"}
    token = os.environ.get(TOKEN_ENVIRONMENT_VARIABLE)
    if token:
        headers["X-AIAstra-Token"] = token

    body = None
    if payload is not None:
        headers["Content-Type"] = "application/json; charset=utf-8"
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    request = urllib.request.Request(base_url + path, data=body, headers=headers, method=method)
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        _NoRedirectHandler(),
    )
    try:
        with opener.open(request, timeout=timeout) as response:
            if response.status != 200:
                raise BridgeError(f"AI Astra backend returned HTTP {response.status} for {path}.")
            content = response.read(MAX_HTTP_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as exc:
        status = exc.code
        exc.close()
        if status in {301, 302, 303, 307, 308}:
            raise BridgeError(f"AI Astra backend redirect refused for {path}; redirects are disabled.") from None
        if status in {401, 403}:
            raise BridgeError(
                f"AI Astra backend returned HTTP {status}; check whether {TOKEN_ENVIRONMENT_VARIABLE} is configured."
            ) from None
        raise BridgeError(f"AI Astra backend returned HTTP {status} for {path}.") from None
    except urllib.error.URLError as exc:
        reason = exc.reason
        if isinstance(reason, TimeoutError):
            raise BridgeError(f"AI Astra backend request timed out after {timeout:g} seconds.") from None
        raise BridgeError(f"Could not reach AI Astra backend at {base_url}: {reason}.") from None
    except TimeoutError:
        raise BridgeError(f"AI Astra backend request timed out after {timeout:g} seconds.") from None
    except OSError as exc:
        raise BridgeError(f"Could not reach AI Astra backend at {base_url}: {exc}.") from None
    except http.client.HTTPException as exc:
        raise BridgeError(
            f"AI Astra backend at {base_url} sent a malformed HTTP response ({type(exc).__name__})."
        ) from None

    if len(content) > MAX_HTTP_RESPONSE_BYTES:
        raise BridgeError(f"AI Astra backend response exceeded the {MAX_HTTP_RESPONSE_BYTES}-byte limit.")
    try:
        document = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise BridgeError("AI Astra backend returned invalid UTF-8 JSON.") from None
    if not isinstance(document, dict):
        raise BridgeError("AI Astra backend returned an invalid JSON object.")
    return document


def get_health(endpoint: str = DEFAULT_ENDPOINT) -> dict[str, Any]:
    """Fetch health and return only fields safe for the status display."""
    document = _request_json(endpoint, "/health", "GET")
    if document.get("ok") is not True:
        raise BridgeError("AI Astra backend health check did not report ok=true.")
    summary: dict[str, Any] = {"ok": True}
    for key in ("transport", "api_name", "api_version"):
        value = document.get(key)
        if isinstance(value, str):
            summary[key] = value
    return summary


def ask_ai_astra(text: str, endpoint: str = DEFAULT_ENDPOINT) -> str:
    """Ask AI Astra for a response; this can mutate AI Astra conversation state."""
    if not isinstance(text, str) or not text.strip():
        raise BridgeError("Ask text must not be blank.")
    if len(text) > MAX_ASK_CHARS:
        raise BridgeError(f"Ask text must be at most {MAX_ASK_CHARS} characters.")
    request_id = str(uuid.uuid4())
    envelope = {
        "id": request_id,
        "method": "generate_response",
        "params": {"user_input": text},
    }
    document = _request_json(endpoint, "/api", "POST", envelope)
    if document.get("id") != request_id:
        raise BridgeError("AI Astra backend response id did not match the request.")
    if document.get("ok") is not True:
        raise BridgeError("AI Astra backend reported that response generation failed.")
    result = document.get("result")
    if not isinstance(result, dict) or not isinstance(result.get("response"), str):
        raise BridgeError("AI Astra backend response did not contain a text result.")
    return result["response"]


def _is_disallowed_control(ch: str) -> bool:
    code = ord(ch)
    return ch != "\n" and (code < 0x20 or 0x7F <= code <= 0x9F)


def printable_response(text: str) -> str:
    """Escape terminal control characters in model output before display."""
    return "".join(
        ch if ch in "\n\t" or not _is_disallowed_control(ch) else f"\\x{ord(ch):02x}"
        for ch in text
    )


def validate_chat_text(text: str) -> tuple[str, int, int]:
    """Normalize line endings and enforce VRChat chatbox text limits."""
    if not isinstance(text, str):
        raise BridgeError("Chat text must be a string.")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if "\0" in normalized:
        raise BridgeError("Chat text must not contain NUL characters.")
    if any(_is_disallowed_control(ch) for ch in normalized):
        raise BridgeError("Chat text must not contain control characters other than newline.")
    if not normalized.strip():
        raise BridgeError("Chat text must not be blank.")
    line_count = len(normalized.split("\n"))
    if line_count > MAX_CHAT_LINES:
        raise BridgeError(f"Chat text must contain at most {MAX_CHAT_LINES} lines.")
    try:
        utf16_units = len(normalized.encode("utf-16-le", errors="strict")) // 2
        normalized.encode("utf-8", errors="strict")
    except UnicodeEncodeError:
        raise BridgeError("Chat text contains invalid Unicode characters.") from None
    if utf16_units > MAX_CHAT_UTF16_UNITS:
        raise BridgeError(f"Chat text is {utf16_units} UTF-16 units; the maximum is {MAX_CHAT_UTF16_UNITS}.")
    return normalized, utf16_units, line_count


def _osc_string(value: str) -> bytes:
    encoded = value.encode("utf-8") + b"\0"
    return encoded + (b"\0" * (-len(encoded) % 4))


def build_osc_packet(text: str) -> tuple[bytes, str, int, int]:
    """Prepare the VRChat ``/chatbox/input`` OSC packet without sending it."""
    normalized, utf16_units, line_count = validate_chat_text(text)
    packet = _osc_string(OSC_ADDRESS) + _osc_string(OSC_TYPE_TAGS) + _osc_string(normalized)
    return packet, normalized, utf16_units, line_count


def preview_chat(text: str) -> dict[str, Any]:
    """Return a reviewable OSC draft without opening a socket."""
    packet, normalized, utf16_units, line_count = build_osc_packet(text)
    return {
        "address": OSC_ADDRESS,
        "type_tags": OSC_TYPE_TAGS,
        "text": normalized,
        "utf16_code_units": utf16_units,
        "lines": line_count,
        "immediate_send": False,
        "notification": False,
        "destination": {"host": OSC_HOST, "port": OSC_PORT},
        "osc_packet_hex": packet.hex(),
    }


def queue_chat(text: str, *, port: int = OSC_PORT) -> int:
    """Send a chatbox draft to a loopback UDP port and return sent bytes."""
    if not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535:
        raise BridgeError("OSC port must be between 1 and 65535.")
    packet, _, _, _ = build_osc_packet(text)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sent = sock.sendto(packet, (OSC_HOST, port))
    except OSError as exc:
        raise BridgeError(f"Could not queue OSC draft to local port {port}: {exc}.") from None
    if sent != len(packet):
        raise BridgeError("The local UDP socket did not accept the complete OSC draft.")
    return sent


def _add_endpoint_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help=f"AI Astra loopback HTTP base URL (default: {DEFAULT_ENDPOINT})",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local AI Astra-to-VRChat text bridge.")
    commands = parser.add_subparsers(dest="command", required=True)

    status = commands.add_parser("status", help="Check the AI Astra backend health endpoint.")
    _add_endpoint_argument(status)

    ask = commands.add_parser("ask", help="Ask AI Astra for text; output is not sent to VRChat.")
    _add_endpoint_argument(ask)
    ask.add_argument("text", help="Text to send to AI Astra")

    preview = commands.add_parser("preview", help="Validate and prepare an OSC chat draft without networking.")
    preview.add_argument("text", help="Text to preview")

    queue = commands.add_parser("queue", help="Queue reviewed text to the local VRChat chatbox.")
    queue.add_argument("text", help="Reviewed text to queue")
    queue.add_argument(
        "--port",
        type=int,
        default=OSC_PORT,
        help=f"Local OSC input port (default: VRChat's {OSC_PORT}; tests use a stub port)",
    )
    return parser


def _use_utf8_stdio() -> None:
    """Avoid UnicodeEncodeError when Windows redirects output to a cp1252 pipe."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="backslashreplace")
            except (ValueError, OSError):
                pass


def main(argv: list[str] | None = None) -> int:
    _use_utf8_stdio()
    args = build_parser().parse_args(argv)
    try:
        if args.command == "status":
            summary = get_health(args.endpoint)
            print(json.dumps(summary, ensure_ascii=False, separators=(",", ":")))
        elif args.command == "ask":
            response = ask_ai_astra(args.text, args.endpoint)
            sys.stdout.write(f"AI Astra: {printable_response(response)}\n")
            print("(Not sent to VRChat. Review it, then use: queue \"<approved text>\")", file=sys.stderr)
        elif args.command == "preview":
            print(json.dumps(preview_chat(args.text), ensure_ascii=False, separators=(",", ":")))
        elif args.command == "queue":
            sent = queue_chat(args.text, port=args.port)
            print(
                f"Sent one {sent}-byte chatbox draft packet to 127.0.0.1:{args.port} (UDP). "
                "UDP gives no delivery receipt: check the chatbox in VRChat and send it there yourself."
            )
        return 0
    except BridgeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
