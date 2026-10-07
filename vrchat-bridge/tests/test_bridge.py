"""Offline tests for bridge.py using loopback stubs on ephemeral ports.

Nothing here talks to a real Mai instance or to VRChat's live OSC port 9000.
Run from the astra-vrchat folder:  python -m unittest discover -s tests -v
"""

from __future__ import annotations

import ast
import contextlib
import io
import json
import os
import socket
import sys
import threading
import time
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import bridge  # noqa: E402

LIVE_VRCHAT_PORT = 9000
TOKEN = "test-token-not-real"


class StubMai:
    """A tiny loopback HTTP server whose behaviour each test sets."""

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.behaviour = lambda handler, body: (200, {"ok": True}, None)
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # silence
                pass

            def _handle(self):
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else b""
                stub.requests.append(
                    {"method": self.command, "path": self.path, "headers": dict(self.headers), "body": body}
                )
                status, document, extra_headers = stub.behaviour(self, body)
                raw = document if isinstance(document, bytes) else json.dumps(document).encode("utf-8")
                self.send_response(status)
                for key, value in (extra_headers or {}).items():
                    self.send_header(key, value)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            do_GET = _handle
            do_POST = _handle

        class QuietServer(ThreadingHTTPServer):
            daemon_threads = True

            def handle_error(self, request, client_address):
                pass  # e.g. BrokenPipe after the client timed out on purpose

        self.server = QuietServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        assert self.port != LIVE_VRCHAT_PORT
        self.endpoint = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def echo_success(response_text="Hello from stub"):
    def behaviour(handler, body):
        request = json.loads(body)
        return 200, {"id": request["id"], "ok": True, "result": {"response": response_text, "extra": 1}}, None

    return behaviour


class HttpTests(unittest.TestCase):
    def setUp(self):
        self.stub = StubMai()
        self.addCleanup(self.stub.close)
        patcher = mock.patch.dict(os.environ, {}, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop(bridge.TOKEN_ENVIRONMENT_VARIABLE, None)

    # --- health -------------------------------------------------------------
    def test_health_summary_only_exposes_safe_fields(self):
        self.stub.behaviour = lambda h, b: (
            200,
            {"ok": True, "transport": "http", "api_name": "mai", "api_version": "2", "state_dir": "C:/secret",
             "brain_rows": 123},
            None,
        )
        summary = bridge.get_health(self.stub.endpoint)
        self.assertEqual(summary, {"ok": True, "transport": "http", "api_name": "mai", "api_version": "2"})
        self.assertEqual(self.stub.requests[0]["method"], "GET")
        self.assertEqual(self.stub.requests[0]["path"], "/health")

    def test_health_not_ok(self):
        self.stub.behaviour = lambda h, b: (200, {"ok": False}, None)
        with self.assertRaisesRegex(bridge.BridgeError, "ok=true"):
            bridge.get_health(self.stub.endpoint)

    # --- ask: contract ------------------------------------------------------
    def test_ask_sends_exact_contract_and_returns_response(self):
        self.stub.behaviour = echo_success("Hi Astra ✨")
        self.assertEqual(bridge.ask_mai("hello ✨", self.stub.endpoint), "Hi Astra ✨")
        request = self.stub.requests[0]
        self.assertEqual((request["method"], request["path"]), ("POST", "/api"))
        body = json.loads(request["body"].decode("utf-8"))
        self.assertEqual(set(body), {"id", "method", "params"})
        self.assertEqual(body["method"], "generate_response")
        self.assertEqual(body["params"], {"user_input": "hello ✨"})
        self.assertTrue(body["id"])
        self.assertTrue(request["headers"]["Content-Type"].startswith("application/json"))
        self.assertNotIn("X-Mai-Token", request["headers"])

    def test_ask_uses_unique_ids(self):
        self.stub.behaviour = echo_success()
        bridge.ask_mai("one", self.stub.endpoint)
        bridge.ask_mai("two", self.stub.endpoint)
        ids = [json.loads(r["body"])["id"] for r in self.stub.requests]
        self.assertNotEqual(ids[0], ids[1])

    def test_ask_rejects_mismatched_id(self):
        self.stub.behaviour = lambda h, b: (200, {"id": "someone-else", "ok": True, "result": {"response": "x"}}, None)
        with self.assertRaisesRegex(bridge.BridgeError, "id did not match"):
            bridge.ask_mai("hello", self.stub.endpoint)

    def test_ask_rejects_missing_id(self):
        self.stub.behaviour = lambda h, b: (200, {"ok": True, "result": {"response": "x"}}, None)
        with self.assertRaisesRegex(bridge.BridgeError, "id did not match"):
            bridge.ask_mai("hello", self.stub.endpoint)

    def test_ask_requires_ok_strictly_true(self):
        for ok_value in (False, "true", 1, None):
            with self.subTest(ok=ok_value):
                def behaviour(h, b, ok_value=ok_value):
                    request_id = json.loads(b)["id"]
                    return 200, {"id": request_id, "ok": ok_value, "result": {"response": "x"}}, None

                self.stub.behaviour = behaviour
                with self.assertRaisesRegex(bridge.BridgeError, "failed"):
                    bridge.ask_mai("hello", self.stub.endpoint)

    def test_ask_rejects_bad_result_shapes(self):
        for result in (None, "text", {"response": 5}, {"text": "x"}, []):
            with self.subTest(result=result):
                def behaviour(h, b, result=result):
                    return 200, {"id": json.loads(b)["id"], "ok": True, "result": result}, None

                self.stub.behaviour = behaviour
                with self.assertRaisesRegex(bridge.BridgeError, "text result"):
                    bridge.ask_mai("hello", self.stub.endpoint)

    def test_ask_input_validation_makes_no_request(self):
        for text in ("", "   \n", "x" * (bridge.MAX_ASK_CHARS + 1)):
            with self.subTest(length=len(text)):
                with self.assertRaises(bridge.BridgeError):
                    bridge.ask_mai(text, self.stub.endpoint)
        self.assertEqual(self.stub.requests, [])

    # --- auth ---------------------------------------------------------------
    def test_token_header_sent_when_configured_and_never_echoed(self):
        os.environ[bridge.TOKEN_ENVIRONMENT_VARIABLE] = TOKEN
        self.stub.behaviour = echo_success()
        bridge.ask_mai("hello", self.stub.endpoint)
        self.assertEqual(self.stub.requests[0]["headers"].get("X-Mai-Token"), TOKEN)

        self.stub.behaviour = lambda h, b: (401, {"ok": False, "error": "unauthorized"}, None)
        with self.assertRaises(bridge.BridgeError) as caught:
            bridge.ask_mai("hello", self.stub.endpoint)
        self.assertIn("401", str(caught.exception))
        self.assertIn(bridge.TOKEN_ENVIRONMENT_VARIABLE, str(caught.exception))
        self.assertNotIn(TOKEN, str(caught.exception))

    def test_forbidden_status(self):
        self.stub.behaviour = lambda h, b: (403, {"ok": False}, None)
        with self.assertRaisesRegex(bridge.BridgeError, "403"):
            bridge.get_health(self.stub.endpoint)

    # --- redirects ----------------------------------------------------------
    def test_redirect_refused_and_token_not_forwarded(self):
        other = StubMai()
        self.addCleanup(other.close)
        other.behaviour = lambda h, b: (200, {"ok": True}, None)
        os.environ[bridge.TOKEN_ENVIRONMENT_VARIABLE] = TOKEN
        for status in (301, 302, 303, 307, 308):
            with self.subTest(status=status):
                self.stub.behaviour = lambda h, b, s=status: (s, {}, {"Location": other.endpoint + "/health"})
                with self.assertRaisesRegex(bridge.BridgeError, "redirect refused"):
                    bridge.get_health(self.stub.endpoint)
        self.assertEqual(other.requests, [], "redirect target must never be contacted")

    # --- other failures -----------------------------------------------------
    def test_server_error(self):
        self.stub.behaviour = lambda h, b: (500, {"ok": False}, None)
        with self.assertRaisesRegex(bridge.BridgeError, "HTTP 500"):
            bridge.get_health(self.stub.endpoint)

    def test_non_200_success_status(self):
        self.stub.behaviour = lambda h, b: (202, {"ok": True}, None)
        with self.assertRaisesRegex(bridge.BridgeError, "HTTP 202"):
            bridge.get_health(self.stub.endpoint)

    def test_invalid_json_and_non_object(self):
        for raw in (b"not json", b"\xff\xfe", b"[1,2]", b'"text"'):
            with self.subTest(raw=raw):
                self.stub.behaviour = lambda h, b, raw=raw: (200, raw, None)
                with self.assertRaisesRegex(bridge.BridgeError, "invalid"):
                    bridge.get_health(self.stub.endpoint)

    def test_oversized_response(self):
        big = b'{"ok":true,"pad":"' + b"a" * (bridge.MAX_HTTP_RESPONSE_BYTES + 10) + b'"}'
        self.stub.behaviour = lambda h, b: (200, big, None)
        with self.assertRaisesRegex(bridge.BridgeError, "byte limit"):
            bridge.get_health(self.stub.endpoint)

    def test_timeout(self):
        def slow(h, b):
            time.sleep(1.0)
            return 200, {"ok": True}, None

        self.stub.behaviour = slow
        with self.assertRaisesRegex(bridge.BridgeError, "timed out"):
            bridge._request_json(self.stub.endpoint, "/health", "GET", timeout=0.2)

    def test_proxy_environment_is_ignored(self):
        self.stub.behaviour = lambda h, b: (200, {"ok": True}, None)
        with mock.patch.dict(os.environ, {"HTTP_PROXY": "http://127.0.0.1:1", "http_proxy": "http://127.0.0.1:1",
                                          "NO_PROXY": "", "no_proxy": ""}):
            self.assertEqual(bridge.get_health(self.stub.endpoint), {"ok": True})

    def test_only_two_operations_allowed(self):
        for path, method in (("/api", "GET"), ("/health", "POST"), ("/admin", "GET"), ("/api/../x", "POST")):
            with self.subTest(path=path, method=method):
                with self.assertRaises(bridge.BridgeError):
                    bridge._request_json(self.stub.endpoint, path, method, {} if method == "POST" else None)
        self.assertEqual(self.stub.requests, [])


class RawSocketFailureTests(unittest.TestCase):
    def test_connection_refused(self):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        with self.assertRaisesRegex(bridge.BridgeError, "Could not reach"):
            bridge.get_health(f"http://127.0.0.1:{port}")

    def test_malformed_http_response(self):
        server = socket.socket()
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        self.addCleanup(server.close)

        def serve():
            connection, _ = server.accept()
            with connection:
                connection.recv(4096)
                connection.sendall(b"GARBAGE\r\n\r\n")

        threading.Thread(target=serve, daemon=True).start()
        with self.assertRaisesRegex(bridge.BridgeError, "malformed|Could not reach"):
            bridge.get_health(f"http://127.0.0.1:{port}")


class EndpointTests(unittest.TestCase):
    def test_accepted_and_normalized(self):
        self.assertEqual(bridge.normalize_endpoint("http://127.0.0.1:8765"), "http://127.0.0.1:8765")
        self.assertEqual(bridge.normalize_endpoint("http://localhost:8765/"), "http://127.0.0.1:8765")
        self.assertEqual(bridge.normalize_endpoint("HTTP://LOCALHOST"), "http://127.0.0.1:80")

    def test_rejected(self):
        for endpoint in (
            "", "https://127.0.0.1:8765", "http://example.com:8765", "http://192.168.1.5:8765",
            "http://127.0.0.1:8765/api", "http://user:pw@127.0.0.1:8765", "http://127.0.0.1:8765?x=1",
            "http://127.0.0.1:8765#frag", "http://127.0.0.1:99999", "http://127.0.0.1:0", "http://127.0.0.1:abc",
            "http://[::1]:8765", "ftp://127.0.0.1", "http://127.0.0.1.evil.com:8765",
        ):
            with self.subTest(endpoint=endpoint):
                with self.assertRaises(bridge.BridgeError):
                    bridge.normalize_endpoint(endpoint)


def decode_osc(packet: bytes) -> list:
    """Minimal decoder used to check the encoding independently."""
    def read_string(offset):
        end = packet.index(b"\0", offset)
        value = packet[offset:end].decode("utf-8")
        return value, (end + 4) & ~3

    self_check = len(packet) % 4 == 0
    address, offset = read_string(0)
    tags, offset = read_string(offset)
    values = []
    for tag in tags[1:]:
        if tag == "s":
            value, offset = read_string(offset)
            values.append(value)
        elif tag == "T":
            values.append(True)
        elif tag == "F":
            values.append(False)
        else:
            raise AssertionError(tag)
    assert offset == len(packet)
    return [self_check, address, tags, *values]


class ChatTests(unittest.TestCase):
    def test_length_limit_in_utf16_units(self):
        self.assertEqual(bridge.validate_chat_text("a" * 144)[1], 144)
        with self.assertRaisesRegex(bridge.BridgeError, "145 UTF-16"):
            bridge.validate_chat_text("a" * 145)
        # Emoji outside the BMP count as two UTF-16 units each.
        self.assertEqual(bridge.validate_chat_text("💖" * 72)[1], 144)
        with self.assertRaises(bridge.BridgeError):
            bridge.validate_chat_text("💖" * 73)
        # BMP accented and CJK characters count once.
        self.assertEqual(bridge.validate_chat_text("é" * 144)[1], 144)
        self.assertEqual(bridge.validate_chat_text("星" * 144)[1], 144)

    def test_line_limit_and_newline_normalization(self):
        text, _, lines = bridge.validate_chat_text("\r\n".join("abcdefghi"))
        self.assertEqual(lines, 9)
        self.assertNotIn("\r", text)
        self.assertEqual(bridge.validate_chat_text("a\rb")[0], "a\nb")
        with self.assertRaisesRegex(bridge.BridgeError, "9 lines"):
            bridge.validate_chat_text("\n".join("abcdefghij"))

    def test_rejects_blank_nul_controls_and_bad_unicode(self):
        for text in ("", "  \n ", "a\0b", "a\x1b[31mred", "tab\there", "\x85next", "bad \ud83d surrogate"):
            with self.subTest(text=repr(text)):
                with self.assertRaises(bridge.BridgeError):
                    bridge.validate_chat_text(text)
        with self.assertRaises(bridge.BridgeError):
            bridge.validate_chat_text(None)  # type: ignore[arg-type]

    def test_osc_packet_exact_bytes(self):
        packet, *_ = bridge.build_osc_packet("hi")
        self.assertEqual(
            packet,
            b"/chatbox/input\0\0" + b",sFF\0\0\0\0" + b"hi\0\0",
        )

    def test_osc_padding_for_every_length_mod_four(self):
        for length in range(1, 9):
            with self.subTest(length=length):
                packet, *_ = bridge.build_osc_packet("x" * length)
                self.assertEqual(len(packet) % 4, 0)
                self.assertEqual(decode_osc(packet), [True, "/chatbox/input", ",sFF", "x" * length, False, False])

    def test_osc_unicode_round_trip(self):
        text = "Astra ✨ こんにちは\nñ 💖"
        packet, normalized, units, lines = bridge.build_osc_packet(text)
        self.assertEqual(decode_osc(packet)[3], text)
        self.assertIn("✨".encode("utf-8"), packet)
        self.assertEqual(lines, 2)
        self.assertEqual(units, len(text.encode("utf-16-le")) // 2)


class NoNetworkGuard:
    """Make any socket or urllib use fail loudly."""

    def __enter__(self):
        def refuse(*args, **kwargs):
            raise AssertionError("network used during preview")

        self.patches = [
            mock.patch.object(socket, "socket", refuse),
            mock.patch.object(socket, "create_connection", refuse),
            mock.patch.object(urllib.request.OpenerDirector, "open", refuse),
            mock.patch.object(urllib.request, "urlopen", refuse),
        ]
        for patch in self.patches:
            patch.start()
        return self

    def __exit__(self, *exc):
        for patch in reversed(self.patches):
            patch.stop()


class PreviewTests(unittest.TestCase):
    def test_preview_makes_no_network_calls(self):
        with NoNetworkGuard():
            result = bridge.preview_chat("Hello ✨")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = bridge.main(["preview", "Hello ✨"])
        self.assertEqual(code, 0)
        printed = json.loads(stdout.getvalue())
        self.assertEqual(printed, result)
        self.assertEqual(result["immediate_send"], False)
        self.assertEqual(result["notification"], False)
        self.assertEqual(result["type_tags"], ",sFF")
        self.assertEqual(bytes.fromhex(result["osc_packet_hex"]), bridge.build_osc_packet("Hello ✨")[0])

    def test_guard_actually_blocks(self):
        with NoNetworkGuard():
            with self.assertRaises(AssertionError):
                bridge.queue_chat("x", port=9)


class UdpStub:
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.settimeout(2)
        self.port = self.sock.getsockname()[1]
        assert self.port != LIVE_VRCHAT_PORT

    def receive(self):
        data, _ = self.sock.recvfrom(65535)
        return data

    def close(self):
        self.sock.close()


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.udp = UdpStub()
        self.addCleanup(self.udp.close)

    def test_queue_sends_exactly_one_draft_packet(self):
        sent = bridge.queue_chat("Queued ✨", port=self.udp.port)
        data = self.udp.receive()
        self.assertEqual(sent, len(data))
        self.assertEqual(data, bridge.build_osc_packet("Queued ✨")[0])
        self.assertEqual(decode_osc(data)[2:], [",sFF", "Queued ✨", False, False])
        self.udp.sock.settimeout(0.2)
        with self.assertRaises(socket.timeout):
            self.udp.receive()

    def test_queue_cli_with_stub_port(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = bridge.main(["queue", "CLI test", "--port", str(self.udp.port)])
        self.assertEqual(code, 0)
        self.assertEqual(decode_osc(self.udp.receive())[3], "CLI test")
        self.assertIn("no delivery receipt", stdout.getvalue())

    def test_queue_validation_sends_nothing(self):
        for text in ("a" * 145, "", "\n".join("abcdefghij")):
            with self.subTest(length=len(text)):
                with self.assertRaises(bridge.BridgeError):
                    bridge.queue_chat(text, port=self.udp.port)
        self.udp.sock.settimeout(0.2)
        with self.assertRaises(socket.timeout):
            self.udp.receive()

    def test_queue_rejects_bad_ports(self):
        for port in (0, 65536, -1, True, "9000"):
            with self.subTest(port=port):
                with self.assertRaises(bridge.BridgeError):
                    bridge.queue_chat("x", port=port)  # type: ignore[arg-type]

    def test_default_queue_port_is_vrchat_input(self):
        # Checked by value only; the tests never send to the live port.
        self.assertEqual(bridge.OSC_PORT, LIVE_VRCHAT_PORT)
        self.assertEqual(bridge.build_parser().parse_args(["queue", "x"]).port, LIVE_VRCHAT_PORT)


class CliTests(unittest.TestCase):
    def setUp(self):
        self.stub = StubMai()
        self.addCleanup(self.stub.close)

    def run_cli(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = bridge.main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_status(self):
        self.stub.behaviour = lambda h, b: (200, {"ok": True, "api_name": "mai", "secret": "x"}, None)
        code, out, _ = self.run_cli(["status", "--endpoint", self.stub.endpoint])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out), {"ok": True, "api_name": "mai"})

    def test_ask_escapes_terminal_controls_and_does_not_queue(self):
        self.stub.behaviour = echo_success("line1\n\x1b[2Jclear\ttab")
        with mock.patch.object(bridge, "queue_chat", side_effect=AssertionError("ask must not queue")):
            code, out, err = self.run_cli(["ask", "hi", "--endpoint", self.stub.endpoint])
        self.assertEqual(code, 0)
        self.assertEqual(out, "Mai: line1\n\\x1b[2Jclear\ttab\n")
        self.assertIn("Not sent to VRChat", err)

    def test_error_exit_code(self):
        self.stub.behaviour = lambda h, b: (500, {}, None)
        code, out, err = self.run_cli(["status", "--endpoint", self.stub.endpoint])
        self.assertEqual(code, 2)
        self.assertEqual(out, "")
        self.assertIn("HTTP 500", err)


class DependencyTests(unittest.TestCase):
    def test_bridge_imports_only_standard_library(self):
        tree = ast.parse((ROOT / "bridge.py").read_text(encoding="utf-8"))
        modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                modules.add(node.module.split(".")[0])
        stdlib = getattr(sys, "stdlib_module_names", None)
        if stdlib is None:
            self.skipTest("sys.stdlib_module_names needs Python 3.10+")
        self.assertEqual(sorted(modules - set(stdlib)), [])


if __name__ == "__main__":
    unittest.main()
