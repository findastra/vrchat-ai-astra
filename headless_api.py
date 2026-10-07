import argparse
import contextlib
import hmac
import io
import inspect
import ipaddress
import json
import math
import os
import sys
import threading
import time
import urllib.parse
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

EXCLUDED_REMOTE_METHODS = {
    'attach_brain',
    'blend_student_attention',
    'close_brain_instance',
    'create_clone_brain',
}

_RUNTIME_MODULE = None
MAX_HTTP_BODY_BYTES = 8 * 1024 * 1024


def _load_runtime_module():
    global _RUNTIME_MODULE
    if _RUNTIME_MODULE is None:
        try:
            from . import backend_runtime as runtime_module
        except ImportError:
            import backend_runtime as runtime_module
        _RUNTIME_MODULE = runtime_module
    return _RUNTIME_MODULE


def _get_app_dir() -> str:
    return str(getattr(_load_runtime_module(), 'APP_DIR'))


def _json_dump(payload) -> str:
    return json.dumps(_serialize_payload(payload), indent=2, ensure_ascii=True, allow_nan=False, default=str)


def _json_line_dump(payload) -> str:
    return json.dumps(
        _serialize_payload(payload),
        ensure_ascii=True,
        separators=(',', ':'),
        allow_nan=False,
        default=str,
    )


def _reject_json_constant(value: str):
    raise ValueError(f'Invalid JSON numeric constant: {value}')


def _json_loads(payload: str):
    return json.loads(payload, parse_constant=_reject_json_constant)


def _serialize_payload(payload):
    if hasattr(payload, 'to_dict') and callable(payload.to_dict):
        return _serialize_payload(payload.to_dict())
    if isinstance(payload, dict):
        return {str(key): _serialize_payload(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [_serialize_payload(item) for item in payload]
    if isinstance(payload, float):
        return payload if math.isfinite(payload) else None
    if isinstance(payload, (str, int, bool)) or payload is None:
        return payload
    return str(payload)


def _is_loopback_host(host: str) -> bool:
    normalized = str(host or '').strip().lower()
    if normalized == 'localhost':
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _error_payload(request_id, error_code: str, error: str, method: str | None = None, **details) -> dict:
    payload = {
        'id': request_id,
        'ok': False,
        'error_code': error_code,
        'error': error,
    }
    if method:
        payload['method'] = method
    for key, value in details.items():
        if value is not None:
            payload[key] = _serialize_payload(value)
    return payload


def _build_remote_method_map(api) -> dict[str, callable]:
    methods = {}
    transport_names = None
    get_transport_method_names = getattr(api, 'get_transport_method_names', None)
    if callable(get_transport_method_names):
        try:
            transport_names = list(get_transport_method_names())
        except Exception:
            transport_names = None

    candidate_names = transport_names or dir(api)
    for name in candidate_names:
        if name.startswith('_') or name in EXCLUDED_REMOTE_METHODS:
            continue
        candidate = getattr(api, name, None)
        if callable(candidate):
            methods[name] = candidate
    return methods


def _build_method_descriptions(methods: dict[str, callable]) -> dict[str, dict[str, object]]:
    descriptions = {}
    for name, method in methods.items():
        try:
            signature = str(inspect.signature(method))
        except (TypeError, ValueError):
            signature = "()"
        descriptions[name] = {
            'signature': signature,
            'doc': inspect.getdoc(method) or "",
        }
    return descriptions


def _extract_top_level_errors(payload) -> list[dict]:
    if isinstance(payload, dict):
        if payload.get('ok') is False:
            return [payload]
        return []
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict) and item.get('ok') is False]
    return []


def _create_api_with_startup_logs():
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        runtime_module = _load_runtime_module()
        api = runtime_module.create_headless_backend_api()
    startup_logs = [line.strip() for line in buffer.getvalue().splitlines() if line.strip()]
    return api, startup_logs


def _shutdown_api(api) -> None:
    runtime_module = _load_runtime_module()
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        runtime_module.shutdown_headless_backend_api(api)
    shutdown_logs = [line.strip() for line in buffer.getvalue().splitlines() if line.strip()]
    _write_stderr_lines(shutdown_logs)


def _write_stderr_lines(lines: list[str]) -> None:
    for line in lines:
        print(line, file=sys.stderr, flush=True)


def _emit_backend_logs(method_name: str, lines: list[str]) -> None:
    for line in lines:
        print(f"[backend:{method_name}] {line}", file=sys.stderr, flush=True)


class BackendServiceHost:
    def __init__(self, api, transport_name: str = 'in_process'):
        self.api = api
        self.lock = threading.RLock()
        self.transport_name = transport_name
        self.session_id = uuid.uuid4().hex
        self.started_at = time.time()
        self.process_id = os.getpid()
        self.methods = _build_remote_method_map(api)
        self.method_signatures: dict[str, inspect.Signature] = {}
        for name, method in self.methods.items():
            try:
                self.method_signatures[name] = inspect.signature(method)
            except (TypeError, ValueError):
                continue
        get_transport_method_specs = getattr(api, 'get_transport_method_specs', None)
        if callable(get_transport_method_specs):
            try:
                self.method_descriptions = _serialize_payload(get_transport_method_specs())
            except Exception:
                self.method_descriptions = _build_method_descriptions(self.methods)
        else:
            self.method_descriptions = _build_method_descriptions(self.methods)
        get_transport_control_specs = getattr(api, 'get_transport_control_specs', None)
        if callable(get_transport_control_specs):
            try:
                self.control_descriptions = _serialize_payload(get_transport_control_specs())
            except Exception:
                self.control_descriptions = {}
        else:
            self.control_descriptions = {}
        self.max_batch_size = int(getattr(api, 'api_config', {}).get('transport_max_batch_size', 32) or 32)

    def list_methods(self) -> dict[str, dict[str, object]]:
        return self.method_descriptions

    def list_controls(self) -> dict[str, dict[str, object]]:
        return self.control_descriptions

    def get_method_description(self, name: str) -> dict[str, object] | None:
        return self.method_descriptions.get(name) or self.control_descriptions.get(name)

    def get_session_info(self, transport: str | None = None) -> dict[str, object]:
        uptime_seconds = max(0.0, time.time() - self.started_at)
        return {
            'session_id': self.session_id,
            'process_id': self.process_id,
            'started_at_epoch': self.started_at,
            'uptime_seconds': round(uptime_seconds, 3),
            'stateful_process': True,
            'shared_backend_instance': True,
            'transport': transport or self.transport_name,
            'max_batch_size': self.max_batch_size,
            'method_count': len(self.methods),
            'control_method_count': len(self.control_descriptions),
        }

    def _handle_control_request(self, request_id, method_name: str, params) -> tuple[dict, bool]:
        if params is None:
            params = {}
        if not isinstance(params, dict):
            return _error_payload(
                request_id,
                'invalid_request',
                'params must be a JSON object.',
                method=method_name,
            ), False
        if params:
            return _error_payload(
                request_id,
                'invalid_params',
                f'{method_name} does not accept parameters.',
                method=method_name,
            ), False
        if method_name == 'shutdown':
            return {
                'id': request_id,
                'ok': True,
                'result': {'message': 'Backend shutdown requested.'},
            }, True
        if method_name == 'get_session_info':
            return {
                'id': request_id,
                'ok': True,
                'result': self.get_session_info(),
            }, False
        return _error_payload(
            request_id,
            'unknown_method',
            f'Unknown control method: {method_name}',
            method=method_name,
        ), False

    def _invoke_backend_method(self, method_name: str, method, params: dict) -> Any:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            result = method(**params)
        log_lines = [line.strip() for line in buffer.getvalue().splitlines() if line.strip()]
        if log_lines:
            _emit_backend_logs(method_name, log_lines)
        return result

    def handle_request(self, request: dict) -> dict:
        request_id = request.get('id')
        method_name = request.get('method')
        params = request.get('params', {})
        if not isinstance(method_name, str) or not method_name:
            return _error_payload(request_id, 'invalid_request', 'Request must include a non-empty method name.')
        method = self.methods.get(method_name)
        if method is None:
            return _error_payload(request_id, 'unknown_method', f'Unknown method: {method_name}', method=method_name)

        if params is None:
            params = {}
        if not isinstance(params, dict):
            return _error_payload(request_id, 'invalid_request', 'params must be a JSON object.', method=method_name)

        signature = self.method_signatures.get(method_name)
        if signature is not None:
            try:
                signature.bind(**params)
            except TypeError as e:
                return _error_payload(
                    request_id,
                    'invalid_params',
                    f'Invalid parameters for {method_name}: {e}',
                    method=method_name,
                    signature=self.method_descriptions.get(method_name, {}).get('signature'),
                )

        try:
            with self.lock:
                result = self._invoke_backend_method(method_name, method, params)
            return {'id': request_id, 'ok': True, 'result': _serialize_payload(result)}
        except (TypeError, ValueError) as e:
            return _error_payload(
                request_id,
                'invalid_params',
                f'Invalid parameters for {method_name}: {e}',
                method=method_name,
                exception_type=type(e).__name__,
            )
        except Exception as e:
            return _error_payload(
                request_id,
                'method_failed',
                f'{method_name} failed: {e}',
                method=method_name,
                exception_type=type(e).__name__,
            )

    def handle_payload(self, payload) -> tuple[dict, bool]:
        if isinstance(payload, dict):
            if payload.get('method') in {'shutdown', 'get_session_info'}:
                return self._handle_control_request(payload.get('id'), payload.get('method'), payload.get('params', {}))
            return self.handle_request(payload), False

        if isinstance(payload, list):
            if not payload:
                return _error_payload(None, 'invalid_request', 'Batch requests must be a non-empty JSON array.'), False
            if len(payload) > self.max_batch_size:
                return _error_payload(
                    None,
                    'invalid_request',
                    f'Batch requests may contain at most {self.max_batch_size} items.',
                    max_items=self.max_batch_size,
                ), False

            results = []
            should_shutdown = False
            for item in payload:
                if not isinstance(item, dict):
                    results.append(_error_payload(None, 'invalid_request', 'Each batch item must be a JSON object.'))
                    continue
                if item.get('method') in {'shutdown', 'get_session_info'}:
                    control_response, control_should_shutdown = self._handle_control_request(
                        item.get('id'),
                        item.get('method'),
                        item.get('params', {}),
                    )
                    if control_should_shutdown and isinstance(control_response, dict) and control_response.get('ok'):
                        control_response = {
                            'id': item.get('id'),
                            'ok': True,
                            'result': {'message': 'Backend shutdown queued after batch.'},
                        }
                    should_shutdown = should_shutdown or control_should_shutdown
                    results.append(control_response)
                    continue
                results.append(self.handle_request(item))

            error_count = sum(1 for item in results if isinstance(item, dict) and item.get('ok') is False)
            return {
                'ok': True,
                'batch': True,
                'item_count': len(results),
                'error_count': error_count,
                'success_count': len(results) - error_count,
                'has_errors': error_count > 0,
                'results': results,
            }, should_shutdown

        return _error_payload(
            None,
            'invalid_request',
            'Request body must be a JSON object or a non-empty JSON array.',
        ), False


def cmd_status(args) -> int:
    api, startup_logs = _create_api_with_startup_logs()
    try:
        _write_stderr_lines(startup_logs)
        snapshot = api.get_runtime_bootstrap_snapshot()
        print(_json_dump(snapshot))
        return 0
    finally:
        _shutdown_api(api)


def cmd_manifest(args) -> int:
    api, startup_logs = _create_api_with_startup_logs()
    try:
        _write_stderr_lines(startup_logs)
        manifest = api.get_api_manifest()
        print(_json_dump(manifest))
        return 0
    finally:
        _shutdown_api(api)


def cmd_method(args) -> int:
    api, startup_logs = _create_api_with_startup_logs()
    try:
        _write_stderr_lines(startup_logs)
        try:
            payload = api.get_transport_method_spec(args.name)
            print(_json_dump(payload))
            return 0
        except ValueError as e:
            print(_json_dump(_error_payload(None, 'unknown_method', f'Unknown method: {args.name}', method=args.name, exception_type=type(e).__name__)))
            return 1
        except Exception as e:
            print(_json_dump(_error_payload(None, 'method_failed', f'Could not describe method: {args.name}', method=args.name, exception_type=type(e).__name__)))
            return 1
    finally:
        _shutdown_api(api)


def cmd_session(args) -> int:
    api, startup_logs = _create_api_with_startup_logs()
    try:
        _write_stderr_lines(startup_logs)
        host = BackendServiceHost(api, transport_name='cli')
        print(_json_dump(host.get_session_info()))
        return 0
    finally:
        _shutdown_api(api)


def cmd_generate(args) -> int:
    api, startup_logs = _create_api_with_startup_logs()
    try:
        _write_stderr_lines(startup_logs)
        result = api.generate_response(args.prompt)
        payload = result.to_dict() if hasattr(result, 'to_dict') else result
        print(_json_dump(payload))
        return 0
    finally:
        _shutdown_api(api)


def cmd_train(args) -> int:
    api, startup_logs = _create_api_with_startup_logs()
    try:
        _write_stderr_lines(startup_logs)
        discovery = api.collect_training_files(args.paths)
        for file_path in discovery.get('paths', []):
            plan = api.get_training_plan(file_path)
            if plan.get('success'):
                print(f"Training {file_path}", file=sys.stderr)
                print(
                    f"  chunk_size={int(plan.get('chunk_size', 0) or 0):,} "
                    f"estimated_chunks={int(plan.get('estimated_chunks', 0) or 0):,}",
                    file=sys.stderr,
                )
        payload = api.train_files(args.paths)
        print(_json_dump(payload))
        return 0 if payload.get('success') else 1
    finally:
        _shutdown_api(api)


def cmd_gated_learn(args) -> int:
    response_text = str(args.response or '').strip()
    if args.response_file:
        response_text = Path(args.response_file).read_text(encoding='utf-8')
    user_text = str(args.user or '').strip()
    try:
        from evaluation_harness import assess_learning_hygiene
    except ImportError:
        from maimain.evaluation_harness import assess_learning_hygiene
    early_hygiene = assess_learning_hygiene(response_text or user_text, user_text)
    if not early_hygiene.get('accept', False) and (response_text or user_text):
        payload = {
            'success': True,
            'accepted': False,
            'committed': False,
            'reason': f"hygiene_{early_hygiene.get('reason', 'reject')}",
            'hygiene': early_hygiene,
            'duration_seconds': 0.0,
        }
        print(_json_dump(payload))
        return 2

    api, startup_logs = _create_api_with_startup_logs()
    try:
        _write_stderr_lines(startup_logs)
        payload = api.propose_gated_learning(
            response_text=response_text,
            user_text=user_text,
            commit=not bool(args.dry_run),
            priority_boost=int(args.priority_boost or 2),
            seed=int(args.seed or 1729),
            prompts_path=str(args.prompts or ''),
            baseline_path=str(args.baseline or ''),
            require_absolute_pass=not bool(args.skip_absolute_pass),
        )
        print(_json_dump(payload))
        if not payload.get('success'):
            return 1
        return 0 if payload.get('accepted') else 2
    finally:
        _shutdown_api(api)


def cmd_migrate_vocab(args) -> int:
    api, startup_logs = _create_api_with_startup_logs()
    try:
        _write_stderr_lines(startup_logs)
        batch_size = max(1, int(args.batch_size or 5000))
        max_batches = max(1, int(args.max_batches or 1))
        offset = max(0, int(args.offset_rowid or 0))
        batches = []
        done = False
        for _ in range(max_batches):
            payload = api.migrate_vocab_id_storage(batch_size=batch_size, offset_rowid=offset)
            batches.append(payload)
            if not payload.get('success'):
                print(_json_dump({
                    'success': False,
                    'batches': batches,
                    'snapshot': api.get_vocab_id_storage_snapshot(),
                }))
                return 1
            offset = int(payload.get('next_rowid', offset) or offset)
            done = bool(payload.get('done'))
            if done:
                break
        print(_json_dump({
            'success': True,
            'done': done,
            'next_rowid': offset,
            'batch_count': len(batches),
            'migrated_total': sum(int(item.get('migrated', 0) or 0) for item in batches),
            'batches': batches,
            'snapshot': api.get_vocab_id_storage_snapshot(),
        }))
        return 0
    finally:
        _shutdown_api(api)


def cmd_serve_stdio(args) -> int:
    api, startup_logs = _create_api_with_startup_logs()
    host = BackendServiceHost(api, transport_name='stdio')
    manifest = _serialize_payload(api.get_api_manifest())
    print(_json_line_dump({
        'ok': True,
        'event': 'ready',
        'transport': 'stdio',
        'app_dir': _get_app_dir(),
        'startup_logs': startup_logs,
        'session': host.get_session_info(),
        'manifest': manifest,
        'methods': host.list_methods(),
        'controls': host.list_controls(),
    }), flush=True)
    try:
        for raw_line in sys.stdin:
            line = raw_line.strip()
            if not line:
                continue
            try:
                request = _json_loads(line)
            except (json.JSONDecodeError, ValueError) as e:
                print(_json_line_dump(_error_payload(None, 'invalid_json', f'Invalid JSON: {e}')), flush=True)
                continue
            response, should_shutdown = host.handle_payload(request)
            print(_json_line_dump(response), flush=True)
            if should_shutdown:
                return 0
        return 0
    finally:
        _shutdown_api(api)


def cmd_serve_http(args) -> int:
    bind_host = str(args.host or '').strip()
    auth_token = str(args.auth_token or '').strip()
    max_body_bytes = max(1024, int(args.max_body_bytes or MAX_HTTP_BODY_BYTES))
    if not _is_loopback_host(bind_host) and not auth_token:
        print(
            'Refusing to expose the backend on a non-loopback interface without an authentication token.',
            file=sys.stderr,
        )
        return 2

    api, startup_logs = _create_api_with_startup_logs()
    host = BackendServiceHost(api, transport_name='http')
    port = int(args.port)

    class Handler(BaseHTTPRequestHandler):
        server_version = "AIAstraBackendHTTP/1.0"
        sys_version = ""

        def _is_authorized(self) -> bool:
            if not auth_token:
                return not bool(self.headers.get('Origin'))
            supplied_token = str(self.headers.get('X-AIAstra-Token', '') or '')
            return bool(supplied_token) and hmac.compare_digest(supplied_token, auth_token)

        def _require_authorization(self) -> bool:
            if self._is_authorized():
                return True
            self._send_json(403, _error_payload(None, 'forbidden', 'Backend authentication failed.'))
            return False

        def _send_json(self, status_code: int, payload):
            body = _json_dump(payload).encode('utf-8')
            self.send_response(status_code)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):
            self._send_json(403, _error_payload(None, 'forbidden', 'Browser cross-origin access is disabled.'))

        def do_GET(self):
            if not self._require_authorization():
                return
            parsed = urllib.parse.urlsplit(self.path)
            path = parsed.path
            if path == '/health':
                manifest = _serialize_payload(api.get_api_manifest())
                self._send_json(200, {
                    'ok': True,
                    'transport': 'http',
                    'app_dir': _get_app_dir(),
                    'api_name': manifest.get('name'),
                    'api_version': manifest.get('api_version'),
                    'startup_logs': startup_logs,
                    'session': host.get_session_info(),
                })
                return
            if path == '/session':
                self._send_json(200, {'ok': True, 'session': host.get_session_info()})
                return
            if path == '/manifest':
                self._send_json(200, {'ok': True, 'manifest': _serialize_payload(api.get_api_manifest())})
                return
            if path == '/methods':
                self._send_json(200, {'ok': True, 'methods': host.list_methods(), 'controls': host.list_controls()})
                return
            if path.startswith('/methods/'):
                method_name = urllib.parse.unquote(path[len('/methods/'):])
                if not method_name:
                    self._send_json(404, _error_payload(None, 'unknown_method', 'Unknown method: ', method=''))
                    return
                method_description = host.get_method_description(method_name)
                if method_description is None:
                    self._send_json(404, _error_payload(None, 'unknown_method', f'Unknown method: {method_name}', method=method_name))
                    return
                self._send_json(200, {'ok': True, 'method': method_name, 'spec': method_description})
                return
            self._send_json(404, _error_payload(None, 'not_found', f'Unknown route: {path}', path=path))

        def do_POST(self):
            if not self._require_authorization():
                return
            parsed = urllib.parse.urlsplit(self.path)
            path = parsed.path
            if path not in ('/api', '/api/batch'):
                self._send_json(404, _error_payload(None, 'not_found', f'Unknown route: {path}', path=path))
                return
            raw_content_length = self.headers.get('Content-Length')
            if raw_content_length is None:
                self._send_json(411, _error_payload(None, 'length_required', 'Content-Length is required.'))
                return
            try:
                content_length = int(raw_content_length)
            except (TypeError, ValueError):
                self._send_json(400, _error_payload(None, 'invalid_content_length', 'Content-Length must be an integer.'))
                return
            if content_length < 0:
                self._send_json(400, _error_payload(None, 'invalid_content_length', 'Content-Length cannot be negative.'))
                return
            if content_length > max_body_bytes:
                self.close_connection = True
                self._send_json(413, _error_payload(
                    None,
                    'request_too_large',
                    f'Request body exceeds the {max_body_bytes}-byte limit.',
                    max_body_bytes=max_body_bytes,
                ))
                return
            raw_body = self.rfile.read(content_length) if content_length > 0 else b'{}'
            try:
                request = _json_loads(raw_body.decode('utf-8'))
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as e:
                self._send_json(400, _error_payload(None, 'invalid_json', f'Invalid JSON: {e}'))
                return
            response, should_shutdown = host.handle_payload(request)
            top_level_errors = _extract_top_level_errors(response)
            status_code = 200 if not top_level_errors else 400
            self._send_json(status_code, response)
            if should_shutdown:
                threading.Thread(target=self.server.shutdown, daemon=True).start()

        def log_message(self, format, *args):
            print(f"[http] {self.address_string()} - {format % args}")

    server = ThreadingHTTPServer((bind_host, port), Handler)
    print(f"AI Astra backend HTTP service listening on http://{bind_host}:{port}", flush=True)
    try:
        server.serve_forever()
        return 0
    finally:
        server.server_close()
        _shutdown_api(api)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Headless runner for AI Astra's in-process backend API.",
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    status_parser = subparsers.add_parser('status', help='Print a headless backend bootstrap snapshot.')
    status_parser.set_defaults(func=cmd_status)

    manifest_parser = subparsers.add_parser('manifest', help='Print the backend API manifest and transport contract.')
    manifest_parser.set_defaults(func=cmd_manifest)

    method_parser = subparsers.add_parser('method', help='Print the contract metadata for one backend API method.')
    method_parser.add_argument('name', help='Backend method name to describe.')
    method_parser.set_defaults(func=cmd_method)

    session_parser = subparsers.add_parser('session', help='Print transport session metadata for the current headless backend process.')
    session_parser.set_defaults(func=cmd_session)

    generate_parser = subparsers.add_parser('generate', help='Generate a response without launching the GUI.')
    generate_parser.add_argument('prompt', nargs='?', default='', help='Prompt to send to the backend.')
    generate_parser.set_defaults(func=cmd_generate)

    train_parser = subparsers.add_parser('train', help='Train the backend on files or directories without launching the GUI.')
    train_parser.add_argument('paths', nargs='+', help='Text files or directories to train on.')
    train_parser.set_defaults(func=cmd_train)

    gated_parser = subparsers.add_parser(
        'gated-learn',
        help='Trial-learn text on a clone, re-run the held-out gate, and commit only if behavior does not regress.',
    )
    gated_parser.add_argument('--user', default='', help='Optional user/context text for the learning pair.')
    gated_parser.add_argument('--response', default='', help='Candidate response text to learn from.')
    gated_parser.add_argument('--response-file', help='Read candidate response text from a UTF-8 file.')
    gated_parser.add_argument('--dry-run', action='store_true', help='Evaluate the gate without committing to the live brain.')
    gated_parser.add_argument('--priority-boost', type=int, default=2, help='Learning priority boost. Defaults to 2.')
    gated_parser.add_argument('--seed', type=int, default=1729, help='Deterministic evaluation seed.')
    gated_parser.add_argument('--prompts', help='Optional gate prompt JSON path. Defaults to evaluation_prompts_gate.json.')
    gated_parser.add_argument('--baseline', help='Optional baseline JSON path. Defaults to evaluation_baseline_generative.json.')
    gated_parser.add_argument(
        '--skip-absolute-pass',
        action='store_true',
        help='Skip absolute gate thresholds and only require no pre/post or baseline regression.',
    )
    gated_parser.set_defaults(func=cmd_gated_learn)

    migrate_parser = subparsers.add_parser(
        'migrate-vocab',
        help='Copy TEXT n-gram chains into the compact vocabulary-ID table in batches.',
    )
    migrate_parser.add_argument('--batch-size', type=int, default=5000, help='Rows per migration batch. Defaults to 5000.')
    migrate_parser.add_argument('--max-batches', type=int, default=1, help='Maximum batches to run in this invocation.')
    migrate_parser.add_argument('--offset-rowid', type=int, default=0, help='Continue from this TEXT-table rowid.')
    migrate_parser.set_defaults(func=cmd_migrate_vocab)

    stdio_parser = subparsers.add_parser('serve-stdio', help='Run the backend as a persistent stdio JSON service.')
    stdio_parser.set_defaults(func=cmd_serve_stdio)

    http_parser = subparsers.add_parser('serve-http', help='Run the backend as a persistent localhost HTTP JSON service.')
    http_parser.add_argument('--host', default='127.0.0.1', help='Host/interface to bind. Defaults to 127.0.0.1.')
    http_parser.add_argument('--port', type=int, default=8765, help='Port to bind. Defaults to 8765.')
    http_parser.add_argument(
        '--auth-token',
        default=os.environ.get('AI_ASTRA_BACKEND_AUTH_TOKEN', ''),
        help='Optional shared secret. Defaults to AI_ASTRA_BACKEND_AUTH_TOKEN.',
    )
    http_parser.add_argument(
        '--max-body-bytes',
        type=int,
        default=MAX_HTTP_BODY_BYTES,
        help=f'Maximum HTTP request body size. Defaults to {MAX_HTTP_BODY_BYTES} bytes.',
    )
    http_parser.set_defaults(func=cmd_serve_http)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == '__main__':
    raise SystemExit(main())
