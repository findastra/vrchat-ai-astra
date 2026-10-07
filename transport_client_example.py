import argparse
import json
import subprocess
import sys
import time
import urllib.request


KNOWLEDGE_SAMPLE_TEXT = 'Mai is a statistical intelligence. Mai is part of the sgm system.'


def _http_get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.loads(response.read().decode('utf-8'))


def _http_post(url: str, payload: dict, timeout: int = 60) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode('utf-8'))


def _wait_for_http(process: subprocess.Popen[str], port: int) -> None:
    ready = f"http://127.0.0.1:{port}"
    deadline = time.time() + 25
    while time.time() < deadline:
        line = process.stdout.readline()
        if line and ready in line:
            return
    raise RuntimeError("HTTP backend did not become ready.")


def _read_stdio_json(process: subprocess.Popen[str], timeout_seconds: int = 120) -> dict:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        line = process.stdout.readline()
        if not line:
            if process.poll() is not None:
                break
            continue
        stripped = line.strip()
        if not stripped:
            continue
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            continue
    raise RuntimeError("stdio backend did not return JSON.")


def _write_stdio_json(process: subprocess.Popen[str], payload: dict) -> None:
    if process.stdin is None:
        raise RuntimeError("stdio backend stdin is unavailable.")
    process.stdin.write(json.dumps(payload) + '\n')
    process.stdin.flush()


def run_http_example(port: int, prompt: str) -> dict:
    process = subprocess.Popen(
        [sys.executable, '-m', 'maimain.headless_api', 'serve-http', '--port', str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        _wait_for_http(process, port)
        base_url = f"http://127.0.0.1:{port}"
        session = _http_get(base_url + '/session')
        manifest = _http_get(base_url + '/manifest')
        methods = _http_get(base_url + '/methods')
        method_spec = _http_get(base_url + '/methods/generate_response')
        knowledge_train = _http_post(
            base_url + '/api',
            {
                'id': 3,
                'method': 'learn_from_training_chunk',
                'params': {
                    'chunk': KNOWLEDGE_SAMPLE_TEXT,
                    'source_label': 'transport_client_example',
                    'source_category': 'reference_docs',
                    'source_weight': 1.15,
                },
            },
        )
        knowledge_snapshot = _http_post(
            base_url + '/api',
            {
                'id': 4,
                'method': 'get_knowledge_snapshot',
                'params': {'limit': 4},
            },
        )
        response_plan = _http_post(
            base_url + '/api',
            {
                'id': 5,
                'method': 'get_response_plan_preview',
                'params': {'user_input': 'what is mai?', 'limit': 3},
            },
        )
        hardware_profile = _http_post(
            base_url + '/api',
            {
                'id': 6,
                'method': 'get_hardware_profile',
                'params': {},
            },
        )
        response = _http_post(
            base_url + '/api',
            [
                {'id': 1, 'method': 'get_runtime_bootstrap_snapshot', 'params': {}},
                {'id': 2, 'method': 'generate_response', 'params': {'user_input': prompt}},
            ],
            timeout=300,
        )
        _http_post(base_url + '/api', {'id': 2, 'method': 'shutdown', 'params': {}})
        results = response.get('results', [])
        bootstrap = results[0].get('result', {}) if len(results) > 0 else {}
        generation = results[1].get('result', {}) if len(results) > 1 else {}
        return {
            'transport': 'http',
            'manifest_method_count': manifest.get('manifest', {}).get('transport', {}).get('method_count'),
            'manifest_categories': sorted((manifest.get('manifest', {}).get('transport', {}) or {}).get('method_categories', {}).keys()),
            'manifest_workflows': sorted((manifest.get('manifest', {}).get('transport', {}) or {}).get('frontend_workflows', {}).keys()),
            'session_transport': (session.get('session', {}) or {}).get('transport'),
            'control_method_count': len(methods.get('controls', {}) or {}),
            'method_detail_summary': (method_spec.get('spec', {}) or {}).get('summary'),
            'knowledge_train_ok': knowledge_train.get('ok'),
            'knowledge_fact_count': ((knowledge_snapshot.get('result', {}) or {}).get('fact_count')),
            'response_plan_intent': ((response_plan.get('result', {}) or {}).get('intent')),
            'hardware_tier': ((hardware_profile.get('result', {}) or {}).get('tier')),
            'hardware_recommended_workers': ((hardware_profile.get('result', {}) or {}).get('recommended_parallel_workers')),
            'hardware_gpu_count': ((hardware_profile.get('result', {}) or {}).get('total_gpus')),
            'batch_ok': response.get('ok'),
            'bootstrap_storage_mode': (bootstrap.get('storage', {}) or {}).get('mode'),
            'bootstrap_hardware_tier': (bootstrap.get('hardware', {}) or {}).get('tier'),
            'response': generation.get('response', ''),
        }
    finally:
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def run_stdio_example(prompt: str) -> dict:
    process = subprocess.Popen(
        [sys.executable, '-m', 'maimain.headless_api', 'serve-stdio'],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.PIPE,
        text=True,
    )
    try:
        ready = _read_stdio_json(process)
        _write_stdio_json(
            process,
            {'id': 9, 'method': 'get_session_info', 'params': {}},
        )
        session = _read_stdio_json(process)
        _write_stdio_json(
            process,
            {'id': 10, 'method': 'get_transport_method_spec', 'params': {'name': 'generate_response'}},
        )
        method_spec = _read_stdio_json(process)
        _write_stdio_json(
            process,
            {
                'id': 11,
                'method': 'learn_from_training_chunk',
                'params': {
                    'chunk': KNOWLEDGE_SAMPLE_TEXT,
                    'source_label': 'transport_client_example',
                    'source_category': 'reference_docs',
                    'source_weight': 1.15,
                },
            },
        )
        knowledge_train = _read_stdio_json(process)
        _write_stdio_json(
            process,
            {'id': 12, 'method': 'get_knowledge_snapshot', 'params': {'limit': 4}},
        )
        knowledge_snapshot = _read_stdio_json(process)
        _write_stdio_json(
            process,
            {'id': 13, 'method': 'get_response_plan_preview', 'params': {'user_input': 'what is mai?', 'limit': 3}},
        )
        response_plan = _read_stdio_json(process)
        _write_stdio_json(
            process,
            {'id': 14, 'method': 'get_hardware_profile', 'params': {}},
        )
        hardware_profile = _read_stdio_json(process)
        _write_stdio_json(
            process,
            [
                {'id': 1, 'method': 'get_runtime_bootstrap_snapshot', 'params': {}},
                {'id': 2, 'method': 'generate_response', 'params': {'user_input': prompt}},
            ],
        )
        response = _read_stdio_json(process, timeout_seconds=300)
        _write_stdio_json(process, {'id': 3, 'method': 'shutdown', 'params': {}})
        _read_stdio_json(process)
        results = response.get('results', [])
        bootstrap = results[0].get('result', {}) if len(results) > 0 else {}
        generation = results[1].get('result', {}) if len(results) > 1 else {}
        return {
            'transport': 'stdio',
            'manifest_method_count': (ready.get('manifest', {}).get('transport', {}) or {}).get('method_count'),
            'manifest_categories': sorted((ready.get('manifest', {}).get('transport', {}) or {}).get('method_categories', {}).keys()),
            'manifest_workflows': sorted((ready.get('manifest', {}).get('transport', {}) or {}).get('frontend_workflows', {}).keys()),
            'session_transport': ((session.get('result', {}) or {}).get('transport')),
            'control_method_count': len(ready.get('controls', {}) or {}),
            'method_detail_summary': ((method_spec.get('result', {}) or {}).get('spec', {}) or {}).get('summary'),
            'knowledge_train_ok': knowledge_train.get('ok'),
            'knowledge_fact_count': (((knowledge_snapshot.get('result', {}) or {}).get('fact_count'))),
            'response_plan_intent': (((response_plan.get('result', {}) or {}).get('intent'))),
            'hardware_tier': (((hardware_profile.get('result', {}) or {}).get('tier'))),
            'hardware_recommended_workers': (((hardware_profile.get('result', {}) or {}).get('recommended_parallel_workers'))),
            'hardware_gpu_count': (((hardware_profile.get('result', {}) or {}).get('total_gpus'))),
            'batch_ok': response.get('ok'),
            'bootstrap_storage_mode': (bootstrap.get('storage', {}) or {}).get('mode'),
            'bootstrap_hardware_tier': (bootstrap.get('hardware', {}) or {}).get('tier'),
            'response': generation.get('response', ''),
        }
    finally:
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Example client for Mai's headless transports.")
    parser.add_argument('--transport', choices=('http', 'stdio'), default='stdio')
    parser.add_argument('--port', type=int, default=8773)
    parser.add_argument('--prompt', default='what is the sgm model?')
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.transport == 'http':
        result = run_http_example(args.port, args.prompt)
    else:
        result = run_stdio_example(args.prompt)
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
