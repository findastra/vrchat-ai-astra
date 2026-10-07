import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request


KNOWLEDGE_SAMPLE_TEXT = 'Mai is a statistical intelligence. Mai is part of the sgm system.'


def _get_json(url: str) -> dict:
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as exc:
        return json.loads(exc.read().decode('utf-8'))


def _post_json(url: str, payload: dict, timeout: int = 120) -> dict:
    data = json.dumps(payload).encode('utf-8')
    request = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as exc:
        return json.loads(exc.read().decode('utf-8'))


def _wait_for_http_ready(process: subprocess.Popen[str], port: int, timeout_seconds: int = 25) -> list[str]:
    lines: list[str] = []
    ready_text = f"http://127.0.0.1:{port}"
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        line = process.stdout.readline()
        if not line:
            if process.poll() is not None:
                break
            continue
        stripped = line.rstrip()
        lines.append(stripped)
        if ready_text in stripped:
            return lines
    raise RuntimeError(f"HTTP service did not become ready. Output: {' | '.join(lines[-10:])}")


def _read_json_line(process: subprocess.Popen[str], timeout_seconds: int = 25) -> dict:
    seen_output: list[str] = []
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
            seen_output.append(stripped)
            continue
    raise RuntimeError(f"No JSON response received from stdio backend. Output: {' | '.join(seen_output[-10:])}")


def _write_json_line(process: subprocess.Popen[str], payload: dict) -> None:
    if process.stdin is None:
        raise RuntimeError("stdio backend stdin is not available.")
    process.stdin.write(json.dumps(payload, ensure_ascii=True) + '\n')
    process.stdin.flush()


def run_http_smoke_test(port: int, prompt: str) -> dict:
    command = [sys.executable, '-m', 'maimain.headless_api', 'serve-http', '--port', str(port)]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        startup_output = _wait_for_http_ready(process, port)
        base_url = f"http://127.0.0.1:{port}"
        health = _get_json(base_url + '/health?probe=1')
        session = _get_json(base_url + '/session')
        manifest = _get_json(base_url + '/manifest')
        methods = _get_json(base_url + '/methods?include=controls')
        method_detail = _get_json(base_url + '/methods/generate_response?view=full')
        control_detail = _get_json(base_url + '/methods/get_session_info')
        not_found = _get_json(base_url + '/definitely-not-a-route')
        invalid_control = _post_json(
            base_url + '/api',
            {
                'id': 6,
                'method': 'get_session_info',
                'params': 'oops',
            },
        )
        knowledge_train = _post_json(
            base_url + '/api',
            {
                'id': 7,
                'method': 'learn_from_training_chunk',
                'params': {
                    'chunk': KNOWLEDGE_SAMPLE_TEXT,
                    'source_label': 'transport_smoke',
                    'source_category': 'reference_docs',
                    'source_weight': 1.15,
                },
            },
            timeout=60,
        )
        knowledge_snapshot = _post_json(
            base_url + '/api',
            {
                'id': 8,
                'method': 'get_knowledge_snapshot',
                'params': {'limit': 4},
            },
        )
        knowledge_facts = _post_json(
            base_url + '/api',
            {
                'id': 9,
                'method': 'get_knowledge_facts',
                'params': {'query': 'mai', 'limit': 4},
            },
        )
        identity_traits = _post_json(
            base_url + '/api',
            {
                'id': 10,
                'method': 'get_knowledge_identity_traits',
                'params': {'limit': 4},
            },
        )
        response_plan = _post_json(
            base_url + '/api',
            {
                'id': 11,
                'method': 'get_response_plan_preview',
                'params': {'user_input': 'what is mai?', 'limit': 3},
            },
        )
        reasoning_preview = _post_json(
            base_url + '/api',
            {
                'id': 111,
                'method': 'get_graph_reasoning_preview',
                'params': {'user_input': 'why is mai part of the sgm system?', 'limit': 2, 'max_depth': 2},
            },
        )
        hardware_profile = _post_json(
            base_url + '/api',
            {
                'id': 12,
                'method': 'get_hardware_profile',
                'params': {},
            },
        )
        learning_health = _post_json(
            base_url + '/api',
            {
                'id': 18,
                'method': 'get_learning_health_snapshot',
                'params': {},
            },
        )
        generate = _post_json(
            base_url + '/api',
            [
                {
                    'id': 1,
                    'method': 'get_runtime_bootstrap_snapshot',
                    'params': {},
                },
                {
                    'id': 2,
                    'method': 'generate_response',
                    'params': {'user_input': prompt},
                },
                {
                    'id': 21,
                    'method': 'generate_response',
                    'params': {'user_input': 'how does memory replay help mai learn over time?'},
                },
            ],
        )
        invalid_params = _post_json(
            base_url + '/api',
            {
                'id': 4,
                'method': 'generate_correction_response',
                'params': {},
            },
        )
        method_failed = _post_json(
            base_url + '/api',
            {
                'id': 5,
                'method': 'get_transport_method_spec',
                'params': {'name': 'not_a_real_method'},
            },
        )
        shutdown = _post_json(
            base_url + '/api',
            {
                'id': 3,
                'method': 'shutdown',
                'params': {},
            },
        )
        batch_results = generate.get('results', [])
        bootstrap = batch_results[0].get('result', {}) if len(batch_results) > 0 else {}
        generation = batch_results[1].get('result', {}) if len(batch_results) > 1 else {}
        open_generation = batch_results[2].get('result', {}) if len(batch_results) > 2 else {}
        return {
            'success': True,
            'startup_output': startup_output[-10:],
            'health': health,
            'session_ok': session.get('ok'),
            'session_transport': (session.get('session', {}) or {}).get('transport'),
            'manifest_method_count': manifest.get('manifest', {}).get('transport', {}).get('method_count'),
            'manifest_category_count': len((manifest.get('manifest', {}).get('transport', {}) or {}).get('method_categories', {})),
            'manifest_workflow_count': len((manifest.get('manifest', {}).get('transport', {}) or {}).get('frontend_workflows', {})),
            'manifest_batch_route': (manifest.get('manifest', {}).get('transport', {}).get('http_routes', {}) or {}).get('batch'),
            'manifest_session_route': (manifest.get('manifest', {}).get('transport', {}).get('http_routes', {}) or {}).get('session'),
            'manifest_session_control_method': (manifest.get('manifest', {}).get('transport', {}).get('session_model', {}) or {}).get('session_control_method'),
            'methods_count': len(methods.get('methods', {})),
            'controls_count': len(methods.get('controls', {})),
            'method_detail_ok': method_detail.get('ok'),
            'method_detail_summary': (method_detail.get('spec', {}) or {}).get('summary'),
            'control_detail_ok': control_detail.get('ok'),
            'control_detail_summary': (control_detail.get('spec', {}) or {}).get('summary'),
            'not_found_code': not_found.get('error_code'),
            'invalid_control_code': invalid_control.get('error_code'),
            'knowledge_train_ok': knowledge_train.get('ok'),
            'knowledge_snapshot_ok': knowledge_snapshot.get('ok'),
            'knowledge_fact_count': ((knowledge_snapshot.get('result', {}) or {}).get('fact_count')),
            'knowledge_query_count': len((knowledge_facts.get('result', {}) or {}).get('rows', [])),
            'identity_trait_count': len((identity_traits.get('result', {}) or {}).get('rows', [])),
            'response_plan_ok': response_plan.get('ok'),
            'response_plan_intent': ((response_plan.get('result', {}) or {}).get('intent')),
            'reasoning_preview_ok': reasoning_preview.get('ok'),
            'reasoning_path_count': len((reasoning_preview.get('result', {}) or {}).get('paths', [])),
            'reasoning_mode': ((reasoning_preview.get('result', {}) or {}).get('mode')),
            'hardware_profile_ok': hardware_profile.get('ok'),
            'hardware_tier': ((hardware_profile.get('result', {}) or {}).get('tier')),
            'hardware_total_gpus': ((hardware_profile.get('result', {}) or {}).get('total_gpus')),
            'hardware_recommended_workers': ((hardware_profile.get('result', {}) or {}).get('recommended_parallel_workers')),
            'learning_health_ok': learning_health.get('ok'),
            'learning_health_mode': ((learning_health.get('result', {}) or {}).get('mode')),
            'learning_health_gate': ((learning_health.get('result', {}) or {}).get('recommended_min_quality')),
            'bootstrap_hardware_tier': (bootstrap.get('hardware', {}) or {}).get('tier'),
            'generate_ok': generate.get('ok'),
            'batch_mode': generate.get('batch'),
            'batch_error_count': generate.get('error_count'),
            'bootstrap_storage_mode': (bootstrap.get('storage', {}) or {}).get('mode'),
            'response_preview': generation.get('response', '')[:160],
            'open_realization_mode': (((open_generation.get('metadata', {}) or {}).get('realization_trace', {}) or {}).get('mode')),
            'open_response_preview': open_generation.get('response', '')[:160],
            'reasoning_trace_mode': (((generation.get('metadata', {}) or {}).get('reasoning_trace', {}) or {}).get('mode')),
            'invalid_params_code': invalid_params.get('error_code'),
            'method_failed_code': method_failed.get('error_code'),
            'shutdown_ok': shutdown.get('ok'),
        }
    finally:
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def run_stdio_smoke_test(prompt: str) -> dict:
    command = [sys.executable, '-m', 'maimain.headless_api', 'serve-stdio']
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.PIPE,
        text=True,
    )
    try:
        ready = _read_json_line(process)
        _write_json_line(
            process,
            {
                'id': 9,
                'method': 'get_session_info',
                'params': {},
            },
        )
        session = _read_json_line(process)
        _write_json_line(
            process,
            {
                'id': 11,
                'method': 'get_session_info',
                'params': 'oops',
            },
        )
        invalid_control = _read_json_line(process)
        _write_json_line(
            process,
            {
                'id': 10,
                'method': 'get_transport_method_spec',
                'params': {'name': 'generate_response'},
            },
        )
        method_detail = _read_json_line(process)
        _write_json_line(
            process,
            {
                'id': 12,
                'method': 'learn_from_training_chunk',
                'params': {
                    'chunk': KNOWLEDGE_SAMPLE_TEXT,
                    'source_label': 'transport_smoke',
                    'source_category': 'reference_docs',
                    'source_weight': 1.15,
                },
            },
        )
        knowledge_train = _read_json_line(process)
        _write_json_line(
            process,
            {
                'id': 13,
                'method': 'get_knowledge_snapshot',
                'params': {'limit': 4},
            },
        )
        knowledge_snapshot = _read_json_line(process)
        _write_json_line(
            process,
            {
                'id': 14,
                'method': 'get_knowledge_facts',
                'params': {'query': 'mai', 'limit': 4},
            },
        )
        knowledge_facts = _read_json_line(process)
        _write_json_line(
            process,
            {
                'id': 15,
                'method': 'get_knowledge_identity_traits',
                'params': {'limit': 4},
            },
        )
        identity_traits = _read_json_line(process)
        _write_json_line(
            process,
            {
                'id': 16,
                'method': 'get_response_plan_preview',
                'params': {'user_input': 'what is mai?', 'limit': 3},
            },
        )
        response_plan = _read_json_line(process)
        _write_json_line(
            process,
            {
                'id': 161,
                'method': 'get_graph_reasoning_preview',
                'params': {'user_input': 'why is mai part of the sgm system?', 'limit': 2, 'max_depth': 2},
            },
        )
        reasoning_preview = _read_json_line(process)
        _write_json_line(
            process,
            {
                'id': 17,
                'method': 'get_hardware_profile',
                'params': {},
            },
        )
        hardware_profile = _read_json_line(process)
        _write_json_line(
            process,
            {
                'id': 18,
                'method': 'get_learning_health_snapshot',
                'params': {},
            },
        )
        learning_health = _read_json_line(process)
        _write_json_line(
            process,
            [
                {
                    'id': 1,
                    'method': 'get_runtime_bootstrap_snapshot',
                    'params': {},
                },
                {
                    'id': 2,
                    'method': 'generate_response',
                    'params': {'user_input': prompt},
                },
                {
                    'id': 21,
                    'method': 'generate_response',
                    'params': {'user_input': 'how does memory replay help mai learn over time?'},
                },
            ],
        )
        generate = _read_json_line(process, timeout_seconds=300)
        _write_json_line(
            process,
            {
                'id': 4,
                'method': 'generate_correction_response',
                'params': {},
            },
        )
        invalid_params = _read_json_line(process)
        _write_json_line(
            process,
            {
                'id': 5,
                'method': 'get_transport_method_spec',
                'params': {'name': 'not_a_real_method'},
            },
        )
        method_failed = _read_json_line(process)
        _write_json_line(
            process,
            {
                'id': 3,
                'method': 'shutdown',
                'params': {},
            },
        )
        shutdown = _read_json_line(process)
        manifest = ready.get('manifest', {})
        methods = ready.get('methods', {})
        batch_results = generate.get('results', [])
        bootstrap = batch_results[0].get('result', {}) if len(batch_results) > 0 else {}
        generation = batch_results[1].get('result', {}) if len(batch_results) > 1 else {}
        open_generation = batch_results[2].get('result', {}) if len(batch_results) > 2 else {}
        return {
            'success': True,
            'ready_ok': ready.get('ok'),
            'transport': ready.get('transport'),
            'session_ok': session.get('ok'),
            'session_transport': ((session.get('result', {}) or {}).get('transport')),
            'manifest_method_count': (manifest.get('transport', {}) or {}).get('method_count'),
            'manifest_category_count': len((manifest.get('transport', {}) or {}).get('method_categories', {})),
            'manifest_workflow_count': len((manifest.get('transport', {}) or {}).get('frontend_workflows', {})),
            'manifest_batch_route': (manifest.get('transport', {}).get('http_routes', {}) or {}).get('batch'),
            'manifest_session_route': (manifest.get('transport', {}).get('http_routes', {}) or {}).get('session'),
            'manifest_session_control_method': (manifest.get('transport', {}).get('session_model', {}) or {}).get('session_control_method'),
            'methods_count': len(methods),
            'controls_count': len(ready.get('controls', {}) or {}),
            'method_detail_ok': method_detail.get('ok'),
            'method_detail_summary': ((method_detail.get('result', {}) or {}).get('spec', {}) or {}).get('summary'),
            'control_detail_ok': ((ready.get('controls', {}) or {}).get('get_session_info') is not None),
            'control_detail_summary': (((ready.get('controls', {}) or {}).get('get_session_info', {}) or {}).get('summary')),
            'invalid_control_code': invalid_control.get('error_code'),
            'knowledge_train_ok': knowledge_train.get('ok'),
            'knowledge_snapshot_ok': knowledge_snapshot.get('ok'),
            'knowledge_fact_count': (((knowledge_snapshot.get('result', {}) or {}).get('fact_count'))),
            'knowledge_query_count': len(((knowledge_facts.get('result', {}) or {}).get('rows', []))),
            'identity_trait_count': len(((identity_traits.get('result', {}) or {}).get('rows', []))),
            'response_plan_ok': response_plan.get('ok'),
            'response_plan_intent': (((response_plan.get('result', {}) or {}).get('intent'))),
            'reasoning_preview_ok': reasoning_preview.get('ok'),
            'reasoning_path_count': len(((reasoning_preview.get('result', {}) or {}).get('paths', []))),
            'reasoning_mode': (((reasoning_preview.get('result', {}) or {}).get('mode'))),
            'hardware_profile_ok': hardware_profile.get('ok'),
            'hardware_tier': (((hardware_profile.get('result', {}) or {}).get('tier'))),
            'hardware_total_gpus': (((hardware_profile.get('result', {}) or {}).get('total_gpus'))),
            'hardware_recommended_workers': (((hardware_profile.get('result', {}) or {}).get('recommended_parallel_workers'))),
            'learning_health_ok': learning_health.get('ok'),
            'learning_health_mode': (((learning_health.get('result', {}) or {}).get('mode'))),
            'learning_health_gate': (((learning_health.get('result', {}) or {}).get('recommended_min_quality'))),
            'bootstrap_hardware_tier': (bootstrap.get('hardware', {}) or {}).get('tier'),
            'generate_ok': generate.get('ok'),
            'batch_mode': generate.get('batch'),
            'batch_error_count': generate.get('error_count'),
            'bootstrap_storage_mode': (bootstrap.get('storage', {}) or {}).get('mode'),
            'response_preview': generation.get('response', '')[:160],
            'open_realization_mode': ((((open_generation.get('metadata', {}) or {}).get('realization_trace', {}) or {}).get('mode'))),
            'open_response_preview': open_generation.get('response', '')[:160],
            'reasoning_trace_mode': ((((generation.get('metadata', {}) or {}).get('reasoning_trace', {}) or {}).get('mode'))),
            'invalid_params_code': invalid_params.get('error_code'),
            'method_failed_code': method_failed.get('error_code'),
            'shutdown_ok': shutdown.get('ok'),
        }
    finally:
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-test Mai's headless transports.")
    parser.add_argument('--port', type=int, default=8771, help='HTTP port to use for the temporary backend.')
    parser.add_argument(
        '--transport',
        choices=('http', 'stdio', 'both'),
        default='both',
        help='Which transport to verify.',
    )
    parser.add_argument(
        '--prompt',
        default='what is the sgm model?',
        help='Prompt to use for the generation check.',
    )
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.transport == 'http':
        result = {'http': run_http_smoke_test(args.port, args.prompt)}
    elif args.transport == 'stdio':
        result = {'stdio': run_stdio_smoke_test(args.prompt)}
    else:
        result = {
            'http': run_http_smoke_test(args.port, args.prompt),
            'stdio': run_stdio_smoke_test(args.prompt),
        }
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0 if all(bool(item.get('success')) for item in result.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
