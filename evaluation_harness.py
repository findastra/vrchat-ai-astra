"""Deterministic, non-learning evaluation for Mai's current persisted brain."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np


APP_DIR = Path(__file__).resolve().parent
DEFAULT_PROMPTS_FILE = APP_DIR / 'evaluation_prompts.json'
DEFAULT_GENERATIVE_PROMPTS_FILE = APP_DIR / 'evaluation_prompts_generative.json'
DEFAULT_GATE_PROMPTS_FILE = APP_DIR / 'evaluation_prompts_gate.json'
DEFAULT_RESULTS_DIR = APP_DIR / 'evaluation_results'
WORD_RE = re.compile(r"[a-z0-9']+")
PROFANITY = {'fuck', 'fucking', 'shit', 'damn', 'wtf'}
GENERIC_PREFIXES = (
    "a useful way to frame",
    "the main idea is that",
    "that's an interesting question",
    "i'm still learning",
    "i'm here to chat",
    "hello!",
    "hi there!",
)


def _words(value: str) -> list[str]:
    return WORD_RE.findall(str(value or '').lower())


def _longest_prompt_echo_ratio(prompt: str, response: str) -> float:
    prompt_words = _words(prompt)
    response_words = _words(response)
    if not prompt_words or not response_words:
        return 0.0
    previous = [0] * (len(response_words) + 1)
    longest = 0
    for prompt_word in prompt_words:
        current = [0] * (len(response_words) + 1)
        for index, response_word in enumerate(response_words, start=1):
            if prompt_word == response_word:
                current[index] = previous[index - 1] + 1
                longest = max(longest, current[index])
        previous = current
    return longest / len(prompt_words)


def calculate_surface_metrics(response: str, prompt: str = '') -> dict[str, Any]:
    words = _words(response)
    bigrams = list(zip(words, words[1:]))
    repeated_bigrams = len(bigrams) - len(set(bigrams))
    prompt_echo_ratio = _longest_prompt_echo_ratio(prompt, response)
    return {
        'word_count': len(words),
        'unique_word_ratio': round(len(set(words)) / max(1, len(words)), 4),
        'repeated_bigram_ratio': round(repeated_bigrams / max(1, len(bigrams)), 4),
        'has_terminal_punctuation': bool(str(response or '').rstrip().endswith(('.', '!', '?'))),
        'generic_fallback': str(response or '').strip().lower().startswith(GENERIC_PREFIXES),
        'prompt_echo_ratio': round(prompt_echo_ratio, 4),
        'high_prompt_echo': prompt_echo_ratio >= 0.6,
        'profanity_terms': sorted(set(words) & PROFANITY),
    }


DEFAULT_PASS_THRESHOLDS = {
    'min_mean_quality_score': 0.55,
    'max_mean_prompt_echo_ratio': 0.40,
    'max_high_prompt_echo_count': 0,
    'max_generic_fallback_count': 0,
    'max_profanity_response_count': 0,
    'max_missing_terminal_punctuation_count': 1,
}

GENERATIVE_PASS_THRESHOLDS = {
    'min_mean_quality_score': 0.30,
    'max_mean_prompt_echo_ratio': 0.50,
    'max_high_prompt_echo_count': 1,
    'max_generic_fallback_count': 0,
    'max_profanity_response_count': 0,
    'max_missing_terminal_punctuation_count': 2,
    'max_symbolic_hit_rate': 0.25,
    'min_cognitive_action_presence_rate': 0.50,
    'max_unexpected_symbolic_count': 0,
}


def assess_learning_hygiene(response: str, prompt: str = '') -> dict[str, Any]:
    """Decide whether a live response is clean enough to enter durable memory."""
    surface = calculate_surface_metrics(response, prompt)
    failures = []
    if surface['generic_fallback']:
        failures.append('generic_fallback')
    if surface['high_prompt_echo']:
        failures.append('high_prompt_echo')
    if surface['profanity_terms']:
        failures.append('profanity')
    if surface['word_count'] < 3:
        failures.append('too_short')
    if surface['repeated_bigram_ratio'] >= 0.45 and surface['word_count'] >= 6:
        failures.append('high_repetition')
    return {
        'accept': not failures,
        'reason': 'accepted' if not failures else failures[0],
        'failures': failures,
        'surface': surface,
    }


def evaluate_verdict(summary: dict[str, Any], thresholds: dict[str, Any] | None = None) -> dict[str, Any]:
    """Turn an evaluation summary into an explicit pass/fail gate."""
    limits = dict(DEFAULT_PASS_THRESHOLDS)
    if thresholds:
        limits.update(thresholds)
    checks = [
        (
            'mean_quality_score',
            float(summary.get('mean_quality_score') or 0.0) >= float(limits['min_mean_quality_score']),
            summary.get('mean_quality_score'),
            limits['min_mean_quality_score'],
        ),
        (
            'mean_prompt_echo_ratio',
            float(summary.get('mean_prompt_echo_ratio') or 0.0) <= float(limits['max_mean_prompt_echo_ratio']),
            summary.get('mean_prompt_echo_ratio'),
            limits['max_mean_prompt_echo_ratio'],
        ),
        (
            'high_prompt_echo_count',
            int(summary.get('high_prompt_echo_count') or 0) <= int(limits['max_high_prompt_echo_count']),
            summary.get('high_prompt_echo_count'),
            limits['max_high_prompt_echo_count'],
        ),
        (
            'generic_fallback_count',
            int(summary.get('generic_fallback_count') or 0) <= int(limits['max_generic_fallback_count']),
            summary.get('generic_fallback_count'),
            limits['max_generic_fallback_count'],
        ),
        (
            'profanity_response_count',
            int(summary.get('profanity_response_count') or 0) <= int(limits['max_profanity_response_count']),
            summary.get('profanity_response_count'),
            limits['max_profanity_response_count'],
        ),
        (
            'missing_terminal_punctuation_count',
            int(summary.get('missing_terminal_punctuation_count') or 0)
            <= int(limits['max_missing_terminal_punctuation_count']),
            summary.get('missing_terminal_punctuation_count'),
            limits['max_missing_terminal_punctuation_count'],
        ),
    ]
    if 'max_symbolic_hit_rate' in limits and summary.get('symbolic_hit_rate') is not None:
        checks.append((
            'symbolic_hit_rate',
            float(summary.get('symbolic_hit_rate') or 0.0) <= float(limits['max_symbolic_hit_rate']),
            summary.get('symbolic_hit_rate'),
            limits['max_symbolic_hit_rate'],
        ))
    if 'min_cognitive_action_presence_rate' in limits and summary.get('cognitive_action_presence_rate') is not None:
        checks.append((
            'cognitive_action_presence_rate',
            float(summary.get('cognitive_action_presence_rate') or 0.0)
            >= float(limits['min_cognitive_action_presence_rate']),
            summary.get('cognitive_action_presence_rate'),
            limits['min_cognitive_action_presence_rate'],
        ))
    if 'max_unexpected_symbolic_count' in limits and summary.get('unexpected_symbolic_count') is not None:
        checks.append((
            'unexpected_symbolic_count',
            int(summary.get('unexpected_symbolic_count') or 0) <= int(limits['max_unexpected_symbolic_count']),
            summary.get('unexpected_symbolic_count'),
            limits['max_unexpected_symbolic_count'],
        ))
    failed = [
        {
            'metric': name,
            'observed': observed,
            'limit': limit,
        }
        for name, passed, observed, limit in checks
        if not passed
    ]
    return {
        'passed': not failed,
        'failed_checks': failed,
        'thresholds': limits,
    }


def compare_to_baseline(current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    """Fail when key held-out metrics regress relative to a saved baseline."""
    current_summary = current.get('summary') if isinstance(current.get('summary'), dict) else current
    baseline_summary = baseline.get('summary') if isinstance(baseline.get('summary'), dict) else baseline
    regressions = []

    def _num(payload, key, default=0.0):
        try:
            return float(payload.get(key, default) or default)
        except (TypeError, ValueError):
            return float(default)

    quality_delta = _num(current_summary, 'mean_quality_score') - _num(baseline_summary, 'mean_quality_score')
    if quality_delta < -0.05:
        regressions.append({
            'metric': 'mean_quality_score',
            'delta': round(quality_delta, 4),
            'current': current_summary.get('mean_quality_score'),
            'baseline': baseline_summary.get('mean_quality_score'),
        })

    echo_delta = _num(current_summary, 'mean_prompt_echo_ratio') - _num(baseline_summary, 'mean_prompt_echo_ratio')
    if echo_delta > 0.08:
        regressions.append({
            'metric': 'mean_prompt_echo_ratio',
            'delta': round(echo_delta, 4),
            'current': current_summary.get('mean_prompt_echo_ratio'),
            'baseline': baseline_summary.get('mean_prompt_echo_ratio'),
        })

    for count_key in (
        'high_prompt_echo_count',
        'generic_fallback_count',
        'profanity_response_count',
    ):
        current_count = int(_num(current_summary, count_key))
        baseline_count = int(_num(baseline_summary, count_key))
        if current_count > baseline_count:
            regressions.append({
                'metric': count_key,
                'delta': current_count - baseline_count,
                'current': current_count,
                'baseline': baseline_count,
            })

    return {
        'passed': not regressions,
        'regressions': regressions,
    }


def _finite_mean(values: list[float]) -> float | None:
    clean = [float(value) for value in values if isinstance(value, (int, float)) and math.isfinite(float(value))]
    return round(statistics.fmean(clean), 4) if clean else None


def _load_prompts(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, list) or not payload:
        raise ValueError('Evaluation prompt file must contain a non-empty JSON array.')
    prompts = []
    seen_ids = set()
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f'Prompt entry {index} must be an object.')
        prompt_id = str(item.get('id') or f'prompt_{index + 1}').strip()
        prompt = str(item.get('prompt') or '').strip()
        if not prompt:
            raise ValueError(f'Prompt entry {prompt_id} has no prompt text.')
        if prompt_id in seen_ids:
            raise ValueError(f'Duplicate prompt id: {prompt_id}')
        seen_ids.add(prompt_id)
        entry = {
            'id': prompt_id,
            'prompt': prompt,
            'expect_non_symbolic': bool(item.get('expect_non_symbolic', False)),
        }
        prompts.append(entry)
    return prompts


def _extract_path_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    payload = metadata if isinstance(metadata, dict) else {}
    realization = payload.get('realization_trace') if isinstance(payload.get('realization_trace'), dict) else {}
    cognitive = payload.get('cognitive_trace') if isinstance(payload.get('cognitive_trace'), dict) else {}
    if not cognitive:
        plan = payload.get('response_plan') if isinstance(payload.get('response_plan'), dict) else {}
        control = plan.get('cognitive_control') if isinstance(plan.get('cognitive_control'), dict) else {}
        cognitive = control
    mode = str(realization.get('mode', '') or '').strip()
    action = str(cognitive.get('action', '') or '').strip()
    goal = str(cognitive.get('goal', '') or '').strip()
    return {
        'realization_mode': mode,
        'symbolic_hit': mode == 'symbolic_reasoning',
        'cognitive_action': action,
        'cognitive_goal': goal,
        'has_cognitive_control': bool(action),
    }


def _resolve_pass_thresholds(prompts: list[dict[str, Any]], thresholds: dict[str, Any] | None = None) -> dict[str, Any]:
    limits = dict(DEFAULT_PASS_THRESHOLDS)
    if any(bool(item.get('expect_non_symbolic')) for item in prompts):
        limits.update(GENERATIVE_PASS_THRESHOLDS)
    if thresholds:
        limits.update(thresholds)
    return limits


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    try:
        with temp_path.open('w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def evaluate_with_api(
    api: Any,
    prompts: list[dict[str, Any]],
    *,
    label: str,
    seed: int,
    thresholds: dict[str, Any] | None = None,
    close_brain: bool = False,
) -> dict[str, Any]:
    """Run held-out prompts against an existing API/brain without durable learning."""
    try:
        from . import backend_runtime
    except ImportError:
        import backend_runtime

    random.seed(seed)
    np.random.seed(seed)
    started = time.time()
    brain = api.brain
    original_memory_update = getattr(brain, 'update_conversation_memory', None)
    brain.update_conversation_memory = lambda *_args, **_kwargs: None

    rows = []
    knowledge_before = {}
    generation_before = {}
    try:
        knowledge_before = api.get_knowledge_snapshot(limit=0)
        generation_before = api.get_generation_statistics_snapshot()
        for index, item in enumerate(prompts):
            prompt_seed = seed + index
            random.seed(prompt_seed)
            np.random.seed(prompt_seed)
            if hasattr(brain, 'conversation_memory') and brain.conversation_memory is not None:
                brain.conversation_memory.clear()
            if hasattr(brain, 'current_topics') and isinstance(brain.current_topics, dict):
                brain.current_topics.clear()
            if hasattr(brain, 'topic_entities') and isinstance(brain.topic_entities, dict):
                brain.topic_entities.clear()
            if hasattr(brain, 'topic_transition_history') and isinstance(brain.topic_transition_history, list):
                brain.topic_transition_history.clear()
            prompt_started = time.perf_counter()
            result = api.generate_response(item['prompt'])
            latency = time.perf_counter() - prompt_started
            surface = calculate_surface_metrics(result.response, item['prompt'])
            path_meta = _extract_path_metadata(result.metadata)
            expect_non_symbolic = bool(item.get('expect_non_symbolic'))
            unexpected_symbolic = bool(expect_non_symbolic and path_meta['symbolic_hit'])
            rows.append({
                'id': item['id'],
                'prompt': item['prompt'],
                'expect_non_symbolic': expect_non_symbolic,
                'response': result.response,
                'quality_score': round(float(result.quality_score), 4),
                'latency_seconds': round(latency, 4),
                'surface': surface,
                'path': path_meta,
                'unexpected_symbolic': unexpected_symbolic,
                'metadata': result.metadata,
            })
        knowledge_after = api.get_knowledge_snapshot(limit=0)
        if knowledge_before.get('fact_count') != knowledge_after.get('fact_count'):
            raise RuntimeError('Evaluation unexpectedly changed the knowledge fact count.')
    finally:
        if callable(original_memory_update):
            brain.update_conversation_memory = original_memory_update
        if close_brain:
            close = getattr(api, 'close_brain_instance', None)
            if callable(close):
                close(brain)

    prompt_count = max(1, len(rows))
    symbolic_hits = sum(1 for row in rows if row['path']['symbolic_hit'])
    cognitive_hits = sum(1 for row in rows if row['path']['has_cognitive_control'])
    summary = {
        'prompt_count': len(rows),
        'mean_quality_score': _finite_mean([row['quality_score'] for row in rows]),
        'mean_latency_seconds': _finite_mean([row['latency_seconds'] for row in rows]),
        'mean_unique_word_ratio': _finite_mean([row['surface']['unique_word_ratio'] for row in rows]),
        'mean_prompt_echo_ratio': _finite_mean([row['surface']['prompt_echo_ratio'] for row in rows]),
        'high_prompt_echo_count': sum(bool(row['surface']['high_prompt_echo']) for row in rows),
        'generic_fallback_count': sum(bool(row['surface']['generic_fallback']) for row in rows),
        'profanity_response_count': sum(bool(row['surface']['profanity_terms']) for row in rows),
        'missing_terminal_punctuation_count': sum(not row['surface']['has_terminal_punctuation'] for row in rows),
        'symbolic_hit_count': symbolic_hits,
        'symbolic_hit_rate': round(symbolic_hits / prompt_count, 4),
        'cognitive_action_presence_count': cognitive_hits,
        'cognitive_action_presence_rate': round(cognitive_hits / prompt_count, 4),
        'unexpected_symbolic_count': sum(1 for row in rows if row['unexpected_symbolic']),
    }
    pass_thresholds = _resolve_pass_thresholds(prompts, thresholds)
    verdict = evaluate_verdict(summary, pass_thresholds)
    return {
        'schema_version': 3,
        'label': label,
        'seed': seed,
        'suite': 'generative' if any(item.get('expect_non_symbolic') for item in prompts) else 'symbolic',
        'created_at_epoch': time.time(),
        'duration_seconds': round(time.time() - started, 4),
        'brain': {
            'db_file': str(getattr(backend_runtime, 'DB_FILE', '')),
            'chain_count': generation_before.get('chain_count'),
            'association_count': generation_before.get('association_count'),
            'unique_words': generation_before.get('unique_words'),
            'knowledge_fact_count': knowledge_before.get('fact_count'),
            'knowledge_concept_count': knowledge_before.get('concept_count'),
        },
        'summary': summary,
        'verdict': verdict,
        'results': rows,
    }


def evaluate(
    prompts: list[dict[str, Any]],
    *,
    label: str,
    seed: int,
    thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        from . import backend_runtime
        from .backend_api import MaiBackendAPI
    except ImportError:
        import backend_runtime
        from backend_api import MaiBackendAPI

    brain = backend_runtime.HybridBrain(backend_runtime.DB_FILE, is_clone=True, use_hsb_backend=False)
    api = object.__new__(MaiBackendAPI)
    api.brain = brain
    api.settings_manager = backend_runtime.settings_manager
    api.memory_manager = backend_runtime.memory_manager
    api.api_config = backend_runtime.build_backend_api_config()
    return evaluate_with_api(
        api,
        prompts,
        label=label,
        seed=seed,
        thresholds=thresholds,
        close_brain=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prompts', type=Path, default=None)
    parser.add_argument(
        '--suite',
        choices=('symbolic', 'generative', 'gate'),
        default='symbolic',
        help='Select the held-out prompt suite when --prompts is omitted.',
    )
    parser.add_argument('--label', default='evaluation')
    parser.add_argument('--seed', type=int, default=1729)
    parser.add_argument('--output', type=Path)
    parser.add_argument(
        '--baseline',
        type=Path,
        help='Optional prior evaluation JSON. Exit non-zero when held-out metrics regress.',
    )
    parser.add_argument(
        '--require-pass',
        action='store_true',
        help='Exit non-zero when absolute pass thresholds fail.',
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    prompts_path = args.prompts
    if prompts_path is None:
        if args.suite == 'generative':
            prompts_path = DEFAULT_GENERATIVE_PROMPTS_FILE
        elif args.suite == 'gate':
            prompts_path = DEFAULT_GATE_PROMPTS_FILE
        else:
            prompts_path = DEFAULT_PROMPTS_FILE
    prompts = _load_prompts(prompts_path.resolve())
    label = str(args.label)
    if label == 'evaluation' and args.prompts is None:
        if args.suite == 'generative':
            label = 'generative_check'
        elif args.suite == 'gate':
            label = 'gate_check'
    payload = evaluate(prompts, label=label, seed=int(args.seed))
    baseline_comparison = None
    if args.baseline is not None:
        baseline_payload = json.loads(args.baseline.resolve().read_text(encoding='utf-8'))
        baseline_comparison = compare_to_baseline(payload, baseline_payload)
        payload['baseline_comparison'] = baseline_comparison
        if not baseline_comparison['passed']:
            payload['verdict'] = {
                **payload['verdict'],
                'passed': False,
                'failed_checks': list(payload['verdict'].get('failed_checks') or []) + [
                    {
                        'metric': item['metric'],
                        'observed': item.get('current'),
                        'limit': item.get('baseline'),
                        'kind': 'baseline_regression',
                    }
                    for item in baseline_comparison['regressions']
                ],
            }
    output = args.output or (DEFAULT_RESULTS_DIR / f'{label}.json')
    output = output.resolve()
    _atomic_write_json(output, payload)
    report = {
        'output': str(output),
        'passed': bool(payload['verdict']['passed']),
        **payload['summary'],
        'failed_checks': payload['verdict'].get('failed_checks') or [],
    }
    if baseline_comparison is not None:
        report['baseline_passed'] = bool(baseline_comparison['passed'])
        report['regressions'] = baseline_comparison.get('regressions') or []
    print(json.dumps(report, indent=2, allow_nan=False))
    if args.require_pass and not payload['verdict']['passed']:
        return 1
    if baseline_comparison is not None and not baseline_comparison['passed']:
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
