"""Propose durable learning only when a held-out gate does not regress.

This is the explicit learn -> hygiene -> evaluate -> commit/rollback loop for Mai.
Live chat still uses cheap hygiene gates; this module is for intentional memory writes
that should survive only when they preserve coherent held-out behavior.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from evaluation_harness import (
    APP_DIR,
    DEFAULT_GATE_PROMPTS_FILE,
    _load_prompts,
    assess_learning_hygiene,
    compare_to_baseline,
    evaluate_with_api,
)


DEFAULT_GATE_BASELINE = APP_DIR / 'evaluation_baseline_generative.json'


def _build_trial_api(live_api: Any):
    try:
        from . import backend_runtime
        from .backend_api import MaiBackendAPI
    except ImportError:
        import backend_runtime
        from backend_api import MaiBackendAPI

    brain = backend_runtime.HybridBrain(backend_runtime.DB_FILE, is_clone=True, use_hsb_backend=False)
    trial = object.__new__(MaiBackendAPI)
    trial.brain = brain
    trial.settings_manager = getattr(live_api, 'settings_manager', backend_runtime.settings_manager)
    trial.memory_manager = getattr(live_api, 'memory_manager', backend_runtime.memory_manager)
    trial.api_config = getattr(live_api, 'api_config', None) or backend_runtime.build_backend_api_config()
    return trial


def _learning_text(user_text: str, response_text: str) -> str:
    user = str(user_text or '').strip()
    response = str(response_text or '').strip()
    if user and response:
        return f"[user] {user} [bot] {response}"
    return response or user


def propose_gated_learning(
    live_api: Any,
    *,
    response_text: str,
    user_text: str = '',
    commit: bool = True,
    priority_boost: int = 2,
    seed: int = 1729,
    prompts_path: Path | None = None,
    baseline_path: Path | None = None,
    require_absolute_pass: bool = True,
) -> dict[str, Any]:
    """Trial-learn on a clone, compare held-out behavior, then optionally commit live."""
    started = time.time()
    response = str(response_text or '').strip()
    user = str(user_text or '').strip()
    if not response and not user:
        return {
            'success': False,
            'accepted': False,
            'committed': False,
            'reason': 'empty_text',
            'duration_seconds': 0.0,
        }

    hygiene = assess_learning_hygiene(response or user, user)
    if not hygiene.get('accept', False):
        return {
            'success': True,
            'accepted': False,
            'committed': False,
            'reason': f"hygiene_{hygiene.get('reason', 'reject')}",
            'hygiene': hygiene,
            'duration_seconds': round(time.time() - started, 4),
        }

    prompts_file = Path(prompts_path) if prompts_path else DEFAULT_GATE_PROMPTS_FILE
    prompts = _load_prompts(prompts_file.resolve())
    learning_payload = _learning_text(user, response)
    boost = max(1, int(priority_boost or 2))

    trial = None
    before = None
    after = None
    words = 0
    try:
        trial = _build_trial_api(live_api)
        before = evaluate_with_api(trial, prompts, label='gated_before', seed=seed, close_brain=False)
        words = int(trial.brain.learn_from_text_optimized(learning_payload, base_priority_boost=boost) or 0)
        after = evaluate_with_api(trial, prompts, label='gated_after', seed=seed, close_brain=False)
    finally:
        if trial is not None:
            close = getattr(live_api, 'close_brain_instance', None) or getattr(trial, 'close_brain_instance', None)
            if callable(close):
                try:
                    close(trial.brain)
                except Exception:
                    try:
                        trial.brain.con.close()
                    except Exception:
                        pass

    if before is None or after is None:
        return {
            'success': False,
            'accepted': False,
            'committed': False,
            'reason': 'trial_eval_failed',
            'hygiene': hygiene,
            'duration_seconds': round(time.time() - started, 4),
        }

    delta_comparison = compare_to_baseline(after, before)
    baseline_comparison = None
    baseline_file = Path(baseline_path) if baseline_path else DEFAULT_GATE_BASELINE
    if baseline_file.exists():
        import json
        baseline_payload = json.loads(baseline_file.read_text(encoding='utf-8'))
        baseline_comparison = compare_to_baseline(after, baseline_payload)

    absolute_ok = bool(after.get('verdict', {}).get('passed', False)) if require_absolute_pass else True
    delta_ok = bool(delta_comparison.get('passed', False))
    baseline_ok = True if baseline_comparison is None else bool(baseline_comparison.get('passed', False))
    accepted = absolute_ok and delta_ok and baseline_ok

    committed = False
    commit_words = 0
    if not absolute_ok:
        reason = 'absolute_gate_failed'
    elif not delta_ok:
        reason = 'pre_post_regression'
    elif not baseline_ok:
        reason = 'baseline_regression'
    elif not commit:
        reason = 'accepted_dry_run'
    else:
        commit_words = int(
            live_api.brain.learn_from_text_optimized(learning_payload, base_priority_boost=boost) or 0
        )
        committed = True
        reason = 'committed'

    return {
        'success': True,
        'accepted': accepted,
        'committed': committed,
        'reason': reason,
        'hygiene': hygiene,
        'learning_text_preview': learning_payload[:240],
        'priority_boost': boost,
        'trial_words_processed': words,
        'commit_words_processed': commit_words,
        'before': {
            'summary': before.get('summary'),
            'verdict': before.get('verdict'),
        },
        'after': {
            'summary': after.get('summary'),
            'verdict': after.get('verdict'),
        },
        'delta_comparison': delta_comparison,
        'baseline_comparison': baseline_comparison,
        'duration_seconds': round(time.time() - started, 4),
    }
