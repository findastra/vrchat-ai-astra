import copy
import hashlib
import importlib
import inspect
import json
import math
import os
import re
import shutil
import sqlite3
import sys
from dataclasses import asdict, dataclass, field
import psutil
import time
from pathlib import Path
from typing import Any


@dataclass
class SerializablePayload:
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _stringify_contract_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_stringify_contract_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _stringify_contract_value(item) for key, item in value.items()}
    return str(value)


@dataclass
class GenerationResult(SerializablePayload):
    response: str
    quality_score: float
    generation_stats: str = "N/A"
    memory_usage_mb: float = 0.0
    semantic_coherence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FeedbackResult(SerializablePayload):
    success: bool
    message: str


@dataclass
class StatusSnapshot(SerializablePayload):
    recent_quality: float | None
    cluster_count: int
    cooccurrence_count: int
    generation_stats: str
    memory_usage_mb: float | None
    available_memory_mb: float | None
    memory_monitoring_enabled: bool
    conversation_memory_count: int
    topic_count: int
    hierarchical_memory_stats: dict[str, Any] = field(default_factory=dict)


def process_text_chunk_task(chunk_data, runtime_config: dict[str, Any] | None = None) -> dict[str, Any]:
    chunk_brain = None
    runtime = dict(runtime_config or {})
    try:
        chunk_text, chunk_id, total_chunks = chunk_data
        chunk_brain = _create_chunk_brain(runtime)
        chunk_brain.setup_database()
        _prepare_chunk_training_tables(chunk_brain)
        _clear_chunk_training_tables(chunk_brain)
        chunk_brain._bonus_caches_loaded = False

        if not hasattr(chunk_brain, 'batch_operations'):
            chunk_brain.batch_operations = []
            chunk_brain.batch_count = 0

        words = chunk_brain.clean_text(chunk_text)
        base_priority_boost = int(runtime.get('base_priority_boost', 2) or 2)
        processed_words = chunk_brain.learn_from_text_optimized(chunk_text, base_priority_boost=base_priority_boost)
        knowledge_result = {'success': False, 'fact_count': 0}
        if hasattr(chunk_brain, 'learn_knowledge_from_text'):
            knowledge_result = chunk_brain.learn_knowledge_from_text(
                chunk_text,
                source_type='training_chunk',
                source_path=str(runtime.get('source_path', '') or ''),
                source_label=str(runtime.get('source_label', '') or f'chunk_{chunk_id}'),
                source_category=str(runtime.get('source_category', 'general_text') or 'general_text'),
                source_weight=float(runtime.get('source_weight', 1.0) or 1.0),
            )
        knowledge_rows = {}
        knowledge_store = getattr(chunk_brain, 'knowledge_store', None)
        if knowledge_store is not None:
            try:
                knowledge_rows = knowledge_store.export_rows()
            except Exception:
                knowledge_rows = {}

        chunk_brain.cur.execute("""
            SELECT context_len, word1, word2, word3, word4, word5, word6, word7, word8, next_word, priority, success_rate, usage_count
            FROM dynamic_word_chain
        """)
        patterns = chunk_brain.cur.fetchall()

        chunk_brain.cur.execute("""
            SELECT source_word, next_word, priority, success_rate, usage_count
            FROM word_associations
        """)
        associations = chunk_brain.cur.fetchall()

        sample_words = words[::5] if len(words) > 5000 else words
        return {
            'chunk_id': chunk_id,
            'word_count': processed_words,
            'patterns': patterns,
            'associations': associations,
            'chunk_words': sample_words[:1000],
            'knowledge': knowledge_rows,
            'knowledge_summary': knowledge_result,
            'success': True,
        }
    except Exception as e:
        return {
            'chunk_id': chunk_data[1] if isinstance(chunk_data, (list, tuple)) and len(chunk_data) > 1 else -1,
            'error': str(e),
            'success': False,
        }
    finally:
        _close_chunk_brain(chunk_brain)


def _create_chunk_brain(runtime_config: dict[str, Any]):
    brain_factory = _load_runtime_brain_factory(runtime_config)
    db_file = runtime_config.get('db_file')
    if not isinstance(db_file, str) or not db_file:
        raise RuntimeError('Chunk processing database path is not configured.')

    kwargs = {'is_clone': True}
    if 'use_hsb_backend' in runtime_config:
        kwargs['use_hsb_backend'] = runtime_config.get('use_hsb_backend')
    try:
        return brain_factory(db_file, **kwargs)
    except TypeError:
        return brain_factory(db_file, is_clone=True)


def _load_runtime_brain_factory(runtime_config: dict[str, Any]):
    app_dir = runtime_config.get('app_dir')
    if isinstance(app_dir, str) and app_dir:
        app_dir = os.path.abspath(app_dir)
        parent_dir = os.path.dirname(app_dir)
        if parent_dir and parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        if app_dir not in sys.path:
            sys.path.insert(0, app_dir)

    module_candidates = []
    configured_module = runtime_config.get('runtime_module') or runtime_config.get('desktop_module')
    if isinstance(configured_module, str) and configured_module:
        module_candidates.append(configured_module)
    module_candidates.extend([
        'ai_astra_main.backend_runtime',
        'backend_runtime',
    ])

    last_error = None
    for module_name in module_candidates:
        try:
            module = importlib.import_module(module_name)
            brain_factory = getattr(module, 'HybridBrain', None)
            if callable(brain_factory):
                return brain_factory
        except Exception as e:
            last_error = e
    raise RuntimeError(f'Could not load HybridBrain for chunk processing: {last_error}')


def _prepare_chunk_training_tables(chunk_brain) -> None:
    chunk_brain.cur.execute("""
        CREATE TABLE IF NOT EXISTS sentence_patterns (
            pattern_type TEXT,
            pattern_text TEXT,
            priority REAL DEFAULT 1.0,
            usage_count INTEGER DEFAULT 0,
            PRIMARY KEY (pattern_type, pattern_text)
        )
    """)
    chunk_brain.cur.execute("""
        CREATE TABLE IF NOT EXISTS grammar_patterns (
            pattern_type TEXT,
            word1 TEXT,
            word2 TEXT,
            priority REAL DEFAULT 1.0,
            usage_count INTEGER DEFAULT 0,
            PRIMARY KEY (pattern_type, word1, word2)
        )
    """)
    chunk_brain.cur.execute("""
        CREATE TABLE IF NOT EXISTS phrase_patterns (
            phrase_text TEXT PRIMARY KEY,
            priority REAL DEFAULT 1.0,
            usage_count INTEGER DEFAULT 0
        )
    """)
    chunk_brain.cur.execute("""
        CREATE TABLE IF NOT EXISTS semantic_relationships (
            word1 TEXT,
            word2 TEXT,
            strength REAL DEFAULT 1.0,
            usage_count INTEGER DEFAULT 0,
            PRIMARY KEY (word1, word2)
        )
    """)


def _clear_chunk_training_tables(chunk_brain) -> None:
    chunk_brain.cur.execute("DELETE FROM dynamic_word_chain")
    chunk_brain.cur.execute("DELETE FROM word_associations")
    chunk_brain.cur.execute("DELETE FROM sentence_patterns")
    chunk_brain.cur.execute("DELETE FROM grammar_patterns")
    chunk_brain.cur.execute("DELETE FROM phrase_patterns")
    chunk_brain.cur.execute("DELETE FROM semantic_relationships")
    chunk_brain.con.commit()


def _close_chunk_brain(chunk_brain) -> None:
    if chunk_brain is not None and getattr(chunk_brain, 'con', None) is not None:
        try:
            chunk_brain.con.close()
        except Exception:
            pass


class AIAstraBackendAPI:
    """
    Backend-facing service boundary for the AI Astra model runtime.

    The current PySide UI can call this directly in-process, and a future
    Electron shell can expose the same methods over IPC or HTTP without
    depending on widget code or Qt objects.
    """

    def __init__(self, brain, settings_manager, memory_manager, api_config: dict[str, Any] | None = None):
        self.brain = brain
        self.settings_manager = settings_manager
        self.memory_manager = memory_manager
        self.api_config = dict(api_config or {})
        self._startup_hardware_adaptation = self._apply_startup_hardware_adaptation()
        self._startup_hardware_runtime = self._sync_hardware_runtime()
        self._startup_runtime_changes = self._sync_runtime_configuration()

    def attach_brain(self, brain) -> None:
        self.brain = brain
        self._sync_runtime_configuration()

    def get_transport_method_names(self) -> list[str]:
        return [
            'get_api_manifest',
            'get_transport_method_spec',
            'get_runtime_bootstrap_snapshot',
            'get_feature_runtime_snapshot',
            'get_learning_health_snapshot',
            'get_status_snapshot',
            'get_settings_snapshot',
            'get_hardware_profile',
            'save_settings',
            'reset_settings',
            'update_setting',
            'apply_performance_profile',
            'set_feature_flag',
            'get_storage_snapshot',
            'get_response_display_state',
            'get_feedback_display_state',
            'export_conversation_text',
            'generate_response',
            'generate_correction_prompt',
            'generate_correction_response',
            'apply_feedback',
            'reinforce_generated_response',
            'discourage_generated_response',
            'teach_corrected_response',
            'get_brain_patterns',
            'get_response_plan_preview',
            'get_graph_reasoning_preview',
            'get_knowledge_snapshot',
            'get_knowledge_concepts',
            'get_knowledge_facts',
            'get_knowledge_evidence',
            'get_knowledge_identity_traits',
            'get_attention_snapshot',
            'get_context_performance_snapshot',
            'get_semantic_cluster_snapshot',
            'get_generation_statistics_snapshot',
            'get_vocab_id_storage_snapshot',
            'migrate_vocab_id_storage',
            'get_memory_statistics_snapshot',
            'save_recent_patterns_to_initial_knowledge',
            'get_memory_pressure_state',
            'force_garbage_collection',
            'get_chunk_processing_config',
            'collect_training_files',
            'get_training_plan',
            'learn_from_training_chunk',
            'propose_gated_learning',
            'train_files',
            'flush_training_batches',
            'finalize_training_run',
            'merge_chunk_training_results',
            'rebuild_semantic_clusters_from_words',
            'get_training_summary',
            'evaluate_self_training_response',
            'apply_self_training_feedback',
            'clean_brain_for_new_training',
            'sync_brain_state',
            'perform_memory_replay',
            'export_brain_bundle',
            'import_brain_bundle',
            'export_hsb_copy',
            'export_brain_to_hsb',
            'import_brain_from_hsb',
        ]

    def get_transport_control_names(self) -> list[str]:
        return ['shutdown', 'get_session_info']

    def get_transport_control_specs(self) -> dict[str, dict[str, Any]]:
        return {
            'shutdown': {
                'category': 'transport_control',
                'mutates_state': True,
                'summary': 'Request a clean shutdown of the current transport process.',
                'signature': '(id?: Any) -> control result',
                'params': [],
                'returns': 'dict[str, Any]',
                'doc': 'Transport-level control method. Stops the current stdio or HTTP service after responding.',
            },
            'get_session_info': {
                'category': 'transport_control',
                'mutates_state': False,
                'summary': 'Return process-level session metadata for the current transport host.',
                'signature': '(id?: Any) -> dict[str, Any]',
                'params': [],
                'returns': 'dict[str, Any]',
                'doc': 'Transport-level control method. Reports session ID, PID, uptime, transport name, and batch limits.',
            },
        }

    def get_transport_method_specs(self) -> dict[str, dict[str, Any]]:
        specs: dict[str, dict[str, Any]] = {}
        for name in self.get_transport_method_names():
            method = getattr(self, name, None)
            if not callable(method):
                continue
            try:
                signature = inspect.signature(method)
            except (TypeError, ValueError):
                signature = None

            params: list[dict[str, Any]] = []
            return_annotation = ""
            if signature is not None:
                for parameter in signature.parameters.values():
                    if parameter.name == 'self':
                        continue
                    params.append({
                        'name': parameter.name,
                        'kind': parameter.kind.name.lower(),
                        'required': parameter.default is inspect.Signature.empty,
                        'default': None if parameter.default is inspect.Signature.empty else _stringify_contract_value(parameter.default),
                        'annotation': self._format_contract_annotation(parameter.annotation),
                    })
                return_annotation = self._format_contract_annotation(signature.return_annotation)

            doc = inspect.getdoc(method) or ""
            specs[name] = {
                'category': self._classify_transport_method(name),
                'mutates_state': self._transport_method_mutates_state(name),
                'summary': self._summarize_transport_method(name, doc),
                'signature': str(signature) if signature is not None else "()",
                'params': params,
                'returns': return_annotation,
                'doc': doc,
            }
        return specs

    def get_transport_method_spec(self, name: str) -> dict[str, Any]:
        specs = self.get_transport_method_specs()
        spec = specs.get(name)
        if spec is None:
            spec = self.get_transport_control_specs().get(name)
        if spec is None:
            raise ValueError(f'Unknown transport method: {name}')
        return {
            'name': name,
            'spec': spec,
        }

    def get_transport_category_map(self) -> dict[str, list[str]]:
        categories: dict[str, list[str]] = {}
        for name in self.get_transport_method_names():
            category = self._classify_transport_method(name)
            categories.setdefault(category, []).append(name)
        return {key: sorted(value) for key, value in sorted(categories.items())}

    def get_transport_workflows(self) -> dict[str, list[str]]:
        return {
            'bootstrap': [
                'get_api_manifest',
                'get_runtime_bootstrap_snapshot',
                'get_hardware_profile',
                'get_feature_runtime_snapshot',
            ],
            'chat_round_trip': [
                'get_response_plan_preview',
                'get_graph_reasoning_preview',
                'generate_response',
                'get_response_display_state',
                'apply_feedback',
            ],
            'correction_flow': [
                'generate_correction_prompt',
                'generate_correction_response',
                'teach_corrected_response',
            ],
            'training_flow': [
                'collect_training_files',
                'get_training_plan',
                'train_files',
                'finalize_training_run',
            ],
            'knowledge_inspection': [
                'get_knowledge_snapshot',
                'get_knowledge_concepts',
                'get_knowledge_facts',
                'get_knowledge_evidence',
                'get_knowledge_identity_traits',
            ],
            'bundle_portability': [
                'export_brain_bundle',
                'import_brain_bundle',
                'sync_brain_state',
            ],
            'settings_sync': [
                'get_settings_snapshot',
                'get_hardware_profile',
                'update_setting',
                'set_feature_flag',
                'get_feature_runtime_snapshot',
            ],
        }

    def _format_contract_annotation(self, annotation: Any) -> str:
        if annotation is inspect.Signature.empty:
            return ""
        if isinstance(annotation, type):
            return annotation.__name__
        return str(annotation).replace('typing.', '')

    def _classify_transport_method(self, name: str) -> str:
        settings_methods = {
            'get_settings_snapshot',
            'get_hardware_profile',
            'save_settings',
            'reset_settings',
            'update_setting',
            'apply_performance_profile',
            'set_feature_flag',
        }
        generation_methods = {
            'get_response_plan_preview',
            'get_graph_reasoning_preview',
            'generate_response',
            'generate_correction_prompt',
            'generate_correction_response',
        }
        feedback_methods = {
            'apply_feedback',
            'reinforce_generated_response',
            'discourage_generated_response',
            'teach_corrected_response',
        }
        inspection_methods = {
            'get_brain_patterns',
            'get_knowledge_snapshot',
            'get_knowledge_concepts',
            'get_knowledge_facts',
            'get_knowledge_evidence',
            'get_knowledge_identity_traits',
            'get_attention_snapshot',
            'get_context_performance_snapshot',
            'get_semantic_cluster_snapshot',
            'get_generation_statistics_snapshot',
            'get_vocab_id_storage_snapshot',
            'migrate_vocab_id_storage',
            'get_memory_statistics_snapshot',
            'get_status_snapshot',
            'get_storage_snapshot',
            'get_response_display_state',
            'get_feedback_display_state',
            'get_memory_pressure_state',
            'get_feature_runtime_snapshot',
            'get_learning_health_snapshot',
            'get_runtime_bootstrap_snapshot',
        }
        training_methods = {
            'save_recent_patterns_to_initial_knowledge',
            'get_chunk_processing_config',
            'collect_training_files',
            'get_training_plan',
            'learn_from_training_chunk',
            'train_files',
            'propose_gated_learning',
            'flush_training_batches',
            'finalize_training_run',
            'merge_chunk_training_results',
            'rebuild_semantic_clusters_from_words',
            'get_training_summary',
            'evaluate_self_training_response',
            'apply_self_training_feedback',
        }
        persistence_methods = {
            'clean_brain_for_new_training',
            'sync_brain_state',
            'perform_memory_replay',
            'export_conversation_text',
            'export_brain_bundle',
            'import_brain_bundle',
            'export_hsb_copy',
            'export_brain_to_hsb',
            'import_brain_from_hsb',
            'force_garbage_collection',
        }

        if name in {'get_api_manifest', 'get_transport_method_spec'}:
            return 'transport'
        if name in settings_methods:
            return 'settings'
        if name in generation_methods:
            return 'generation'
        if name in feedback_methods:
            return 'feedback'
        if name in inspection_methods:
            return 'inspection'
        if name in training_methods:
            return 'training'
        if name in persistence_methods:
            return 'persistence'
        return 'runtime'

    def _transport_method_mutates_state(self, name: str) -> bool:
        read_only_methods = {
            'get_api_manifest',
            'get_transport_method_spec',
            'get_runtime_bootstrap_snapshot',
            'get_feature_runtime_snapshot',
            'get_status_snapshot',
            'get_settings_snapshot',
            'get_hardware_profile',
            'get_storage_snapshot',
            'get_response_display_state',
            'get_feedback_display_state',
            'generate_correction_prompt',
            'get_brain_patterns',
            'get_response_plan_preview',
            'get_knowledge_snapshot',
            'get_knowledge_concepts',
            'get_knowledge_facts',
            'get_knowledge_evidence',
            'get_knowledge_identity_traits',
            'get_attention_snapshot',
            'get_context_performance_snapshot',
            'get_semantic_cluster_snapshot',
            'get_generation_statistics_snapshot',
            'get_vocab_id_storage_snapshot',
            'get_memory_statistics_snapshot',
            'get_memory_pressure_state',
            'get_chunk_processing_config',
            'collect_training_files',
            'get_training_plan',
            'get_training_summary',
            'evaluate_self_training_response',
            'export_brain_bundle',
            'export_hsb_copy',
        }
        return name not in read_only_methods

    def _summarize_transport_method(self, name: str, doc: str) -> str:
        if doc:
            return doc.splitlines()[0].strip()

        summaries = {
            'get_api_manifest': 'Describe the transport surface, routes, and method contract.',
            'get_transport_method_spec': 'Return the contract metadata for one transport method.',
            'get_runtime_bootstrap_snapshot': 'Return a frontend-friendly snapshot of settings, storage, status, generation, memory, and features.',
            'get_feature_runtime_snapshot': 'Describe which higher-level backend features are configured and currently active.',
            'get_learning_health_snapshot': 'Return adaptive live-learning health, acceptance rate, and quality gating guidance.',
            'get_status_snapshot': 'Return a compact runtime health snapshot.',
            'get_settings_snapshot': 'Return the saved runtime settings.',
            'get_hardware_profile': 'Return detected hardware, recommended runtime settings, and the active GPU/runtime selection.',
            'save_settings': 'Persist the current settings to disk.',
            'reset_settings': 'Restore default settings and sync the live runtime.',
            'update_setting': 'Change one setting and apply any live runtime updates.',
            'apply_performance_profile': 'Apply a named performance profile and sync the backend runtime.',
            'set_feature_flag': 'Enable or disable one feature flag and sync the live runtime.',
            'get_storage_snapshot': 'Describe the current storage backend and active brain files.',
            'get_response_display_state': 'Return frontend display hints for a quality score.',
            'get_feedback_display_state': 'Return frontend display hints for feedback actions.',
            'export_conversation_text': 'Render the current conversation log as plain text inside an app-local export folder.',
            'generate_response': 'Generate a response from the live brain and update conversation state.',
            'generate_correction_prompt': 'Prepare the correction prompt used by the teaching workflow.',
            'generate_correction_response': 'Generate a response for a correction/teaching prompt.',
            'apply_feedback': 'Apply end-user feedback to the latest response.',
            'reinforce_generated_response': 'Reward the latest generated response.',
            'discourage_generated_response': 'Penalize the latest generated response.',
            'teach_corrected_response': 'Teach the model a preferred corrected response.',
            'get_brain_patterns': 'Return learned pattern rows for inspection tooling.',
            'get_response_plan_preview': 'Build a non-mutating claim-based response plan preview from concepts, facts, identity traits, and planner goals.',
            'get_graph_reasoning_preview': 'Build a non-mutating graph reasoning preview from concept relations and multi-hop fact paths.',
            'get_knowledge_snapshot': 'Return concept, fact, evidence, and identity counts from the knowledge layer.',
            'get_knowledge_concepts': 'Return canonical concepts captured by the knowledge layer.',
            'get_knowledge_facts': 'Return extracted facts with multi-dimensional confidence and provenance metadata.',
            'get_knowledge_evidence': 'Return provenance rows supporting one fact or one knowledge query.',
            'get_knowledge_identity_traits': 'Return durable identity-level traits extracted for AI Astra.',
            'get_attention_snapshot': 'Return analog-attention weights and summaries.',
            'get_context_performance_snapshot': 'Return context-length performance metrics.',
            'get_semantic_cluster_snapshot': 'Return semantic cluster summaries and related stats.',
            'get_generation_statistics_snapshot': 'Return generation counters and quality history.',
            'get_vocab_id_storage_snapshot': 'Return lexicon and ID-chain storage counters for compact n-gram storage.',
            'migrate_vocab_id_storage': 'Copy a batch of TEXT n-gram chains into the compact vocabulary-ID table.',
            'get_memory_statistics_snapshot': 'Return memory pressure, worker, and chunking guidance.',
            'save_recent_patterns_to_initial_knowledge': 'Copy recent learned patterns into the initial knowledge file.',
            'get_memory_pressure_state': 'Measure current memory pressure against the configured threshold.',
            'force_garbage_collection': 'Force a memory cleanup pass and return the resulting pressure snapshot.',
            'get_chunk_processing_config': 'Describe the runtime settings needed by parallel chunk workers.',
            'collect_training_files': 'Resolve directories and file paths into a deduplicated training file list.',
            'get_training_plan': 'Estimate chunking and processing strategy for one training file.',
            'learn_from_training_chunk': 'Train the live brain on a single chunk of text.',
            'propose_gated_learning': 'Trial-learn on a clone, re-run the held-out gate, and commit only if behavior does not regress.',
            'train_files': 'Run a full backend-owned training pass over one or more files.',
            'flush_training_batches': 'Flush pending batched training writes to storage.',
            'finalize_training_run': 'Persist and summarize the current training session.',
            'merge_chunk_training_results': 'Merge parallel chunk results into the live brain state.',
            'rebuild_semantic_clusters_from_words': 'Rebuild semantic clusters from a supplied word set.',
            'get_training_summary': 'Return a summary of the current training state.',
            'evaluate_self_training_response': 'Score a self-training response before applying it.',
            'apply_self_training_feedback': 'Apply scored self-training feedback to the live brain.',
            'clean_brain_for_new_training': 'Reset the active brain to a clean state for new training.',
            'sync_brain_state': 'Force the backend to flush state to its persistent storage.',
            'perform_memory_replay': 'Run the backend memory replay pass.',
            'export_brain_bundle': 'Export the full brain bundle to a portable archive.',
            'import_brain_bundle': 'Import a portable brain bundle into the active backend.',
            'export_hsb_copy': 'Export a copy of the active HSB file.',
            'export_brain_to_hsb': 'Export the active brain state into HSB format.',
            'import_brain_from_hsb': 'Import HSB content into the active brain.',
        }
        if name in summaries:
            return summaries[name]
        return name.replace('_', ' ').capitalize() + '.'

    def _resolve_app_relative_folder(self, folder_name: str | None, default: str = 'conversations') -> tuple[str, str]:
        app_dir = os.path.abspath(self.api_config.get('app_dir') or os.path.dirname(os.path.abspath(__file__)))
        requested_name = (folder_name or default).strip()
        if not requested_name:
            requested_name = default

        normalized_name = requested_name.replace('/', os.sep).replace('\\', os.sep)
        if os.path.isabs(normalized_name):
            raise ValueError('folder_name must be relative to the app directory.')

        candidate = os.path.abspath(os.path.join(app_dir, normalized_name))
        try:
            common_path = os.path.commonpath([app_dir, candidate])
        except ValueError:
            common_path = ''
        if common_path != app_dir:
            raise ValueError('folder_name must stay within the app directory.')

        relative_name = os.path.relpath(candidate, app_dir)
        if relative_name.startswith('..'):
            raise ValueError('folder_name must stay within the app directory.')

        return candidate, relative_name

    def get_settings_snapshot(self) -> dict[str, Any]:
        try:
            return copy.deepcopy(getattr(self.settings_manager, 'settings', {}) or {})
        except Exception:
            return {}

    def get_hardware_profile(self) -> dict[str, Any]:
        return self._get_hardware_profile()

    def _normalize_setting_value(self, key: str, value: Any) -> Any:
        defaults = getattr(self.settings_manager, 'default_settings', {}) or {}
        if key not in defaults:
            raise ValueError(f'Unsupported setting: {key}')
        if key == 'features':
            raise ValueError('Use set_feature_flag to update individual feature flags.')

        boolean_keys = {
            'gpu_acceleration',
            'gpu_acceleration_enabled',
            'hardware_adaptive_mode',
            'enable_memory_monitoring',
            'enable_garbage_collection',
            'enhanced_intelligence',
            'advanced_reasoning',
            'adaptive_learning',
            'save_hsb_copy',
            'use_hsb_backend',
            'show_low_confidence_notice',
            'debug_logging',
        }
        if key in boolean_keys:
            if not isinstance(value, bool):
                raise ValueError(f'{key} must be a boolean.')
            return value

        enum_values = {
            'gpu_acceleration_type': {'auto', 'cuda', 'opencl'},
            'gpu_batch_size': {'auto', 'small', 'medium', 'large'},
            'log_level': {'debug', 'info', 'warning', 'error', 'critical'},
            'vocab_id_storage_mode': {'off', 'dual', 'ids'},
        }
        if key in enum_values:
            normalized = str(value or '').strip().lower()
            if normalized not in enum_values[key]:
                choices = ', '.join(sorted(enum_values[key]))
                raise ValueError(f'{key} must be one of: {choices}.')
            return normalized

        auto_integer_ranges = {
            'gpu_device_index': (0, 255),
            'cpu_worker_reserve': (0, 255),
            'parallel_workers': (1, 255),
            'chunk_size': (1, 10_000_000),
        }
        if key in auto_integer_ranges:
            if isinstance(value, str) and value.strip().lower() == 'auto':
                return 'auto'
            if isinstance(value, bool):
                raise ValueError(f'{key} must be "auto" or an integer.')
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                raise ValueError(f'{key} must be "auto" or an integer.') from None
            minimum, maximum = auto_integer_ranges[key]
            if not math.isfinite(numeric) or not numeric.is_integer() or not minimum <= numeric <= maximum:
                raise ValueError(f'{key} must be "auto" or an integer from {minimum} to {maximum}.')
            return str(int(numeric))

        integer_ranges = {
            'max_response_length': (1, 512),
            'min_generation_attempts': (1, 64),
            'auto_save_interval': (0, 86_400),
            'auto_save_interval_minutes': (0, 10_080),
            'memory_replay_interval': (1, 10_000_000),
        }
        if key in integer_ranges:
            if isinstance(value, bool):
                raise ValueError(f'{key} must be an integer.')
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                raise ValueError(f'{key} must be an integer.') from None
            minimum, maximum = integer_ranges[key]
            if not math.isfinite(numeric) or not numeric.is_integer() or not minimum <= numeric <= maximum:
                raise ValueError(f'{key} must be an integer from {minimum} to {maximum}.')
            return int(numeric)

        number_ranges = {
            'gpu_memory_limit': (0.05, 1.0),
            'memory_threshold': (0.05, 0.99),
            'self_training_interval': (0.1, 86_400.0),
            'quality_threshold': (0.0, 1.0),
            'creativity_factor': (0.0, 1.0),
            'memory_compression_threshold': (0.0, 1.0),
            'context_weight': (0.0, 1.0),
            'semantic_weight': (0.0, 1.0),
            'pattern_weight': (0.0, 1.0),
        }
        if key in number_ranges:
            if isinstance(value, bool):
                raise ValueError(f'{key} must be a number.')
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                raise ValueError(f'{key} must be a number.') from None
            minimum, maximum = number_ranges[key]
            if not math.isfinite(numeric) or not minimum <= numeric <= maximum:
                raise ValueError(f'{key} must be from {minimum} to {maximum}.')
            return numeric

        raise ValueError(f'Unsupported setting: {key}')

    def save_settings(self) -> dict[str, Any]:
        saved = self.settings_manager.save_settings()
        return {
            'success': saved is not False,
            'settings': self.get_settings_snapshot(),
            'message': 'Settings saved.' if saved is not False else 'Settings could not be saved.',
        }

    def reset_settings(self) -> dict[str, Any]:
        self.settings_manager.reset_to_defaults()
        hardware_adaptation = self._apply_startup_hardware_adaptation()
        hardware_runtime = self._sync_hardware_runtime()
        runtime_changes = self._sync_runtime_configuration()
        return {
            'success': True,
            'settings': self.get_settings_snapshot(),
            'hardware_profile': self.get_hardware_profile(),
            'hardware_adaptation': hardware_adaptation,
            'hardware_runtime': hardware_runtime,
            'runtime_changes': runtime_changes,
            'feature_runtime': self.get_feature_runtime_snapshot(),
        }

    def update_setting(self, key: str, value: Any) -> dict[str, Any]:
        if key == 'performance_profile':
            return self.apply_performance_profile(str(value), self._get_hardware_profile())
        value = self._normalize_setting_value(key, value)
        self.settings_manager.set(key, value)
        hardware_adaptation = {}
        hardware_runtime = {}
        if key == 'enhanced_intelligence' and hasattr(self.brain, 'enhanced_intelligence_enabled'):
            self.brain.enhanced_intelligence_enabled = bool(value)
        elif key == 'memory_replay_interval' and hasattr(self.brain, 'hierarchical_memory'):
            self.brain.hierarchical_memory.memory_replay_interval = value
        elif key in ('creativity_factor', 'context_weight', 'semantic_weight', 'pattern_weight'):
            adaptive_learning = getattr(self.brain, 'adaptive_learning', None)
            adaptive_parameters = getattr(adaptive_learning, 'adaptive_parameters', None)
            if isinstance(adaptive_parameters, dict):
                adaptive_parameters[key] = value
        elif key == 'max_response_length':
            setter = self.api_config.get('set_max_response_length')
            if callable(setter):
                setter(int(value))
        elif key == 'hardware_adaptive_mode' and bool(value):
            hardware_adaptation = self._apply_startup_hardware_adaptation()
        if key in {'gpu_acceleration_enabled', 'gpu_acceleration_type', 'gpu_device_index', 'hardware_adaptive_mode'}:
            hardware_runtime = self._sync_hardware_runtime()
        runtime_changes = self._sync_runtime_configuration()
        return {
            'success': True,
            'setting': key,
            'value': value,
            'restart_required': key in {'use_hsb_backend', 'vocab_id_storage_mode'},
            'hardware_profile': self.get_hardware_profile(),
            'hardware_adaptation': hardware_adaptation,
            'hardware_runtime': hardware_runtime,
            'runtime_changes': runtime_changes,
            'feature_runtime': self.get_feature_runtime_snapshot(),
        }

    def apply_performance_profile(self, profile_name: str, system_tier: dict[str, Any] | None = None) -> dict[str, Any]:
        if not profile_name:
            return {'success': False, 'message': 'No performance profile provided.'}

        profile = profile_name.strip().lower()
        if profile not in ('low', 'medium', 'max'):
            return {'success': False, 'message': f'Unsupported performance profile: {profile_name}'}

        hardware_profile = dict(system_tier or {})
        if 'profiles' not in hardware_profile or 'recommended_parallel_workers' not in hardware_profile:
            hardware_profile = self._get_hardware_profile()
        tier = str(hardware_profile.get('tier', 'medium')).lower()
        has_gpu = bool(hardware_profile.get('has_gpu', False))

        profile_result = self._apply_profile_settings(profile, hardware_profile)
        hardware_runtime = self._sync_hardware_runtime()
        runtime_changes = self._sync_runtime_configuration()
        return {
            'success': True,
            'profile': profile,
            'tier': tier,
            'has_gpu': has_gpu,
            'hardware_profile': self.get_hardware_profile(),
            'hardware_runtime': hardware_runtime,
            'applied_settings': profile_result.get('settings', {}),
            'settings': self.get_settings_snapshot(),
            'runtime_changes': runtime_changes,
            'feature_runtime': self.get_feature_runtime_snapshot(),
            'message': f"Applied {profile_name.strip()} profile for your system ({tier} spec).",
        }

    def set_feature_flag(self, name: str, enabled: bool) -> dict[str, Any]:
        if not isinstance(enabled, bool):
            return {
                'success': False,
                'feature': name,
                'enabled': False,
                'message': 'enabled must be a boolean.',
            }
        known_features = self._get_known_feature_flags()
        if known_features and name not in known_features:
            return {
                'success': False,
                'feature': name,
                'enabled': False,
                'message': f'Unsupported feature flag: {name}',
            }
        self.settings_manager.set_feature(name, enabled)
        runtime_changes = self._sync_runtime_configuration()
        snapshot = self.get_feature_runtime_snapshot()
        features = snapshot.get('feature_flags', {})
        runtime_feature_flags = snapshot.get('runtime_feature_flags', {})
        return {
            'success': True,
            'feature': name,
            'enabled': bool(features.get(name, enabled)),
            'runtime_enabled': bool(runtime_feature_flags.get(name, False)),
            'runtime_changes': runtime_changes,
            'feature_runtime': snapshot,
        }

    def get_storage_snapshot(self) -> dict[str, Any]:
        storage_backend = getattr(self.brain, '_storage_backend', None)
        if storage_backend is not None:
            hsb_file = getattr(storage_backend, 'hsb_file', self._resolve_hsb_default_path())
            return {
                'mode': 'hsb',
                'hsb_file': hsb_file,
                'label': f"Storage: HSB ({os.path.basename(hsb_file)})",
            }
        return {
            'mode': 'sqlite',
            'hsb_file': self._resolve_hsb_default_path(),
            'label': 'Storage: SQLite',
        }

    def get_response_display_state(self, quality: float | None = None) -> dict[str, Any]:
        indicator_color = ""
        indicator_level = "none"
        if quality is not None:
            if quality > 0.7:
                indicator_color = "#22c55e"
                indicator_level = "high"
            elif quality > 0.5:
                indicator_color = "#eab308"
                indicator_level = "medium"
            else:
                indicator_color = "#ef4444"
                indicator_level = "low"

        try:
            threshold = float(self.settings_manager.get('quality_threshold', 0.5))
        except (TypeError, ValueError):
            threshold = 0.5

        show_notice = bool(self.settings_manager.get('show_low_confidence_notice', True))
        low_confidence = bool(show_notice and quality is not None and quality < threshold)
        return {
            'quality': quality,
            'indicator_color': indicator_color,
            'indicator_level': indicator_level,
            'quality_threshold': threshold,
            'show_low_confidence_notice': show_notice,
            'low_confidence': low_confidence,
            'low_confidence_message': "Low confidence — more training may improve answers." if low_confidence else "",
        }

    def get_feedback_display_state(self, sentence: str, is_positive: bool) -> dict[str, Any]:
        feedback_type = "positive" if is_positive else "negative"
        preview = (sentence[:40] + "...") if len(sentence) > 40 else sentence
        return {
            'feedback_type': feedback_type,
            'preview': preview,
            'message': f'(Memory optimized {feedback_type} pattern feedback recorded for: "{preview}")',
        }

    def export_conversation_text(self, text: str, folder_name: str = 'conversations') -> dict[str, Any]:
        app_dir = os.path.abspath(self.api_config.get('app_dir') or os.path.dirname(os.path.abspath(__file__)))
        folder, relative_folder = self._resolve_app_relative_folder(folder_name)
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, time.strftime('%Y%m%d_%H%M%S') + '_chat.txt')
        with open(path, 'w', encoding='utf-8') as f:
            f.write(text)
        return {
            'success': True,
            'app_dir': app_dir,
            'relative_folder': relative_folder,
            'folder': folder,
            'path': path,
            'message': f'Conversation saved to:\n{path}',
        }

    def generate_response(self, user_input: str = "") -> GenerationResult:
        response = self.brain.generate_response(user_input) if getattr(self, 'brain', None) else None
        response = response or ""
        metadata: dict[str, Any] = {}
        get_plan = getattr(self.brain, 'get_last_response_plan', None)
        if callable(get_plan):
            try:
                plan = get_plan() or {}
                if plan:
                    metadata['response_plan'] = plan
            except Exception:
                pass
        get_critic = getattr(self.brain, 'get_last_critic_assessment', None)
        if callable(get_critic):
            try:
                critic_assessment = get_critic() or {}
                if critic_assessment:
                    metadata['critic_assessment'] = critic_assessment
            except Exception:
                pass
        get_reasoning = getattr(self.brain, 'get_last_reasoning_trace', None)
        if callable(get_reasoning):
            try:
                reasoning_trace = get_reasoning() or {}
                if reasoning_trace:
                    metadata['reasoning_trace'] = reasoning_trace
            except Exception:
                pass
        get_realization = getattr(self.brain, 'get_last_realization_trace', None)
        if callable(get_realization):
            try:
                realization_trace = get_realization() or {}
                if realization_trace:
                    metadata['realization_trace'] = realization_trace
            except Exception:
                pass
        get_cognitive = getattr(self.brain, 'get_last_cognitive_trace', None)
        if callable(get_cognitive):
            try:
                cognitive_trace = get_cognitive() or {}
                if cognitive_trace:
                    metadata['cognitive_trace'] = cognitive_trace
            except Exception:
                pass
        try:
            from evaluation_harness import assess_learning_hygiene
        except ImportError:
            try:
                from .evaluation_harness import assess_learning_hygiene
            except ImportError:
                assess_learning_hygiene = None
        if assess_learning_hygiene is not None:
            try:
                metadata['learning_hygiene'] = assess_learning_hygiene(response, user_input)
            except Exception:
                pass
        return GenerationResult(
            response=response,
            quality_score=self._calculate_quality(response, user_input),
            generation_stats=self._get_generation_stats(),
            memory_usage_mb=self._get_memory_usage_mb(),
            metadata=metadata,
        )

    def generate_correction_prompt(self) -> GenerationResult:
        return self.generate_response("")

    def generate_correction_response(self, user_input: str) -> GenerationResult:
        result = self.generate_response(user_input)
        response_words = self._clean_text(result.response)
        context_words = self._get_structured_context(history_turns=2)
        result.semantic_coherence = self._calculate_semantic_coherence(response_words, context_words)
        return result

    def apply_feedback(self, sentence: str, is_positive: bool) -> FeedbackResult:
        self.brain.apply_feedback(sentence, is_positive=is_positive)
        feedback_type = "positive" if is_positive else "negative"
        return FeedbackResult(True, f"{feedback_type.title()} feedback recorded.")

    def reinforce_generated_response(self, context: str, response: str) -> FeedbackResult:
        sentence = f"{context} {response}".strip()
        self.brain.apply_feedback(sentence, is_positive=True)
        return FeedbackResult(True, "Generated response pattern reinforced.")

    def discourage_generated_response(self, context: str, response: str) -> FeedbackResult:
        sentence = f"{context} {response}".strip()
        self.brain.apply_feedback(sentence, is_positive=False)
        return FeedbackResult(True, "Generated response pattern discouraged.")

    def teach_corrected_response(self, context: str, bad_response: str, correction: str) -> FeedbackResult:
        bad_sentence = f"{context} {bad_response}".strip()
        good_sentence = f"{context} {correction}".strip()
        self.brain.apply_feedback(bad_sentence, is_positive=False)
        self.brain.apply_feedback(good_sentence, is_positive=True)
        return FeedbackResult(True, "Corrected response learned as a statistical pattern.")

    def get_status_snapshot(self) -> StatusSnapshot:
        recent_quality = None
        hist = getattr(self.brain, 'response_quality_history', None) or []
        nums = [float(x) for x in hist[-5:] if isinstance(x, (int, float))]
        if nums:
            recent_quality = sum(nums) / max(1, len(nums))

        semantic_memory = getattr(self.brain, 'semantic_memory', None)
        cluster_count = len(getattr(semantic_memory, 'clusters', None) or [])
        cooccurrence_count = len(getattr(semantic_memory, 'word_cooccurrence', None) or [])
        memory_monitoring_enabled = bool(self.settings_manager.get('enable_memory_monitoring', True))

        memory_usage_mb = self._get_memory_usage_mb() if memory_monitoring_enabled else None
        available_memory_mb = self._get_available_memory_mb() if memory_monitoring_enabled else None

        memory_stats = {}
        hierarchical_memory = getattr(self.brain, 'hierarchical_memory', None)
        if hierarchical_memory is not None:
            try:
                memory_stats = hierarchical_memory.get_memory_stats() or {}
            except Exception:
                memory_stats = {}

        return StatusSnapshot(
            recent_quality=recent_quality,
            cluster_count=cluster_count,
            cooccurrence_count=cooccurrence_count,
            generation_stats=self._get_generation_stats(),
            memory_usage_mb=memory_usage_mb,
            available_memory_mb=available_memory_mb,
            memory_monitoring_enabled=memory_monitoring_enabled,
            conversation_memory_count=len(getattr(self.brain, 'conversation_memory', None) or []),
            topic_count=len(getattr(self.brain, 'current_topics', None) or {}),
            hierarchical_memory_stats=memory_stats,
        )

    def get_knowledge_snapshot(self, limit: int = 8) -> dict[str, Any]:
        knowledge_store = getattr(self.brain, 'knowledge_store', None)
        if knowledge_store is None:
            return {'success': False, 'message': 'Knowledge store not available.'}
        snapshot = knowledge_store.get_snapshot(limit=limit)
        snapshot['generation_stats'] = self._get_generation_stats()
        return snapshot

    def get_knowledge_concepts(self, query: str = '', limit: int = 25) -> dict[str, Any]:
        knowledge_store = getattr(self.brain, 'knowledge_store', None)
        if knowledge_store is None:
            return {'success': False, 'rows': [], 'message': 'Knowledge store not available.'}
        result = knowledge_store.get_concepts(query=query, limit=limit)
        result['generation_stats'] = self._get_generation_stats()
        return result

    def get_knowledge_facts(self, query: str = '', relation_type: str = '', limit: int = 25) -> dict[str, Any]:
        knowledge_store = getattr(self.brain, 'knowledge_store', None)
        if knowledge_store is None:
            return {'success': False, 'rows': [], 'message': 'Knowledge store not available.'}
        result = knowledge_store.get_facts(query=query, relation_type=relation_type, limit=limit)
        result['generation_stats'] = self._get_generation_stats()
        return result

    def get_knowledge_evidence(self, fact_key: str = '', query: str = '', limit: int = 25) -> dict[str, Any]:
        knowledge_store = getattr(self.brain, 'knowledge_store', None)
        if knowledge_store is None:
            return {'success': False, 'rows': [], 'message': 'Knowledge store not available.'}
        result = knowledge_store.get_fact_evidence(fact_key=fact_key, query=query, limit=limit)
        result['generation_stats'] = self._get_generation_stats()
        return result

    def get_knowledge_identity_traits(self, limit: int = 15) -> dict[str, Any]:
        knowledge_store = getattr(self.brain, 'knowledge_store', None)
        if knowledge_store is None:
            return {'success': False, 'rows': [], 'message': 'Knowledge store not available.'}
        result = knowledge_store.get_identity_traits(limit=limit)
        result['generation_stats'] = self._get_generation_stats()
        return result

    def get_response_plan_preview(self, user_input: str, limit: int = 3) -> dict[str, Any]:
        knowledge_store = getattr(self.brain, 'knowledge_store', None)
        if knowledge_store is None:
            return {'success': False, 'message': 'Knowledge store not available.'}
        result = knowledge_store.build_response_plan(user_input, limit=limit)
        result['generation_stats'] = self._get_generation_stats()
        return result

    def get_graph_reasoning_preview(self, user_input: str, limit: int = 3, max_depth: int = 2) -> dict[str, Any]:
        knowledge_store = getattr(self.brain, 'knowledge_store', None)
        if knowledge_store is None:
            return {'success': False, 'message': 'Knowledge store not available.'}
        result = knowledge_store.get_graph_reasoning_preview(user_input, limit=limit, max_depth=max_depth)
        result['generation_stats'] = self._get_generation_stats()
        return result

    def get_brain_patterns(self, limit: int = 100, context_len: int = 8) -> dict[str, Any]:
        rows = []
        data = None
        if getattr(self.brain, '_storage_backend', None):
            try:
                data = self.brain._storage_backend.get_patterns_sample(limit)
                data = [r for r in data if len(r) >= 13 and r[0] == context_len]
            except Exception as e:
                return {'error': str(e), 'rows': []}
        else:
            read_con = None
            try:
                read_con = sqlite3.connect(self.brain.db_file, check_same_thread=False)
                read_cur = read_con.cursor()
                read_cur.execute("""
                    SELECT context_len, word1, word2, word3, word4, word5, word6, word7, word8, next_word, priority, success_rate, usage_count
                    FROM dynamic_word_chain
                    WHERE context_len = ?
                    ORDER BY priority DESC
                    LIMIT ?
                """, (context_len, limit))
                data = read_cur.fetchall()
            finally:
                if read_con is not None:
                    try:
                        read_con.close()
                    except Exception:
                        pass

        for row in data or []:
            if not isinstance(row, (list, tuple)) or len(row) < 13:
                continue
            rows.append({
                'context_len': int(row[0]),
                'words': [row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8]],
                'next_word': row[9],
                'priority': int(row[10] or 0),
                'success_rate': float(row[11] or 0.0),
                'usage_count': int(row[12] or 0),
            })

        return {
            'rows': rows,
            'generation_stats': self._get_generation_stats(),
            'memory_usage_mb': self._get_memory_usage_mb(),
            'available_memory_mb': self._get_available_memory_mb(),
        }

    def get_attention_snapshot(self, limit: int = 50) -> dict[str, Any]:
        attention = getattr(self.brain, 'attention', None)
        weights = getattr(attention, 'focus_weights', None) or {}
        adaptive_rates = getattr(attention, 'adaptive_rates', None) or {}
        success_history = getattr(attention, 'success_history', None) or {}
        word_to_cluster = getattr(getattr(self.brain, 'semantic_memory', None), 'word_to_cluster', None) or {}
        rows = []
        for word, weight in sorted(weights.items(), key=lambda item: item[1], reverse=True)[:limit]:
            history = success_history.get(word, [])
            success_pct = (sum(history) / len(history) * 100) if history else 50.0
            rows.append({
                'word': word,
                'weight': float(weight),
                'adaptive_rate': float(adaptive_rates.get(word, 0.1)),
                'success_pct': float(success_pct),
                'cluster_id': word_to_cluster.get(word, "None"),
            })

        return {
            'rows': rows,
            'generation_stats': self._get_generation_stats(),
            'memory_usage_mb': self._get_memory_usage_mb(),
        }

    def get_context_performance_snapshot(self) -> dict[str, Any]:
        scorer = getattr(self.brain, 'context_scorer', None)
        context_levels = self.api_config.get('context_levels', [8, 6, 4, 2])
        rows = []
        for context_len in context_levels:
            score = scorer.get_context_score(context_len) if scorer else 0.0
            usage = getattr(scorer, 'context_usage_count', {}).get(str(context_len), 0) if scorer else 0
            if score > 0.7:
                performance = "Excellent"
            elif score > 0.5:
                performance = "Good"
            elif score > 0.3:
                performance = "Fair"
            else:
                performance = "Poor"
            rows.append({
                'context_len': context_len,
                'score': float(score),
                'usage_count': int(usage),
                'performance': performance,
            })

        semantic_memory = getattr(self.brain, 'semantic_memory', None)
        return {
            'rows': rows,
            'best_order': scorer.get_best_context_order() if scorer else [],
            'cluster_count': len(getattr(semantic_memory, 'clusters', None) or []),
            'cooccurrence_count': len(getattr(semantic_memory, 'word_cooccurrence', None) or []),
            'conversation_coherence': float(getattr(self.brain, 'conversation_coherence_score', 0.0)),
            'memory_usage_mb': self._get_memory_usage_mb(),
            'available_memory_mb': self._get_available_memory_mb(),
            'memory_safety_threshold': self.api_config.get('memory_safety_threshold'),
            'min_chunk_size': self.api_config.get('min_chunk_size'),
            'max_chunk_size': self.api_config.get('max_chunk_size'),
            'parallel_workers': self._safe_parallel_workers(),
            'generation_stats': self._get_generation_stats(),
        }

    def get_semantic_cluster_snapshot(self, limit: int = 20, example_words: int = 5, related_limit: int = 5) -> dict[str, Any]:
        semantic_memory = getattr(self.brain, 'semantic_memory', None)
        if semantic_memory is None:
            return {'rows': [], 'cluster_count': 0, 'cooccurrence_count': 0, 'example_relationships': []}

        clusters = getattr(semantic_memory, 'clusters', None) or {}
        cluster_strength = getattr(semantic_memory, 'cluster_strength', None) or {}
        rows = []
        for cluster_id, words in sorted(clusters.items(), key=lambda x: cluster_strength.get(x[0], 0), reverse=True)[:limit]:
            word_list = list(words)
            rows.append({
                'cluster_id': cluster_id,
                'strength': cluster_strength.get(cluster_id, 0),
                'size': len(word_list),
                'words': word_list,
            })

        example_relationships = []
        word_to_cluster = getattr(semantic_memory, 'word_to_cluster', None) or {}
        for word in list(word_to_cluster.keys())[:example_words]:
            try:
                related = list(semantic_memory.get_related_words(word, related_limit))
            except Exception:
                related = []
            if related:
                example_relationships.append({'word': word, 'related_words': related})

        total_words = sum(len(list(words)) for words in clusters.values())
        cluster_count = len(clusters)
        return {
            'rows': rows,
            'cluster_count': cluster_count,
            'total_words_in_clusters': total_words,
            'average_cluster_size': (total_words / cluster_count) if cluster_count else 0.0,
            'cooccurrence_count': len(getattr(semantic_memory, 'word_cooccurrence', None) or []),
            'example_relationships': example_relationships,
            'memory_usage_mb': self._get_memory_usage_mb(),
            'available_memory_mb': self._get_available_memory_mb(),
            'generation_stats': self._get_generation_stats(),
        }

    def get_generation_statistics_snapshot(self) -> dict[str, Any]:
        if getattr(self.brain, '_storage_backend', None):
            chain_count = self.brain._storage_backend.get_pattern_count()
            assoc_count = self.brain._storage_backend.get_association_count()
            unique_words = self.brain._storage_backend.get_unique_next_words_count()
            storage_mode = 'hsb'
        else:
            chain_count = assoc_count = unique_words = 0
            read_con = None
            db_file = self._get_db_file()
            if db_file:
                try:
                    read_con = sqlite3.connect(db_file, check_same_thread=False)
                    read_cur = read_con.cursor()
                    read_cur.execute("SELECT COUNT(*) FROM dynamic_word_chain")
                    r1 = read_cur.fetchone()
                    read_cur.execute("SELECT COUNT(*) FROM word_associations")
                    r2 = read_cur.fetchone()
                    read_cur.execute("SELECT COUNT(DISTINCT next_word) FROM dynamic_word_chain")
                    r3 = read_cur.fetchone()
                    chain_count = r1[0] if r1 is not None else 0
                    assoc_count = r2[0] if r2 is not None else 0
                    unique_words = r3[0] if r3 is not None else 0
                finally:
                    if read_con is not None:
                        try:
                            read_con.close()
                        except Exception:
                            pass
                storage_mode = 'sqlite'
            else:
                storage_mode = 'unavailable'

        semantic_memory = getattr(self.brain, 'semantic_memory', None)
        response_quality_history = getattr(self.brain, 'response_quality_history', None) or []
        quality_numbers = [float(x) for x in response_quality_history if isinstance(x, (int, float))]
        context_levels = self.api_config.get('context_levels', [8, 6, 4, 2])
        context_rows = []
        scorer = getattr(self.brain, 'context_scorer', None)
        for context_len in context_levels:
            score = scorer.get_context_score(context_len) if scorer else 0.0
            usage = getattr(scorer, 'context_usage_count', {}).get(str(context_len), 0) if scorer else 0
            context_rows.append({
                'context_len': context_len,
                'score': float(score),
                'usage_count': int(usage),
            })

        return {
            'generation_stats': self._get_generation_stats(),
            'storage_mode': storage_mode,
            'chain_count': int(chain_count),
            'association_count': int(assoc_count),
            'unique_words': int(unique_words),
            'cluster_count': len(getattr(semantic_memory, 'clusters', None) or []),
            'cooccurrence_count': len(getattr(semantic_memory, 'word_cooccurrence', None) or []),
            'average_quality': (sum(quality_numbers) / len(quality_numbers)) if quality_numbers else 0.0,
            'recent_quality': (sum(quality_numbers[-10:]) / len(quality_numbers[-10:])) if quality_numbers[-10:] else 0.0,
            'quality_history_length': len(response_quality_history),
            'context_rows': context_rows,
            'conversation_memory_count': len(getattr(self.brain, 'conversation_memory', None) or []),
            'topic_words_count': len(getattr(self.brain, 'topic_words', None) or {}),
            'current_topics_count': len(getattr(self.brain, 'current_topics', None) or {}),
            'conversation_coherence': float(getattr(self.brain, 'conversation_coherence_score', 0.0)),
            'model_stats': {
                'vocab_size': getattr(getattr(self.brain, 'model', None), 'vocab_size', 'N/A'),
                'context_size': getattr(getattr(self.brain, 'model', None), 'context_size', 'N/A'),
                'hidden_size': getattr(getattr(self.brain, 'model', None), 'hidden_size', 'N/A'),
                'input_size': getattr(getattr(self.brain, 'model', None), 'input_size', 'N/A'),
            },
            'memory_usage_mb': self._get_memory_usage_mb(),
            'available_memory_mb': self._get_available_memory_mb(),
            'memory_usage_percent': self._get_memory_usage_percent(),
            'memory_safety_threshold': self.api_config.get('memory_safety_threshold'),
            'base_chunk_size_words': self.api_config.get('chunk_size_words'),
            'min_chunk_size': self.api_config.get('min_chunk_size'),
            'max_chunk_size': self.api_config.get('max_chunk_size'),
            'optimal_chunk_size': self._safe_optimal_chunk_size(1000000),
            'parallel_workers': self._safe_parallel_workers(),
            'batch_count': int(getattr(self.brain, 'batch_count', 0)),
            'feature_runtime': self.get_feature_runtime_snapshot(),
        }

    def get_vocab_id_storage_snapshot(self) -> dict[str, Any]:
        getter = getattr(self.brain, 'get_vocab_id_storage_snapshot', None)
        if callable(getter):
            payload = getter() or {}
            payload['success'] = True
            return payload
        return {
            'success': True,
            'mode': 'off',
            'active': False,
            'lexicon_size': 0,
            'id_chain_count': 0,
            'text_chain_count': 0,
        }

    def migrate_vocab_id_storage(self, batch_size: int = 5000, offset_rowid: int = 0) -> dict[str, Any]:
        migrator = getattr(self.brain, 'migrate_vocab_id_storage', None)
        if not callable(migrator):
            return {
                'success': False,
                'message': 'Vocab ID migration is unavailable on this brain.',
                'migrated': 0,
                'done': True,
            }
        return migrator(batch_size=int(batch_size or 5000), offset_rowid=int(offset_rowid or 0))

    def get_feature_runtime_snapshot(self) -> dict[str, Any]:
        feature_flags = copy.deepcopy(getattr(self.settings_manager, 'settings', {}).get('features', {}) or {})
        enabled_feature_flags = sorted([name for name, enabled in feature_flags.items() if enabled])
        feature_objects = {
            'reasoning_engine': getattr(self.brain, 'reasoning_engine', None),
            'adaptive_learning': getattr(self.brain, 'adaptive_learning', None),
            'creative_generator': getattr(self.brain, 'creative_generator', None),
            'hierarchical_memory': getattr(self.brain, 'hierarchical_memory', None),
            'critic': getattr(self.brain, 'critic', None),
            'confidence_gate': getattr(self.brain, 'confidence_gate', None),
            'anti_loop_filter': getattr(self.brain, 'anti_loop_filter', None),
            'meta_memory': getattr(self.brain, 'meta_memory', None),
            'curiosity': getattr(self.brain, 'curiosity', None),
            'env_feedback': getattr(self.brain, 'env_feedback', None),
            'autotune': getattr(self.brain, 'autotune', None),
            'response_learning': getattr(self.brain, 'response_learning', None),
            'truth_fact_table': getattr(self.brain, 'truth_fact_table', None),
            'topic_detection': getattr(self.brain, 'topic_detection', None),
            'knowledge_store': getattr(self.brain, 'knowledge_store', None),
            'cognitive_core': getattr(self.brain, 'cognitive_core', None),
        }
        initialized_features = sorted([name for name, feature in feature_objects.items() if feature is not None])
        runtime_feature_flags = {
            'critic': feature_objects.get('critic') is not None,
            'confidence_gate': feature_objects.get('confidence_gate') is not None,
            'anti_loop_filter': feature_objects.get('anti_loop_filter') is not None,
            'meta_memory': feature_objects.get('meta_memory') is not None,
            'curiosity': feature_objects.get('curiosity') is not None,
            'env_feedback': feature_objects.get('env_feedback') is not None,
            'autotune': feature_objects.get('autotune') is not None,
            'response_learning': feature_objects.get('response_learning') is not None,
            'truth_fact_table': feature_objects.get('truth_fact_table') is not None,
            'topic_detection': feature_objects.get('topic_detection') is not None,
            'knowledge_store': feature_objects.get('knowledge_store') is not None,
            'cognitive_core': feature_objects.get('cognitive_core') is not None,
        }
        active_feature_flags = sorted([name for name, active in runtime_feature_flags.items() if active])
        runtime_mismatches = sorted([
            name for name, enabled in feature_flags.items()
            if bool(enabled) != bool(runtime_feature_flags.get(name, False))
        ])
        return {
            'enhanced_intelligence_enabled': bool(getattr(self.brain, 'enhanced_intelligence_enabled', False)),
            'advanced_reasoning_enabled': bool(self.settings_manager.get('advanced_reasoning', True)),
            'adaptive_learning_enabled': bool(self.settings_manager.get('adaptive_learning', True)),
            'feature_flags': feature_flags,
            'enabled_feature_flags': enabled_feature_flags,
            'runtime_feature_flags': runtime_feature_flags,
            'active_feature_flags': active_feature_flags,
            'runtime_mismatches': runtime_mismatches,
            'reasoning_engine_active': feature_objects.get('reasoning_engine') is not None,
            'adaptive_learning_active': feature_objects.get('adaptive_learning') is not None,
            'creative_generator_active': feature_objects.get('creative_generator') is not None,
            'initialized_features': initialized_features,
            'initialized_count': len(initialized_features),
            'generation_stats': self._get_generation_stats(),
            'adaptive_learning_summary': self._safe_feature_call(feature_objects.get('adaptive_learning'), 'get_performance_summary', ""),
            'adaptive_learning_health': self._safe_feature_call(feature_objects.get('adaptive_learning'), 'get_learning_health_snapshot', {'success': False}),
            'hierarchical_memory_stats': self._safe_feature_call(feature_objects.get('hierarchical_memory'), 'get_memory_stats', {}),
            'meta_memory_stats': self._safe_feature_call(feature_objects.get('meta_memory'), 'get_weakness_stats', {}),
            'environment_feedback_stats': self._safe_feature_call(feature_objects.get('env_feedback'), 'get_feedback_stats', {}),
            'autotune_stats': self._safe_feature_call(feature_objects.get('autotune'), 'get_tuning_stats', {}),
            'response_learning_stats': self._safe_feature_call(feature_objects.get('response_learning'), 'get_learning_stats', {}),
            'truth_fact_stats': self._safe_feature_call(feature_objects.get('truth_fact_table'), 'get_fact_stats', {}),
            'topic_detection_stats': self._safe_feature_call(feature_objects.get('topic_detection'), 'get_topic_stats', {}),
            'knowledge_stats': self._safe_feature_call(feature_objects.get('knowledge_store'), 'get_snapshot', {'success': False}),
            'cognitive_core_stats': self._safe_feature_call(feature_objects.get('cognitive_core'), 'get_snapshot', {'active': False}),
        }

    def get_learning_health_snapshot(self) -> dict[str, Any]:
        adaptive_learning = getattr(self.brain, 'adaptive_learning', None)
        if adaptive_learning is None:
            return {
                'success': False,
                'enabled': bool(self.settings_manager.get('adaptive_learning', True)),
                'active': False,
                'message': 'Adaptive learning is not active in the current runtime.',
            }
        snapshot = self._safe_feature_call(adaptive_learning, 'get_learning_health_snapshot', {'success': False})
        if isinstance(snapshot, dict):
            snapshot.setdefault('enabled', bool(self.settings_manager.get('adaptive_learning', True)))
            snapshot.setdefault('active', True)
            return snapshot
        return {
            'success': False,
            'enabled': bool(self.settings_manager.get('adaptive_learning', True)),
            'active': True,
            'message': 'Adaptive learning did not return a structured health snapshot.',
        }

    def get_runtime_bootstrap_snapshot(self) -> dict[str, Any]:
        status = self.get_status_snapshot()
        return {
            'settings': self.get_settings_snapshot(),
            'hardware': self.get_hardware_profile(),
            'storage': self.get_storage_snapshot(),
            'vocab_id_storage': self.get_vocab_id_storage_snapshot(),
            'status': status.to_dict() if hasattr(status, 'to_dict') else asdict(status),
            'feature_runtime': self.get_feature_runtime_snapshot(),
            'generation': self.get_generation_statistics_snapshot(),
            'memory': self.get_memory_statistics_snapshot(),
            'knowledge': self.get_knowledge_snapshot(limit=6),
        }

    def get_memory_statistics_snapshot(self) -> dict[str, Any]:
        hardware_profile = self._get_hardware_profile()
        total_memory_mb = psutil.virtual_memory().total / 1024 / 1024
        cpu_logical = psutil.cpu_count() or 0
        cpu_physical = psutil.cpu_count(logical=False) or 0
        parallel_worker_limit = self.api_config.get('parallel_worker_limit')
        file_sizes = [10000, 50000, 100000, 500000, 1000000, 2000000, 5000000]
        file_recommendations = []
        parallel_workers = self._safe_parallel_workers()
        for file_size in file_sizes:
            chunk_size = self._safe_optimal_chunk_size(file_size)
            chunks_needed = (file_size + chunk_size - 1) // max(chunk_size, 1)
            file_recommendations.append({
                'file_size_words': file_size,
                'chunk_size': chunk_size,
                'chunks_needed': chunks_needed,
                'processing_mode': "Parallel" if parallel_workers > 1 and chunks_needed > 1 else "Sequential",
            })

        safety_threshold = float(self.api_config.get('memory_safety_threshold', 0.85) or 0.85)
        min_chunk_size = int(self.api_config.get('min_chunk_size', 10000) or 10000)
        max_chunk_size = int(self.api_config.get('max_chunk_size', 500000) or 500000)
        pressure_scenarios = []
        for scenario_name, available_mb in [
            ("Low Memory (1GB available)", 1024),
            ("Medium Memory (4GB available)", 4096),
            ("High Memory (8GB+ available)", 8192),
        ]:
            safe_memory_mb = available_mb * (1 - safety_threshold)
            estimated_mb_per_word = 0.001
            safe_chunk_size = int(safe_memory_mb / estimated_mb_per_word)
            safe_chunk_size = max(min_chunk_size, min(max_chunk_size, safe_chunk_size))
            workers = 1 if available_mb < 2000 else (min(2, parallel_worker_limit or 2) if available_mb < 4000 else parallel_worker_limit)
            pressure_scenarios.append({
                'scenario': scenario_name,
                'chunk_size': safe_chunk_size,
                'parallel_workers': workers,
                'processing_mode': 'Parallel' if (workers or 0) > 1 else 'Sequential',
            })

        return {
            'memory_usage_mb': self._get_memory_usage_mb(),
            'available_memory_mb': self._get_available_memory_mb(),
            'memory_usage_percent': self._get_memory_usage_percent(),
            'total_memory_mb': total_memory_mb,
            'memory_safety_threshold': safety_threshold,
            'batch_size': self.api_config.get('batch_size'),
            'large_file_threshold': self.api_config.get('large_file_threshold'),
            'chunk_size_words': self.api_config.get('chunk_size_words'),
            'min_chunk_size': min_chunk_size,
            'max_chunk_size': max_chunk_size,
            'parallel_worker_limit': parallel_worker_limit,
            'optimal_chunk_size': self._safe_optimal_chunk_size(1000000),
            'parallel_workers': parallel_workers,
            'recommended_parallel_workers': hardware_profile.get('recommended_parallel_workers'),
            'recommended_chunk_size': hardware_profile.get('recommended_chunk_size'),
            'recommended_memory_threshold': hardware_profile.get('recommended_memory_threshold'),
            'recommended_gpu_type': hardware_profile.get('recommended_gpu_type'),
            'recommended_gpu_index': hardware_profile.get('recommended_gpu_index'),
            'should_reduce_batch_size': self._safe_should_reduce_batch_size(),
            'cpu_logical': cpu_logical,
            'cpu_physical': cpu_physical,
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'file_recommendations': file_recommendations,
            'pressure_scenarios': pressure_scenarios,
        }

    def save_recent_patterns_to_initial_knowledge(self, max_patterns: int = 15) -> dict[str, Any]:
        chains = []
        if getattr(self.brain, '_storage_backend', None):
            try:
                rows = self.brain._storage_backend.get_patterns_sample(1000)
                rows = [r for r in rows if len(r) >= 13 and r[0] == 8 and r[10] > 15 and r[12] > 2]
                rows.sort(key=lambda r: float(r[10]) * math.log(int(r[12]) + 1), reverse=True)
                chains = [r[1:9] + (r[9], r[10], r[11], r[12]) for r in rows[:30]]
            except Exception:
                chains = []
        else:
            read_con = None
            try:
                read_con = sqlite3.connect(self.brain.db_file, check_same_thread=False)
                read_cur = read_con.cursor()
                read_cur.execute("""
                    SELECT word1, word2, word3, word4, word5, word6, word7, word8, next_word, priority, success_rate, usage_count
                    FROM dynamic_word_chain
                    WHERE priority > 15 AND context_len = 8 AND usage_count > 2
                    ORDER BY priority DESC
                    LIMIT 1000
                """)
                chains = read_cur.fetchall() or []
                chains.sort(key=lambda r: float(r[9]) * math.log(int(r[11]) + 1), reverse=True)
                chains = chains[:30]
            finally:
                if read_con is not None:
                    try:
                        read_con.close()
                    except Exception:
                        pass

        learned_sentences = []
        for chain in chains or []:
            if not chain or len(chain) < 3:
                continue
            words = [w for w in chain[:-3] if w != '<PAD>']
            words.append(chain[-3])
            if len(words) > 4:
                sentence = " ".join(words).strip()
                if sentence and sentence not in learned_sentences:
                    learned_sentences.append(sentence)

        knowledge_file = self.api_config.get('initial_knowledge_file')
        if not knowledge_file or not isinstance(knowledge_file, str):
            return {'success': False, 'saved_count': 0, 'message': 'Initial knowledge file path is not configured.'}

        current_content = ""
        if os.path.exists(knowledge_file):
            try:
                with open(knowledge_file, 'r', encoding='utf-8') as f:
                    current_content = f.read().strip()
            except (OSError, IOError, UnicodeDecodeError):
                current_content = ""

        if not learned_sentences:
            return {'success': True, 'saved_count': 0, 'file_path': knowledge_file, 'message': 'No high-quality patterns found to save.'}

        semantic_memory = getattr(self.brain, 'semantic_memory', None)
        cluster_count = len(getattr(semantic_memory, 'clusters', {}) or {})
        try:
            gen_stats = self.brain.get_generation_stats()
        except Exception:
            gen_stats = "N/A"
        memory_usage = self._get_memory_usage_mb()

        new_content = current_content + f"\n\n# Learned patterns ({time.strftime('%Y-%m-%d')}):\n"
        new_content += f"# {gen_stats}\n"
        new_content += f"# Semantic clusters: {cluster_count}\n"
        new_content += f"# Memory usage: {memory_usage:.1f}MB\n"
        new_content += "\n".join(learned_sentences[:max_patterns])

        os.makedirs(os.path.dirname(knowledge_file), exist_ok=True)
        with open(knowledge_file, 'w', encoding='utf-8') as f:
            f.write(new_content)

        saved_count = len(learned_sentences[:max_patterns])
        return {
            'success': True,
            'saved_count': saved_count,
            'file_path': knowledge_file,
            'message': f"Added {saved_count} high-quality statistical patterns to initial knowledge file.",
        }

    def create_clone_brain(self):
        brain_factory = self.api_config.get('brain_factory')
        db_file = self._get_db_file()
        if not callable(brain_factory):
            raise RuntimeError('Clone brain factory is not configured.')
        if not db_file:
            raise RuntimeError('Brain database path is not configured.')
        return brain_factory(db_file, is_clone=True)

    def close_brain_instance(self, brain) -> None:
        if brain is None:
            return
        con = getattr(brain, 'con', None)
        if con is not None:
            try:
                con.close()
            except Exception:
                pass

    def get_memory_pressure_state(self, threshold: float | None = None) -> dict[str, Any]:
        threshold_value = threshold
        if threshold_value is None:
            threshold_value = self.api_config.get('memory_safety_threshold', 0.85)
        try:
            threshold_value = float(threshold_value or 0.85)
        except Exception:
            threshold_value = 0.85
        memory_usage_percent = self._get_memory_usage_percent()
        return {
            'memory_usage_mb': self._get_memory_usage_mb(),
            'available_memory_mb': self._get_available_memory_mb(),
            'memory_usage_percent': memory_usage_percent,
            'threshold': threshold_value,
            'memory_pressure': memory_usage_percent > threshold_value,
        }

    def force_garbage_collection(self) -> dict[str, Any]:
        if hasattr(self.memory_manager, 'force_garbage_collection'):
            self.memory_manager.force_garbage_collection()
        snapshot = self.get_memory_pressure_state()
        snapshot['success'] = True
        return snapshot

    def get_chunk_processing_config(self) -> dict[str, Any]:
        return {
            'app_dir': self.api_config.get('app_dir') or os.path.dirname(os.path.abspath(__file__)),
            'db_file': self._get_db_file(),
            'runtime_module': self.api_config.get('runtime_module', self.api_config.get('desktop_module', 'ai_astra_main.backend_runtime')),
            'desktop_module': self.api_config.get('desktop_module', self.api_config.get('runtime_module', 'ai_astra_main.backend_runtime')),
            'use_hsb_backend': bool(self.settings_manager.get('use_hsb_backend', False)),
            'base_priority_boost': int(self.api_config.get('training_chunk_priority_boost', 2) or 2),
        }

    def get_api_manifest(self) -> dict[str, Any]:
        transport_methods = self.get_transport_method_names()
        method_specs = self.get_transport_method_specs()
        control_specs = self.get_transport_control_specs()
        category_map = self.get_transport_category_map()
        max_batch_size = int(self.api_config.get('transport_max_batch_size', 32) or 32)
        return {
            'name': 'ai_astra_backend_api',
            'api_version': 1,
            'app_dir': self.api_config.get('app_dir') or os.path.dirname(os.path.abspath(__file__)),
            'storage': self.get_storage_snapshot(),
            'training_extensions': self._get_training_extensions(),
            'capabilities': {
                'generation': True,
                'response_planning': True,
                'graph_reasoning': True,
                'feedback': True,
                'inspection': True,
                'hardware_adaptation': True,
                'live_learning_health': True,
                'goal_driven_cognition': True,
                'homeostatic_control': True,
                'knowledge_inspection': True,
                'training_low_level': True,
                'training_high_level': True,
                'import_export': True,
                'runtime_bootstrap_snapshot': True,
                'feature_runtime_snapshot': True,
                'transport_session_info': True,
            },
            'transport': {
                'protocols': ['stdio-json', 'http-json'],
                'http_routes': {
                    'health': '/health',
                    'session': '/session',
                    'manifest': '/manifest',
                    'methods': '/methods',
                    'method_detail': '/methods/{name}',
                    'invoke': '/api',
                    'batch': '/api/batch',
                },
                'control_methods': self.get_transport_control_names(),
                'control_method_specs': control_specs,
                'session_model': {
                    'stateful_process': True,
                    'shared_backend_instance': True,
                    'session_control_method': 'get_session_info',
                    'shutdown_control_method': 'shutdown',
                },
                'request_envelope': {
                    'single': {
                        'id': 1,
                        'method': 'generate_response',
                        'params': {'user_input': 'what is the sgm model?'},
                    },
                    'batch': [
                        {'id': 1, 'method': 'get_runtime_bootstrap_snapshot', 'params': {}},
                        {'id': 2, 'method': 'generate_response', 'params': {'user_input': 'hello'}},
                    ],
                },
                'response_envelope': {
                    'single': {
                        'id': 1,
                        'ok': True,
                        'result': {'response': '...'},
                    },
                    'batch': {
                        'ok': True,
                        'batch': True,
                        'results': [
                            {'id': 1, 'ok': True, 'result': {}},
                            {'id': 2, 'ok': True, 'result': {}},
                        ],
                    },
                    'error': {
                        'id': 1,
                        'ok': False,
                        'error_code': 'invalid_params',
                        'error': 'Human-readable explanation',
                        'method': 'generate_response',
                    },
                },
                'batch_requests': {
                    'supported': True,
                    'style': 'json-array',
                    'max_items': max_batch_size,
                },
                'method_categories': category_map,
                'read_only_methods': sorted([name for name in transport_methods if not method_specs.get(name, {}).get('mutates_state')]),
                'mutating_methods': sorted([name for name in transport_methods if method_specs.get(name, {}).get('mutates_state')]),
                'frontend_workflows': self.get_transport_workflows(),
                'method_count': len(transport_methods),
                'methods': transport_methods,
                'method_specs': method_specs,
            },
        }

    def collect_training_files(self, paths: list[str], extensions: list[str] | None = None) -> dict[str, Any]:
        normalized_extensions = self._normalize_training_extensions(extensions)
        collected: list[str] = []
        missing: list[str] = []
        skipped: list[str] = []

        for raw_path in paths or []:
            if not isinstance(raw_path, str) or not raw_path.strip():
                continue
            path = os.path.abspath(raw_path)
            if os.path.isdir(path):
                for root, _, files in os.walk(path):
                    for name in sorted(files):
                        ext = os.path.splitext(name)[1].lower()
                        if ext in normalized_extensions:
                            collected.append(os.path.join(root, name))
                        else:
                            skipped.append(os.path.join(root, name))
            elif os.path.isfile(path):
                ext = os.path.splitext(path)[1].lower()
                if ext in normalized_extensions:
                    collected.append(path)
                else:
                    skipped.append(path)
            else:
                missing.append(path)

        deduped: list[str] = []
        seen: set[str] = set()
        for path in collected:
            norm = os.path.normcase(os.path.abspath(path))
            if norm in seen:
                continue
            seen.add(norm)
            deduped.append(path)

        return {
            'success': True,
            'paths': deduped,
            'count': len(deduped),
            'extensions': sorted(normalized_extensions),
            'missing': missing,
            'skipped': skipped,
        }

    def get_training_plan(self, file_path: str) -> dict[str, Any]:
        if not file_path:
            return {'success': False, 'message': 'No training file provided.'}
        if not os.path.exists(file_path):
            return {'success': False, 'message': f'File does not exist: {file_path}'}

        file_size_bytes = os.path.getsize(file_path)
        estimated_word_count = self._estimate_word_count_from_sample(file_path)
        chunk_size = max(1, self._safe_optimal_chunk_size(estimated_word_count))
        # SQLite has a single writer. Parsing in multiple processes currently
        # clones the entire brain per worker, so advertising those workers here
        # was both misleading and slower than the coalescing single-writer path.
        parallel_workers = 1
        estimated_chunks = max(1, math.ceil(estimated_word_count / chunk_size))
        parallel_chunk_limit = 1
        gpu_accelerator = self.api_config.get('gpu_accelerator')
        use_gpu = bool(self.settings_manager.get('gpu_acceleration_enabled', False) and getattr(gpu_accelerator, 'gpu_available', False))
        source_profile = self._get_training_source_profile(file_path)
        content_signature = self._get_training_content_signature(file_path) if file_size_bytes > 0 else 'empty'
        return {
            'success': True,
            'file_path': file_path,
            'file_size_bytes': file_size_bytes,
            'estimated_word_count': estimated_word_count,
            'chunk_size': chunk_size,
            'parallel_workers': parallel_workers,
            'estimated_chunks': estimated_chunks,
            'parallel_chunk_limit': parallel_chunk_limit,
            'write_strategy': 'coalesced_single_writer',
            'use_gpu': use_gpu,
            'gpu_type': getattr(gpu_accelerator, 'gpu_type', 'CPU') if use_gpu else 'CPU',
            'enable_garbage_collection': bool(self.settings_manager.get('enable_garbage_collection', True)),
            'source_category': source_profile['category'],
            'source_weight': source_profile['weight'],
            'content_signature': content_signature,
        }

    def learn_from_training_chunk(
        self,
        chunk: str,
        base_priority_boost: int = 2,
        source_path: str = '',
        source_label: str = '',
        source_category: str = 'general_text',
        source_weight: float = 1.0,
    ) -> dict[str, Any]:
        words_processed = int(self.brain.learn_from_text_optimized(chunk, base_priority_boost=base_priority_boost))
        knowledge_result = {'success': False, 'fact_count': 0}
        if hasattr(self.brain, 'learn_knowledge_from_text'):
            knowledge_result = self.brain.learn_knowledge_from_text(
                chunk,
                source_type='training_text',
                source_path=source_path,
                source_label=source_label,
                source_category=source_category,
                source_weight=source_weight,
            )
        return {
            'success': True,
            'words_processed': words_processed,
            'knowledge': knowledge_result,
            'memory': self.get_memory_pressure_state(),
        }

    def propose_gated_learning(
        self,
        response_text: str = '',
        user_text: str = '',
        commit: bool = True,
        priority_boost: int = 2,
        seed: int = 1729,
        prompts_path: str = '',
        baseline_path: str = '',
        require_absolute_pass: bool = True,
    ) -> dict[str, Any]:
        """Propose durable learning only when the held-out gate stays healthy."""
        try:
            from gated_learning import propose_gated_learning as _propose
        except ImportError:
            from .gated_learning import propose_gated_learning as _propose
        return _propose(
            self,
            response_text=response_text,
            user_text=user_text,
            commit=bool(commit),
            priority_boost=int(priority_boost or 2),
            seed=int(seed or 1729),
            prompts_path=Path(prompts_path) if prompts_path else None,
            baseline_path=Path(baseline_path) if baseline_path else None,
            require_absolute_pass=bool(require_absolute_pass),
        )

    def train_files(
        self,
        paths: list[str],
        extensions: list[str] | None = None,
        chunk_size_override: int | None = None,
        base_priority_boost: int = 2,
    ) -> dict[str, Any]:
        discovery = self.collect_training_files(paths, extensions=extensions)
        file_paths = discovery.get('paths', [])
        if not file_paths:
            return {
                'success': False,
                'message': 'No text or markdown files found to train on.',
                'discovery': discovery,
            }

        total_words_processed = 0
        file_summaries: list[dict[str, Any]] = []
        seen_signatures: dict[str, str] = {}
        self._ensure_training_history_table()
        for file_path in file_paths:
            plan = self.get_training_plan(file_path)
            if not plan.get('success'):
                file_summaries.append({
                    'file_path': file_path,
                    'success': False,
                    'message': plan.get('message', 'Training plan failed.'),
                })
                continue

            source_category = str(plan.get('source_category', 'general_text') or 'general_text')
            try:
                source_weight = float(plan.get('source_weight', 1.0) or 1.0)
            except (TypeError, ValueError):
                source_weight = 1.0
            content_signature = str(plan.get('content_signature', '') or '')
            effective_priority_boost = self._get_effective_training_priority_boost(base_priority_boost, source_weight)

            prior_state = self._get_training_history(content_signature)
            if prior_state and prior_state.get('status') == 'completed':
                file_summaries.append({
                    'file_path': file_path,
                    'success': True,
                    'trained': False,
                    'words_processed': 0,
                    'chunks': 0,
                    'skipped': 'already_trained',
                    'previous_words_processed': int(prior_state.get('words_processed', 0) or 0),
                    'source_category': source_category,
                    'source_weight': source_weight,
                    'effective_priority_boost': effective_priority_boost,
                })
                continue

            if os.path.getsize(file_path) == 0:
                file_summaries.append({
                    'file_path': file_path,
                    'success': True,
                    'trained': False,
                    'words_processed': 0,
                    'chunks': 0,
                    'skipped': 'empty',
                    'source_category': source_category,
                    'source_weight': source_weight,
                    'effective_priority_boost': effective_priority_boost,
                })
                continue

            if content_signature and content_signature != 'empty':
                duplicate_of = seen_signatures.get(content_signature)
                if duplicate_of:
                    file_summaries.append({
                        'file_path': file_path,
                        'success': True,
                        'trained': False,
                        'words_processed': 0,
                        'chunks': 0,
                        'skipped': 'duplicate_content',
                        'duplicate_of': duplicate_of,
                        'source_category': source_category,
                        'source_weight': source_weight,
                        'effective_priority_boost': effective_priority_boost,
                    })
                    continue
                seen_signatures[content_signature] = file_path

            chunk_size = chunk_size_override if chunk_size_override is not None else int(plan.get('chunk_size', 10000) or 10000)
            words_processed = 0
            chunk_count = 0
            knowledge_fact_count = 0
            completed_chunks = 0

            self._record_training_history(
                content_signature,
                file_path,
                status='running',
                words_processed=0,
                chunks_completed=0,
                chunk_size=chunk_size,
            )

            try:
                for chunk_text in self._iter_text_chunks_from_file(file_path, chunk_size):
                    if not chunk_text.strip():
                        continue
                    chunk_count += 1
                    chunk_result = self.learn_from_training_chunk(
                        chunk_text,
                        base_priority_boost=effective_priority_boost,
                        source_path=file_path,
                        source_label=os.path.basename(file_path),
                        source_category=source_category,
                        source_weight=source_weight,
                    )
                    chunk_words = int(chunk_result.get('words_processed', 0))
                    words_processed += chunk_words
                    total_words_processed += chunk_words
                    knowledge_fact_count += int((chunk_result.get('knowledge') or {}).get('fact_count', 0))
                    completed_chunks = chunk_count
                    self._record_training_history(
                        content_signature,
                        file_path,
                        status='running',
                        words_processed=words_processed,
                        chunks_completed=completed_chunks,
                        chunk_size=chunk_size,
                    )
                self.flush_training_batches()
                self._record_training_history(
                    content_signature,
                    file_path,
                    status='completed',
                    words_processed=words_processed,
                    chunks_completed=completed_chunks,
                    chunk_size=chunk_size,
                )
                file_summaries.append({
                    'file_path': file_path,
                    'success': True,
                    'trained': True,
                    'words_processed': words_processed,
                    'chunks': chunk_count,
                    'estimated_chunks': int(plan.get('estimated_chunks', 0) or 0),
                    'chunk_size': chunk_size,
                    'source_category': source_category,
                    'source_weight': source_weight,
                    'effective_priority_boost': effective_priority_boost,
                    'knowledge_fact_count': knowledge_fact_count,
                })
            except Exception as e:
                self._record_training_history(
                    content_signature,
                    file_path,
                    status='failed',
                    words_processed=words_processed,
                    chunks_completed=completed_chunks,
                    chunk_size=chunk_size,
                    error_message=str(e),
                )
                file_summaries.append({
                    'file_path': file_path,
                    'success': False,
                    'message': str(e),
                    'chunks': chunk_count,
                    'words_processed': words_processed,
                    'source_category': source_category,
                    'source_weight': source_weight,
                    'effective_priority_boost': effective_priority_boost,
                    'knowledge_fact_count': knowledge_fact_count,
                })

            memory_state = self.get_memory_pressure_state()
            if memory_state.get('memory_pressure'):
                self.force_garbage_collection()

        finalize = self.finalize_training_run()
        trained_count = sum(1 for item in file_summaries if item.get('trained'))
        skipped_count = sum(1 for item in file_summaries if item.get('skipped'))
        success_count = sum(1 for item in file_summaries if item.get('success'))
        failure_count = sum(1 for item in file_summaries if not item.get('success'))
        finalize_success = bool(finalize.get('success', True))
        overall_success = bool(file_summaries) and failure_count == 0 and finalize_success
        if overall_success and trained_count:
            message = f"Training completed across {trained_count} file(s)."
            if skipped_count:
                message += f" {skipped_count} file(s) were skipped."
        elif overall_success and skipped_count:
            message = "No new training content was processed; all files were skipped."
        elif success_count > 0:
            message = f"Training completed with partial failures ({success_count} succeeded, {failure_count} failed)."
        else:
            message = "Training did not complete successfully."
        return {
            'success': overall_success,
            'message': message,
            'discovery': discovery,
            'files': file_summaries,
            'success_count': success_count,
            'failure_count': failure_count,
            'trained_count': trained_count,
            'skipped_count': skipped_count,
            'knowledge_fact_count': sum(int(item.get('knowledge_fact_count', 0) or 0) for item in file_summaries),
            'total_words_processed': total_words_processed,
            'finalize': finalize,
        }

    def flush_training_batches(self) -> dict[str, Any]:
        if hasattr(self.brain, '_flush_batch_operations'):
            self.brain._flush_batch_operations()
        return {'success': True}

    def finalize_training_run(self) -> dict[str, Any]:
        result = self.sync_brain_state()
        summary = self.get_training_summary()
        result['summary'] = summary
        return result

    def merge_chunk_training_results(self, chunk_results: list[dict[str, Any]]) -> dict[str, Any]:
        aggregated_patterns = {}
        aggregated_associations = {}
        knowledge_fact_count = 0
        for result in chunk_results:
            if not result.get('success'):
                continue
            knowledge_summary = result.get('knowledge_summary') or {}
            knowledge_fact_count += int(knowledge_summary.get('fact_count', 0) or 0)
            for pattern in result.get('patterns', []):
                if not isinstance(pattern, (list, tuple)) or len(pattern) < 13:
                    continue
                key = tuple(pattern[:10])
                bucket = aggregated_patterns.setdefault(key, {'priority': 0, 'usage_count': 0, 'success_weight': 0.0})
                try:
                    priority = int(pattern[10] or 0)
                except (TypeError, ValueError):
                    priority = 0
                try:
                    success_rate = float(pattern[11] or 0.5)
                except (TypeError, ValueError):
                    success_rate = 0.5
                try:
                    usage_count = int(pattern[12] or 0)
                except (TypeError, ValueError):
                    usage_count = 0
                weight = max(usage_count, 1)
                bucket['priority'] += priority
                bucket['usage_count'] += usage_count
                bucket['success_weight'] += success_rate * weight
            for association in result.get('associations', []):
                if not isinstance(association, (list, tuple)) or len(association) < 5:
                    continue
                key = tuple(association[:2])
                bucket = aggregated_associations.setdefault(key, {'priority': 0, 'usage_count': 0, 'success_weight': 0.0})
                try:
                    priority = int(association[2] or 0)
                except (TypeError, ValueError):
                    priority = 0
                try:
                    success_rate = float(association[3] or 0.5)
                except (TypeError, ValueError):
                    success_rate = 0.5
                try:
                    usage_count = int(association[4] or 0)
                except (TypeError, ValueError):
                    usage_count = 0
                weight = max(usage_count, 1)
                bucket['priority'] += priority
                bucket['usage_count'] += usage_count
                bucket['success_weight'] += success_rate * weight

        merge_con = None
        owns_merge_con = False
        pattern_data = []
        assoc_data = []
        try:
            merge_con = getattr(self.brain, 'con', None)
            if merge_con is None:
                merge_con = sqlite3.connect(self.brain.db_file, check_same_thread=False)
                owns_merge_con = True
            else:
                try:
                    merge_con.commit()
                except Exception:
                    pass
            merge_cur = merge_con.cursor()
            merge_cur.execute("PRAGMA journal_mode=WAL;")
            merge_cur.execute("PRAGMA synchronous=NORMAL;")
            merge_cur.execute("BEGIN TRANSACTION")
            for key, values in aggregated_patterns.items():
                usage_count = int(values['usage_count'])
                success_weight = float(values['success_weight'])
                weight = max(usage_count, 1)
                success_rate = success_weight / weight
                pattern_data.append((*key, int(values['priority']), success_rate, usage_count))
            for key, values in aggregated_associations.items():
                usage_count = int(values['usage_count'])
                success_weight = float(values['success_weight'])
                weight = max(usage_count, 1)
                success_rate = success_weight / weight
                assoc_data.append((*key, int(values['priority']), success_rate, usage_count))
            if pattern_data:
                merge_cur.executemany("""
                    INSERT INTO dynamic_word_chain (context_len, word1, word2, word3, word4, word5, word6, word7, word8, next_word, priority, success_rate, usage_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(context_len, word1, word2, word3, word4, word5, word6, word7, word8, next_word) DO UPDATE SET
                        priority = dynamic_word_chain.priority + excluded.priority,
                        success_rate = CASE
                            WHEN (dynamic_word_chain.usage_count + excluded.usage_count) > 0 THEN
                                ((dynamic_word_chain.success_rate * dynamic_word_chain.usage_count) + (excluded.success_rate * excluded.usage_count))
                                / (dynamic_word_chain.usage_count + excluded.usage_count)
                            ELSE excluded.success_rate
                        END,
                        usage_count = dynamic_word_chain.usage_count + excluded.usage_count;
                """, pattern_data)
            if assoc_data:
                merge_cur.executemany("""
                    INSERT INTO word_associations (source_word, next_word, priority, success_rate, usage_count)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(source_word, next_word) DO UPDATE SET
                        priority = word_associations.priority + excluded.priority,
                        success_rate = CASE
                            WHEN (word_associations.usage_count + excluded.usage_count) > 0 THEN
                                ((word_associations.success_rate * word_associations.usage_count) + (excluded.success_rate * excluded.usage_count))
                                / (word_associations.usage_count + excluded.usage_count)
                            ELSE excluded.success_rate
                        END,
                        usage_count = word_associations.usage_count + excluded.usage_count;
                """, assoc_data)
            merge_cur.execute("COMMIT")
        except Exception:
            if merge_con is not None:
                try:
                    merge_con.rollback()
                except Exception:
                    pass
            raise
        finally:
            if owns_merge_con and merge_con is not None:
                try:
                    merge_con.close()
                except Exception:
                    pass

        storage_backend = getattr(self.brain, '_storage_backend', None)
        if storage_backend is not None:
            for row in pattern_data:
                context_len, w1, w2, w3, w4, w5, w6, w7, w8, next_word, priority, success_rate, usage_count = row
                pad = (w1, w2, w3, w4, w5, w6, w7, w8)
                storage_backend.add_pattern_batch(context_len, pad, next_word, priority, success_rate, usage_count)
            for row in assoc_data:
                src, nxt, priority, success_rate, usage_count = row
                storage_backend.add_association_batch(src, nxt, priority, success_rate, usage_count)
            storage_backend.save_to_hsb()

        con = getattr(self.brain, 'con', None)
        if con is not None:
            try:
                con.commit()
            except Exception:
                pass

        knowledge_store = getattr(self.brain, 'knowledge_store', None)
        if knowledge_store is not None:
            for result in chunk_results:
                if not result.get('success'):
                    continue
                knowledge_rows = result.get('knowledge')
                if not knowledge_rows:
                    continue
                try:
                    knowledge_store.import_rows(knowledge_rows)
                except Exception:
                    pass
            if con is not None:
                try:
                    con.commit()
                except Exception:
                    pass

        return {
            'success': True,
            'pattern_count': len(pattern_data),
            'association_count': len(assoc_data),
            'knowledge_fact_count': knowledge_fact_count,
        }

    def rebuild_semantic_clusters_from_words(self, words: list[str], use_gpu: bool = False, max_words: int = 100000) -> dict[str, Any]:
        semantic_memory = getattr(self.brain, 'semantic_memory', None)
        if semantic_memory is None:
            return {'success': False, 'message': 'Semantic memory not available.', 'cluster_count': 0}

        limited_words = list(words[:max_words]) if words else []
        semantic_memory.update_cooccurrence(limited_words, force_minimal=True)

        gpu_accelerator = self.api_config.get('gpu_accelerator')
        if use_gpu and gpu_accelerator is not None:
            cluster_patterns = list(getattr(semantic_memory, 'clusters', {}).keys())
            if cluster_patterns:
                gpu_accelerator.parallel_process_patterns(cluster_patterns, "clustering")

        semantic_memory.rebuild_clusters()
        semantic_clusters_file = self.api_config.get('semantic_clusters_file')
        if semantic_clusters_file:
            semantic_memory.save_clusters(semantic_clusters_file)
        return {
            'success': True,
            'cluster_count': len(getattr(semantic_memory, 'clusters', {}) or {}),
            'words_used': len(limited_words),
        }

    def get_training_summary(self, quality_history: list[float] | None = None, recent_limit: int = 10) -> dict[str, Any]:
        history = [float(x) for x in (quality_history or []) if isinstance(x, (int, float))]
        cluster_count = len(getattr(getattr(self.brain, 'semantic_memory', None), 'clusters', {}) or {})
        return {
            'average_quality': (sum(history) / len(history)) if history else None,
            'recent_quality': (sum(history[-recent_limit:]) / len(history[-recent_limit:])) if history[-recent_limit:] else None,
            'quality_count': len(history),
            'cluster_count': cluster_count,
            'generation_stats': self._get_generation_stats(),
            'memory_usage_mb': self._get_memory_usage_mb(),
            'available_memory_mb': self._get_available_memory_mb(),
        }

    def evaluate_self_training_response(self, teacher_text: str, student_response: str, student_brain=None) -> dict[str, Any]:
        response_words = self._clean_text(student_response)
        base_quality = len(set(response_words)) / max(len(response_words), 1)
        length_bonus = 1.0 if 5 <= len(response_words) <= 20 else 0.5
        coherence_bonus = 0.5 if any(word in (teacher_text or "").lower() for word in response_words) else 0.0

        semantic_bonus = 0.0
        semantic_source = getattr(student_brain, 'semantic_memory', None)
        if semantic_source is not None:
            for word in response_words:
                try:
                    related_words = semantic_source.get_related_words(word, 3)
                except Exception:
                    related_words = []
                if any(related in (teacher_text or "").lower() for related in related_words):
                    semantic_bonus += 0.1

        quality_score = (base_quality + length_bonus + coherence_bonus + semantic_bonus) / 4
        return {
            'success': True,
            'quality_score': quality_score,
            'base_quality': base_quality,
            'length_bonus': length_bonus,
            'coherence_bonus': coherence_bonus,
            'semantic_bonus': semantic_bonus,
            'response_words': response_words,
        }

    def apply_self_training_feedback(self, sentence: str, quality_score: float, threshold: float | None = None) -> dict[str, Any]:
        threshold_value = threshold
        if threshold_value is None:
            threshold_value = self.api_config.get('self_training_quality_threshold', 0.55)
        try:
            threshold_value = float(threshold_value or 0.55)
        except Exception:
            threshold_value = 0.55
        if quality_score > threshold_value:
            self.brain.apply_feedback(sentence, is_positive=True)
            return {'success': True, 'reinforced': True, 'threshold': threshold_value}
        return {'success': True, 'reinforced': False, 'threshold': threshold_value}

    def blend_student_attention(self, student_brain, quality_score: float) -> dict[str, Any]:
        main_attention = getattr(getattr(self.brain, 'attention', None), 'focus_weights', None)
        student_attention = getattr(getattr(student_brain, 'attention', None), 'focus_weights', None)
        if not isinstance(main_attention, dict) or not isinstance(student_attention, dict):
            return {'success': False, 'updated_words': 0}

        updated_words = 0
        quality_factor = min(1.0, float(quality_score) + 0.3)
        for word, weight in student_attention.items():
            if word not in main_attention:
                continue
            current_weight = main_attention[word]
            main_attention[word] = (current_weight + weight * quality_factor) / (1 + quality_factor)
            updated_words += 1
        return {'success': True, 'updated_words': updated_words, 'quality_factor': quality_factor}

    def clean_brain_for_new_training(self, auto_save_minutes: int = 5) -> dict[str, Any]:
        brain = getattr(self, 'brain', None)
        if brain is None:
            return {'success': False, 'message': 'Brain not available.'}

        self.sync_brain_state()
        self.close_brain_instance(brain)

        app_dir = self.api_config.get('app_dir') or os.path.dirname(os.path.abspath(__file__))
        backup_dir = os.path.join(app_dir, "brain_backup_" + time.strftime("%Y%m%d_%H%M%S"))
        os.makedirs(backup_dir, exist_ok=True)

        warnings = []
        backed_up_files = []
        removed_files = []
        for path in self._managed_brain_files():
            if os.path.isfile(path):
                try:
                    shutil.copy2(path, os.path.join(backup_dir, os.path.basename(path)))
                    backed_up_files.append(path)
                except Exception as e:
                    warnings.append(f"Backup copy warning for {path}: {e}")
        for path in self._managed_brain_files():
            if os.path.isfile(path):
                try:
                    os.remove(path)
                    removed_files.append(path)
                except Exception as e:
                    warnings.append(f"Could not remove {path}: {e}")

        brain_factory = self.api_config.get('brain_factory')
        db_file = self._get_db_file()
        if not callable(brain_factory) or not db_file:
            return {'success': False, 'message': 'Brain factory or database path is not configured.', 'backup_dir': backup_dir}

        use_hsb_backend = bool(self.settings_manager.get('use_hsb_backend', False))
        new_brain = brain_factory(db_file, use_hsb_backend=use_hsb_backend)
        self.attach_brain(new_brain)

        self.update_setting('auto_save_interval_minutes', auto_save_minutes)
        self.save_settings()
        storage = self.get_storage_snapshot()
        return {
            'success': True,
            'backup_dir': backup_dir,
            'backed_up_files': backed_up_files,
            'removed_files': removed_files,
            'warnings': warnings,
            'auto_save_minutes': auto_save_minutes,
            'storage': storage,
            'message': (
                f"Old brain backed up to:\n{backup_dir}\n\n"
                f"Fresh brain created. Auto-save enabled (every {auto_save_minutes} min). Ready for new training."
            ),
        }

    def sync_brain_state(self) -> dict[str, Any]:
        brain = getattr(self, 'brain', None)
        if brain is None:
            return {'success': False, 'message': 'Brain not available.'}

        if hasattr(brain, '_flush_batch_operations'):
            brain._flush_batch_operations()

        if hasattr(brain, '_save_state_threaded'):
            brain._save_state_threaded()
        elif hasattr(brain, 'save_state'):
            brain.save_state()

        return {'success': True, 'message': 'Brain state synchronized.'}

    def perform_memory_replay(self) -> dict[str, Any]:
        hierarchical_memory = getattr(self.brain, 'hierarchical_memory', None)
        if hierarchical_memory is None:
            return {'success': False, 'message': 'Hierarchical memory system not available.'}

        hierarchical_memory._perform_memory_replay()
        memory_stats = {}
        if hasattr(hierarchical_memory, 'get_memory_stats'):
            try:
                memory_stats = hierarchical_memory.get_memory_stats() or {}
            except Exception:
                memory_stats = {}
        return {
            'success': True,
            'message': 'Memory replay completed successfully.',
            'memory_stats': memory_stats,
        }

    def export_brain_bundle(self, file_path: str) -> dict[str, Any]:
        brain = getattr(self, 'brain', None)
        target_db = self._get_db_file()
        if brain is None or not target_db:
            return {'success': False, 'message': 'Brain database is not available.'}
        if not file_path:
            return {'success': False, 'message': 'No export path provided.'}
        if os.path.normcase(os.path.abspath(file_path)) == os.path.normcase(os.path.abspath(target_db)):
            return {'success': False, 'message': 'Choose a different file name than the active brain database.'}

        self.sync_brain_state()

        dest_dir = os.path.dirname(os.path.abspath(file_path))
        if dest_dir:
            os.makedirs(dest_dir, exist_ok=True)

        source_con = None
        dest_con = None
        try:
            active_con = getattr(brain, 'con', None)
            if active_con is not None:
                source_con = active_con
            else:
                source_con = sqlite3.connect(target_db, check_same_thread=False)
            dest_con = sqlite3.connect(file_path, check_same_thread=False)
            source_con.backup(dest_con)
        finally:
            if dest_con is not None:
                dest_con.close()
            if source_con is not None and source_con is not getattr(brain, 'con', None):
                source_con.close()

        bundle_paths, exported_sidecars = self._export_brain_sidecars(file_path)
        metadata = self._collect_export_metadata(exported_sidecars)
        metadata_file = bundle_paths['metadata']
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        extra_files = len(exported_sidecars) + 1
        return {
            'success': True,
            'file_path': file_path,
            'metadata_file': metadata_file,
            'metadata': metadata,
            'exported_sidecars': exported_sidecars,
            'extra_file_count': extra_files,
            'message': f"Memory optimized brain exported to {file_path}\nBundle files written: {extra_files}",
        }

    def import_brain_bundle(self, file_path: str) -> dict[str, Any]:
        brain = getattr(self, 'brain', None)
        target_db = self._get_db_file()
        if brain is None or not target_db:
            return {'success': False, 'message': 'Brain database is not available.'}
        if not file_path:
            return {'success': False, 'message': 'No import path provided.'}
        if not os.path.isfile(file_path):
            return {'success': False, 'message': f'Import file not found: {file_path}'}
        if os.path.normcase(os.path.abspath(file_path)) == os.path.normcase(os.path.abspath(target_db)):
            return {'success': False, 'message': 'That file is already the active brain database.'}

        active_con = getattr(brain, 'con', None)
        if active_con is not None:
            try:
                active_con.close()
            except Exception:
                pass
            try:
                brain.con = None
            except Exception:
                pass
            try:
                brain.cur = None
            except Exception:
                pass

        for stale_path in (target_db, target_db + '-wal', target_db + '-shm'):
            if os.path.exists(stale_path):
                os.remove(stale_path)

        source_con = None
        dest_con = None
        try:
            source_con = sqlite3.connect(file_path, check_same_thread=False)
            dest_con = sqlite3.connect(target_db, check_same_thread=False)
            source_con.backup(dest_con)
        finally:
            if dest_con is not None:
                dest_con.close()
            if source_con is not None:
                source_con.close()

        metadata_file, imported_sidecars, removed_sidecars = self._import_brain_sidecars(file_path)
        metadata = {}
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

        cluster_info = ""
        if 'semantic_clusters' in metadata:
            cluster_info = f"Clusters: {metadata.get('semantic_clusters', 'unknown')}, "
        gen_info = "Statistical Generation: YES" if metadata.get('statistical_generation', False) else "Statistical Generation: UNKNOWN"
        mem_info = " (Memory Optimized)" if metadata.get('memory_optimized', False) else ""
        sidecar_info = f"\nSupporting files imported: {', '.join(sorted(imported_sidecars.keys()))}" if imported_sidecars else ""
        removal_info = "\nMissing supporting files were cleared to avoid stale local state." if removed_sidecars else ""
        if metadata:
            message = (
                f"Memory optimized brain imported successfully.\n"
                f"Version: {metadata.get('version', 'unknown')}\n"
                f"{cluster_info}{gen_info}{mem_info}{sidecar_info}{removal_info}\n"
                f"Please restart the application."
            )
        else:
            message = f"Brain imported successfully.{sidecar_info}{removal_info}\nPlease restart the application."

        return {
            'success': True,
            'metadata_file': metadata_file,
            'metadata': metadata,
            'imported_sidecars': imported_sidecars,
            'removed_sidecars': removed_sidecars,
            'restart_required': True,
            'invalidates_runtime_state': True,
            'message': message,
        }

    def export_hsb_copy(self) -> dict[str, Any]:
        try:
            return self.export_brain_to_hsb(self._resolve_hsb_default_path(), create_backup=True)
        except Exception as e:
            return {'success': False, 'message': f'Could not save HSB copy: {e}'}

    def export_brain_to_hsb(self, file_path: str, create_backup: bool = True) -> dict[str, Any]:
        brain = getattr(self, 'brain', None)
        if brain is None:
            return {'success': False, 'message': 'Brain not available.'}
        if not file_path:
            return {'success': False, 'message': 'No HSB export path provided.'}

        self.sync_brain_state()

        if create_backup and os.path.exists(file_path):
            backup = file_path + '.bak.' + time.strftime('%Y%m%d_%H%M%S')
            try:
                shutil.copy2(file_path, backup)
            except Exception:
                pass

        storage_backend = getattr(brain, '_storage_backend', None)
        if storage_backend is not None:
            storage_backend.save_to_hsb(file_path)
            return {
                'success': True,
                'file_path': file_path,
                'message': f"Brain exported to {file_path}\nYou can open this file in the Newtype HSB Viewer.",
            }

        create_hsb_brain_from_data = self._load_hsb_writer()
        if create_hsb_brain_from_data is None:
            return {'success': False, 'message': 'Newtype/HSB support not available.'}

        patterns, associations, clusters = self._collect_hsb_export_data()
        create_hsb_brain_from_data(file_path, patterns, associations, clusters)
        return {
            'success': True,
            'file_path': file_path,
            'patterns': len(patterns),
            'associations': len(associations),
            'message': (
                f"Brain exported to {file_path}\n"
                f"Patterns: {len(patterns)}, Associations: {len(associations)}\n"
                f"You can open this file in the Newtype HSB Viewer."
            ),
        }

    def import_brain_from_hsb(self, file_path: str) -> dict[str, Any]:
        read_hsb_brain = self._load_hsb_reader()
        if read_hsb_brain is None:
            return {'success': False, 'message': 'Newtype/HSB support not available.'}
        if not file_path:
            return {'success': False, 'message': 'No HSB import path provided.'}

        brain = getattr(self, 'brain', None)
        cur = getattr(brain, 'cur', None)
        con = getattr(brain, 'con', None)
        if brain is None or cur is None or con is None:
            return {'success': False, 'message': 'Brain database is not available for HSB import.'}

        if hasattr(brain, '_flush_batch_operations'):
            brain._flush_batch_operations()

        reader = read_hsb_brain(file_path)
        try:
            patterns = reader.get_patterns()
            associations = reader.get_associations()
            clusters_data = reader.get_semantic_clusters()
        finally:
            reader.close()

        backend = getattr(brain, '_storage_backend', None)
        for pattern in patterns:
            context_len, words, next_word, priority, success_rate, usage_count = pattern
            padded = list(words) + [''] * (8 - len(words))
            cur.execute("""
                INSERT OR REPLACE INTO dynamic_word_chain
                (context_len, word1, word2, word3, word4, word5, word6, word7, word8, next_word, priority, success_rate, usage_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (context_len, *padded[:8], next_word, priority, success_rate or 0.5, usage_count or 0))
            if backend is not None:
                backend.add_pattern_batch(context_len, tuple(padded[:8]), next_word, priority, success_rate or 0.5, usage_count or 0)

        for association in associations:
            src, nxt, priority, success_rate, usage_count = (
                association[0],
                association[1],
                association[2],
                association[3] if len(association) > 3 else 0.5,
                association[4] if len(association) > 4 else 0,
            )
            cur.execute("""
                INSERT OR REPLACE INTO word_associations (source_word, next_word, priority, success_rate, usage_count)
                VALUES (?, ?, ?, ?, ?)
            """, (src, nxt, priority, success_rate, usage_count))
            if backend is not None:
                backend.add_association_batch(src, nxt, priority, success_rate, usage_count)

        con.commit()
        if backend is not None:
            backend.save_to_hsb()

        semantic_clusters_file = self.api_config.get('semantic_clusters_file')
        semantic_memory = getattr(brain, 'semantic_memory', None)
        if clusters_data and semantic_memory is not None:
            for key, value in clusters_data.get('clusters', {}).items():
                try:
                    cluster_id = int(key)
                    semantic_memory.clusters[cluster_id] = set(value) if not isinstance(value, set) else value
                except (ValueError, TypeError):
                    pass
            semantic_memory.word_to_cluster.update(clusters_data.get('word_to_cluster', {}))
            for key, value in clusters_data.get('cluster_strength', {}).items():
                try:
                    semantic_memory.cluster_strength[int(key)] = value
                except (ValueError, TypeError):
                    pass
            for key, value in clusters_data.get('cluster_coherence', {}).items():
                try:
                    semantic_memory.cluster_coherence[int(key)] = value
                except (ValueError, TypeError):
                    pass
            if semantic_clusters_file:
                semantic_memory.save_clusters(semantic_clusters_file)

        return {
            'success': True,
            'patterns': len(patterns),
            'associations': len(associations),
            'message': (
                f"Imported {len(patterns)} patterns and {len(associations)} associations from HSB.\n"
                f"Restart recommended to ensure full consistency."
            ),
        }

    def _get_generation_stats(self) -> str:
        try:
            stats = self.brain.get_generation_stats()
            return str(stats) if stats is not None else "N/A"
        except Exception as e:
            return f"Stats unavailable: {e}"

    def _safe_feature_call(self, feature, method_name: str, default: Any):
        if feature is None:
            return default
        method = getattr(feature, method_name, None)
        if not callable(method):
            return default
        try:
            result = method()
            return default if result is None else result
        except Exception:
            return default

    def _get_known_feature_flags(self) -> set[str]:
        default_features = getattr(self.settings_manager, 'default_settings', {}).get('features', {}) or {}
        current_features = getattr(self.settings_manager, 'settings', {}).get('features', {}) or {}
        return set(default_features.keys()) | set(current_features.keys())

    def _get_hardware_profile(self) -> dict[str, Any]:
        provider = self.api_config.get('get_hardware_profile')
        if callable(provider):
            try:
                profile = provider()
                if isinstance(profile, dict):
                    return copy.deepcopy(profile)
            except Exception:
                pass
        provider = self.api_config.get('get_system_tier')
        if callable(provider):
            try:
                profile = provider()
                if isinstance(profile, dict):
                    fallback = copy.deepcopy(profile)
                    fallback.setdefault('recommended_parallel_workers', self._safe_parallel_workers())
                    fallback.setdefault('recommended_chunk_size', self._safe_optimal_chunk_size(1000000))
                    fallback.setdefault('recommended_memory_threshold', self.api_config.get('memory_safety_threshold', 0.85))
                    fallback.setdefault('recommended_gpu_type', 'auto')
                    fallback.setdefault('recommended_gpu_index', 'auto')
                    fallback.setdefault('profiles', {})
                    return fallback
            except Exception:
                pass
        return {
            'tier': 'medium',
            'total_ram_mb': 8000,
            'available_ram_mb': 4000,
            'has_gpu': False,
            'recommended_parallel_workers': self._safe_parallel_workers(),
            'recommended_chunk_size': self._safe_optimal_chunk_size(1000000),
            'recommended_memory_threshold': self.api_config.get('memory_safety_threshold', 0.85),
            'recommended_gpu_type': 'auto',
            'recommended_gpu_index': 'auto',
            'profiles': {},
        }

    def _apply_profile_settings(self, profile_name: str, hardware_profile: dict[str, Any] | None = None) -> dict[str, Any]:
        profile = str(profile_name or 'medium').strip().lower()
        if profile not in {'low', 'medium', 'max'}:
            profile = 'medium'
        hardware = dict(hardware_profile or self._get_hardware_profile())
        profile_settings = dict((hardware.get('profiles', {}) or {}).get(profile, {}))
        settings = getattr(self.settings_manager, 'settings', None)
        if not isinstance(settings, dict):
            settings = {}
            self.settings_manager.settings = settings
        previous_settings = copy.deepcopy(settings)

        applied_settings: dict[str, Any] = {}

        def _set_value(key: str, value: Any) -> None:
            if settings.get(key) != value:
                settings[key] = value
                applied_settings[key] = value

        _set_value('performance_profile', profile)
        for key, value in profile_settings.items():
            _set_value(key, value)

        enable_all_features = profile == 'max'
        default_features = getattr(self.settings_manager, 'default_settings', {}).get('features', {}) or {}
        current_features = settings.get('features', {}) or {}
        feature_keys = list(default_features.keys()) or list(current_features.keys())
        if feature_keys:
            next_features = {key: enable_all_features for key in feature_keys}
            if current_features != next_features:
                settings['features'] = next_features
                applied_settings['features'] = next_features

        if self.settings_manager.save_settings() is False:
            self.settings_manager.settings = previous_settings
            raise OSError('Performance profile could not be saved.')
        return {
            'profile': profile,
            'tier': hardware.get('tier', 'medium'),
            'has_gpu': bool(hardware.get('has_gpu', False)),
            'settings': applied_settings,
        }

    def _apply_startup_hardware_adaptation(self) -> dict[str, Any]:
        settings = self.get_settings_snapshot()
        enabled = bool(settings.get('hardware_adaptive_mode', True))
        profile_name = str(settings.get('performance_profile', 'medium') or 'medium')
        if not enabled:
            return {
                'applied': False,
                'reason': 'hardware_adaptive_mode_disabled',
                'profile': profile_name,
            }
        result = self._apply_profile_settings(profile_name, self._get_hardware_profile())
        result['applied'] = True
        return result

    def _sync_hardware_runtime(self) -> dict[str, Any]:
        gpu_accelerator = self.api_config.get('gpu_accelerator')
        if gpu_accelerator is None:
            return {'success': False, 'message': 'GPU accelerator is not configured.'}
        reinitializer = getattr(gpu_accelerator, 'reinitialize_gpu', None)
        if not callable(reinitializer):
            snapshot = getattr(gpu_accelerator, 'get_snapshot', None)
            return snapshot() if callable(snapshot) else {}
        try:
            snapshot = reinitializer()
            if isinstance(snapshot, dict):
                result = dict(snapshot)
            else:
                result = {'snapshot': snapshot}
            result.setdefault('success', True)
            return result
        except Exception as exc:
            return {'success': False, 'error': str(exc)}

    def _sync_runtime_configuration(self) -> dict[str, str]:
        brain = getattr(self, 'brain', None)
        if brain is None:
            return {}

        settings = self.get_settings_snapshot()
        feature_flags = settings.get('features', {}) or {}
        enhanced_enabled = bool(settings.get('enhanced_intelligence', True))
        advanced_reasoning_enabled = bool(settings.get('advanced_reasoning', True)) and enhanced_enabled
        adaptive_learning_enabled = bool(settings.get('adaptive_learning', True)) and enhanced_enabled

        try:
            brain.enhanced_intelligence_enabled = enhanced_enabled
        except Exception:
            pass

        factories = self._get_runtime_feature_factories()
        changes: dict[str, str] = {}
        self._sync_runtime_attr('reasoning_engine', advanced_reasoning_enabled, factories.get('reasoning_engine'), changes)
        self._sync_runtime_attr('adaptive_learning', adaptive_learning_enabled, factories.get('adaptive_learning'), changes)
        self._sync_runtime_attr(
            'creative_generator',
            enhanced_enabled and (advanced_reasoning_enabled or adaptive_learning_enabled),
            factories.get('creative_generator'),
            changes,
        )

        for feature_name in sorted(self._get_known_feature_flags()):
            attr_name = feature_name
            enabled = enhanced_enabled and bool(feature_flags.get(feature_name, False))
            self._sync_runtime_attr(attr_name, enabled, factories.get(attr_name), changes)

        return changes

    def _sync_runtime_attr(self, attr_name: str, enabled: bool, factory, changes: dict[str, str]) -> None:
        brain = getattr(self, 'brain', None)
        if brain is None:
            return

        current = getattr(brain, attr_name, None)
        exists = hasattr(brain, attr_name)
        if enabled:
            if current is None:
                if callable(factory):
                    setattr(brain, attr_name, factory())
                    changes[attr_name] = 'created'
            return

        if exists:
            try:
                delattr(brain, attr_name)
                changes[attr_name] = 'removed'
            except Exception:
                try:
                    setattr(brain, attr_name, None)
                    changes[attr_name] = 'disabled'
                except Exception:
                    pass

    def _get_runtime_feature_factories(self) -> dict[str, Any]:
        classes = self._load_backend_feature_classes()
        brain = getattr(self, 'brain', None)
        context_window = int(self.api_config.get('reasoning_context_window', 10) or 10)
        adaptation_rate = float(self.api_config.get('reasoning_adaptation_rate', 0.1) or 0.1)
        parallel_candidates = int(self.api_config.get('generation_parallel_candidates', 1) or 1)
        return {
            'reasoning_engine': lambda: classes['AdvancedReasoningEngine'](
                context_window=context_window,
                adaptation_rate=adaptation_rate,
                brain=brain,
            ),
            'adaptive_learning': lambda: classes['AdaptiveLearningSystem'](brain),
            'creative_generator': lambda: classes['CreativeResponseGenerator'](
                brain,
                parallel_candidates=parallel_candidates,
                context_window=context_window,
                adaptation_rate=adaptation_rate,
            ),
            'critic': lambda: classes['Critic'](brain),
            'confidence_gate': lambda: classes['ConfidenceGate'](brain),
            'anti_loop_filter': lambda: classes['AntiLoopFilter'](brain),
            'meta_memory': lambda: classes['MetaMemory'](brain),
            'curiosity': lambda: classes['Curiosity'](brain),
            'env_feedback': lambda: classes['EnvironmentFeedback'](brain),
            'autotune': lambda: classes['Autotune'](brain),
            'response_learning': lambda: classes['ResponseLearningSystem'](brain),
            'truth_fact_table': lambda: classes['TruthFactTable'](brain),
            'topic_detection': lambda: classes['TopicDetectionSystem'](brain),
        }

    def _load_backend_feature_classes(self) -> dict[str, Any]:
        try:
            from .backend_features import (
                AdaptiveLearningSystem,
                AdvancedReasoningEngine,
                AntiLoopFilter,
                Autotune,
                ConfidenceGate,
                CreativeResponseGenerator,
                Critic,
                Curiosity,
                EnvironmentFeedback,
                MetaMemory,
                ResponseLearningSystem,
                TopicDetectionSystem,
                TruthFactTable,
            )
        except ImportError:
            from backend_features import (  # type: ignore
                AdaptiveLearningSystem,
                AdvancedReasoningEngine,
                AntiLoopFilter,
                Autotune,
                ConfidenceGate,
                CreativeResponseGenerator,
                Critic,
                Curiosity,
                EnvironmentFeedback,
                MetaMemory,
                ResponseLearningSystem,
                TopicDetectionSystem,
                TruthFactTable,
            )
        return {
            'AdaptiveLearningSystem': AdaptiveLearningSystem,
            'AdvancedReasoningEngine': AdvancedReasoningEngine,
            'AntiLoopFilter': AntiLoopFilter,
            'Autotune': Autotune,
            'ConfidenceGate': ConfidenceGate,
            'CreativeResponseGenerator': CreativeResponseGenerator,
            'Critic': Critic,
            'Curiosity': Curiosity,
            'EnvironmentFeedback': EnvironmentFeedback,
            'MetaMemory': MetaMemory,
            'ResponseLearningSystem': ResponseLearningSystem,
            'TopicDetectionSystem': TopicDetectionSystem,
            'TruthFactTable': TruthFactTable,
        }

    def _get_memory_usage_mb(self) -> float:
        try:
            return float(self.memory_manager.get_memory_usage_mb())
        except Exception:
            return 0.0

    def _get_available_memory_mb(self) -> float:
        try:
            return float(self.memory_manager.get_available_memory_mb())
        except Exception:
            return 0.0

    def _get_memory_usage_percent(self) -> float:
        try:
            return float(self.memory_manager.get_memory_usage_percent() or 0.0)
        except Exception:
            return 0.0

    def _calculate_quality(self, response: str, user_input: str) -> float:
        try:
            if hasattr(self.brain, '_calculate_response_quality'):
                return float(self.brain._calculate_response_quality(response, user_input))
        except Exception:
            pass
        return 0.5

    def _clean_text(self, text: str) -> list[str]:
        try:
            return list(self.brain.clean_text(text))
        except Exception:
            return []

    def _get_structured_context(self, history_turns: int = 2) -> list[str]:
        try:
            return list(self.brain.get_structured_conversation_context(history_turns=history_turns))
        except Exception:
            return []

    def _calculate_semantic_coherence(self, response_words: list[str], context_words: list[str]) -> float:
        semantic_memory = getattr(self.brain, 'semantic_memory', None)
        word_to_cluster = getattr(semantic_memory, 'word_to_cluster', None) or {}
        if not response_words or not word_to_cluster:
            return 0.0

        semantic_matches = 0
        for word in response_words:
            cluster_id = word_to_cluster.get(word)
            if cluster_id is None:
                continue
            for context_word in context_words:
                if word_to_cluster.get(context_word) == cluster_id:
                    semantic_matches += 1
                    break
        return semantic_matches / max(len(response_words), 1)

    def _safe_parallel_workers(self) -> int:
        try:
            return int(self.memory_manager.calculate_parallel_workers())
        except Exception:
            return 1

    def _safe_optimal_chunk_size(self, estimated_words: int) -> int:
        try:
            return int(self.memory_manager.calculate_optimal_chunk_size(estimated_words))
        except Exception:
            return int(self.api_config.get('chunk_size_words', 100000))

    def _safe_should_reduce_batch_size(self) -> bool:
        try:
            return bool(self.memory_manager.should_reduce_batch_size())
        except Exception:
            return False

    def _normalize_training_extensions(self, extensions: list[str] | None = None) -> set[str]:
        normalized = set()
        values = extensions if extensions else self.api_config.get('training_extensions')
        if not values:
            values = ['.md', '.rst', '.text', '.txt']
        for ext in values:
            if not isinstance(ext, str):
                continue
            clean = ext.strip().lower()
            if not clean:
                continue
            if not clean.startswith('.'):
                clean = '.' + clean
            normalized.add(clean)
        return normalized or {'.md', '.rst', '.text', '.txt'}

    def _get_training_extensions(self) -> list[str]:
        return sorted(self._normalize_training_extensions())

    def _estimate_word_count_from_sample(self, file_path: str, sample_chars: int = 24000) -> int:
        fallback_size = max(1, os.path.getsize(file_path))
        fallback_estimate = max(1, fallback_size // 6)
        try:
            collected: list[str] = []
            total_chars = 0
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as handle:
                while total_chars < sample_chars:
                    chunk = handle.read(min(4096, sample_chars - total_chars))
                    if not chunk:
                        break
                    collected.append(chunk)
                    total_chars += len(chunk)
            sample_text = "".join(collected)
            sample_words = len(sample_text.split())
            if sample_words <= 0 or total_chars <= 0:
                return fallback_estimate
            scale = fallback_size / max(total_chars, 1)
            return max(sample_words, int(round(sample_words * scale)))
        except Exception:
            return fallback_estimate

    def _get_training_source_profile(self, file_path: str) -> dict[str, Any]:
        normalized_path = os.path.normpath(file_path).lower()
        filename = os.path.basename(normalized_path)
        parts = set(part for part in re.split(r"[\\/_\-. ]+", normalized_path) if part)

        if {'training', 'seed'} & parts or 'initial_knowledge' in normalized_path or {'initial', 'knowledge'} <= parts:
            return {'category': 'seed_corpus', 'weight': 1.35}
        if 'docs' in parts or filename.startswith('readme') or {'guide', 'manual', 'reference'} & parts:
            return {'category': 'reference_docs', 'weight': 1.2}
        if {'conversation', 'chat', 'dialog', 'transcript'} & parts:
            return {'category': 'conversation', 'weight': 1.05}
        if {'draft', 'scratch', 'todo', 'notes', 'note'} & parts:
            return {'category': 'working_notes', 'weight': 0.85}
        return {'category': 'general_text', 'weight': 1.0}

    def _get_effective_training_priority_boost(self, base_priority_boost: int, source_weight: float) -> int:
        try:
            base_value = int(base_priority_boost)
        except (TypeError, ValueError):
            base_value = 2
        try:
            weight_value = float(source_weight)
        except (TypeError, ValueError):
            weight_value = 1.0

        if weight_value >= 1.3:
            return max(1, base_value + 2)
        if weight_value >= 1.15:
            return max(1, base_value + 1)
        if weight_value <= 0.9:
            return max(1, base_value - 1)
        return max(1, base_value)

    def _get_training_content_signature(self, file_path: str) -> str:
        digest = hashlib.sha1()
        has_content = False
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as handle:
            for line in handle:
                normalized = " ".join(line.lower().split())
                if not normalized:
                    continue
                has_content = True
                digest.update(normalized.encode('utf-8', errors='ignore'))
                digest.update(b'\n')
        return digest.hexdigest() if has_content else 'empty'

    def _ensure_training_history_table(self) -> None:
        if getattr(self, '_training_history_ready', False):
            return
        con = getattr(self.brain, 'con', None)
        cur = getattr(self.brain, 'cur', None)
        if con is None or cur is None:
            return
        cur.execute('''
            CREATE TABLE IF NOT EXISTS training_history (
                content_signature TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                status TEXT NOT NULL,
                words_processed INTEGER NOT NULL DEFAULT 0,
                chunks_completed INTEGER NOT NULL DEFAULT 0,
                chunk_size INTEGER NOT NULL DEFAULT 0,
                error_message TEXT NOT NULL DEFAULT '',
                updated_at REAL NOT NULL
            )
        ''')
        con.commit()
        self._training_history_ready = True

    def _get_training_history(self, content_signature: str) -> dict[str, Any] | None:
        if not content_signature or content_signature == 'empty':
            return None
        self._ensure_training_history_table()
        cur = getattr(self.brain, 'cur', None)
        if cur is None:
            return None
        row = cur.execute('''
            SELECT content_signature, file_path, status, words_processed,
                   chunks_completed, chunk_size, error_message, updated_at
            FROM training_history
            WHERE content_signature = ?
        ''', (content_signature,)).fetchone()
        if not row:
            return None
        keys = (
            'content_signature', 'file_path', 'status', 'words_processed',
            'chunks_completed', 'chunk_size', 'error_message', 'updated_at',
        )
        return dict(zip(keys, row))

    def _record_training_history(
        self,
        content_signature: str,
        file_path: str,
        *,
        status: str,
        words_processed: int,
        chunks_completed: int,
        chunk_size: int,
        error_message: str = '',
    ) -> None:
        if not content_signature or content_signature == 'empty':
            return
        self._ensure_training_history_table()
        con = getattr(self.brain, 'con', None)
        cur = getattr(self.brain, 'cur', None)
        if con is None or cur is None:
            return
        cur.execute('''
            INSERT INTO training_history (
                content_signature, file_path, status, words_processed,
                chunks_completed, chunk_size, error_message, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(content_signature) DO UPDATE SET
                file_path = excluded.file_path,
                status = excluded.status,
                words_processed = excluded.words_processed,
                chunks_completed = excluded.chunks_completed,
                chunk_size = excluded.chunk_size,
                error_message = excluded.error_message,
                updated_at = excluded.updated_at
        ''', (
            content_signature,
            os.path.abspath(file_path),
            status,
            max(0, int(words_processed or 0)),
            max(0, int(chunks_completed or 0)),
            max(1, int(chunk_size or 1)),
            str(error_message or '')[:1000],
            time.time(),
        ))
        con.commit()

    def _iter_text_chunks_from_file(self, file_path: str, chunk_size_words: int):
        buffer: list[str] = []
        chunk_size_words = max(1, int(chunk_size_words or 1))
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as handle:
            for line in handle:
                words = line.split()
                if not words:
                    continue
                buffer.extend(words)
                while len(buffer) >= chunk_size_words:
                    yield " ".join(buffer[:chunk_size_words])
                    buffer = buffer[chunk_size_words:]
        if buffer:
            yield " ".join(buffer)

    def _get_db_file(self) -> str | None:
        db_file = self.api_config.get('db_file')
        if isinstance(db_file, str) and db_file:
            return db_file
        brain_db_file = getattr(self.brain, 'db_file', None)
        return brain_db_file if isinstance(brain_db_file, str) and brain_db_file else None

    def _resolve_hsb_default_path(self) -> str:
        configured = self.api_config.get('hsb_file')
        if isinstance(configured, str) and configured:
            return configured
        db_file = self._get_db_file()
        if db_file and db_file.endswith('.db'):
            return db_file[:-3] + '.hsb'
        app_dir = self.api_config.get('app_dir') or os.path.dirname(os.path.abspath(__file__))
        return os.path.join(app_dir, 'ai_astra_phoenix_brain.hsb')

    def _sidecar_targets(self) -> dict[str, str]:
        return {
            'model': str(self.api_config.get('nn_model_file', '')),
            'vocab': str(self.api_config.get('vocab_file', '')),
            'attention': str(self.api_config.get('attention_file', '')),
            'context_scores': str(self.api_config.get('context_scores_file', '')),
            'semantic_clusters': str(self.api_config.get('semantic_clusters_file', '')),
            'hsb': self._resolve_hsb_default_path(),
        }

    def _managed_brain_files(self) -> list[str]:
        db_file = self._get_db_file()
        paths = [
            db_file,
            (db_file + '-wal') if db_file else '',
            (db_file + '-shm') if db_file else '',
            self.api_config.get('nn_model_file', ''),
            self.api_config.get('vocab_file', ''),
            self.api_config.get('attention_file', ''),
            self.api_config.get('context_scores_file', ''),
            self.api_config.get('semantic_clusters_file', ''),
            self._resolve_hsb_default_path(),
        ]
        normalized = []
        for path in paths:
            if isinstance(path, str) and path:
                normalized.append(path)
        return normalized

    def _brain_bundle_paths(self, db_path: str) -> dict[str, str]:
        root, _ = os.path.splitext(db_path)
        return {
            'metadata': root + '_metadata.json',
            'model': root + '_model.json',
            'vocab': root + '_vocab.json',
            'attention': root + '_attention_weights.json',
            'context_scores': root + '_context_scores.json',
            'semantic_clusters': root + '_semantic_clusters.json',
            'hsb': root + '.hsb',
        }

    def _export_brain_sidecars(self, db_path: str) -> tuple[dict[str, str], dict[str, str]]:
        bundle_paths = self._brain_bundle_paths(db_path)
        exported = {}
        sidecars = self._sidecar_targets()
        for key in ('model', 'vocab', 'attention', 'context_scores', 'semantic_clusters'):
            src = sidecars.get(key)
            if src and os.path.isfile(src):
                shutil.copy2(src, bundle_paths[key])
                exported[key] = bundle_paths[key]

        hsb_source = None
        storage_backend = getattr(self.brain, '_storage_backend', None)
        if storage_backend is not None:
            hsb_source = getattr(storage_backend, 'hsb_file', None)
        else:
            candidate = self._resolve_hsb_default_path()
            if os.path.isfile(candidate):
                hsb_source = candidate
        if hsb_source and os.path.isfile(hsb_source):
            shutil.copy2(hsb_source, bundle_paths['hsb'])
            exported['hsb'] = bundle_paths['hsb']

        return bundle_paths, exported

    def _import_brain_sidecars(self, db_path: str) -> tuple[str, dict[str, str], list[str]]:
        source_bundle = self._brain_bundle_paths(db_path)
        target_bundle = self._sidecar_targets()
        imported = {}
        removed = []
        for key, target_path in target_bundle.items():
            if not target_path:
                continue
            source_path = source_bundle[key]
            if os.path.isfile(source_path):
                target_dir = os.path.dirname(os.path.abspath(target_path))
                if target_dir:
                    os.makedirs(target_dir, exist_ok=True)
                shutil.copy2(source_path, target_path)
                imported[key] = target_path
            elif os.path.exists(target_path):
                os.remove(target_path)
                removed.append(target_path)
        return source_bundle['metadata'], imported, removed

    def _collect_export_metadata(self, exported_sidecars: dict[str, str]) -> dict[str, Any]:
        brain = self.brain
        context_scorer = getattr(brain, 'context_scorer', None)
        semantic_memory = getattr(brain, 'semantic_memory', None)
        return {
            'version': '4.5_consolidated_build',
            'export_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'generation_stats': self._get_generation_stats(),
            'context_performance': getattr(context_scorer, 'context_performance', {}) if context_scorer else {},
            'semantic_clusters': len(getattr(semantic_memory, 'clusters', {}) or {}),
            'word_cooccurrence_patterns': len(getattr(semantic_memory, 'word_cooccurrence', {}) or {}),
            'no_preset_responses': True,
            'statistical_generation': True,
            'memory_optimized': True,
            'memory_optimization_settings': {
                'memory_safety_threshold': self.api_config.get('memory_safety_threshold'),
                'chunk_size_words': self.api_config.get('chunk_size_words'),
                'min_chunk_size': self.api_config.get('min_chunk_size'),
                'max_chunk_size': self.api_config.get('max_chunk_size'),
                'parallel_worker_limit': self.api_config.get('parallel_worker_limit'),
                'large_file_threshold': self.api_config.get('large_file_threshold'),
            },
            'memory_stats_at_export': {
                'memory_usage_mb': self._get_memory_usage_mb(),
                'available_memory_mb': self._get_available_memory_mb(),
                'optimal_chunk_size': self._safe_optimal_chunk_size(1000000),
                'parallel_workers': self._safe_parallel_workers(),
            },
            'conversation_stats': {
                'memory_length': len(getattr(brain, 'conversation_memory', []) or []),
                'topics_tracked': len(getattr(brain, 'current_topics', {}) or {}),
                'quality_history_length': len(getattr(brain, 'response_quality_history', []) or []),
            },
            'uses_hsb_backend': bool(getattr(brain, '_storage_backend', None)),
            'exported_sidecars': sorted(exported_sidecars.keys()),
        }

    def _collect_hsb_export_data(self) -> tuple[list[tuple[Any, ...]], list[tuple[Any, ...]], dict[str, Any] | None]:
        db_file = self._get_db_file()
        if not db_file:
            raise ValueError('Brain database path not set.')

        read_con = sqlite3.connect(db_file, check_same_thread=False)
        try:
            read_cur = read_con.cursor()
            read_cur.execute("""
                SELECT context_len, word1, word2, word3, word4, word5, word6, word7, word8, next_word, priority, success_rate, usage_count
                FROM dynamic_word_chain
            """)
            rows = read_cur.fetchall() or []
            patterns = []
            for row in rows:
                try:
                    if not row or len(row) < 13:
                        continue
                    context_len, w1, w2, w3, w4, w5, w6, w7, w8, next_word, priority, success_rate, usage_count = row[:13]
                    words = tuple(word for word in (w1, w2, w3, w4, w5, w6, w7, w8) if word)
                    patterns.append((context_len, words, next_word, priority, success_rate or 0.5, usage_count or 0))
                except (TypeError, ValueError, IndexError):
                    continue

            read_cur.execute("SELECT source_word, next_word, priority, success_rate, usage_count FROM word_associations")
            assoc_rows = read_cur.fetchall() or []
            associations = []
            for row in assoc_rows:
                try:
                    if row and len(row) >= 5:
                        associations.append((row[0], row[1], row[2], row[3] or 0.5, row[4] or 0))
                except (TypeError, IndexError):
                    continue
        finally:
            read_con.close()

        semantic_memory = getattr(self.brain, 'semantic_memory', None)
        clusters = None
        if semantic_memory is not None:
            clusters = {
                'clusters': {str(key): list(value) for key, value in getattr(semantic_memory, 'clusters', {}).items()},
                'word_to_cluster': getattr(semantic_memory, 'word_to_cluster', {}),
                'cluster_strength': {str(key): value for key, value in getattr(semantic_memory, 'cluster_strength', {}).items()},
                'cluster_coherence': {str(key): value for key, value in getattr(semantic_memory, 'cluster_coherence', {}).items()},
            }
        return patterns, associations, clusters

    def _load_hsb_writer(self):
        app_dir = self.api_config.get('app_dir') or os.path.dirname(os.path.abspath(__file__))
        newtype_dir = os.path.join(app_dir, 'Newtype')
        if newtype_dir not in sys.path and os.path.isdir(newtype_dir):
            sys.path.insert(0, newtype_dir)
        try:
            from hsb_format import create_hsb_brain_from_data  # type: ignore[reportMissingImports]
        except ImportError:
            return None
        return create_hsb_brain_from_data

    def _load_hsb_reader(self):
        app_dir = self.api_config.get('app_dir') or os.path.dirname(os.path.abspath(__file__))
        newtype_dir = os.path.join(app_dir, 'Newtype')
        if newtype_dir not in sys.path and os.path.isdir(newtype_dir):
            sys.path.insert(0, newtype_dir)
        try:
            from hsb_format import read_hsb_brain  # type: ignore[reportMissingImports]
        except ImportError:
            return None
        return read_hsb_brain
