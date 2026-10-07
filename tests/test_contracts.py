import ast
import json
import math
import pathlib
import re
import sqlite3
import unittest
from types import SimpleNamespace

from backend_api import MaiBackendAPI
from backend_runtime import HybridBrain, PlainMLP, settings_manager
from backend_features import AdaptiveLearningSystem
from cognitive_core import NornCognitiveCore
from evaluation_harness import (
    assess_learning_hygiene,
    calculate_surface_metrics,
    compare_to_baseline,
    evaluate_verdict,
    _extract_path_metadata,
    GENERATIVE_PASS_THRESHOLDS,
)
from headless_api import BackendServiceHost, _json_dump, _json_loads, _serialize_payload


ROOT = pathlib.Path(__file__).resolve().parents[1]


class _FakeApi:
    api_config = {'transport_max_batch_size': 2}

    def get_transport_method_names(self):
        return ['echo', 'reject']

    def get_transport_method_specs(self):
        return {
            'echo': {'signature': '(value)', 'summary': 'Return a value.'},
            'reject': {'signature': '(value)', 'summary': 'Reject a value.'},
        }

    def get_transport_control_specs(self):
        return {}

    def echo(self, value):
        return value

    def reject(self, value):
        raise ValueError(f'bad value: {value}')


class JsonContractTests(unittest.TestCase):
    def test_non_finite_values_are_emitted_as_valid_json_nulls(self):
        payload = {'nan': math.nan, 'positive': math.inf, 'negative': -math.inf}
        serialized = _serialize_payload(payload)
        self.assertEqual(serialized, {'nan': None, 'positive': None, 'negative': None})
        self.assertEqual(json.loads(_json_dump(payload)), serialized)

    def test_non_standard_json_constants_are_rejected(self):
        with self.assertRaises(ValueError):
            _json_loads('{"value": NaN}')


class TransportContractTests(unittest.TestCase):
    def setUp(self):
        self.host = BackendServiceHost(_FakeApi())

    def test_invalid_arguments_are_rejected_before_invocation(self):
        response = self.host.handle_request({'id': 1, 'method': 'echo', 'params': {}})
        self.assertFalse(response['ok'])
        self.assertEqual(response['error_code'], 'invalid_params')

    def test_batch_limit_is_enforced(self):
        response, should_shutdown = self.host.handle_payload([
            {'method': 'echo', 'params': {'value': 1}},
            {'method': 'echo', 'params': {'value': 2}},
            {'method': 'echo', 'params': {'value': 3}},
        ])
        self.assertFalse(response['ok'])
        self.assertFalse(should_shutdown)

    def test_user_value_errors_are_reported_as_invalid_params(self):
        response = self.host.handle_request({'id': 1, 'method': 'reject', 'params': {'value': -1}})
        self.assertFalse(response['ok'])
        self.assertEqual(response['error_code'], 'invalid_params')


class SettingsContractTests(unittest.TestCase):
    def setUp(self):
        defaults = {
            'features': {},
            'max_response_length': 25,
            'quality_threshold': 0.5,
            'parallel_workers': 'auto',
            'gpu_acceleration_enabled': False,
        }
        self.api = object.__new__(MaiBackendAPI)
        self.api.settings_manager = SimpleNamespace(default_settings=defaults)

    def test_setting_values_are_normalized_before_persistence(self):
        self.assertEqual(self.api._normalize_setting_value('max_response_length', 64.0), 64)
        self.assertEqual(self.api._normalize_setting_value('parallel_workers', 8), '8')
        self.assertEqual(self.api._normalize_setting_value('quality_threshold', 0.75), 0.75)

    def test_invalid_or_unknown_settings_are_rejected(self):
        invalid_values = [
            ('max_response_length', -1),
            ('quality_threshold', math.nan),
            ('gpu_acceleration_enabled', 'false'),
            ('parallel_workers', 0),
            ('features', {}),
            ('not_a_setting', True),
        ]
        for key, value in invalid_values:
            with self.subTest(key=key, value=value):
                with self.assertRaises(ValueError):
                    self.api._normalize_setting_value(key, value)


class ArchitectureTests(unittest.TestCase):
    def test_core_runtime_has_no_tensor_framework_imports(self):
        forbidden_roots = {'torch', 'tensorflow', 'keras', 'jax'}
        imported_roots = set()
        for filename in (
            'backend_api.py',
            'backend_features.py',
            'backend_knowledge.py',
            'backend_runtime.py',
            'cognitive_core.py',
            'gated_learning.py',
            'evaluation_harness.py',
            'vocab_storage.py',
        ):
            tree = ast.parse((ROOT / filename).read_text(encoding='utf-8-sig'), filename=filename)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported_roots.update(alias.name.split('.')[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported_roots.add(node.module.split('.')[0])
        self.assertFalse(imported_roots & forbidden_roots)

    def test_optional_mlp_is_lazy_until_explicitly_used(self):
        model = PlainMLP(100)
        self.assertFalse(model._weights_initialized)
        self.assertEqual(model.w1, [])
        self.assertEqual(model.w2, [])
        restored = PlainMLP(2)
        restored.from_dict(model.to_dict())
        self.assertFalse(restored._weights_initialized)
        self.assertEqual(restored.trained_steps, 0)

    def test_open_response_rejects_one_word_topic_matches(self):
        brain = object.__new__(HybridBrain)
        score = brain._score_realization_text(
            'Every sentence I construct is a product of learned patterns.',
            'Explain why liquid water freezes in one concise sentence.',
            {'intent': 'explanation'},
            {'overall': 0.9, 'source_reliability': 0.9, 'stability': 0.9},
        )
        self.assertEqual(score, float('-inf'))

    def test_self_description_scaffold_describes_real_architecture(self):
        brain = object.__new__(HybridBrain)
        response = brain._build_scaffolded_open_response(
            'What kind of AI are you?',
            {'intent': 'self_description'},
        )
        self.assertIn('non-tensor', response.lower())
        self.assertIn('bounded chemistry-like control signals', response.lower())
        self.assertIn('learned goals', response.lower())
        direct_response = brain._build_symbolic_response('What kind of AI are you?')
        self.assertIn('bounded chemistry-like control signals', direct_response.lower())

    def test_symbolic_syllogism_composes_rules(self):
        brain = object.__new__(HybridBrain)
        response = brain._build_symbolic_response(
            'If every glint is a moth and no moth is silent, could any glint be silent?'
        )
        self.assertTrue(response.startswith('No;'))
        self.assertIn('no glint can be silent', response.lower())

    def test_symbolic_responses_cover_evidence_and_constraints(self):
        brain = object.__new__(HybridBrain)
        evidence = brain._build_symbolic_response(
            'What evidence is needed before claiming a treatment works?'
        )
        self.assertIn('independent replication', evidence.lower())
        greeting = brain._build_symbolic_response(
            "Reply to the greeting 'hello' without using any word twice."
        )
        words = re.findall(r"[a-z]+", greeting.lower())
        self.assertEqual(len(words), len(set(words)))

    def test_corpus_artifact_fragments_are_not_prompt_neutral(self):
        brain = object.__new__(HybridBrain)
        self.assertTrue(
            brain._is_corpus_artifact_response(
                'The ground and her tail wags happily as she crossed the road.',
                'Explain how to profile a backend service.',
            )
        )
        self.assertFalse(
            brain._is_corpus_artifact_response(
                'The bird crossed the ground and its tail wags in the wind.',
                'Describe the ground and tail wags in the story.',
            )
        )


class CognitiveCoreTests(unittest.TestCase):
    def setUp(self):
        self.con = sqlite3.connect(':memory:')
        self.core = NornCognitiveCore(self.con, event_limit=100)

    def tearDown(self):
        self.con.close()

    def test_appraisal_selects_goal_and_response_strategy(self):
        clarification = self.core.begin_turn('What about it?')
        self.assertEqual(clarification['goal'], 'resolve_uncertainty')
        self.assertEqual(clarification['action'], 'ask_clarifying')
        self.core.observe_response(quality=0.7, coherence=0.8, response_text='What part should I examine?')

        repair = self.core.begin_turn('That answer was wrong and repeated the same snippet again.')
        self.assertEqual(repair['goal'], 'restore_alignment')
        self.assertEqual(repair['action'], 'repair_alignment')

        self.core.observe_response(quality=0.7, coherence=0.8, response_text='I will change the reasoning strategy.')
        creative = self.core.begin_turn('Invent a novel memory architecture for a small AI.')
        self.assertEqual(creative['goal'], 'create_novel_idea')
        self.assertEqual(creative['action'], 'create_connection')

    def test_learned_goal_progress_biases_future_selection(self):
        first = self.core.begin_turn('Invent a novel idea about curiosity and memory.')
        self.assertEqual(first['goal'], 'create_novel_idea')
        self.core.observe_response(
            quality=0.95,
            coherence=0.9,
            response_text='A testable connection lets curiosity reshape which memory relations stay active.',
        )
        progress = self.con.execute(
            "SELECT progress, activations FROM cognitive_goals WHERE goal_key='create_novel_idea'"
        ).fetchone()
        self.assertIsNotNone(progress)
        self.assertGreater(progress[0], 0.5)
        self.assertEqual(progress[1], 1)

        learned = self.core._learned_goal_values()
        self.assertIn('create_novel_idea', learned)
        scores = self.core._goal_scores(self.core._appraise('Tell me something.'))
        boosted = dict(scores)
        for goal_key, base in scores.items():
            item = learned.get(goal_key)
            if not item:
                continue
            boosted[goal_key] = (
                base
                + ((item['progress'] - 0.5) * 0.18)
                + ((item['priority'] - 0.5) * 0.10)
                + (0.03 / (1.0 + item['activations']) ** 0.5)
            )
        self.assertGreater(boosted['create_novel_idea'], scores['create_novel_idea'])

        again = self.core.begin_turn('Invent another novel idea about memory.')
        self.assertEqual(again['goal'], 'create_novel_idea')
        self.assertIsNotNone(again.get('goal_learning'))
        self.assertIn('prior goal progress influenced selection', again.get('reasons', []))

    def test_explicit_feedback_changes_context_action_value(self):
        turn = self.core.begin_turn('Explain causal memory replay in a small AI.')
        self.assertEqual(
            self.con.execute('SELECT COUNT(*) FROM cognitive_events').fetchone()[0],
            0,
            'Appraisal should remain in memory until the response completes.',
        )
        self.core.observe_response(
            quality=0.45,
            coherence=0.55,
            response_text='Memory replay revisits an earlier event and updates its relations.',
        )
        before = self.con.execute(
            'SELECT value FROM cognitive_action_values WHERE context_key=? AND action_key=?',
            (turn['context_key'], turn['action']),
        ).fetchone()[0]
        self.core.apply_feedback(True)
        after = self.con.execute(
            'SELECT value, attempts FROM cognitive_action_values WHERE context_key=? AND action_key=?',
            (turn['context_key'], turn['action']),
        ).fetchone()
        self.assertGreater(after[0], before)
        self.assertEqual(after[1], 1)
        self.assertGreater(self.core.get_snapshot()['learned_relations'], 0)

    def test_chemistry_is_bounded_persistent_and_event_log_is_capped(self):
        self.core.ACTION_VALUE_LIMIT = 20
        self.core.RELATION_LIMIT = 25
        for index in range(115):
            self.core.begin_turn(
                f"Why did topic{index} fail and how could we invent a repair?",
                memory_salience=(index % 4) / 3,
                recent_quality=(index % 5) / 4,
            )
            self.core.observe_response(
                quality=(index % 7) / 6,
                coherence=(index % 3) / 2,
                response_text=f"Attempt {index} produced a bounded response with a distinct repair path.",
            )

        snapshot = self.core.get_snapshot()
        self.assertTrue(all(0.0 <= value <= 1.0 for value in snapshot['state'].values()))
        self.assertLessEqual(snapshot['stored_events'], 100)
        self.assertLessEqual(snapshot['learned_contexts'], 20)
        self.assertLessEqual(snapshot['learned_relations'], 25)

        restored = NornCognitiveCore(self.con, event_limit=100)
        self.assertEqual(restored.get_snapshot()['state'], snapshot['state'])

    def test_cognitive_fallback_realizes_a_clarification_action(self):
        brain = object.__new__(HybridBrain)
        brain.cognitive_core = None
        brain.last_cognitive_trace = {
            'goal': 'resolve_uncertainty',
            'action': 'ask_clarifying',
            'appraisal': {'ambiguity': 0.8, 'topics': ['memory', 'chemistry']},
        }
        response = brain._build_cognitive_strategy_response('What about it?', {})
        self.assertIn('memory', response.lower())
        self.assertTrue(response.endswith('?'))

    def test_cognitive_actions_change_scoring_and_response_mode(self):
        brain = object.__new__(HybridBrain)
        brain.cognitive_core = None
        brain.last_cognitive_trace = {}
        brain.last_episode_memory = None
        brain.conversation_memory = []

        clarify_plan = {
            'cognitive_control': {
                'action': 'ask_clarifying',
                'appraisal': {'ambiguity': 0.55, 'topics': ['protocol'], 'memory_salience': 0.1},
            }
        }
        self.assertTrue(brain._cognitive_strategy_should_preempt(clarify_plan))
        clarify = brain._build_cognitive_strategy_response('What about that?', clarify_plan)
        self.assertIsNotNone(clarify)
        self.assertTrue(clarify.endswith('?'))

        recall_plan = {
            'cognitive_control': {
                'action': 'recall_episode',
                'appraisal': {'ambiguity': 0.2, 'topics': ['compass', 'protocol'], 'memory_salience': 0.05},
            }
        }
        self.assertTrue(brain._cognitive_strategy_should_preempt(recall_plan))
        recall = brain._build_cognitive_strategy_response(
            'What did we decide yesterday about the green compass protocol?',
            recall_plan,
        )
        self.assertIsNotNone(recall)
        self.assertIn('matching stored episode', recall.lower())

        causal_claim = 'Liquid water freezes when molecules form a lattice.'
        identity_claim = 'Mai prefers concise replies.'
        causal_bias = brain._cognitive_unit_score_bias('fact', 'explain_causally', causal_claim)
        identity_bias = brain._cognitive_unit_score_bias('identity', 'explain_causally', identity_claim)
        self.assertGreater(causal_bias, identity_bias)

        direct_claim_bias = brain._cognitive_unit_score_bias('claim', 'answer_directly', identity_claim)
        direct_reason_bias = brain._cognitive_unit_score_bias('reasoning', 'answer_directly', causal_claim)
        self.assertGreater(direct_claim_bias, direct_reason_bias)

    def test_episode_memory_recall_uses_conversation_history(self):
        brain = object.__new__(HybridBrain)
        brain.cognitive_core = None
        brain.conversation_memory = [
            {
                'user_input': 'We decided the green compass protocol uses two keys.',
                'bot_response': 'Agreed: two keys and a checksum for the green compass protocol.',
                'quality': 0.9,
                'topics': ['green', 'compass', 'protocol'],
                'timestamp': 1.0,
            }
        ]
        brain.last_episode_memory = None
        episode = brain._retrieve_episode_memory('What did we decide about the green compass protocol?')
        self.assertIsNotNone(episode)
        self.assertGreaterEqual(episode['score'], 0.25)
        brain.last_episode_memory = episode
        response = brain._build_cognitive_strategy_response(
            'What did we decide about the green compass protocol?',
            {
                'cognitive_control': {
                    'action': 'recall_episode',
                    'appraisal': {
                        'ambiguity': 0.2,
                        'topics': ['green', 'compass'],
                        'memory_salience': episode['score'],
                    },
                }
            },
        )
        self.assertIsNotNone(response)
        self.assertIn('stored episode memory', response.lower())
        self.assertIn('two keys', response.lower())

    def test_goal_continuity_persists_unfinished_goals(self):
        first = self.core.begin_turn('Explain how episode memory stores prior turns.')
        self.assertFalse(first.get('goal_continuity'))
        prior_goal = first['goal']
        self.core.observe_response(
            quality=0.35,
            coherence=0.4,
            response_text='Episode memory keeps recent turns for later recall.',
        )
        continued = self.core.begin_turn('Keep explaining how those prior turns are scored.')
        self.assertEqual(continued['goal'], prior_goal)
        self.assertTrue(continued.get('goal_continuity'))
        self.assertIn('continuing an unfinished prior goal', continued.get('reasons', []))

        interrupted = self.core.begin_turn('That answer was wrong and missed the scoring rule.')
        self.assertEqual(interrupted['goal'], 'restore_alignment')
        self.assertFalse(interrupted.get('goal_continuity'))

    def test_evidence_led_sentences_avoid_generic_hygiene_prefixes(self):
        brain = object.__new__(HybridBrain)
        sentence = brain._evidence_led_sentence(
            'Persistent short and long chains preserve both local fluency and distant relations',
            cognitive_action='explain_causally',
        )
        self.assertFalse(sentence.lower().startswith('a useful way to frame'))
        self.assertFalse(sentence.lower().startswith('the main idea is that'))
        self.assertTrue(sentence.endswith('.'))
        hypothesis = brain._evidence_led_sentence(
            'Curiosity can lengthen answers while confidence is low',
            cognitive_action='explore_hypothesis',
            mark_uncertainty=True,
        )
        self.assertIn('hypothesis', hypothesis.lower())


class TrainingBatchTests(unittest.TestCase):
    def setUp(self):
        self.brain = object.__new__(HybridBrain)
        self.brain.con = sqlite3.connect(':memory:')
        self.brain.cur = self.brain.con.cursor()
        self.brain._storage_backend = None
        self.brain.batch_operations = []
        self.brain.batch_count = 0
        self.brain._bonus_caches_loaded = True
        self.brain.cur.execute('''
            CREATE TABLE dynamic_word_chain (
                context_len INTEGER, word1 TEXT, word2 TEXT, word3 TEXT, word4 TEXT,
                word5 TEXT, word6 TEXT, word7 TEXT, word8 TEXT, next_word TEXT,
                priority REAL, success_rate REAL, usage_count INTEGER,
                PRIMARY KEY (context_len, word1, word2, word3, word4, word5, word6, word7, word8, next_word)
            )
        ''')
        self.brain.cur.execute('''
            CREATE TABLE word_associations (
                source_word TEXT, next_word TEXT, priority REAL,
                success_rate REAL, usage_count INTEGER,
                PRIMARY KEY (source_word, next_word)
            )
        ''')
        self.brain.cur.execute('''
            CREATE TABLE phrase_patterns (
                phrase_text TEXT PRIMARY KEY, priority REAL, usage_count INTEGER
            )
        ''')

    def tearDown(self):
        self.brain.con.close()

    def test_duplicate_observations_are_coalesced_and_accumulated(self):
        context = ('<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', 'radical', 'ai')
        self.brain.batch_operations = [
            {'type': 'chain', 'data': (2, *context, 'learns', 2, 0.5, 1)},
            {'type': 'chain', 'data': (2, *context, 'learns', 3, 0.75, 1)},
            {'type': 'phrase_2', 'data': ('radical', 'ai', 2.0)},
            {'type': 'phrase_2', 'data': ('radical', 'ai', 3.0)},
        ]
        self.brain._flush_batch_operations()

        priority, success_rate, usage_count = self.brain.cur.execute(
            'SELECT priority, success_rate, usage_count FROM dynamic_word_chain'
        ).fetchone()
        self.assertEqual(priority, 5)
        self.assertAlmostEqual(success_rate, 0.625)
        self.assertEqual(usage_count, 2)
        self.assertEqual(
            self.brain.cur.execute('SELECT priority, usage_count FROM phrase_patterns').fetchone(),
            (5, 2),
        )

        self.brain.batch_operations = [
            {'type': 'chain', 'data': (2, *context, 'learns', 4, 1.0, 1)},
        ]
        self.brain._flush_batch_operations()
        priority, success_rate, usage_count = self.brain.cur.execute(
            'SELECT priority, success_rate, usage_count FROM dynamic_word_chain'
        ).fetchone()
        self.assertEqual(priority, 9)
        self.assertAlmostEqual(success_rate, 0.75)
        self.assertEqual(usage_count, 3)

    def test_accumulated_evidence_bonus_is_bounded(self):
        self.assertEqual(HybridBrain._bounded_evidence_bonus(0, 0.5), 1.0)
        self.assertLessEqual(HybridBrain._bounded_evidence_bonus(1_000_000, 0.5), 4.0)
        self.assertGreater(
            HybridBrain._bounded_evidence_bonus(100, 0.5),
            HybridBrain._bounded_evidence_bonus(10, 0.5),
        )


class TrainingHistoryTests(unittest.TestCase):
    def setUp(self):
        self.api = object.__new__(MaiBackendAPI)
        con = sqlite3.connect(':memory:')
        self.api.brain = SimpleNamespace(con=con, cur=con.cursor())

    def tearDown(self):
        self.api.brain.con.close()

    def test_completed_corpus_fingerprint_is_persisted(self):
        self.api._record_training_history(
            'abc123',
            'training.txt',
            status='completed',
            words_processed=42,
            chunks_completed=2,
            chunk_size=21,
        )
        state = self.api._get_training_history('abc123')
        self.assertEqual(state['status'], 'completed')
        self.assertEqual(state['words_processed'], 42)
        self.assertEqual(state['chunks_completed'], 2)


class EvaluationMetricTests(unittest.TestCase):
    def test_surface_metrics_detect_repetition_and_profanity(self):
        metrics = calculate_surface_metrics('Damn, maps map maps map.', 'Compare maps and memory.')
        self.assertEqual(metrics['profanity_terms'], ['damn'])
        self.assertLess(metrics['unique_word_ratio'], 1.0)
        self.assertTrue(metrics['has_terminal_punctuation'])

    def test_surface_metrics_detect_prompt_echo_scaffolds(self):
        prompt = 'Why does water freeze in winter?'
        response = 'A useful way to frame why does water freeze in winter is in terms of stability.'
        metrics = calculate_surface_metrics(response, prompt)
        self.assertTrue(metrics['generic_fallback'])
        self.assertTrue(metrics['high_prompt_echo'])

    def test_learning_hygiene_rejects_echo_and_generic_scaffolds(self):
        prompt = 'Why does water freeze in winter?'
        rejected = assess_learning_hygiene(
            'A useful way to frame why does water freeze in winter is in terms of stability.',
            prompt,
        )
        self.assertFalse(rejected['accept'])
        self.assertIn(rejected['reason'], {'generic_fallback', 'high_prompt_echo'})
        accepted = assess_learning_hygiene(
            'Liquid water freezes when molecules lose enough energy to form a lattice.',
            prompt,
        )
        self.assertTrue(accepted['accept'])

    def test_evaluation_verdict_and_baseline_regression(self):
        healthy = {
            'mean_quality_score': 0.73,
            'mean_prompt_echo_ratio': 0.18,
            'high_prompt_echo_count': 0,
            'generic_fallback_count': 0,
            'profanity_response_count': 0,
            'missing_terminal_punctuation_count': 0,
        }
        verdict = evaluate_verdict(healthy)
        self.assertTrue(verdict['passed'])

        degraded = dict(healthy)
        degraded['mean_quality_score'] = 0.40
        degraded['generic_fallback_count'] = 2
        failed = evaluate_verdict(degraded)
        self.assertFalse(failed['passed'])
        self.assertTrue(any(item['metric'] == 'generic_fallback_count' for item in failed['failed_checks']))

        comparison = compare_to_baseline(
            {'summary': {'mean_quality_score': 0.60, 'mean_prompt_echo_ratio': 0.30, 'high_prompt_echo_count': 1,
                         'generic_fallback_count': 0, 'profanity_response_count': 0}},
            {'summary': healthy},
        )
        self.assertFalse(comparison['passed'])
        self.assertTrue(any(item['metric'] == 'mean_quality_score' for item in comparison['regressions']))

    def test_path_metadata_and_generative_verdict_gates(self):
        path = _extract_path_metadata({
            'realization_trace': {'mode': 'cognitive_strategy'},
            'cognitive_trace': {'goal': 'resolve_uncertainty', 'action': 'ask_clarifying'},
        })
        self.assertFalse(path['symbolic_hit'])
        self.assertTrue(path['has_cognitive_control'])
        self.assertEqual(path['cognitive_action'], 'ask_clarifying')

        generative_summary = {
            'mean_quality_score': 0.42,
            'mean_prompt_echo_ratio': 0.20,
            'high_prompt_echo_count': 0,
            'generic_fallback_count': 0,
            'profanity_response_count': 0,
            'missing_terminal_punctuation_count': 0,
            'symbolic_hit_rate': 0.0,
            'cognitive_action_presence_rate': 1.0,
            'unexpected_symbolic_count': 0,
        }
        passed = evaluate_verdict(generative_summary, GENERATIVE_PASS_THRESHOLDS)
        self.assertTrue(passed['passed'])

        leaked = dict(generative_summary)
        leaked['unexpected_symbolic_count'] = 2
        leaked['symbolic_hit_rate'] = 0.5
        failed = evaluate_verdict(leaked, GENERATIVE_PASS_THRESHOLDS)
        self.assertFalse(failed['passed'])
        failed_metrics = {item['metric'] for item in failed['failed_checks']}
        self.assertIn('unexpected_symbolic_count', failed_metrics)


class LearningGateTests(unittest.TestCase):
    def test_adaptive_learning_rejects_hygiene_failures(self):
        system = AdaptiveLearningSystem(brain=None)
        decision = system.should_accept_live_learning(
            0.9,
            source_kind='conversation',
            response_text='A useful way to frame why does water freeze in winter is still unclear.',
            context_text='Why does water freeze in winter?',
        )
        self.assertFalse(decision['accept'])
        self.assertTrue(str(decision['reason']).startswith('hygiene_'))

    def test_adaptive_learning_accepts_clean_high_quality_response(self):
        system = AdaptiveLearningSystem(brain=None)
        decision = system.should_accept_live_learning(
            0.9,
            source_kind='conversation',
            response_text='Liquid water freezes when molecules lose enough energy to form a lattice.',
            context_text='Why does water freeze?',
        )
        self.assertTrue(decision['accept'])
        self.assertEqual(decision['reason'], 'accepted')


class VocabIdStorageTests(unittest.TestCase):
    def setUp(self):
        from vocab_storage import VocabIdStore

        self.con = sqlite3.connect(':memory:')
        self.con.execute('''
            CREATE TABLE dynamic_word_chain (
                context_len INTEGER, word1 TEXT, word2 TEXT, word3 TEXT, word4 TEXT,
                word5 TEXT, word6 TEXT, word7 TEXT, word8 TEXT, next_word TEXT,
                priority REAL, success_rate REAL, usage_count INTEGER,
                PRIMARY KEY (context_len, word1, word2, word3, word4, word5, word6, word7, word8, next_word)
            )
        ''')
        self.store = VocabIdStore(self.con, context_size=8)

    def tearDown(self):
        self.con.close()

    def test_encode_roundtrip_and_lookup(self):
        row = (2, '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', 'radical', 'ai', 'learns', 3.0, 0.8, 2)
        written = self.store.upsert_pattern_rows([row])
        self.assertEqual(written, 1)
        counts = self.store.fetch_next_counts(
            2,
            ('<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', 'radical', 'ai'),
        )
        self.assertEqual(counts, [('learns', 3.0)])
        self.assertGreaterEqual(self.store.stats()['lexicon_size'], 4)

    def test_migration_copies_text_chains(self):
        self.con.execute(
            '''INSERT INTO dynamic_word_chain
               VALUES (2, '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', 'memory', 'replay', 'helps', 2.0, 0.7, 1)'''
        )
        self.con.commit()
        first = self.store.migrate_text_batch(limit=10, offset_rowid=0)
        self.assertEqual(first['migrated'], 1)
        self.assertTrue(first['done'])
        counts = self.store.fetch_next_counts(
            2,
            ('<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', 'memory', 'replay'),
        )
        self.assertEqual(counts, [('helps', 2.0)])

    def test_brain_dual_write_mode_persists_id_chains(self):
        previous = settings_manager.get('vocab_id_storage_mode', 'off')
        try:
            settings_manager.set('vocab_id_storage_mode', 'dual')
            brain = object.__new__(HybridBrain)
            brain.is_clone = True
            brain.con = sqlite3.connect(':memory:')
            brain.cur = brain.con.cursor()
            brain._storage_backend = None
            brain.batch_operations = []
            brain.batch_count = 0
            brain._bonus_caches_loaded = True
            brain.setup_database()
            brain.vocab_id_storage_mode = 'dual'
            brain._setup_vocab_id_storage()
            self.assertIsNotNone(brain.vocab_id_store)
            context = ('<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', 'curious', 'mind')
            brain.batch_operations = [
                {'type': 'chain', 'data': (2, *context, 'learns', 2, 0.6, 1)},
            ]
            brain._flush_batch_operations()
            text_count = brain.cur.execute('SELECT COUNT(*) FROM dynamic_word_chain').fetchone()[0]
            id_count = brain.cur.execute('SELECT COUNT(*) FROM dynamic_word_chain_ids').fetchone()[0]
            self.assertEqual(text_count, 1)
            self.assertEqual(id_count, 1)
            probs = brain._kn_next_probs(['curious', 'mind'])
            self.assertIn('learns', probs)
        finally:
            settings_manager.set('vocab_id_storage_mode', previous)


class GatedLearningTests(unittest.TestCase):
    def test_empty_text_is_rejected(self):
        from gated_learning import propose_gated_learning

        result = propose_gated_learning(SimpleNamespace(brain=SimpleNamespace()), response_text='', user_text='')
        self.assertFalse(result['success'])
        self.assertEqual(result['reason'], 'empty_text')

    def test_hygiene_rejects_before_trial_clone(self):
        from gated_learning import propose_gated_learning

        result = propose_gated_learning(
            SimpleNamespace(brain=SimpleNamespace()),
            response_text='A useful way to frame why does water freeze in winter is still unclear.',
            user_text='Why does water freeze in winter?',
            commit=False,
        )
        self.assertTrue(result['success'])
        self.assertFalse(result['accepted'])
        self.assertFalse(result['committed'])
        self.assertTrue(str(result['reason']).startswith('hygiene_'))

    def test_gate_commits_only_when_trial_does_not_regress(self):
        from gated_learning import propose_gated_learning
        import gated_learning as gated_mod

        healthy_summary = {
            'mean_quality_score': 0.70,
            'mean_prompt_echo_ratio': 0.10,
            'high_prompt_echo_count': 0,
            'generic_fallback_count': 0,
            'profanity_response_count': 0,
            'missing_terminal_punctuation_count': 0,
            'symbolic_hit_rate': 0.0,
            'cognitive_action_presence_rate': 1.0,
            'unexpected_symbolic_count': 0,
        }
        healthy = {
            'summary': healthy_summary,
            'verdict': {'passed': True, 'failed_checks': []},
        }
        live_brain = SimpleNamespace(learn_calls=[])

        def live_learn(text, base_priority_boost=2):
            live_brain.learn_calls.append((text, base_priority_boost))
            return 12

        live_brain.learn_from_text_optimized = live_learn
        live_api = SimpleNamespace(brain=live_brain, close_brain_instance=lambda brain: None)

        trial_brain = SimpleNamespace(learn_calls=[])
        trial_brain.learn_from_text_optimized = lambda text, base_priority_boost=2: 9
        trial_api = SimpleNamespace(brain=trial_brain)

        original_build = gated_mod._build_trial_api
        original_eval = gated_mod.evaluate_with_api
        original_baseline = gated_mod.DEFAULT_GATE_BASELINE
        try:
            gated_mod._build_trial_api = lambda api: trial_api
            gated_mod.evaluate_with_api = lambda api, prompts, label='', seed=0, close_brain=False: dict(healthy)
            gated_mod.DEFAULT_GATE_BASELINE = pathlib.Path('missing_baseline_for_unit_test.json')

            accepted = propose_gated_learning(
                live_api,
                response_text='Curiosity can shorten answers once a coherent relation is found.',
                user_text='How could curiosity change answer length?',
                commit=True,
                require_absolute_pass=False,
            )
            self.assertTrue(accepted['accepted'])
            self.assertTrue(accepted['committed'])
            self.assertEqual(accepted['reason'], 'committed')
            self.assertEqual(len(live_brain.learn_calls), 1)

            degraded = {
                'summary': {
                    **healthy_summary,
                    'mean_quality_score': 0.40,
                    'generic_fallback_count': 2,
                },
                'verdict': {'passed': False, 'failed_checks': [{'metric': 'mean_quality_score'}]},
            }
            calls = {'n': 0}

            def eval_side_effect(api, prompts, label='', seed=0, close_brain=False):
                calls['n'] += 1
                return healthy if calls['n'] == 1 else degraded

            gated_mod.evaluate_with_api = eval_side_effect
            rejected = propose_gated_learning(
                live_api,
                response_text='Curiosity can shorten answers once a coherent relation is found.',
                user_text='How could curiosity change answer length?',
                commit=True,
                require_absolute_pass=False,
            )
            self.assertFalse(rejected['accepted'])
            self.assertFalse(rejected['committed'])
            self.assertEqual(rejected['reason'], 'pre_post_regression')
            self.assertEqual(len(live_brain.learn_calls), 1)
        finally:
            gated_mod._build_trial_api = original_build
            gated_mod.evaluate_with_api = original_eval
            gated_mod.DEFAULT_GATE_BASELINE = original_baseline


if __name__ == '__main__':
    unittest.main()
