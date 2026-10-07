import hashlib
import random
import re
import time


class EnhancedSemanticMemory:
    """Enhanced semantic memory with advanced clustering and relationship detection."""

    def __init__(self, base_semantic_memory):
        self.base_memory = base_semantic_memory
        self.concept_hierarchies = {}
        self.semantic_relationships = {}
        self.contextual_weights = {}

    def build_concept_hierarchy(self, words):
        words = words if words is not None else []
        for word in words:
            if word not in self.concept_hierarchies:
                self.concept_hierarchies[word] = {
                    'superordinates': set(),
                    'subordinates': set(),
                    'related_concepts': set(),
                    'semantic_density': 0,
                }

    def detect_semantic_relationships(self, word1, word2, relationship_type):
        if word1 not in self.semantic_relationships:
            self.semantic_relationships[word1] = {}
        self.semantic_relationships[word1][word2] = {
            'type': relationship_type,
            'strength': 1.0,
            'context_count': 1,
        }

    def get_semantic_context(self, word, max_related=5):
        context = {
            'clusters': [],
            'relationships': [],
            'hierarchies': [],
            'contextual_usage': [],
        }
        if hasattr(self.base_memory, 'word_to_cluster') and word in self.base_memory.word_to_cluster:
            cluster_id = self.base_memory.word_to_cluster[word]
            if cluster_id in self.base_memory.clusters:
                context['clusters'] = list(self.base_memory.clusters[cluster_id])[:max_related]
        if word in self.semantic_relationships:
            context['relationships'] = list(self.semantic_relationships[word].keys())[:max_related]
        return context


class AdvancedReasoningEngine:
    """Advanced reasoning engine using statistical patterns and logic."""

    def __init__(self, context_window=10, adaptation_rate=0.1, brain=None):
        self.reasoning_patterns = {}
        self.logical_connectors = {
            'cause_effect': ['because', 'therefore', 'thus', 'hence', 'so', 'as a result'],
            'comparison': ['however', 'but', 'although', 'while', 'whereas', 'in contrast'],
            'addition': ['also', 'moreover', 'furthermore', 'in addition', 'besides'],
            'sequence': ['first', 'second', 'then', 'next', 'finally', 'lastly'],
            'example': ['for example', 'such as', 'like', 'including', 'specifically'],
        }
        self.brain = brain
        self.reasoning_memory = []
        self.last_reasoning_trace = {}
        try:
            self.context_window = max(1, int(context_window or 10))
        except (TypeError, ValueError):
            self.context_window = 10
        try:
            self.adaptation_rate = float(adaptation_rate or 0.1)
        except (TypeError, ValueError):
            self.adaptation_rate = 0.1

    def analyze_conversation_context(self, user_input, conversation_history):
        return self.analyze_context(user_input, conversation_history)

    def get_last_reasoning_trace(self):
        trace = getattr(self, 'last_reasoning_trace', None)
        return dict(trace) if isinstance(trace, dict) else {}

    def analyze_context(self, user_input, conversation_history):
        context_words = self._extract_context_words(user_input, conversation_history)
        return {
            'topic': self._identify_topic(context_words),
            'intent': self._identify_intent(user_input),
            'complexity': self._assess_complexity(user_input),
            'sentiment': self._analyze_sentiment(user_input),
            'reasoning_needed': self._determine_reasoning_needed(user_input),
            'logical_connectors': self._find_logical_connectors(user_input),
        }

    def generate_reasoned_response(self, user_input, conversation_history, base_response, response_plan=None):
        analysis = self.analyze_context(user_input, conversation_history)
        trace = {
            'analysis': analysis,
            'mode': 'none',
            'reasoning_needed': bool(isinstance(analysis, dict) and analysis.get('reasoning_needed')),
        }
        response = base_response if isinstance(base_response, str) else ""

        if isinstance(analysis, dict) and analysis.get('reasoning_needed'):
            graph_trace = self._build_graph_reasoning_trace(user_input, response_plan)
            if graph_trace.get('usable'):
                response = self._apply_graph_reasoning(response, graph_trace, analysis)
                trace.update(graph_trace)
                trace['mode'] = 'graph_path'
            else:
                response = self._apply_reasoning_patterns(response, analysis)
                trace['mode'] = 'pattern_reasoning'

        trace['response_preview'] = str(response or '')[:200]
        self.last_reasoning_trace = trace
        return {
            'response': response,
            'trace': trace,
        }

    def _extract_context_words(self, user_input, conversation_history):
        user_input = user_input if user_input is not None else ""
        hist = conversation_history[-3:] if isinstance(conversation_history, (list, tuple)) else []
        all_text = user_input + " " + " ".join([str(h) for h in hist])
        words = re.findall(r'\b[a-zA-Z]+\b', all_text.lower())
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does',
            'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can',
        }
        return [w for w in words if w not in stop_words and len(w) > 2][:self.context_window]

    def _identify_topic(self, context_words):
        if not context_words:
            return "general"
        word_scores = {}
        for word in context_words:
            word_scores[word] = word_scores.get(word, 0) + 1
        for word in word_scores:
            if len(word) > 4:
                word_scores[word] *= 1.5
        return max(word_scores.items(), key=lambda x: x[1])[0] if word_scores else "general"

    def _identify_intent(self, user_input):
        input_lower = (user_input if user_input is not None else "").lower()
        if any(word in input_lower for word in ['what', 'how', 'why', 'when', 'where', 'who']):
            return 'question'
        if any(word in input_lower for word in ['explain', 'tell me', 'describe', 'show']):
            return 'seeking_explanation'
        if any(word in input_lower for word in ['think', 'opinion', 'believe', 'feel']):
            return 'seeking_opinion'
        if any(word in input_lower for word in ['help', 'problem', 'issue', 'trouble']):
            return 'seeking_help'
        if any(word in input_lower for word in ['thanks', 'thank you', 'appreciate']):
            return 'gratitude'
        return 'statement'

    def _assess_complexity(self, user_input):
        if user_input is None or not isinstance(user_input, str):
            return 'low'
        words = user_input.split()
        if not words:
            return 'low'
        avg_word_length = sum(len(word) for word in words) / len(words)
        unique_words = len(set(words))
        total_words = len(words)
        complexity_score = (avg_word_length * 0.3) + (unique_words / total_words * 0.4) + (total_words / 20 * 0.3)
        if complexity_score > 0.7:
            return 'high'
        if complexity_score > 0.4:
            return 'medium'
        return 'low'

    def _analyze_sentiment(self, user_input):
        if user_input is None or not isinstance(user_input, str):
            return 'neutral'
        positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'love', 'like', 'happy', 'excited',
            'positive', 'yes', 'agree', 'correct', 'right',
        }
        negative_words = {
            'bad', 'terrible', 'awful', 'hate', 'dislike', 'sad', 'angry', 'frustrated', 'negative',
            'no', 'disagree', 'wrong', 'problem', 'issue',
        }
        words = set(user_input.lower().split())
        positive_count = len(words.intersection(positive_words))
        negative_count = len(words.intersection(negative_words))
        if positive_count > negative_count:
            return 'positive'
        if negative_count > positive_count:
            return 'negative'
        return 'neutral'

    def _determine_reasoning_needed(self, user_input):
        if user_input is None or not isinstance(user_input, str):
            return False
        reasoning_triggers = ['why', 'how', 'explain', 'because', 'reason', 'logic', 'think', 'opinion', 'believe']
        return any(trigger in user_input.lower() for trigger in reasoning_triggers)

    def _find_logical_connectors(self, user_input):
        found_connectors = []
        if user_input is None or not isinstance(user_input, str):
            return found_connectors
        for category, connectors in self.logical_connectors.items():
            for connector in connectors:
                if connector in user_input.lower():
                    found_connectors.append((category, connector))
        return found_connectors

    def _apply_reasoning_patterns(self, base_response, analysis):
        enhanced_response = (base_response if base_response is not None else "") or ""
        if not isinstance(analysis, dict):
            return enhanced_response
        intent = analysis.get('intent')
        complexity = analysis.get('complexity')
        sentiment = analysis.get('sentiment')
        low = (enhanced_response or "").lower()
        if intent == 'question':
            enhanced_response = f"Based on the context, {low}" if complexity == 'high' else f"I think {low}"
        elif intent == 'seeking_explanation':
            enhanced_response = f"Let me explain: {low}"
        elif intent == 'seeking_opinion':
            enhanced_response = f"In my view, {low}"
        if complexity == 'high':
            enhanced_response = f"Given the complexity of this topic, {(enhanced_response or '').lower()}"
        if sentiment == 'positive':
            enhanced_response = f"That's great! {enhanced_response}"
        elif sentiment == 'negative':
            enhanced_response = f"I understand your concern. {enhanced_response}"
        return enhanced_response

    def _build_graph_reasoning_trace(self, user_input, response_plan):
        plan = response_plan if isinstance(response_plan, dict) else {}
        paths = plan.get('reasoning_paths', []) if isinstance(plan.get('reasoning_paths', []), list) else []
        reasoning_summary = str(plan.get('reasoning_summary', '') or '').strip()

        if not paths:
            knowledge_store = getattr(self.brain, 'knowledge_store', None) if self.brain is not None else None
            if knowledge_store is not None and user_input:
                try:
                    preview = knowledge_store.get_graph_reasoning_preview(user_input, limit=2, max_depth=2)
                except Exception:
                    preview = {}
                if isinstance(preview, dict):
                    paths = preview.get('paths', []) if isinstance(preview.get('paths', []), list) else []
                    if not reasoning_summary:
                        reasoning_summary = str(preview.get('summary', '') or '').strip()

        top_path = paths[0] if paths else {}
        path_score = 0.0
        try:
            path_score = float(top_path.get('path_score', 0.0) or 0.0)
        except (TypeError, ValueError):
            path_score = 0.0

        explanation = str(top_path.get('explanation', '') or reasoning_summary).strip()
        step_summaries = top_path.get('step_summaries', []) if isinstance(top_path.get('step_summaries', []), list) else []
        relation_chain = top_path.get('relation_chain', []) if isinstance(top_path.get('relation_chain', []), list) else []
        usable = bool(explanation and path_score >= 0.68 and self._is_reasoning_surface_usable(explanation))
        return {
            'usable': usable,
            'path_score': path_score,
            'step_count': int(top_path.get('step_count', 0) or len(step_summaries)),
            'relation_chain': relation_chain,
            'step_summaries': step_summaries,
            'reasoning_summary': reasoning_summary,
            'graph_explanation': explanation,
            'seed_concepts': plan.get('reasoning_seed_concepts', []) if isinstance(plan.get('reasoning_seed_concepts', []), list) else [],
        }

    def _is_reasoning_surface_usable(self, text):
        normalized = " ".join(str(text or '').split()).strip()
        if not normalized:
            return False
        lower_text = normalized.lower()
        if any(fragment in lower_text for fragment in ('interesting question let', 'question let me', ' let me think about that ')):
            return False
        stray_letters = [token for token in re.findall(r'\b[a-zA-Z]\b', lower_text) if token not in {'a', 'i'}]
        return len(stray_letters) <= 1

    def _should_replace_with_graph_explanation(self, response):
        normalized = " ".join(str(response or '').split()).strip()
        if not normalized:
            return True
        lower_text = normalized.lower()
        if any(fragment in lower_text for fragment in ('interesting question let', 'question let me', "that's an interesting question let")):
            return True
        stray_letters = [token for token in re.findall(r'\b[a-zA-Z]\b', lower_text) if token not in {'a', 'i'}]
        if len(stray_letters) > 1:
            return True
        tokens = normalized.split()
        if len(tokens) >= 10:
            prefix = " ".join(token.lower() for token in tokens[:4])
            if prefix and prefix in " ".join(token.lower() for token in tokens[4:]):
                return True
        return False

    def _apply_graph_reasoning(self, base_response, graph_trace, analysis):
        response = " ".join(str(base_response or '').split()).strip()
        explanation = " ".join(str(graph_trace.get('graph_explanation', '') or '').split()).strip()
        if not explanation:
            return response

        if self._should_replace_with_graph_explanation(response):
            if analysis.get('intent') == 'seeking_explanation' and graph_trace.get('step_count', 0) >= 1:
                return f"One useful relation is that {explanation[:1].lower() + explanation[1:]}"
            return explanation

        if not response:
            return explanation

        if not response.endswith(('.', '!', '?')):
            response += '.'

        response_lower = response.lower()
        explanation_lower = explanation.lower()
        if explanation_lower in response_lower:
            return response

        if 'causes' in graph_trace.get('relation_chain', []):
            if explanation.endswith('.'):
                explanation = explanation[:-1]
            return f"{response} Because {explanation[:1].lower() + explanation[1:]}."

        if analysis.get('intent') == 'seeking_explanation':
            return f"{response} One useful relation chain is: {explanation}"

        if analysis.get('intent') == 'question' and len(graph_trace.get('step_summaries', [])) >= 2:
            return f"{response} This follows from {explanation}"

        return response


class AdaptiveLearningSystem:
    """Adaptive learning system that improves performance over time."""

    def __init__(self, brain=None):
        self.brain = brain
        self.performance_history = []
        self.learning_event_history = []
        self.learning_rates = {
            'response_quality': 0.1,
            'context_understanding': 0.15,
            'grammar_accuracy': 0.12,
            'semantic_coherence': 0.13,
            'conversation_flow': 0.14,
        }
        self.adaptive_parameters = {
            'context_weight': 0.3,
            'semantic_weight': 0.4,
            'pattern_weight': 0.3,
            'creativity_factor': 0.5,
        }
        self.adaptation_thresholds = {
            'quality_improvement': 0.05,
            'performance_decline': 0.03,
            'learning_rate_adjustment': 0.02,
        }
        self.max_history_size = 100
        self.max_event_history = 200
        self.metric_retention_limit = 2500
        self._persisted_metric_count = 0

    def _safe_float(self, value, default=0.0):
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _build_context_hash(self, metric_type, context):
        payload = f"{metric_type}|{context}".encode('utf-8', errors='ignore')
        return hashlib.sha1(payload).hexdigest()[:16]

    def _persist_metric(self, metric_type, value, context=''):
        brain = getattr(self, 'brain', None)
        cur = getattr(brain, 'cur', None)
        con = getattr(brain, 'con', None)
        if brain is None or cur is None or con is None or getattr(brain, 'is_clone', False):
            return
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return
        try:
            cur.execute(
                """
                INSERT INTO adaptive_learning_metrics (metric_type, value, timestamp, context_hash)
                VALUES (?, ?, ?, ?)
                """,
                (
                    str(metric_type or 'metric'),
                    numeric_value,
                    time.time(),
                    self._build_context_hash(metric_type, context),
                ),
            )
            self._persisted_metric_count += 1
            if self._persisted_metric_count % 25 == 0:
                cur.execute(
                    """
                    DELETE FROM adaptive_learning_metrics
                    WHERE rowid NOT IN (
                        SELECT rowid
                        FROM adaptive_learning_metrics
                        ORDER BY timestamp DESC
                        LIMIT ?
                    )
                    """,
                    (self.metric_retention_limit,),
                )
            con.commit()
        except Exception:
            pass

    def update_learning_parameters(self, current_quality, target_quality, context_key=''):
        entry = {
            'quality': self._safe_float(current_quality, 0.0),
            'target': self._safe_float(target_quality, 0.0),
            'timestamp': time.time(),
            'context': str(context_key or ''),
        }
        self.performance_history.append(entry)
        if len(self.performance_history) > self.max_history_size:
            self.performance_history.pop(0)
        self._persist_metric('response_quality', entry['quality'], context_key)
        self._persist_metric('quality_gap', entry['quality'] - entry['target'], context_key)
        self._analyze_performance_trends()
        self._adjust_learning_rates()

    def record_learning_event(self, event_type, quality_score, accepted, source_kind='conversation', detail=''):
        entry = {
            'event_type': str(event_type or 'learning_event'),
            'quality': self._safe_float(quality_score, 0.0),
            'accepted': bool(accepted),
            'source_kind': str(source_kind or 'conversation'),
            'detail': str(detail or ''),
            'timestamp': time.time(),
        }
        self.learning_event_history.append(entry)
        if len(self.learning_event_history) > self.max_event_history:
            self.learning_event_history.pop(0)
        context = f"{entry['event_type']}|{entry['source_kind']}|{entry['detail']}"
        self._persist_metric('live_learning_acceptance', 1.0 if entry['accepted'] else 0.0, context)
        self._persist_metric(f"{entry['event_type']}_quality", entry['quality'], context)

    def _recent_quality_samples(self, limit=12):
        return [
            self._safe_float(item.get('quality'), 0.0)
            for item in self.performance_history[-max(1, limit):]
            if isinstance(item, dict)
        ]

    def _quality_trend_slope(self, limit=12):
        samples = self._recent_quality_samples(limit=limit)
        if len(samples) < 2:
            return 0.0
        return (samples[-1] - samples[0]) / max(1, len(samples) - 1)

    def get_learning_health_snapshot(self):
        quality_samples = self._recent_quality_samples(limit=12)
        avg_quality = (sum(quality_samples) / len(quality_samples)) if quality_samples else None
        trend_slope = self._quality_trend_slope(limit=12)
        recent_events = self.learning_event_history[-20:]
        acceptance_values = [1.0 if event.get('accepted') else 0.0 for event in recent_events if isinstance(event, dict)]
        acceptance_rate = (sum(acceptance_values) / len(acceptance_values)) if acceptance_values else None

        if avg_quality is None:
            mode = 'balanced'
            recommended_min_quality = 0.66
        elif (
            avg_quality < 0.58
            or trend_slope < -0.025
            or (acceptance_rate is not None and acceptance_rate < 0.50)
        ):
            mode = 'cautious'
            recommended_min_quality = 0.72
        elif (
            avg_quality >= 0.78
            and trend_slope >= -0.005
            and (acceptance_rate is None or acceptance_rate >= 0.70)
        ):
            mode = 'active'
            recommended_min_quality = 0.62
        else:
            mode = 'balanced'
            recommended_min_quality = 0.66

        if trend_slope > 0.01:
            trend = 'improving'
        elif trend_slope < -0.01:
            trend = 'declining'
        else:
            trend = 'stable'

        return {
            'success': True,
            'mode': mode,
            'recent_average_quality': avg_quality,
            'recent_quality_trend': trend,
            'recent_quality_slope': trend_slope,
            'recent_learning_acceptance_rate': acceptance_rate,
            'recent_learning_event_count': len(recent_events),
            'recommended_min_quality': recommended_min_quality,
            'learning_rates': self.learning_rates.copy(),
            'adaptive_parameters': self.adaptive_parameters.copy(),
        }

    def should_accept_live_learning(self, quality_score, source_kind='conversation', response_text='', context_text=''):
        quality_value = self._safe_float(quality_score, 0.0)
        snapshot = self.get_learning_health_snapshot()
        mode = str(snapshot.get('mode', 'balanced') or 'balanced')
        recommended_min_quality = self._safe_float(snapshot.get('recommended_min_quality'), 0.66)

        if quality_value < 0.58:
            return {
                'accept': False,
                'reason': 'quality_floor',
                'mode': mode,
                'recommended_min_quality': recommended_min_quality,
            }

        # Held-out hygiene: refuse durable learning from scaffolded, echoed, or toxic text.
        if source_kind in {'conversation', 'knowledge', 'self_training'}:
            try:
                try:
                    from .evaluation_harness import assess_learning_hygiene
                except ImportError:
                    from evaluation_harness import assess_learning_hygiene
                hygiene = assess_learning_hygiene(response_text, context_text)
            except Exception:
                hygiene = {'accept': True, 'reason': 'hygiene_unavailable', 'failures': [], 'surface': {}}
            if not hygiene.get('accept', True):
                return {
                    'accept': False,
                    'reason': f"hygiene_{hygiene.get('reason', 'reject')}",
                    'mode': mode,
                    'recommended_min_quality': recommended_min_quality,
                    'hygiene': hygiene,
                }

        brain = getattr(self, 'brain', None)
        low_information_check = getattr(brain, '_is_low_information_response', None)
        if callable(low_information_check):
            try:
                if low_information_check(response_text):
                    return {
                        'accept': False,
                        'reason': 'low_information',
                        'mode': mode,
                        'recommended_min_quality': recommended_min_quality,
                    }
            except Exception:
                pass

        if quality_value < recommended_min_quality:
            return {
                'accept': False,
                'reason': 'below_adaptive_gate',
                'mode': mode,
                'recommended_min_quality': recommended_min_quality,
            }

        if mode == 'cautious' and source_kind in {'conversation', 'knowledge'} and quality_value < (recommended_min_quality + 0.04):
            return {
                'accept': False,
                'reason': 'cautious_mode_guard',
                'mode': mode,
                'recommended_min_quality': recommended_min_quality,
            }

        return {
            'accept': True,
            'reason': 'accepted',
            'mode': mode,
            'recommended_min_quality': recommended_min_quality,
        }

    def _analyze_performance_trends(self):
        if len(self.performance_history) < 10:
            return
        recent_performance = self.performance_history[-10:]
        quality_trend = [p.get('quality', 0) for p in recent_performance]
        if len(quality_trend) >= 2:
            trend_slope = (quality_trend[-1] - quality_trend[0]) / len(quality_trend)
            if trend_slope > self.adaptation_thresholds['quality_improvement']:
                return
            if trend_slope < -self.adaptation_thresholds['performance_decline']:
                self._increase_learning_rates()
            else:
                self._fine_tune_learning_rates()

    def _adjust_learning_rates(self):
        if len(self.performance_history) < 5:
            return
        recent_quality = [p.get('quality', 0) for p in self.performance_history[-5:]]
        if not recent_quality:
            return
        avg_quality = sum(recent_quality) / len(recent_quality)
        if avg_quality > 0.8:
            for key in self.learning_rates:
                self.learning_rates[key] *= 0.95
        elif avg_quality < 0.4:
            for key in self.learning_rates:
                self.learning_rates[key] *= 1.1

    def _increase_learning_rates(self):
        for key in self.learning_rates:
            self.learning_rates[key] = min(0.3, self.learning_rates[key] * 1.2)

    def _fine_tune_learning_rates(self):
        for key in self.learning_rates:
            self.learning_rates[key] *= random.uniform(0.98, 1.02)

    def get_learning_rate(self, parameter):
        return self.learning_rates.get(parameter, 0.1)

    def get_performance_summary(self):
        snapshot = self.get_learning_health_snapshot()
        avg_quality = snapshot.get('recent_average_quality')
        if avg_quality is None:
            return "No performance data available"
        trend = snapshot.get('recent_quality_trend', 'stable')
        mode = snapshot.get('mode', 'balanced')
        acceptance_rate = snapshot.get('recent_learning_acceptance_rate')
        if acceptance_rate is None:
            acceptance_text = 'n/a'
        else:
            acceptance_text = f"{acceptance_rate:.0%}"
        return (
            f"Average Quality: {avg_quality:.3f}, Trend: {trend}, "
            f"Mode: {mode}, Acceptance: {acceptance_text}, Learning Rates: {self.learning_rates}"
        )

    def get_adaptive_weights(self, context_analysis):
        if not isinstance(context_analysis, dict):
            return self.adaptive_parameters.copy()
        weights = self.adaptive_parameters.copy()
        if context_analysis.get('complexity') == 'high':
            weights['semantic_weight'] *= 1.2
            weights['context_weight'] *= 1.1
        elif context_analysis.get('complexity') == 'low':
            weights['pattern_weight'] *= 1.2
            weights['creativity_factor'] *= 1.1
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()} if total > 0 else weights

    def adjust_learning_rate(self):
        self._adjust_learning_rates()


class CreativeResponseGenerator:
    """Enhanced response generation with creativity and context awareness."""

    def __init__(self, brain, parallel_candidates=1, context_window=10, adaptation_rate=0.1):
        self.brain = brain
        self.parallel_candidates = max(1, int(parallel_candidates or 1))
        shared_reasoning = getattr(brain, 'reasoning_engine', None)
        self.reasoning_engine = (
            shared_reasoning
            if shared_reasoning is not None
            else AdvancedReasoningEngine(
                context_window=context_window,
                adaptation_rate=adaptation_rate,
                brain=brain,
            )
        )
        self.enhanced_semantic = EnhancedSemanticMemory(getattr(brain, 'semantic_memory', None))
        shared_adaptive_learning = getattr(brain, 'adaptive_learning', None)
        self.adaptive_learning = (
            shared_adaptive_learning
            if shared_adaptive_learning is not None
            else AdaptiveLearningSystem(brain)
        )
        self.creativity_patterns = self._build_creativity_patterns()

    def _build_creativity_patterns(self):
        return {
            'metaphor': ['like', 'as if', 'similar to', 'reminds me of'],
            'analogy': ['just as', 'in the same way', 'comparable to'],
            'perspective_shift': ['from another angle', 'consider this', 'think about it differently'],
            'synthesis': ['combining', 'integrating', 'bringing together', 'connecting'],
            'exploration': ["let's explore", 'what if', 'imagine', 'suppose'],
        }

    def generate_enhanced_response(self, user_input, conversation_history=None):
        context_analysis = self.reasoning_engine.analyze_conversation_context(user_input, conversation_history or [])
        if not isinstance(context_analysis, dict):
            context_analysis = {}
        adaptive_weights = self.adaptive_learning.get_adaptive_weights(context_analysis)
        candidates = []
        for _ in range(self.parallel_candidates):
            candidate = self._generate_candidate_response(user_input, context_analysis, adaptive_weights)
            if candidate:
                candidates.append(candidate)
        best_response = self._select_best_candidate(candidates, user_input, context_analysis)
        if best_response is None:
            best_response = self.brain.generate_response(user_input)
        if context_analysis.get('intent') in ['seeking_opinion', 'seeking_explanation']:
            best_response = self._apply_creativity_enhancement(best_response, context_analysis)
        return best_response

    def _generate_candidate_response(self, user_input, context_analysis, adaptive_weights):
        try:
            if context_analysis.get('intent') == 'question':
                return self._generate_question_response(user_input, adaptive_weights)
            if context_analysis.get('intent') == 'seeking_explanation':
                return self._generate_explanation_response(user_input, adaptive_weights)
            return self._generate_general_response(user_input, adaptive_weights)
        except Exception as e:
            print(f"Error generating candidate response: {e}")
            return None

    def _generate_question_response(self, user_input, adaptive_weights):
        response = self.brain.generate_response(user_input) if getattr(self, 'brain', None) else None
        response = response if response is not None else ""
        if any(word in (user_input or "").lower() for word in ['why', 'how', 'what']):
            connectors = getattr(getattr(self, 'reasoning_engine', None), 'logical_connectors', {}).get('cause_effect', [])
            if connectors:
                response = f"{response} {random.choice(connectors)} this approach considers multiple factors."
        return response

    def _generate_explanation_response(self, user_input, adaptive_weights):
        response = self.brain.generate_response(user_input) if getattr(self, 'brain', None) else None
        response = response if response is not None else ""
        if len((response or "").split()) > 10:
            structure_words = ['First', 'Then', 'Finally']
            sentences = response.split('. ')
            if len(sentences) >= 3:
                structured_response = []
                for i, sentence in enumerate(sentences[:3]):
                    structured_response.append(f"{structure_words[i]}, {sentence}" if i < len(structure_words) else sentence)
                response = '. '.join(structured_response)
        return response

    def _generate_general_response(self, user_input, adaptive_weights):
        out = self.brain.generate_response(user_input) if getattr(self, 'brain', None) else None
        return out if out is not None else ""

    def _select_best_candidate(self, candidates, user_input, context_analysis):
        if not candidates:
            out = self.brain.generate_response(user_input) if getattr(self, 'brain', None) else None
            return out if out is not None else ""
        scored_candidates = [(candidate, self._score_candidate(candidate, user_input, context_analysis)) for candidate in candidates]
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return scored_candidates[0][0] if scored_candidates else None

    def _score_candidate(self, candidate, user_input, context_analysis):
        score = 0.0
        score += self.brain._calculate_response_quality(candidate, user_input) * 0.4
        score += self._calculate_context_relevance(candidate, user_input) * 0.3
        score += self._calculate_semantic_coherence(candidate, user_input) * 0.2
        score += self._calculate_creativity_bonus(candidate, context_analysis) * 0.1
        return score

    def _calculate_context_relevance(self, response, user_input):
        if user_input is None or response is None:
            return 0.5
        try:
            user_words = set(user_input.lower().split())
            response_words = set(response.lower().split())
        except (AttributeError, TypeError):
            return 0.5
        if not user_words:
            return 0.5
        return min(1.0, len(user_words.intersection(response_words)) / len(user_words))

    def _calculate_semantic_coherence(self, response, user_input):
        response_words = self.brain.clean_text(response)
        user_words = self.brain.clean_text(user_input)
        if not response_words or not user_words:
            return 0.5
        semantic_matches = 0
        for word in response_words:
            if word in self.brain.semantic_memory.word_to_cluster:
                cluster_id = self.brain.semantic_memory.word_to_cluster[word]
                for user_word in user_words:
                    if user_word in self.brain.semantic_memory.word_to_cluster and self.brain.semantic_memory.word_to_cluster[user_word] == cluster_id:
                        semantic_matches += 1
                        break
        return semantic_matches / len(response_words) if response_words else 0.5

    def _calculate_creativity_bonus(self, response, context_analysis):
        creativity_score = 0.0
        for patterns in self.creativity_patterns.values():
            if any(pattern in response.lower() for pattern in patterns):
                creativity_score += 0.2
        if context_analysis.get('complexity') == 'high' and len(response.split()) > 15:
            creativity_score += 0.1
        return min(1.0, creativity_score)

    def _apply_creativity_enhancement(self, response, context_analysis):
        if context_analysis.get('intent') == 'seeking_opinion':
            perspectives = ["From my perspective, ", "I think ", "In my view, ", "It seems to me that "]
            if not any(response.startswith(p) for p in perspectives):
                response = random.choice(perspectives) + response.lower()
        return response


class Critic:
    """Quality assessment system for response evaluation."""

    def __init__(self, brain):
        self.brain = brain
        self.quality_history = []
        self.confidence_threshold = 0.7

    def assess(self, user_input, response, context_hints=None):
        try:
            response = response if isinstance(response, str) else ""
            plan = context_hints if isinstance(context_hints, dict) else {}
            length_score = min(len(response.split()) / 10, 1.0)
            coherence_score = self._assess_coherence(response)
            relevance_score = self._assess_relevance(user_input, response, plan)
            creativity_score = self._assess_creativity(response)
            grounding_score = self._assess_grounding(response, plan)
            plan_alignment_score = self._assess_plan_alignment(response, plan)
            focus_score = self._assess_focus(user_input, response, plan)
            surface_score = self._assess_surface_wellformedness(response)
            quality_score = self._assess_quality(user_input, response)
            low_information = self._is_low_information_response(response)
            if (
                low_information
                and plan
                and grounding_score >= 0.75
                and plan_alignment_score >= 0.45
                and len(re.findall(r'\b[a-zA-Z0-9_]+\b', (response or '').lower())) >= 4
            ):
                low_information = False
            weaknesses = self._identify_weaknesses(
                response,
                plan,
                coherence_score,
                relevance_score,
                grounding_score,
                plan_alignment_score,
                focus_score,
                surface_score,
                quality_score,
                low_information,
            )
            confidence = (
                (coherence_score * 0.14)
                + (relevance_score * 0.18)
                + (grounding_score * 0.24)
                + (plan_alignment_score * 0.10)
                + (focus_score * 0.10)
                + (surface_score * 0.12)
                + (quality_score * 0.08)
                + (creativity_score * 0.04)
            )
            assessment = {
                'confidence': confidence,
                'length_score': length_score,
                'coherence_score': coherence_score,
                'relevance_score': relevance_score,
                'creativity_score': creativity_score,
                'grounding_score': grounding_score,
                'plan_alignment_score': plan_alignment_score,
                'focus_score': focus_score,
                'surface_score': surface_score,
                'quality_score': quality_score,
                'low_information': low_information,
                'weaknesses': weaknesses,
                'repair_recommended': bool(weaknesses) or confidence < self.confidence_threshold,
                'timestamp': time.time(),
            }
            self.quality_history.append(assessment)
            if len(self.quality_history) > 100:
                self.quality_history = self.quality_history[-100:]
            return assessment
        except Exception as e:
            print(f"Critic assessment error: {e}")
            return {'confidence': 0.5, 'error': str(e)}

    def repair_response(self, user_input, response, context_hints=None, assessment=None):
        response = response if isinstance(response, str) else ""
        plan = context_hints if isinstance(context_hints, dict) else {}
        assessment = assessment if isinstance(assessment, dict) else self.assess(user_input, response, plan)
        if not response.strip():
            return {'response': response, 'strategy': 'no_response', 'changed': False}
        if not assessment.get('repair_recommended'):
            return {'response': response, 'strategy': 'accepted', 'changed': False}

        repaired = response.strip()
        strategy = 'none'
        weaknesses = set(assessment.get('weaknesses', []) or [])

        if plan and self._plan_matches_prompt(user_input, plan) and weaknesses.intersection({'low_grounding', 'low_plan_alignment', 'low_information', 'low_relevance', 'topic_drift', 'low_surface'}):
            grounded = self._compose_claim_grounded_response(plan, user_input=user_input)
            if grounded:
                repaired = grounded
                strategy = 'claim_grounding'
                grounded_assessment = self.assess(user_input, repaired, plan)
                if grounded_assessment.get('surface_score', 1.0) < 0.7:
                    normalized_grounded = self._normalize_response_surface(repaired, aggressive=True)
                    if normalized_grounded and normalized_grounded.strip():
                        repaired = normalized_grounded.strip()
                        strategy = 'claim_grounding_surface_repair'

        if repaired == response.strip() and weaknesses.intersection({'low_coherence', 'low_information', 'topic_drift', 'low_surface'}):
            normalized = self._normalize_response_surface(response, aggressive='topic_drift' in weaknesses)
            if normalized:
                repaired = normalized
                strategy = 'surface_repair'

        anti_loop_filter = getattr(self.brain, 'anti_loop_filter', None)
        if anti_loop_filter and repaired:
            try:
                filtered = anti_loop_filter.filter_response(repaired)
                if isinstance(filtered, str) and filtered.strip():
                    repaired = filtered.strip()
                    if strategy == 'none':
                        strategy = 'anti_loop_filter'
            except Exception:
                pass

        if not repaired.endswith(('.', '?', '!')):
            repaired += '.'

        return {
            'response': repaired,
            'strategy': strategy if strategy != 'none' else 'accepted',
            'changed': repaired.strip() != response.strip(),
        }

    def _assess_coherence(self, response):
        try:
            words = response.lower().split()
            if len(words) < 2:
                return 0.3
            coherence_score = 0
            for i in range(len(words) - 1):
                word1, word2 = words[i].strip(".,!?;:'\""), words[i + 1].strip(".,!?;:'\"")
                associations = getattr(self.brain, 'word_associations', {})
                if word1 in associations and word2 in associations[word1]:
                    coherence_score += associations[word1][word2]
            return min(coherence_score / (len(words) - 1), 1.0)
        except (ZeroDivisionError, KeyError, TypeError):
            return 0.5

    def _assess_relevance(self, user_input, response, plan=None):
        try:
            input_words = set(re.findall(r'\b[a-zA-Z0-9_]+\b', (user_input or '').lower()))
            response_words = set(re.findall(r'\b[a-zA-Z0-9_]+\b', (response or '').lower()))
            if not input_words or not response_words:
                return 0.3
            overlap_score = min(len(input_words.intersection(response_words)) / len(input_words), 1.0)
            plan_terms = self._collect_plan_terms(plan)
            if plan_terms:
                plan_overlap = len(plan_terms.intersection(response_words)) / max(1, len(plan_terms))
                return min(1.0, (overlap_score * 0.7) + (plan_overlap * 0.3))
            return overlap_score
        except (ZeroDivisionError, AttributeError):
            return 0.5

    def _assess_creativity(self, response):
        try:
            words = response.lower().split()
            if len(words) < 3:
                return 0.3
            unique_words = len(set(words))
            total_words = len(words)
            if total_words == 0:
                return 0.3
            return min((unique_words / total_words) * 1.5, 1.0)
        except (ZeroDivisionError, TypeError):
            return 0.5

    def _assess_quality(self, user_input, response):
        try:
            quality_fn = getattr(self.brain, '_calculate_response_quality', None)
            if callable(quality_fn):
                return max(0.0, min(1.0, float(quality_fn(response, user_input))))
        except Exception:
            pass
        return 0.5

    def _assess_surface_wellformedness(self, response):
        response = " ".join(str(response or '').split()).strip()
        if not response:
            return 0.2

        tokens = re.findall(r"\b[\w']+\b", response)
        if not tokens:
            return 0.15

        lower_response = response.lower()
        lower_tokens = [token.lower() for token in tokens]
        penalties = 0.0

        single_char_alpha = [token for token in lower_tokens if len(token) == 1 and token.isalpha() and token not in {'a', 'i'}]
        if single_char_alpha:
            penalties += min(0.30, 0.16 * len(single_char_alpha))
        if " s " in f" {lower_response} " or " t " in f" {lower_response} ":
            penalties += 0.16

        glitch_patterns = (
            r"\binteresting question let\b",
            r"\bquestion let me\b",
            r"\blet me think about that [a-z]\b",
        )
        if any(re.search(pattern, lower_response) for pattern in glitch_patterns):
            penalties += 0.28

        if len(lower_tokens) >= 8:
            prefix = " ".join(lower_tokens[:4])
            later = " ".join(lower_tokens[4:])
            if prefix and prefix in later:
                penalties += 0.22

            for window_size in (3, 4):
                windows = {}
                for index in range(0, len(lower_tokens) - window_size + 1):
                    key = tuple(lower_tokens[index:index + window_size])
                    previous_index = windows.get(key)
                    if previous_index is not None and index - previous_index >= window_size:
                        penalties += 0.12
                        break
                    windows[key] = index

        trailing_token = re.sub(r"[^a-z0-9']+", "", lower_tokens[-1])
        if trailing_token in {'let', 'and', 'or', 'but', 'because', 'that', 'these', 'those'}:
            penalties += 0.12

        validate_score = 0.78
        try:
            validate_fn = getattr(self.brain, '_validate_response_quality', None)
            if callable(validate_fn):
                validate_score = 0.96 if validate_fn(response) else 0.54
        except Exception:
            pass

        score = max(0.12, min(1.0, validate_score - penalties))
        return score

    def _score_grounding_claim(self, claim, role='support'):
        if not isinstance(claim, dict):
            return 0.0

        text = str(claim.get('text', '') or '').strip()
        if not text:
            return 0.0

        confidence_profile = claim.get('confidence_profile', {}) if isinstance(claim.get('confidence_profile'), dict) else {}
        provenance = claim.get('provenance', {}) if isinstance(claim.get('provenance'), dict) else {}

        overall = float(confidence_profile.get('overall', 0.0) or 0.0)
        source_reliability = float(confidence_profile.get('source_reliability', 0.0) or 0.0)
        stability = float(confidence_profile.get('stability', 0.0) or 0.0)
        consistency = float(confidence_profile.get('consistency', 0.0) or 0.0)
        revisability = float(confidence_profile.get('revisability', 0.0) or 0.0)
        source_count = float(provenance.get('source_count', 0.0) or 0.0)
        evidence_count = float(provenance.get('evidence_count', 0.0) or 0.0)
        source_type_count = float(provenance.get('source_type_count', 0.0) or 0.0)
        latest_source_type = str(provenance.get('latest_source_type', '') or '').strip().lower()
        surface_score = self._assess_surface_wellformedness(text)
        token_count = len(re.findall(r'\b[a-zA-Z0-9_]+\b', text.lower()))

        uncertainty_penalty = 0.0
        for item in claim.get('uncertainties', []) or []:
            lower_item = str(item).lower()
            if 'relatively weak' in lower_item:
                uncertainty_penalty += 0.10
            elif 'weak' in lower_item:
                uncertainty_penalty += 0.06

        length_penalty = 0.0
        if token_count < 3:
            length_penalty += 0.08
        if token_count > 14:
            length_penalty += min(0.18, (token_count - 14) * 0.015)

        provenance_bonus = min(0.08, ((source_count * 0.5) + evidence_count + source_type_count) * 0.01)
        if latest_source_type == 'conversation_bot':
            provenance_bonus -= 0.04

        relation_type = str(claim.get('relation_type', '') or '').strip().lower()
        relation_bonus = 0.03 if relation_type in {'is_a', 'part_of', 'has_trait', 'prefers'} else 0.0

        score = (
            (overall * 0.28)
            + (source_reliability * 0.18)
            + (stability * 0.12)
            + (consistency * 0.12)
            + (surface_score * 0.24)
            + provenance_bonus
            + relation_bonus
        )
        score -= (revisability * 0.05) + uncertainty_penalty + length_penalty
        if surface_score < 0.55:
            score -= 0.14
        if role == 'main':
            score += 0.02
        return max(0.0, min(1.0, score))

    def _iter_grounding_claims(self, plan):
        if not isinstance(plan, dict):
            return []
        claims = []
        main_claim = plan.get('main_claim', {})
        if isinstance(main_claim, dict) and main_claim.get('text'):
            claims.append(('main', main_claim))
        for claim in plan.get('support_claims', []) or []:
            if isinstance(claim, dict) and claim.get('text'):
                claims.append(('support', claim))
        return claims

    def _is_low_information_response(self, response):
        try:
            low_info_fn = getattr(self.brain, '_is_low_information_response', None)
            if callable(low_info_fn):
                return bool(low_info_fn(response))
        except Exception:
            pass
        words = re.findall(r'\b[a-zA-Z0-9_]+\b', (response or '').lower())
        return len(words) < 5

    def _assess_grounding(self, response, plan):
        response_terms = set(re.findall(r'\b[a-zA-Z0-9_]+\b', (response or '').lower()))
        if not response_terms:
            return 0.2
        claim_texts = []
        main_claim = plan.get('main_claim', {}) if isinstance(plan, dict) else {}
        if isinstance(main_claim, dict) and main_claim.get('text'):
            claim_texts.append(str(main_claim.get('text')))
        for claim in plan.get('support_claims', [])[:2] if isinstance(plan, dict) else []:
            if isinstance(claim, dict) and claim.get('text'):
                claim_texts.append(str(claim.get('text')))
        if not claim_texts:
            return 0.55
        overlaps = []
        for claim_text in claim_texts:
            claim_terms = set(re.findall(r'\b[a-zA-Z0-9_]+\b', claim_text.lower()))
            claim_terms = {term for term in claim_terms if len(term) > 2}
            if not claim_terms:
                continue
            overlaps.append(len(claim_terms.intersection(response_terms)) / max(1, len(claim_terms)))
        if not overlaps:
            return 0.45
        return min(1.0, max(overlaps))

    def _assess_plan_alignment(self, response, plan):
        if not isinstance(plan, dict) or not plan:
            return 0.6
        response_terms = set(re.findall(r'\b[a-zA-Z0-9_]+\b', (response or '').lower()))
        if not response_terms:
            return 0.2
        plan_terms = self._collect_plan_terms(plan)
        if not plan_terms:
            return 0.55
        return min(1.0, len(plan_terms.intersection(response_terms)) / max(1, len(plan_terms)))

    def _assess_focus(self, user_input, response, plan):
        response_terms = [term for term in re.findall(r'\b[a-zA-Z0-9_]+\b', (response or '').lower()) if len(term) > 2]
        if not response_terms:
            return 0.2
        anchor_terms = set(term for term in re.findall(r'\b[a-zA-Z0-9_]+\b', (user_input or '').lower()) if len(term) > 2)
        anchor_terms.update(self._collect_plan_terms(plan))
        if not anchor_terms:
            return 0.6

        def _matches_anchor(term):
            if term in anchor_terms:
                return True
            if len(term) < 4:
                return False
            prefix = term[:4]
            return any(anchor.startswith(prefix) or prefix.startswith(anchor[:4]) for anchor in anchor_terms if len(anchor) >= 4)

        on_topic = sum(1 for term in response_terms if _matches_anchor(term))
        return min(1.0, on_topic / max(1, len(response_terms)))

    def _collect_plan_terms(self, plan):
        if not isinstance(plan, dict):
            return set()
        texts = []
        main_claim = plan.get('main_claim', {})
        if isinstance(main_claim, dict) and main_claim.get('text'):
            texts.append(str(main_claim.get('text')))
        for claim in plan.get('support_claims', [])[:2]:
            if isinstance(claim, dict) and claim.get('text'):
                texts.append(str(claim.get('text')))
        for fact in plan.get('supporting_facts', [])[:2]:
            if isinstance(fact, dict) and fact.get('summary'):
                texts.append(str(fact.get('summary')))
        for concept in plan.get('target_concepts', [])[:3]:
            if isinstance(concept, dict) and concept.get('canonical_name'):
                texts.append(str(concept.get('canonical_name')))
        terms = set()
        for text in texts:
            terms.update(term for term in re.findall(r'\b[a-zA-Z0-9_]+\b', text.lower()) if len(term) > 2)
        return terms

    def _collect_prompt_focus_terms(self, user_input):
        prompt_terms = {
            term for term in re.findall(r'\b[a-zA-Z0-9_]+\b', (user_input or '').lower())
            if len(term) > 2
        }
        stop_terms = {
            'what',
            'when',
            'where',
            'which',
            'who',
            'does',
            'did',
            'just',
            'about',
            'your',
            'you',
            'mai',
            'phoenix',
            'sgm',
            'system',
            'over',
            'time',
        }
        focus_terms = {term for term in prompt_terms if len(term) > 3 and term not in stop_terms}
        return focus_terms or prompt_terms

    def _claim_prompt_fit(self, claim, user_input):
        if not isinstance(claim, dict):
            return 0.0
        text = str(claim.get('text', '') or '').strip().lower()
        if not text:
            return 0.0
        focus_terms = self._collect_prompt_focus_terms(user_input)
        if not focus_terms:
            return 0.0
        text_terms = {
            term for term in re.findall(r'\b[a-zA-Z0-9_]+\b', text)
            if len(term) > 2
        }
        if not text_terms:
            return 0.0
        return len(focus_terms.intersection(text_terms)) / max(1, len(focus_terms))

    def _plan_matches_prompt(self, user_input, plan):
        if not isinstance(plan, dict) or not plan:
            return False
        intent = str(plan.get('intent', '') or '').strip().lower()
        if intent in {'definition', 'self_description'}:
            return True
        prompt_terms = set(term for term in re.findall(r'\b[a-zA-Z0-9_]+\b', (user_input or '').lower()) if len(term) > 2)
        if not prompt_terms:
            return False
        plan_terms = self._collect_plan_terms(plan)
        if not plan_terms:
            return False
        overlap = len(prompt_terms.intersection(plan_terms))
        return overlap >= 2 or (overlap >= 1 and len(prompt_terms) <= 3)

    def _identify_weaknesses(
        self,
        response,
        plan,
        coherence_score,
        relevance_score,
        grounding_score,
        plan_alignment_score,
        focus_score,
        surface_score,
        quality_score,
        low_information,
    ):
        weaknesses = []
        if low_information:
            weaknesses.append('low_information')
        if coherence_score < 0.42:
            weaknesses.append('low_coherence')
        if relevance_score < 0.38:
            weaknesses.append('low_relevance')
        if plan and grounding_score < 0.45:
            weaknesses.append('low_grounding')
        if plan and plan_alignment_score < 0.4:
            weaknesses.append('low_plan_alignment')
        if focus_score < 0.55 and len(re.findall(r'\b[a-zA-Z0-9_]+\b', (response or '').lower())) >= 8:
            weaknesses.append('topic_drift')
        if surface_score < 0.55:
            weaknesses.append('low_surface')
        if quality_score < 0.48:
            weaknesses.append('low_quality')
        return weaknesses

    def _compose_claim_grounded_response(self, plan, user_input=''):
        if not isinstance(plan, dict):
            return ''

        intent = str(plan.get('intent', '') or '').strip().lower()
        ranked_claims = []
        for role, claim in self._iter_grounding_claims(plan):
            prompt_fit = self._claim_prompt_fit(claim, user_input)
            if intent in {'answer', 'explanation', 'guidance'} and prompt_fit <= 0.0:
                continue
            score = self._score_grounding_claim(claim, role=role)
            if score <= 0:
                continue
            ranked_claims.append((score + (prompt_fit * 0.55), prompt_fit, role, claim))
        ranked_claims.sort(key=lambda item: item[0], reverse=True)

        parts = []
        for index, (score, prompt_fit, role, claim) in enumerate(ranked_claims):
            text = str(claim.get('text', '') or '').strip()
            if not text:
                continue
            surface_score = self._assess_surface_wellformedness(text)
            minimum_score = 0.50 if index == 0 else 0.64
            minimum_surface = 0.62 if index == 0 else 0.72
            if intent in {'answer', 'explanation', 'guidance'}:
                minimum_prompt_fit = 0.24 if index == 0 else 0.30
                if prompt_fit < minimum_prompt_fit:
                    continue
            if score < minimum_score or surface_score < minimum_surface:
                continue
            parts.append(text)
            if len(parts) >= 2:
                break

        if not parts and ranked_claims:
            top_text = str(ranked_claims[0][3].get('text', '') or '').strip()
            normalized = self._normalize_response_surface(top_text, aggressive=True)
            if normalized and self._assess_surface_wellformedness(normalized) >= 0.62:
                parts.append(normalized)

        uncertainties = [str(item).strip() for item in plan.get('uncertainties', [])[:1] if str(item).strip()]
        plan_confidence = 0.0
        try:
            plan_confidence = float(plan.get('plan_confidence', 0.0) or 0.0)
        except (TypeError, ValueError):
            plan_confidence = 0.0
        if plan_confidence < 0.62 and uncertainties:
            parts.append(uncertainties[0])

        filtered = []
        seen = set()
        for part in parts:
            key = part.lower()
            if not part or key in seen:
                continue
            seen.add(key)
            filtered.append(part.rstrip('.!?') + '.')
        return " ".join(filtered).strip()

    def _normalize_response_surface(self, response, aggressive=False):
        response = " ".join(str(response or '').split()).strip()
        if not response:
            return ''
        lower_response = response.lower()
        if aggressive and lower_response.startswith("that's an interesting question"):
            return "That's an interesting question. Let me think about that."
        tokens = response.split()
        if aggressive and len(tokens) >= 8:
            lower_tokens = [token.lower() for token in tokens]
            for window_size in (4, 3):
                prefix = lower_tokens[:window_size]
                for start in range(window_size, len(lower_tokens) - window_size + 1):
                    if lower_tokens[start:start + window_size] == prefix:
                        response = " ".join(tokens[:start]).strip()
                        break
                else:
                    continue
                break
        sentences = re.split(r'(?<=[.!?])\s+', response)
        cleaned = []
        seen = set()
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if aggressive:
                sentence_tokens = [token for token in sentence.split() if not (len(token) == 1 and token.isalpha() and token.lower() not in {'a', 'i'})]
                sentence = " ".join(sentence_tokens).strip()
                if not sentence:
                    continue
            key = sentence.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(sentence)
            if aggressive or len(cleaned) >= 2:
                break
        return " ".join(cleaned).strip()


class ConfidenceGate:
    """Confidence-based response regeneration system."""

    def __init__(self, brain):
        self.brain = brain
        self.regeneration_threshold = 0.6
        self.max_regenerations = 2

    def should_regenerate(self, confidence, attempt_count=0):
        if attempt_count >= self.max_regenerations:
            return False
        return confidence < self.regeneration_threshold

    def regenerate_response(self, user_input, context, attempt_count=0):
        try:
            creativity_boost = 0.1 + (attempt_count * 0.05)
            context_weight_reduction = 0.05 * attempt_count
            return self.brain.generate_response(
                user_input,
                context,
                creativity_factor=min(1.0, self.brain.settings_manager.get('creativity_factor', 0.7) + creativity_boost),
                context_weight=max(0.1, self.brain.settings_manager.get('context_weight', 0.3) - context_weight_reduction),
            )
        except Exception as e:
            print(f"Response regeneration error: {e}")
            return None


class AntiLoopFilter:
    """Prevent repetitive response patterns."""

    def __init__(self, brain):
        self.brain = brain
        self.recent_responses = []
        self.max_recent = 10
        self.similarity_threshold = 0.8

    def filter_response(self, response):
        try:
            if not response or len(response.strip()) < 5:
                return response
            for recent in self.recent_responses:
                if self._calculate_similarity(response, recent) > self.similarity_threshold:
                    response = self._modify_response(response)
                    break
            self.recent_responses.append(response)
            if len(self.recent_responses) > self.max_recent:
                self.recent_responses = self.recent_responses[-self.max_recent:]
            return response
        except Exception as e:
            print(f"Anti-loop filter error: {e}")
            return response

    def _calculate_similarity(self, response1, response2):
        try:
            words1 = set(response1.lower().split())
            words2 = set(response2.lower().split())
            if not words1 or not words2:
                return 0
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            return intersection / union if union > 0 else 0
        except (ZeroDivisionError, TypeError, KeyError):
            return 0

    def _modify_response(self, response):
        try:
            words = response.split()
            if len(words) < 3:
                return response
            words.insert(random.randint(1, len(words) - 1), random.choice(["indeed", "certainly", "absolutely", "definitely", "surely"]))
            return " ".join(words)
        except (TypeError, AttributeError):
            return response


class MetaMemory:
    """Weakness tracking and improvement system."""

    def __init__(self, brain):
        self.brain = brain
        self.weakness_ledger = {}
        self.improvement_history = []

    def register_weakness(self, tag, context=None):
        try:
            if tag not in self.weakness_ledger:
                self.weakness_ledger[tag] = {'count': 0, 'first_seen': time.time(), 'last_seen': time.time(), 'contexts': []}
            self.weakness_ledger[tag]['count'] += 1
            self.weakness_ledger[tag]['last_seen'] = time.time()
            if context:
                self.weakness_ledger[tag]['contexts'].append(context)
                if len(self.weakness_ledger[tag]['contexts']) > 5:
                    self.weakness_ledger[tag]['contexts'] = self.weakness_ledger[tag]['contexts'][-5:]
        except Exception as e:
            print(f"Meta-memory weakness registration error: {e}")

    def get_weakness_stats(self):
        try:
            total_weaknesses = len(self.weakness_ledger)
            active_weaknesses = sum(1 for w in self.weakness_ledger.values() if w['count'] > 1)
            return {
                'total_weaknesses': total_weaknesses,
                'active_weaknesses': active_weaknesses,
                'most_common': max(self.weakness_ledger.items(), key=lambda x: x[1]['count']) if self.weakness_ledger else None,
            }
        except (KeyError, TypeError, AttributeError):
            return {'total_weaknesses': 0, 'active_weaknesses': 0, 'most_common': None}


class Curiosity:
    """Idle-time curiosity and exploration system."""

    def __init__(self, brain):
        self.brain = brain
        self.last_curiosity_tick = 0
        self.curiosity_interval = 300
        self.exploration_topics = []

    def tick(self):
        try:
            current_time = time.time()
            if current_time - self.last_curiosity_tick < self.curiosity_interval:
                return
            self.last_curiosity_tick = current_time
            self._explore_new_topics()
            self._analyze_patterns()
            self._update_learning_params()
        except Exception as e:
            print(f"Curiosity tick error: {e}")

    def _explore_new_topics(self):
        try:
            recent_topics = self.brain.hierarchical_memory.get_recent_topics(limit=5)
            for topic in recent_topics:
                if topic not in self.exploration_topics:
                    self.exploration_topics.append(topic)
            if len(self.exploration_topics) > 20:
                self.exploration_topics = self.exploration_topics[-20:]
        except Exception as e:
            print(f"Topic exploration error: {e}")

    def _analyze_patterns(self):
        try:
            if hasattr(self.brain, 'conversation_memory') and self.brain.conversation_memory:
                recent_entries = self.brain.conversation_memory[-10:] if isinstance(self.brain.conversation_memory, list) else []
                word_frequencies = {}
                for entry in recent_entries:
                    if not isinstance(entry, dict):
                        continue
                    bot_text = entry.get('response') or entry.get('bot') or ''
                    if isinstance(bot_text, str) and bot_text.strip():
                        for word in bot_text.lower().split():
                            word_frequencies[word] = word_frequencies.get(word, 0) + 1
                if word_frequencies:
                    most_common = max(word_frequencies.items(), key=lambda x: x[1])
                    if most_common[1] > 2 and hasattr(self.brain, 'batch_operations'):
                        self.brain.batch_operations.append({'type': 'assoc', 'data': (most_common[0], "common_pattern", 0.1, 1)})
        except Exception as e:
            print(f"Pattern analysis error: {e}")

    def _update_learning_params(self):
        try:
            if hasattr(self.brain, 'adaptive_learning'):
                self.brain.adaptive_learning.adjust_learning_rate()
        except Exception as e:
            print(f"Learning parameter update error: {e}")


class EnvironmentFeedback:
    """Environment feedback integration system."""

    def __init__(self, brain):
        self.brain = brain
        self.feedback_handlers = {}
        self.feedback_history = []

    def register_handler(self, event_type, handler_func):
        try:
            self.feedback_handlers[event_type] = handler_func
        except Exception as e:
            print(f"Feedback handler registration error: {e}")

    def process_feedback(self, event_type, data):
        try:
            if event_type in self.feedback_handlers:
                result = self.feedback_handlers[event_type](data)
                self.feedback_history.append({'event_type': event_type, 'data': data, 'result': result, 'timestamp': time.time()})
                if len(self.feedback_history) > 50:
                    self.feedback_history = self.feedback_history[-50:]
                return result
            print(f"No handler registered for event type: {event_type}")
            return None
        except Exception as e:
            print(f"Feedback processing error: {e}")
            return None

    def get_feedback_stats(self):
        try:
            event_counts = {}
            for feedback in self.feedback_history:
                event_type = feedback['event_type']
                event_counts[event_type] = event_counts.get(event_type, 0) + 1
            return {
                'total_feedback': len(self.feedback_history),
                'event_counts': event_counts,
                'recent_feedback': self.feedback_history[-5:] if self.feedback_history else [],
            }
        except (KeyError, TypeError, AttributeError):
            return {'total_feedback': 0, 'event_counts': {}, 'recent_feedback': []}


class Autotune:
    """Automatic hyperparameter tuning system."""

    def __init__(self, brain):
        self.brain = brain
        self.tuning_history = []
        self.parameter_ranges = {
            'creativity_factor': (0.1, 1.0),
            'context_weight': (0.1, 0.8),
            'semantic_weight': (0.1, 0.8),
            'pattern_weight': (0.1, 0.8),
        }

    def tune_from_quality(self, confidence_score):
        try:
            if not isinstance(confidence_score, (int, float)) or confidence_score < 0 or confidence_score > 1:
                return
            if confidence_score < 0.6:
                self._adjust_parameters('increase')
            elif confidence_score > 0.8:
                self._adjust_parameters('maintain')
            else:
                self._adjust_parameters('slight')
            self.tuning_history.append({
                'confidence_score': confidence_score,
                'adjustment': 'increase' if confidence_score < 0.6 else 'maintain' if confidence_score > 0.8 else 'slight',
                'timestamp': time.time(),
            })
            if len(self.tuning_history) > 100:
                self.tuning_history = self.tuning_history[-100:]
        except Exception as e:
            print(f"Autotune error: {e}")

    def _adjust_parameters(self, adjustment_type):
        try:
            current_settings = self.brain.settings_manager.settings
            if adjustment_type == 'increase':
                self.brain.settings_manager.set('creativity_factor', min(1.0, current_settings.get('creativity_factor', 0.7) + 0.05))
                self.brain.settings_manager.set('context_weight', max(0.1, current_settings.get('context_weight', 0.3) - 0.02))
            elif adjustment_type == 'slight':
                self.brain.settings_manager.set('creativity_factor', min(1.0, current_settings.get('creativity_factor', 0.7) + 0.02))
                self.brain.settings_manager.set('semantic_weight', min(0.8, current_settings.get('semantic_weight', 0.4) + 0.01))
            self.brain.settings_manager.save_settings()
        except Exception as e:
            print(f"Parameter adjustment error: {e}")

    def get_tuning_stats(self):
        try:
            if not self.tuning_history:
                return {'total_adjustments': 0, 'recent_adjustments': []}
            recent_adjustments = self.tuning_history[-10:]
            adjustment_counts = {}
            for tuning in self.tuning_history:
                adjustment_counts[tuning['adjustment']] = adjustment_counts.get(tuning['adjustment'], 0) + 1
            return {
                'total_adjustments': len(self.tuning_history),
                'adjustment_counts': adjustment_counts,
                'recent_adjustments': recent_adjustments,
            }
        except (KeyError, TypeError, AttributeError):
            return {'total_adjustments': 0, 'recent_adjustments': []}


class ResponseLearningSystem:
    """Learns from human responses to improve conversation patterns."""

    def __init__(self, brain):
        self.brain = brain
        self.response_chains = {}
        self.response_patterns = {}
        self.sentiment_learning = {}
        self.conversation_flow = []

    def learn_from_response(self, user_input, mai_response, human_response):
        try:
            if not user_input or not mai_response or not human_response:
                return
            chain_key = f"{user_input.lower().strip()} -> {mai_response.lower().strip()}"
            if chain_key not in self.response_chains:
                self.response_chains[chain_key] = []
            self.response_chains[chain_key].append({
                'human_response': human_response,
                'timestamp': time.time(),
                'context': self._extract_context(user_input, mai_response),
            })
            self._learn_response_patterns(mai_response, human_response)
            self._learn_sentiment_context(user_input, mai_response, human_response)
            self._update_conversation_flow(user_input, mai_response, human_response)
            if len(self.response_chains[chain_key]) > 10:
                self.response_chains[chain_key] = self.response_chains[chain_key][-10:]
        except Exception as e:
            print(f"Response learning error: {e}")

    def _extract_context(self, user_input, mai_response):
        try:
            return {
                'user_length': len(user_input.split()),
                'mai_length': len(mai_response.split()),
                'user_sentiment': self._analyze_sentiment(user_input),
                'mai_sentiment': self._analyze_sentiment(mai_response),
                'topics': self._extract_topics(user_input + " " + mai_response),
            }
        except (KeyError, TypeError, AttributeError):
            return {}

    def _learn_response_patterns(self, mai_response, human_response):
        try:
            mai_words = mai_response.lower().split()
            human_words = human_response.lower().split()
            for mai_word in mai_words:
                if mai_word not in self.response_patterns:
                    self.response_patterns[mai_word] = {}
                for human_word in human_words:
                    self.response_patterns[mai_word][human_word] = self.response_patterns[mai_word].get(human_word, 0) + 1
        except Exception as e:
            print(f"Pattern learning error: {e}")

    def _learn_sentiment_context(self, user_input, mai_response, human_response):
        try:
            sentiment = self._analyze_sentiment(human_response)
            context_key = f"{self._analyze_sentiment(user_input)}_{self._analyze_sentiment(mai_response)}"
            if context_key not in self.sentiment_learning:
                self.sentiment_learning[context_key] = []
            self.sentiment_learning[context_key].append(sentiment)
            if len(self.sentiment_learning[context_key]) > 20:
                self.sentiment_learning[context_key] = self.sentiment_learning[context_key][-20:]
        except Exception as e:
            print(f"Sentiment learning error: {e}")

    def _update_conversation_flow(self, user_input, mai_response, human_response):
        try:
            self.conversation_flow.append({'user': user_input, 'mai': mai_response, 'human': human_response, 'timestamp': time.time()})
            if len(self.conversation_flow) > 50:
                self.conversation_flow = self.conversation_flow[-50:]
        except Exception as e:
            print(f"Conversation flow update error: {e}")

    def _analyze_sentiment(self, text):
        try:
            positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'like', 'happy', 'pleased']
            negative_words = ['bad', 'terrible', 'awful', 'hate', 'dislike', 'angry', 'sad', 'disappointed', 'frustrated']
            words = text.lower().split()
            positive_count = sum(1 for word in words if word in positive_words)
            negative_count = sum(1 for word in words if word in negative_words)
            if positive_count > negative_count:
                return 'positive'
            if negative_count > positive_count:
                return 'negative'
            return 'neutral'
        except (KeyError, TypeError, AttributeError):
            return 'neutral'

    def _extract_topics(self, text):
        try:
            words = text.lower().split()
            word_freq = {}
            for word in words:
                if len(word) > 3:
                    word_freq[word] = word_freq.get(word, 0) + 1
            return sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:3]
        except (TypeError, AttributeError):
            return []

    def get_response_suggestions(self, user_input, mai_response):
        try:
            suggestions = []
            for chain_key, responses in self.response_chains.items():
                if user_input.lower().strip() in chain_key:
                    for response_data in responses[-3:]:
                        suggestions.append({'response': response_data['human_response'], 'confidence': 0.7, 'source': 'response_chain'})
            for mai_word in mai_response.lower().split():
                if mai_word in self.response_patterns:
                    for human_word, count in sorted(self.response_patterns[mai_word].items(), key=lambda x: x[1], reverse=True)[:3]:
                        suggestions.append({'response': f"Response involving '{human_word}'", 'confidence': min(count / 10, 1.0), 'source': 'pattern'})
            return suggestions[:5]
        except Exception as e:
            print(f"Response suggestion error: {e}")
            return []

    def get_learning_stats(self):
        try:
            return {
                'response_chains': len(self.response_chains),
                'response_patterns': len(self.response_patterns),
                'sentiment_contexts': len(self.sentiment_learning),
                'conversation_flow_length': len(self.conversation_flow),
                'total_learned_responses': sum(len(responses) for responses in self.response_chains.values()),
            }
        except (KeyError, TypeError, AttributeError):
            return {'response_chains': 0, 'response_patterns': 0, 'sentiment_contexts': 0, 'conversation_flow_length': 0, 'total_learned_responses': 0}


class TruthFactTable:
    """Stores and manages factual information about topics."""

    def __init__(self, brain):
        self.brain = brain
        self.fact_table = {}
        self.fact_sources = {}
        self.fact_confidence = {}
        self.topic_frequency = {}

    def add_fact(self, topic, fact, source="conversation", confidence=0.5):
        try:
            topic = topic.lower().strip()
            fact = fact.strip()
            if not topic or not fact:
                return
            if topic not in self.fact_table:
                self.fact_table[topic] = []
            if fact not in self.fact_table[topic]:
                self.fact_table[topic].append(fact)
                self.fact_sources[fact] = source
                self.fact_confidence[fact] = confidence
                self.topic_frequency[topic] = self.topic_frequency.get(topic, 0) + 1
        except Exception as e:
            print(f"Fact addition error: {e}")

    def get_facts(self, topic, limit=3):
        try:
            topic = topic.lower().strip()
            if topic not in self.fact_table:
                return []
            facts = []
            for fact in self.fact_table[topic]:
                facts.append({'fact': fact, 'confidence': self.fact_confidence.get(fact, 0.5), 'source': self.fact_sources.get(fact, 'unknown')})
            facts.sort(key=lambda x: x['confidence'], reverse=True)
            return facts[:limit]
        except Exception as e:
            print(f"Fact retrieval error: {e}")
            return []

    def update_fact_confidence(self, fact, feedback_score):
        try:
            if fact in self.fact_confidence:
                self.fact_confidence[fact] = max(0.1, min(1.0, (self.fact_confidence[fact] + feedback_score) / 2))
        except Exception as e:
            print(f"Confidence update error: {e}")

    def get_topic_frequency(self, topic):
        try:
            return self.topic_frequency.get(topic.lower().strip(), 0)
        except (AttributeError, TypeError):
            return 0

    def get_fact_stats(self):
        try:
            total_facts = sum(len(facts) for facts in self.fact_table.values())
            total_topics = len(self.fact_table)
            avg_confidence = sum(self.fact_confidence.values()) / len(self.fact_confidence) if self.fact_confidence else 0
            return {
                'total_facts': total_facts,
                'total_topics': total_topics,
                'average_confidence': avg_confidence,
                'most_frequent_topic': max(self.topic_frequency.items(), key=lambda x: x[1]) if self.topic_frequency else None,
            }
        except (KeyError, TypeError, AttributeError):
            return {'total_facts': 0, 'total_topics': 0, 'average_confidence': 0, 'most_frequent_topic': None}


class TopicDetectionSystem:
    """Detects when topics need factual context."""

    def __init__(self, brain):
        self.brain = brain
        self.topic_threshold = 0.1
        self.rare_topic_threshold = 0.05
        self.topic_contexts = {}

    def detect_topics_needing_facts(self, text):
        try:
            words = text.lower().split()
            word_freq = {}
            for word in words:
                if len(word) > 3:
                    word_freq[word] = word_freq.get(word, 0) + 1
            total_words = len(words) or 1
            topics_needing_facts = []
            for word, count in word_freq.items():
                frequency = count / total_words
                if frequency < self.rare_topic_threshold:
                    topics_needing_facts.append({'topic': word, 'frequency': frequency, 'reason': 'rare_topic', 'confidence': 0.8})
                elif frequency < self.topic_threshold:
                    topics_needing_facts.append({'topic': word, 'frequency': frequency, 'reason': 'medium_topic', 'confidence': 0.6})
            return topics_needing_facts
        except Exception as e:
            print(f"Topic detection error: {e}")
            return []

    def should_provide_facts(self, topic, context):
        try:
            if topic in self.topic_contexts and self.topic_contexts[topic].get('frequency', 1) < self.rare_topic_threshold:
                return True
            if 'user_input' in context and 'mai_response' in context:
                user_words = set((context.get('user_input') or '').lower().split())
                mai_words = set((context.get('mai_response') or '').lower().split())
                if topic in user_words and topic not in mai_words:
                    return True
            return False
        except Exception as e:
            print(f"Fact provision decision error: {e}")
            return False

    def update_topic_context(self, topic, context_data):
        try:
            self.topic_contexts[topic] = {
                'frequency': context_data.get('frequency', 0),
                'last_seen': time.time(),
                'context_type': context_data.get('context_type', 'general'),
            }
        except Exception as e:
            print(f"Topic context update error: {e}")

    def get_topic_stats(self):
        try:
            return {
                'total_topics_tracked': len(self.topic_contexts),
                'rare_topics': sum(1 for ctx in self.topic_contexts.values() if ctx.get('frequency', 0) < self.rare_topic_threshold),
                'medium_topics': sum(1 for ctx in self.topic_contexts.values() if self.rare_topic_threshold <= ctx.get('frequency', 0) < self.topic_threshold),
            }
        except (KeyError, TypeError, AttributeError):
            return {'total_topics_tracked': 0, 'rare_topics': 0, 'medium_topics': 0}


__all__ = [
    'AdaptiveLearningSystem',
    'AdvancedReasoningEngine',
    'AntiLoopFilter',
    'Autotune',
    'ConfidenceGate',
    'CreativeResponseGenerator',
    'Critic',
    'Curiosity',
    'EnhancedSemanticMemory',
    'EnvironmentFeedback',
    'MetaMemory',
    'ResponseLearningSystem',
    'TopicDetectionSystem',
    'TruthFactTable',
]
