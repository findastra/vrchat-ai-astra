"""Sparse, persistent cognitive control for Mai.

This module intentionally uses ordinary scalars, dictionaries, relations, and
SQLite.  It does not generate language itself.  It appraises a conversation
event, maintains bounded homeostatic signals, chooses a goal and response
strategy, and learns which strategies work in which contexts.
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
from collections import OrderedDict


def _clamp(value, low=0.02, high=0.98):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = (low + high) / 2.0
    if not math.isfinite(numeric):
        numeric = (low + high) / 2.0
    return max(low, min(high, numeric))


class NornCognitiveCore:
    """A small homeostatic controller and context-conditioned action learner."""

    VERSION = 1
    EVENT_LIMIT = 1200
    ACTION_VALUE_LIMIT = 10000
    RELATION_LIMIT = 25000
    STATE_DEFAULTS = OrderedDict((
        ('curiosity', 0.58),
        ('confidence', 0.50),
        ('arousal', 0.32),
        ('affiliation', 0.56),
        ('frustration', 0.10),
        ('satisfaction', 0.46),
        ('fatigue', 0.08),
        ('novelty_need', 0.44),
    ))
    BASELINES = dict(STATE_DEFAULTS)
    ACTION_DIRECTIVES = {
        'answer_directly': 'give a direct, relevant answer before elaborating',
        'explain_causally': 'explain the causal chain in concrete steps',
        'ask_clarifying': 'ask one targeted question that reduces uncertainty',
        'recall_episode': 'retrieve the most relevant earlier episode and distinguish memory from inference',
        'repair_alignment': 'acknowledge the mismatch and change response strategy',
        'create_connection': 'combine relevant concepts into a testable new connection',
        'explore_hypothesis': 'offer a bounded hypothesis and mark uncertainty',
        'acknowledge_socially': 'respond socially and succinctly while preserving continuity',
    }
    STOPWORDS = {
        'a', 'about', 'again', 'an', 'and', 'answer', 'are', 'as', 'at', 'be', 'been', 'but', 'by', 'can',
        'could', 'did', 'do', 'does', 'for', 'from', 'had', 'has', 'have', 'he',
        'her', 'him', 'his', 'how', 'i', 'if', 'in', 'is', 'it', 'its', 'me',
        'my', 'of', 'on', 'or', 'our', 'she', 'so', 'that', 'the', 'their',
        'them', 'they', 'this', 'to', 'was', 'we', 'were', 'what', 'when',
        'where', 'which', 'who', 'why', 'will', 'with', 'would', 'you', 'your',
        'analyze', 'better', 'build', 'compare', 'create', 'design', 'evaluate',
        'explain', 'idea', 'invent', 'make', 'new', 'novel', 'please', 'repeated',
        'same', 'small', 'tell', 'wrong',
    }

    def __init__(self, connection, *, event_limit=None):
        self.con = connection
        self.event_limit = max(100, int(event_limit or self.EVENT_LIMIT))
        self._lock = threading.RLock()
        self.state = dict(self.STATE_DEFAULTS)
        self.active_turn = None
        self.last_completed_turn = None
        self.last_trace = {}
        self._setup_schema()
        self._load_state()

    def _setup_schema(self):
        with self._lock:
            cur = self.con.cursor()
            cur.executescript("""
                CREATE TABLE IF NOT EXISTS cognitive_state (
                    state_key TEXT PRIMARY KEY,
                    state_value REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS cognitive_action_values (
                    context_key TEXT NOT NULL,
                    action_key TEXT NOT NULL,
                    value REAL NOT NULL DEFAULT 0.5,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    positive_outcomes INTEGER NOT NULL DEFAULT 0,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (context_key, action_key)
                );
                CREATE TABLE IF NOT EXISTS cognitive_goals (
                    goal_key TEXT PRIMARY KEY,
                    description TEXT NOT NULL,
                    priority REAL NOT NULL,
                    progress REAL NOT NULL DEFAULT 0.5,
                    activations INTEGER NOT NULL DEFAULT 0,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS cognitive_relations (
                    source TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    target TEXT NOT NULL,
                    strength REAL NOT NULL DEFAULT 0.5,
                    evidence INTEGER NOT NULL DEFAULT 0,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (source, relation, target)
                );
                CREATE TABLE IF NOT EXISTS cognitive_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at REAL NOT NULL,
                    event_type TEXT NOT NULL,
                    context_key TEXT NOT NULL,
                    goal_key TEXT,
                    action_key TEXT,
                    reward REAL,
                    appraisal_json TEXT NOT NULL,
                    state_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_cognitive_events_created
                    ON cognitive_events(created_at);
                CREATE INDEX IF NOT EXISTS idx_cognitive_relations_source
                    ON cognitive_relations(source, relation);
            """)
            self.con.commit()

    def _load_state(self):
        with self._lock:
            rows = self.con.execute(
                'SELECT state_key, state_value FROM cognitive_state'
            ).fetchall()
        for key, value in rows:
            if key in self.state:
                self.state[key] = _clamp(value)

    @staticmethod
    def _tokens(text):
        return re.findall(r"[a-z0-9']+", str(text or '').lower())

    def _topics(self, text):
        topics = []
        for token in self._tokens(text):
            if len(token) < 3 or token in self.STOPWORDS or token.isdigit():
                continue
            if token not in topics:
                topics.append(token)
            if len(topics) >= 5:
                break
        return topics

    def _classify_intent(self, text):
        normalized = ' '.join(self._tokens(text))
        if re.search(r'\b(wrong|incorrect|broken|failed|again|repeating|same snippet|not what|fix that)\b', normalized):
            return 'correction'
        if re.search(r'\b(remember|recall|earlier|before|last time|what did)\b', normalized):
            return 'recall'
        if re.search(r'\b(create|invent|imagine|novel|idea|design|build|make)\b', normalized):
            return 'creative'
        if re.search(r'\b(explain|why|how does|how do|reason|cause|meaning)\b', normalized):
            return 'explanation'
        if re.search(r'\b(analyze|compare|evaluate|diagnose|inspect|test)\b', normalized):
            return 'analysis'
        if re.search(r'\b(hello|hi|hey|thanks|thank you|good morning|good night)\b', normalized):
            return 'social'
        if '?' in str(text or '') or re.match(r'^(what|who|where|when|which|can|could|is|are|do|does)\b', normalized):
            return 'question'
        return 'statement'

    def _appraise(self, text, memory_salience=0.0, recent_quality=None):
        tokens = self._tokens(text)
        topics = self._topics(text)
        normalized = ' '.join(tokens)
        intent = self._classify_intent(text)
        positive_words = {'good', 'great', 'love', 'like', 'helpful', 'correct', 'yes', 'thanks'}
        negative_words = {'bad', 'hate', 'wrong', 'broken', 'slow', 'failed', 'frustrated', 'no'}
        positive = min(1.0, sum(token in positive_words for token in tokens) / 2.0)
        negative = min(1.0, sum(token in negative_words for token in tokens) / 2.0)
        correction = 1.0 if intent == 'correction' else 0.0
        question = 1.0 if intent in {'question', 'explanation', 'analysis'} else 0.0
        creative = 1.0 if intent == 'creative' else 0.0
        recall = 1.0 if intent == 'recall' else 0.0
        social = 1.0 if intent == 'social' else 0.0
        complexity = _clamp(len(tokens) / 40.0, 0.0, 1.0)
        pronoun_only = bool(tokens) and not topics and any(token in {'it', 'that', 'this', 'they'} for token in tokens)
        ambiguity = 0.75 if len(tokens) <= 2 else (0.70 if pronoun_only else 0.10)
        if intent in {'question', 'explanation', 'analysis'} and topics and not pronoun_only:
            ambiguity *= 0.55
        quality_signal = 0.5 if recent_quality is None else _clamp(recent_quality, 0.0, 1.0)
        uncertainty = _clamp(
            (ambiguity * 0.48) + (complexity * 0.22) + ((1.0 - quality_signal) * 0.18)
            + ((1.0 - _clamp(memory_salience, 0.0, 1.0)) * 0.12),
            0.0,
            1.0,
        )
        return {
            'intent': intent,
            'topics': topics,
            'positive': positive,
            'negative': negative,
            'correction': correction,
            'question': question,
            'creative': creative,
            'recall': recall,
            'social': social,
            'complexity': complexity,
            'ambiguity': ambiguity,
            'uncertainty': uncertainty,
            'memory_salience': _clamp(memory_salience, 0.0, 1.0),
            'novelty': 1.0,
            'normalized_preview': normalized[:160],
        }

    def _context_key(self, appraisal):
        topic_key = '_'.join(appraisal.get('topics', [])[:3]) or 'general'
        return f"{appraisal.get('intent', 'statement')}|{topic_key}"[:160]

    def _drift_and_appraise_state(self, appraisal):
        for key, baseline in self.BASELINES.items():
            self.state[key] = _clamp(self.state.get(key, baseline) + ((baseline - self.state.get(key, baseline)) * 0.035))
        self.state['curiosity'] = _clamp(
            self.state['curiosity'] + (appraisal['question'] * 0.035)
            + (appraisal['creative'] * 0.055) + (appraisal['novelty'] * 0.018)
            - (self.state['fatigue'] * 0.018)
        )
        self.state['arousal'] = _clamp(
            self.state['arousal'] + (appraisal['correction'] * 0.075)
            + (appraisal['question'] * 0.018) - 0.018
        )
        self.state['confidence'] = _clamp(
            self.state['confidence'] + (appraisal['memory_salience'] * 0.035)
            - (appraisal['uncertainty'] * 0.045)
        )
        self.state['affiliation'] = _clamp(
            self.state['affiliation'] + (appraisal['positive'] * 0.035)
            - (appraisal['negative'] * 0.012)
        )
        self.state['frustration'] = _clamp(
            self.state['frustration'] + (appraisal['correction'] * 0.035)
            + (appraisal['negative'] * 0.018) - (appraisal['positive'] * 0.025)
        )
        self.state['fatigue'] = _clamp(
            self.state['fatigue'] + 0.004 + (appraisal['complexity'] * 0.006)
        )
        self.state['novelty_need'] = _clamp(
            self.state['novelty_need'] + (0.012 if not appraisal['creative'] else -0.035)
        )

    def _goal_scores(self, appraisal):
        s = self.state
        return {
            'answer_usefully': 0.46 + (appraisal['question'] * 0.28) + (s['affiliation'] * 0.12),
            'resolve_uncertainty': 0.18 + (appraisal['uncertainty'] * 0.62)
            + (appraisal['ambiguity'] * 0.34) + (s['curiosity'] * 0.12),
            'restore_alignment': 0.08 + (appraisal['correction'] * 0.82) + (s['frustration'] * 0.10),
            'continue_shared_context': 0.12 + (appraisal['recall'] * 0.58) + (appraisal['memory_salience'] * 0.28),
            'discover_relation': 0.14 + (appraisal['question'] * 0.20) + (s['curiosity'] * 0.30),
            'create_novel_idea': 0.10 + (appraisal['creative'] * 0.70) + (s['novelty_need'] * 0.18),
            'maintain_connection': 0.12 + (appraisal['social'] * 0.68) + (s['affiliation'] * 0.16),
        }

    def _learned_goal_values(self):
        """Return persisted goal progress/priority so selection can learn over time."""
        with self._lock:
            rows = self.con.execute(
                'SELECT goal_key, priority, progress, activations FROM cognitive_goals'
            ).fetchall()
        learned = {}
        for goal_key, priority, progress, activations in rows:
            learned[str(goal_key)] = {
                'priority': _clamp(priority, 0.0, 1.0),
                'progress': _clamp(progress, 0.0, 1.0),
                'activations': max(0, int(activations or 0)),
            }
        return learned

    def _base_action_scores(self, appraisal):
        s = self.state
        intent = appraisal['intent']
        return {
            'answer_directly': 0.34 + (appraisal['question'] * 0.34) + (s['confidence'] * 0.15) - (appraisal['ambiguity'] * 0.20),
            'explain_causally': 0.22 + (0.55 if intent in {'explanation', 'analysis'} else 0.0) + (s['curiosity'] * 0.12),
            'ask_clarifying': 0.10 + (appraisal['ambiguity'] * 0.66) + (appraisal['uncertainty'] * 0.20),
            'recall_episode': 0.12 + (appraisal['recall'] * 0.68) + (appraisal['memory_salience'] * 0.20),
            'repair_alignment': 0.08 + (appraisal['correction'] * 0.82) + (s['frustration'] * 0.08),
            'create_connection': 0.12 + (appraisal['creative'] * 0.68) + (s['curiosity'] * 0.10) + (s['novelty_need'] * 0.10),
            'explore_hypothesis': 0.16 + (0.24 if intent in {'analysis', 'explanation'} else 0.0) + (s['curiosity'] * 0.22),
            'acknowledge_socially': 0.10 + (appraisal['social'] * 0.72) + (s['affiliation'] * 0.08),
        }

    def _learned_action_values(self, context_key, topics):
        with self._lock:
            rows = self.con.execute(
                'SELECT action_key, value, attempts FROM cognitive_action_values WHERE context_key=?',
                (context_key,),
            ).fetchall()
            relation_rows = []
            if topics:
                placeholders = ','.join('?' for _ in topics)
                relation_rows = self.con.execute(
                    f"SELECT target, AVG(strength) FROM cognitive_relations "
                    f"WHERE relation='effective_action' AND source IN ({placeholders}) GROUP BY target",
                    tuple(topics),
                ).fetchall()
        values = {action: (float(value), int(attempts)) for action, value, attempts in rows}
        relations = {action: float(value) for action, value in relation_rows}
        return values, relations

    def begin_turn(self, user_text, *, memory_salience=0.0, recent_quality=None):
        """Appraise an event and select a goal/action without writing to disk."""
        appraisal = self._appraise(user_text, memory_salience, recent_quality)
        context_key = self._context_key(appraisal)
        learned, relations = self._learned_action_values(context_key, appraisal['topics'])
        total_attempts = sum(attempts for _, attempts in learned.values())
        appraisal['novelty'] = 1.0 / math.sqrt(1.0 + total_attempts)
        self._drift_and_appraise_state(appraisal)

        goal_scores = self._goal_scores(appraisal)
        learned_goals = self._learned_goal_values()
        for candidate_goal, base in list(goal_scores.items()):
            goal_memory = learned_goals.get(candidate_goal)
            if not goal_memory:
                continue
            # Prefer goals that historically delivered progress, without drowning the appraisal.
            progress_bias = (goal_memory['progress'] - 0.5) * 0.18
            priority_bias = (goal_memory['priority'] - 0.5) * 0.10
            exploration = 0.03 / math.sqrt(1.0 + goal_memory['activations'])
            goal_scores[candidate_goal] = base + progress_bias + priority_bias + exploration

        # Multi-turn continuity: keep pursuing an unfinished prior goal unless a strong interrupt arrives.
        continuity_goal = None
        prior = self.last_completed_turn if isinstance(self.last_completed_turn, dict) else None
        if prior and prior.get('goal') in goal_scores:
            interrupt = bool(appraisal.get('correction') or appraisal.get('social') or appraisal.get('creative'))
            prior_progress = float((learned_goals.get(prior['goal']) or {}).get('progress', prior.get('reward', 0.5)) or 0.5)
            if not interrupt and prior_progress < 0.78:
                continuity_goal = prior['goal']
                goal_scores[continuity_goal] = goal_scores[continuity_goal] + 0.16 + ((0.78 - prior_progress) * 0.12)

        goal_key = max(goal_scores, key=goal_scores.get)
        base_scores = self._base_action_scores(appraisal)
        action_scores = {}
        for action, base in base_scores.items():
            learned_value, attempts = learned.get(action, (0.5, 0))
            relation_value = relations.get(action, 0.5)
            exploration = (self.state['curiosity'] * 0.045) / math.sqrt(1.0 + attempts)
            action_scores[action] = base + (learned_value * 0.22) + (relation_value * 0.10) + exploration
        action_key = max(action_scores, key=action_scores.get)

        reasons = [f"intent={appraisal['intent']}"]
        if appraisal['uncertainty'] >= 0.55:
            reasons.append('uncertainty is elevated')
        if appraisal['correction']:
            reasons.append('the user signaled a mismatch')
        if appraisal['memory_salience'] >= 0.45:
            reasons.append('related experience is available')
        if appraisal['creative']:
            reasons.append('the request rewards novelty')
        if learned_goals.get(goal_key, {}).get('activations', 0) > 0:
            reasons.append('prior goal progress influenced selection')
        if continuity_goal and goal_key == continuity_goal:
            reasons.append('continuing an unfinished prior goal')

        turn = {
            'version': self.VERSION,
            'started_at': time.time(),
            'context_key': context_key,
            'goal': goal_key,
            'goal_priority': _clamp(goal_scores[goal_key], 0.0, 1.0),
            'goal_learning': learned_goals.get(goal_key),
            'goal_continuity': bool(continuity_goal and goal_key == continuity_goal),
            'action': action_key,
            'action_score': float(action_scores[action_key]),
            'learned_value': float(learned.get(action_key, (0.5, 0))[0]),
            'appraisal': appraisal,
            'directive': self.ACTION_DIRECTIVES[action_key],
            'reasons': reasons,
            'state_before_outcome': dict(self.state),
        }
        self.active_turn = turn
        self.last_trace = dict(turn)
        return self.get_last_trace()

    @staticmethod
    def _goal_description(goal_key):
        return {
            'answer_usefully': 'Produce an answer that improves the user\'s situation.',
            'resolve_uncertainty': 'Reduce uncertainty before making a strong claim.',
            'restore_alignment': 'Repair a mismatch between the user\'s intent and Mai\'s behavior.',
            'continue_shared_context': 'Use relevant shared experience without pretending to remember more than is stored.',
            'discover_relation': 'Find a useful relation among the active concepts.',
            'create_novel_idea': 'Construct a novel but testable connection.',
            'maintain_connection': 'Maintain conversational continuity and responsiveness.',
        }.get(goal_key, goal_key.replace('_', ' '))

    def _persist_state(self, cur, now):
        cur.executemany(
            """INSERT INTO cognitive_state(state_key, state_value, updated_at)
               VALUES (?, ?, ?)
               ON CONFLICT(state_key) DO UPDATE SET
                   state_value=excluded.state_value, updated_at=excluded.updated_at""",
            [(key, float(value), now) for key, value in self.state.items()],
        )

    def _update_action_value(self, cur, turn, reward, now, learning_rate, *, count_attempt=True):
        context_key = turn['context_key']
        action_key = turn['action']
        row = cur.execute(
            'SELECT value, attempts, positive_outcomes FROM cognitive_action_values WHERE context_key=? AND action_key=?',
            (context_key, action_key),
        ).fetchone()
        old_value, attempts, positives = (0.5, 0, 0) if row is None else (float(row[0]), int(row[1]), int(row[2]))
        new_value = _clamp(old_value + (learning_rate * (reward - old_value)), 0.0, 1.0)
        next_attempts = attempts + int(bool(count_attempt))
        next_positives = positives + (int(reward >= 0.68) if count_attempt else 0)
        cur.execute(
            """INSERT INTO cognitive_action_values
                   (context_key, action_key, value, attempts, positive_outcomes, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(context_key, action_key) DO UPDATE SET
                   value=excluded.value, attempts=excluded.attempts,
                   positive_outcomes=excluded.positive_outcomes, updated_at=excluded.updated_at""",
            (context_key, action_key, new_value, next_attempts, next_positives, now),
        )
        return new_value

    def _update_relations(self, cur, turn, reward, now, learning_rate):
        for topic in turn.get('appraisal', {}).get('topics', [])[:5]:
            cur.execute(
                """INSERT INTO cognitive_relations(source, relation, target, strength, evidence, updated_at)
                   VALUES (?, 'effective_action', ?, ?, 1, ?)
                   ON CONFLICT(source, relation, target) DO UPDATE SET
                       strength=cognitive_relations.strength + ? * (excluded.strength - cognitive_relations.strength),
                       evidence=cognitive_relations.evidence + 1,
                       updated_at=excluded.updated_at""",
                (topic, turn['action'], reward, now, learning_rate),
            )

    def _update_goal(self, cur, turn, reward, now, learning_rate):
        cur.execute(
            """INSERT INTO cognitive_goals
                   (goal_key, description, priority, progress, activations, updated_at)
               VALUES (?, ?, ?, ?, 1, ?)
               ON CONFLICT(goal_key) DO UPDATE SET
                   description=excluded.description,
                   priority=excluded.priority,
                   progress=cognitive_goals.progress + ? * (excluded.progress - cognitive_goals.progress),
                   activations=cognitive_goals.activations + 1,
                   updated_at=excluded.updated_at""",
            (
                turn['goal'], self._goal_description(turn['goal']), turn['goal_priority'],
                reward, now, learning_rate,
            ),
        )

    def _trim_events(self, cur):
        count = cur.execute('SELECT COUNT(*) FROM cognitive_events').fetchone()[0]
        excess = int(count) - self.event_limit
        if excess > 0:
            cur.execute(
                'DELETE FROM cognitive_events WHERE event_id IN '
                '(SELECT event_id FROM cognitive_events ORDER BY event_id ASC LIMIT ?)',
                (excess,),
            )
        action_count = cur.execute('SELECT COUNT(*) FROM cognitive_action_values').fetchone()[0]
        action_excess = int(action_count) - self.ACTION_VALUE_LIMIT
        if action_excess > 0:
            cur.execute(
                'DELETE FROM cognitive_action_values WHERE rowid IN '
                '(SELECT rowid FROM cognitive_action_values ORDER BY updated_at ASC LIMIT ?)',
                (action_excess,),
            )
        relation_count = cur.execute('SELECT COUNT(*) FROM cognitive_relations').fetchone()[0]
        relation_excess = int(relation_count) - self.RELATION_LIMIT
        if relation_excess > 0:
            cur.execute(
                'DELETE FROM cognitive_relations WHERE rowid IN '
                '(SELECT rowid FROM cognitive_relations ORDER BY evidence ASC, updated_at ASC LIMIT ?)',
                (relation_excess,),
            )

    def observe_response(self, *, quality, coherence=0.5, response_text=''):
        """Learn from the completed action using bounded internal evidence."""
        turn = self.active_turn
        if not turn:
            return self.get_snapshot()
        quality = _clamp(quality, 0.0, 1.0)
        coherence = _clamp(coherence, 0.0, 1.0)
        response_tokens = self._tokens(response_text)
        diversity = (len(set(response_tokens)) / len(response_tokens)) if response_tokens else 0.0
        reward = _clamp((quality * 0.62) + (coherence * 0.25) + (diversity * 0.13), 0.0, 1.0)

        delta = reward - 0.5
        self.state['satisfaction'] = _clamp(self.state['satisfaction'] + (delta * 0.11))
        self.state['confidence'] = _clamp(self.state['confidence'] + (delta * 0.075))
        self.state['frustration'] = _clamp(self.state['frustration'] - (delta * 0.065))
        self.state['fatigue'] = _clamp(self.state['fatigue'] + 0.006)
        now = time.time()
        learning_rate = 0.14

        with self._lock:
            cur = self.con.cursor()
            try:
                cur.execute('BEGIN')
                learned_value = self._update_action_value(cur, turn, reward, now, learning_rate)
                self._update_relations(cur, turn, reward, now, learning_rate)
                self._update_goal(cur, turn, reward, now, learning_rate)
                self._persist_state(cur, now)
                cur.execute(
                    """INSERT INTO cognitive_events
                           (created_at, event_type, context_key, goal_key, action_key, reward, appraisal_json, state_json)
                       VALUES (?, 'response_outcome', ?, ?, ?, ?, ?, ?)""",
                    (
                        now, turn['context_key'], turn['goal'], turn['action'], reward,
                        json.dumps(turn['appraisal'], separators=(',', ':'), sort_keys=True),
                        json.dumps(self.state, separators=(',', ':'), sort_keys=True),
                    ),
                )
                self._trim_events(cur)
                self.con.commit()
            except Exception:
                self.con.rollback()
                raise

        completed = dict(turn)
        completed.update({
            'reward': reward,
            'quality': quality,
            'coherence': coherence,
            'learned_value_after': learned_value,
            'state_after_outcome': dict(self.state),
        })
        self.last_completed_turn = completed
        self.last_trace = completed
        self.active_turn = None
        return self.get_snapshot()

    def apply_feedback(self, is_positive):
        """Apply explicit user feedback with more weight than self-evaluation."""
        turn = self.last_completed_turn
        if not turn:
            return self.get_snapshot()
        reward = 0.96 if is_positive else 0.04
        direction = 1.0 if is_positive else -1.0
        self.state['satisfaction'] = _clamp(self.state['satisfaction'] + (direction * 0.10))
        self.state['confidence'] = _clamp(self.state['confidence'] + (direction * 0.055))
        self.state['frustration'] = _clamp(self.state['frustration'] - (direction * 0.07))
        now = time.time()
        learning_rate = 0.34

        with self._lock:
            cur = self.con.cursor()
            try:
                cur.execute('BEGIN')
                learned_value = self._update_action_value(
                    cur, turn, reward, now, learning_rate, count_attempt=False
                )
                self._update_relations(cur, turn, reward, now, learning_rate)
                self._update_goal(cur, turn, reward, now, learning_rate)
                self._persist_state(cur, now)
                cur.execute(
                    """INSERT INTO cognitive_events
                           (created_at, event_type, context_key, goal_key, action_key, reward, appraisal_json, state_json)
                       VALUES (?, 'explicit_feedback', ?, ?, ?, ?, ?, ?)""",
                    (
                        now, turn['context_key'], turn['goal'], turn['action'], reward,
                        json.dumps({'positive': bool(is_positive)}, separators=(',', ':')),
                        json.dumps(self.state, separators=(',', ':'), sort_keys=True),
                    ),
                )
                self._trim_events(cur)
                self.con.commit()
            except Exception:
                self.con.rollback()
                raise
        turn['explicit_feedback'] = bool(is_positive)
        turn['learned_value_after_feedback'] = learned_value
        turn['state_after_feedback'] = dict(self.state)
        self.last_trace = dict(turn)
        return self.get_snapshot()

    def get_last_trace(self):
        return json.loads(json.dumps(self.last_trace)) if self.last_trace else {}

    def get_snapshot(self):
        with self._lock:
            action_rows = self.con.execute(
                'SELECT COUNT(*), COALESCE(SUM(attempts), 0) FROM cognitive_action_values'
            ).fetchone()
            relation_count = self.con.execute('SELECT COUNT(*) FROM cognitive_relations').fetchone()[0]
            event_count = self.con.execute('SELECT COUNT(*) FROM cognitive_events').fetchone()[0]
            goal_rows = self.con.execute(
                'SELECT goal_key, priority, progress, activations FROM cognitive_goals '
                'ORDER BY updated_at DESC LIMIT 5'
            ).fetchall()
        trace = self.get_last_trace()
        return {
            'active': True,
            'version': self.VERSION,
            'state': {key: round(float(value), 4) for key, value in self.state.items()},
            'current_goal': trace.get('goal'),
            'current_action': trace.get('action'),
            'directive': trace.get('directive'),
            'last_reward': trace.get('reward'),
            'learned_contexts': int(action_rows[0] or 0),
            'action_observations': int(action_rows[1] or 0),
            'learned_relations': int(relation_count or 0),
            'stored_events': int(event_count or 0),
            'event_limit': self.event_limit,
            'action_value_limit': self.ACTION_VALUE_LIMIT,
            'relation_limit': self.RELATION_LIMIT,
            'recent_goals': [
                {
                    'goal': goal,
                    'priority': round(float(priority), 4),
                    'progress': round(float(progress), 4),
                    'activations': int(activations),
                }
                for goal, priority, progress, activations in goal_rows
            ],
        }
