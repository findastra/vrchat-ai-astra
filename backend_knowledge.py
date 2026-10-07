import re
import time
from collections import deque
from typing import Any


LEXICON_BOOTSTRAP = [
    ('is a', 'relation_trigger', 'is_a', 1.0, 'medium', 'Taxonomy relation trigger.'),
    ('is an', 'relation_trigger', 'is_a', 1.0, 'medium', 'Taxonomy relation trigger.'),
    ('is', 'relation_trigger', 'has_trait', 0.8, 'medium', 'General identity or trait relation trigger.'),
    ('has', 'relation_trigger', 'has_attribute', 0.9, 'medium', 'Attribute relation trigger.'),
    ('causes', 'relation_trigger', 'causes', 1.0, 'medium', 'Causal relation trigger.'),
    ('leads to', 'relation_trigger', 'causes', 0.95, 'medium', 'Causal relation trigger.'),
    ('part of', 'relation_trigger', 'part_of', 1.0, 'medium', 'Part-whole relation trigger.'),
    ('belongs to', 'relation_trigger', 'belongs_to', 0.95, 'medium', 'Ownership or membership relation trigger.'),
    ('works as', 'relation_trigger', 'role', 0.9, 'medium', 'Role relation trigger.'),
    ('serves as', 'relation_trigger', 'role', 0.9, 'medium', 'Role relation trigger.'),
    ('likes', 'relation_trigger', 'prefers', 0.9, 'medium', 'Preference relation trigger.'),
    ('prefers', 'relation_trigger', 'prefers', 1.0, 'medium', 'Preference relation trigger.'),
    ('dislikes', 'relation_trigger', 'avoids', 0.9, 'medium', 'Negative preference relation trigger.'),
    ('hates', 'relation_trigger', 'avoids', 0.85, 'medium', 'Negative preference relation trigger.'),
    ('can', 'relation_trigger', 'capable_of', 0.8, 'medium', 'Capability relation trigger.'),
    ('mai', 'entity_alias', 'mai', 1.0, 'medium', 'Mai self reference.'),
    ('phoenix', 'entity_alias', 'mai', 0.8, 'medium', 'Mai self reference.'),
    ('i', 'self_reference', 'speaker', 1.0, 'high', 'Speaker self reference.'),
    ('you', 'self_reference', 'listener', 1.0, 'high', 'Listener reference.'),
]

HEDGE_PREFIXES = (
    'i think that ',
    'i think ',
    'i believe that ',
    'i believe ',
    'i feel that ',
    'i feel ',
    'maybe ',
    'perhaps ',
)

RELATION_PATTERNS = [
    {
        'relation_type': 'causes',
        'regex': re.compile(r'^(?P<subject>.+?)\s+(?:causes|lead(?:s)?\s+to|creates|produces)\s+(?P<object>.+)$', re.IGNORECASE),
        'object_kind': 'concept',
        'confidence_bonus': 0.16,
    },
    {
        'relation_type': 'part_of',
        'regex': re.compile(r'^(?P<subject>.+?)\s+is\s+part\s+of\s+(?P<object>.+)$', re.IGNORECASE),
        'object_kind': 'concept',
        'confidence_bonus': 0.15,
    },
    {
        'relation_type': 'belongs_to',
        'regex': re.compile(r'^(?P<subject>.+?)\s+belongs\s+to\s+(?P<object>.+)$', re.IGNORECASE),
        'object_kind': 'concept',
        'confidence_bonus': 0.12,
    },
    {
        'relation_type': 'role',
        'regex': re.compile(r'^(?P<subject>.+?)\s+(?:works|serves)\s+as\s+(?P<object>.+)$', re.IGNORECASE),
        'object_kind': 'value',
        'confidence_bonus': 0.12,
    },
    {
        'relation_type': 'is_a',
        'regex': re.compile(r'^(?P<subject>.+?)\s+is\s+(?:a|an)\s+(?P<object>.+)$', re.IGNORECASE),
        'object_kind': 'concept',
        'confidence_bonus': 0.14,
    },
    {
        'relation_type': 'has_attribute',
        'regex': re.compile(r'^(?P<subject>.+?)\s+has\s+(?P<object>.+)$', re.IGNORECASE),
        'object_kind': 'value',
        'confidence_bonus': 0.08,
    },
    {
        'relation_type': 'prefers',
        'regex': re.compile(r'^(?P<subject>.+?)\s+(?:likes|loves|prefers|enjoys)\s+(?P<object>.+)$', re.IGNORECASE),
        'object_kind': 'concept',
        'confidence_bonus': 0.11,
    },
    {
        'relation_type': 'avoids',
        'regex': re.compile(r'^(?P<subject>.+?)\s+(?:dislikes|hates|avoids)\s+(?P<object>.+)$', re.IGNORECASE),
        'object_kind': 'concept',
        'confidence_bonus': 0.11,
    },
    {
        'relation_type': 'capable_of',
        'regex': re.compile(r'^(?P<subject>.+?)\s+can\s+(?P<object>.+)$', re.IGNORECASE),
        'object_kind': 'value',
        'confidence_bonus': 0.09,
    },
    {
        'relation_type': 'has_trait',
        'regex': re.compile(r'^(?P<subject>.+?)\s+is\s+(?P<object>.+)$', re.IGNORECASE),
        'object_kind': 'value',
        'confidence_bonus': 0.05,
    },
]

STOP_WORDS = {
    'the', 'a', 'an', 'this', 'that', 'these', 'those', 'my', 'your', 'our', 'their',
    'his', 'her', 'its',
}

RELATION_QUERY_HINTS = {
    'prefer': 'prefers',
    'prefers': 'prefers',
    'like': 'prefers',
    'likes': 'prefers',
    'love': 'prefers',
    'enjoy': 'prefers',
    'enjoys': 'prefers',
    'avoid': 'avoids',
    'avoids': 'avoids',
    'hate': 'avoids',
    'hates': 'avoids',
    'dislike': 'avoids',
    'dislikes': 'avoids',
    'part': 'part_of',
    'belongs': 'belongs_to',
    'belong': 'belongs_to',
    'role': 'role',
    'works': 'role',
    'serves': 'role',
    'can': 'capable_of',
    'cause': 'causes',
    'causes': 'causes',
    'because': 'causes',
    'trait': 'has_trait',
}

SOFT_CONFLICT_RELATIONS = {'prefers', 'avoids', 'belongs_to', 'role'}
INVERSE_RELATION_CONFLICTS = {
    'prefers': 'avoids',
    'avoids': 'prefers',
}

GRAPH_REASONING_RELATIONS = {
    'is_a',
    'has_trait',
    'has_attribute',
    'causes',
    'part_of',
    'belongs_to',
    'role',
    'prefers',
    'avoids',
    'capable_of',
}

GRAPH_REVERSE_RELATION_LABELS = {
    'is_a': 'includes',
    'has_trait': 'trait_of',
    'has_attribute': 'attribute_of',
    'causes': 'caused_by',
    'part_of': 'has_part',
    'belongs_to': 'has_member',
    'role': 'fulfilled_by',
    'prefers': 'preferred_by',
    'avoids': 'avoided_by',
    'capable_of': 'can_be_done_by',
}


class KnowledgeStore:
    """Persistent concept, fact, provenance, and identity layer for Mai."""

    def __init__(self, brain):
        self.brain = brain
        self.max_sentences_per_ingest = 24
        self.max_candidates_per_ingest = 16

    def ensure_schema(self) -> None:
        cur = getattr(self.brain, 'cur', None)
        con = getattr(self.brain, 'con', None)
        if cur is None or con is None:
            return

        cur.execute("""
            CREATE TABLE IF NOT EXISTS lexicon_entries (
                token_text TEXT PRIMARY KEY,
                token_type TEXT,
                canonical_form TEXT,
                confidence REAL DEFAULT 1.0,
                editable TEXT DEFAULT 'medium',
                notes TEXT DEFAULT ''
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS concept_nodes (
                concept_id INTEGER PRIMARY KEY AUTOINCREMENT,
                canonical_name TEXT UNIQUE,
                concept_type TEXT DEFAULT 'concept',
                confidence REAL DEFAULT 0.5,
                importance REAL DEFAULT 0.5,
                first_seen REAL DEFAULT 0,
                last_seen REAL DEFAULT 0,
                usage_count INTEGER DEFAULT 0
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS concept_aliases (
                alias_text TEXT PRIMARY KEY,
                concept_id INTEGER,
                alias_type TEXT DEFAULT 'surface',
                confidence REAL DEFAULT 0.5,
                first_seen REAL DEFAULT 0,
                last_seen REAL DEFAULT 0
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS fact_records (
                fact_key TEXT PRIMARY KEY,
                subject_concept_id INTEGER,
                relation_type TEXT,
                object_concept_id INTEGER,
                object_text TEXT,
                object_kind TEXT DEFAULT 'concept',
                confidence REAL DEFAULT 0.5,
                editable TEXT DEFAULT 'high',
                status TEXT DEFAULT 'active',
                support_count INTEGER DEFAULT 1,
                contradiction_count INTEGER DEFAULT 0,
                first_seen REAL DEFAULT 0,
                last_seen REAL DEFAULT 0,
                source_count INTEGER DEFAULT 1
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS fact_evidence (
                evidence_id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact_key TEXT,
                source_type TEXT,
                source_path TEXT DEFAULT '',
                source_label TEXT DEFAULT '',
                source_timestamp REAL DEFAULT 0,
                evidence_text TEXT,
                confidence REAL DEFAULT 0.5
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS identity_traits (
                trait_key TEXT PRIMARY KEY,
                relation_type TEXT,
                trait_value TEXT,
                confidence REAL DEFAULT 0.5,
                editable TEXT DEFAULT 'high',
                support_count INTEGER DEFAULT 1,
                contradiction_count INTEGER DEFAULT 0,
                first_seen REAL DEFAULT 0,
                last_seen REAL DEFAULT 0,
                source_count INTEGER DEFAULT 1
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_fact_subject ON fact_records(subject_concept_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_fact_relation ON fact_records(relation_type)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_fact_object ON fact_records(object_text)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_fact_evidence_fact ON fact_evidence(fact_key)")
        self._bootstrap_lexicon()
        con.commit()

    def ingest_conversation_turn(self, user_input: str, bot_response: str, quality_score: float | None = None) -> dict[str, Any]:
        base_quality = self._safe_float(quality_score, 0.5)
        total_candidates = 0
        total_inserted = 0
        consolidation_promotions = 0
        user_result = self.ingest_text(
            user_input,
            source_type='conversation_user',
            source_label='conversation_user',
            base_confidence=0.72,
            editable='high',
            speaker='user',
        )
        total_candidates += int(user_result.get('candidate_count', 0))
        total_inserted += int(user_result.get('fact_count', 0))

        should_ingest_bot = bool(bot_response and base_quality >= 0.44)
        low_info_check = getattr(self.brain, '_is_low_information_response', None)
        if should_ingest_bot and callable(low_info_check):
            try:
                should_ingest_bot = not bool(low_info_check(bot_response))
            except Exception:
                pass
        if should_ingest_bot:
            bot_confidence = min(0.85, max(0.35, 0.35 + (base_quality * 0.45)))
            bot_result = self.ingest_text(
                bot_response,
                source_type='conversation_bot',
                source_label='conversation_bot',
                base_confidence=bot_confidence,
                editable='high',
                speaker='mai',
            )
            total_candidates += int(bot_result.get('candidate_count', 0))
            total_inserted += int(bot_result.get('fact_count', 0))

        con = getattr(self.brain, 'con', None)
        try:
            consolidation_result = self.consolidate_recent_episodes()
            consolidation_promotions = int(consolidation_result.get('promoted_fact_count', 0)) + int(consolidation_result.get('preference_fact_count', 0))
        except Exception:
            consolidation_promotions = 0
        if con is not None and not getattr(self.brain, 'is_clone', False):
            try:
                con.commit()
            except Exception:
                pass

        return {
            'success': True,
            'candidate_count': total_candidates,
            'fact_count': total_inserted,
            'consolidation_promotions': consolidation_promotions,
        }

    def ingest_training_text(
        self,
        text: str,
        source_path: str = '',
        source_label: str = '',
        source_category: str = 'general_text',
        source_weight: float = 1.0,
    ) -> dict[str, Any]:
        editable = 'high' if source_category == 'working_notes' else 'medium'
        base_confidence = min(0.9, max(0.48, 0.52 + ((self._safe_float(source_weight, 1.0) - 1.0) * 0.4)))
        return self.ingest_text(
            text,
            source_type='training_text',
            source_path=source_path,
            source_label=source_label,
            base_confidence=base_confidence,
            editable=editable,
            speaker='external',
        )

    def ingest_text(
        self,
        text: str,
        source_type: str = 'text',
        source_path: str = '',
        source_label: str = '',
        base_confidence: float = 0.55,
        editable: str = 'high',
        speaker: str = 'external',
        max_sentences: int | None = None,
        max_candidates: int | None = None,
    ) -> dict[str, Any]:
        text = text if isinstance(text, str) else ''
        if not text.strip():
            return {'success': True, 'candidate_count': 0, 'fact_count': 0}

        sentence_limit = max_sentences if max_sentences is not None else self.max_sentences_per_ingest
        candidate_limit = max_candidates if max_candidates is not None else self.max_candidates_per_ingest
        sentences = self._split_sentences(text)[:max(1, int(sentence_limit or 1))]

        fact_count = 0
        candidate_count = 0
        for sentence in sentences:
            if candidate_count >= candidate_limit:
                break
            for candidate in self.extract_fact_candidates(sentence, speaker=speaker, base_confidence=base_confidence):
                candidate_count += 1
                if candidate_count > candidate_limit:
                    break
                self.add_fact(
                    candidate['subject'],
                    candidate['relation_type'],
                    candidate['object'],
                    object_kind=candidate['object_kind'],
                    concept_type=candidate.get('concept_type', 'concept'),
                    source_type=source_type,
                    source_path=source_path,
                    source_label=source_label,
                    confidence=candidate['confidence'],
                    editable=editable,
                    evidence_text=sentence,
                )
                fact_count += 1
        return {
            'success': True,
            'candidate_count': candidate_count,
            'fact_count': fact_count,
        }

    def extract_fact_candidates(self, sentence: str, speaker: str = 'external', base_confidence: float = 0.55) -> list[dict[str, Any]]:
        prepared_sentence, hedge_penalty = self._strip_hedges(sentence)
        if not prepared_sentence or prepared_sentence.endswith('?'):
            return []

        candidates: list[dict[str, Any]] = []
        for pattern in RELATION_PATTERNS:
            if pattern['relation_type'] == 'has_trait':
                if re.search(r'\bis\s+(?:a|an)\b', prepared_sentence, re.IGNORECASE):
                    continue
                if re.search(r'\bis\s+part\s+of\b', prepared_sentence, re.IGNORECASE):
                    continue
                if re.search(r'\bbelongs\s+to\b', prepared_sentence, re.IGNORECASE):
                    continue
                if re.search(r'\b(?:works|serves)\s+as\b', prepared_sentence, re.IGNORECASE):
                    continue
            match = pattern['regex'].match(prepared_sentence)
            if not match:
                continue
            subject_raw = self._clean_phrase(match.group('subject'))
            object_raw = self._clean_phrase(match.group('object'))
            if not subject_raw or not object_raw:
                continue

            subject = self._canonicalize_concept(subject_raw, speaker=speaker)
            object_text = self._canonicalize_object(object_raw, speaker=speaker, relation_type=pattern['relation_type'])
            if not subject or not object_text or subject == object_text:
                continue

            object_kind = pattern.get('object_kind', 'concept')
            if object_kind == 'concept' and len(object_text.split()) > 5:
                object_kind = 'value'
            confidence = min(0.95, max(0.2, base_confidence + float(pattern.get('confidence_bonus', 0.0)) - hedge_penalty))
            concept_type = 'agent' if subject in {'mai', 'current_user'} else 'concept'
            candidates.append({
                'subject': subject,
                'relation_type': pattern['relation_type'],
                'object': object_text,
                'object_kind': object_kind,
                'confidence': confidence,
                'concept_type': concept_type,
            })
            if len(candidates) >= self.max_candidates_per_ingest:
                break
        return candidates

    def add_fact(
        self,
        subject: str,
        relation_type: str,
        obj: str,
        object_kind: str = 'concept',
        concept_type: str = 'concept',
        source_type: str = 'text',
        source_path: str = '',
        source_label: str = '',
        confidence: float = 0.55,
        editable: str = 'high',
        evidence_text: str = '',
        record_evidence: bool = True,
    ) -> dict[str, Any]:
        cur = getattr(self.brain, 'cur', None)
        if cur is None:
            return {'success': False, 'message': 'Knowledge store is not available.'}

        now = time.time()
        subject_name = self._canonicalize_concept(subject)
        if not subject_name:
            return {'success': False, 'message': 'No subject concept could be derived.'}
        subject_id = self._upsert_concept(subject_name, concept_type=concept_type, confidence=confidence, importance=confidence)
        self._upsert_alias(subject, subject_id, confidence=confidence)

        object_text = self._canonicalize_object(obj, relation_type=relation_type)
        object_id = None
        object_key = object_text
        if object_kind == 'concept':
            object_id = self._upsert_concept(object_text, concept_type='concept', confidence=confidence, importance=confidence * 0.9)
            self._upsert_alias(obj, object_id, confidence=confidence)
            object_key = object_text

        fact_key = self._fact_key(subject_name, relation_type, object_key, object_kind)
        existing = cur.execute("""
            SELECT confidence, support_count, contradiction_count, editable, source_count, status
            FROM fact_records
            WHERE fact_key = ?
        """, (fact_key,)).fetchone()

        if existing:
            old_confidence = self._safe_float(existing[0], 0.5)
            support_count = int(existing[1] or 0) + 1
            merged_confidence = self._merge_confidence(old_confidence, confidence, support_count)
            merged_editable = self._merge_editability(existing[3], editable)
            source_count = int(existing[4] or 0) + 1
            merged_status = self._compute_fact_status(
                merged_confidence,
                support_count,
                source_count,
                source_type,
                current_status=str(existing[5] or 'active'),
            )
            cur.execute("""
                UPDATE fact_records
                SET confidence = ?, editable = ?, status = ?, support_count = ?, last_seen = ?, source_count = ?
                WHERE fact_key = ?
            """, (merged_confidence, merged_editable, merged_status, support_count, now, source_count, fact_key))
        else:
            status = self._compute_fact_status(confidence, 1, 1, source_type)
            cur.execute("""
                INSERT INTO fact_records (
                    fact_key, subject_concept_id, relation_type, object_concept_id, object_text, object_kind,
                    confidence, editable, status, support_count, contradiction_count, first_seen, last_seen, source_count
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 0, ?, ?, 1)
            """, (
                fact_key,
                subject_id,
                relation_type,
                object_id,
                object_text,
                object_kind,
                confidence,
                editable,
                status,
                now,
                now,
            ))

        if record_evidence:
            self._insert_evidence(
                fact_key,
                source_type=source_type,
                source_path=source_path,
                source_label=source_label,
                source_timestamp=now,
                evidence_text=evidence_text,
                confidence=confidence,
            )

        if subject_name == 'mai' and relation_type in {'has_trait', 'prefers', 'avoids', 'role', 'capable_of'}:
            self._upsert_identity_trait(relation_type, object_text, confidence, editable, now)

        truth_fact_table = getattr(self.brain, 'truth_fact_table', None)
        if truth_fact_table is not None:
            try:
                truth_fact_table.add_fact(subject_name, f"{relation_type}: {object_text}", source=source_type, confidence=confidence)
            except Exception:
                pass

        try:
            self._apply_fact_conflict_updates(subject_name, relation_type, object_text, object_kind, fact_key)
        except Exception:
            pass

        return {
            'success': True,
            'fact_key': fact_key,
            'subject': subject_name,
            'relation_type': relation_type,
            'object': object_text,
        }

    def import_rows(self, knowledge_payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(knowledge_payload, dict):
            return {'success': False, 'message': 'Knowledge payload must be a mapping.'}
        imported_facts = 0
        concepts = knowledge_payload.get('concepts', []) or []
        aliases = knowledge_payload.get('aliases', []) or []
        facts = knowledge_payload.get('facts', []) or []
        evidence_rows = knowledge_payload.get('evidence', []) or []
        identity_rows = knowledge_payload.get('identity_traits', []) or []
        evidence_fact_keys = {
            str(item.get('fact_key', '') or '')
            for item in evidence_rows
            if isinstance(item, dict) and item.get('fact_key')
        }

        for concept in concepts:
            if not isinstance(concept, dict):
                continue
            self._upsert_concept(
                concept.get('canonical_name', ''),
                concept_type=concept.get('concept_type', 'concept'),
                confidence=self._safe_float(concept.get('confidence'), 0.5),
                importance=self._safe_float(concept.get('importance'), 0.5),
            )
        for alias in aliases:
            if not isinstance(alias, dict):
                continue
            canonical_name = alias.get('canonical_name', '')
            if not canonical_name:
                continue
            concept_id = self._upsert_concept(canonical_name, confidence=self._safe_float(alias.get('confidence'), 0.5))
            self._upsert_alias(alias.get('alias_text', ''), concept_id, alias_type=alias.get('alias_type', 'surface'), confidence=self._safe_float(alias.get('confidence'), 0.5))
        for fact in facts:
            if not isinstance(fact, dict):
                continue
            result = self.add_fact(
                fact.get('subject', ''),
                fact.get('relation_type', ''),
                fact.get('object', ''),
                object_kind=fact.get('object_kind', 'concept'),
                concept_type=fact.get('subject_type', 'concept'),
                source_type=fact.get('source_type', 'training_chunk'),
                source_path=fact.get('source_path', ''),
                source_label=fact.get('source_label', ''),
                confidence=self._safe_float(fact.get('confidence'), 0.5),
                editable=fact.get('editable', 'high'),
                evidence_text=fact.get('evidence_text', ''),
                record_evidence=str(fact.get('fact_key', '') or '') not in evidence_fact_keys,
            )
            if result.get('success'):
                imported_facts += 1
        for evidence in evidence_rows:
            if not isinstance(evidence, dict):
                continue
            fact_key = evidence.get('fact_key')
            if not fact_key:
                continue
            self._insert_evidence(
                str(fact_key),
                source_type=evidence.get('source_type', 'training_chunk'),
                source_path=evidence.get('source_path', ''),
                source_label=evidence.get('source_label', ''),
                source_timestamp=self._safe_float(evidence.get('source_timestamp'), time.time()),
                evidence_text=str(evidence.get('evidence_text', ''))[:400],
                confidence=self._safe_float(evidence.get('confidence'), 0.5),
            )
        for identity in identity_rows:
            if not isinstance(identity, dict):
                continue
            self._upsert_identity_trait(
                identity.get('relation_type', 'has_trait'),
                identity.get('trait_value', ''),
                self._safe_float(identity.get('confidence'), 0.5),
                identity.get('editable', 'high'),
                time.time(),
            )
        return {'success': True, 'fact_count': imported_facts}

    def export_rows(self) -> dict[str, Any]:
        cur = getattr(self.brain, 'cur', None)
        if cur is None:
            return {'concepts': [], 'aliases': [], 'facts': [], 'evidence': [], 'identity_traits': []}

        concepts = []
        for row in cur.execute("""
            SELECT canonical_name, concept_type, confidence, importance, usage_count
            FROM concept_nodes
            ORDER BY usage_count DESC, confidence DESC, canonical_name ASC
        """).fetchall():
            concepts.append({
                'canonical_name': row[0],
                'concept_type': row[1],
                'confidence': self._safe_float(row[2], 0.5),
                'importance': self._safe_float(row[3], 0.5),
                'usage_count': int(row[4] or 0),
            })

        aliases = []
        for row in cur.execute("""
            SELECT ca.alias_text, ca.alias_type, ca.confidence, cn.canonical_name
            FROM concept_aliases ca
            JOIN concept_nodes cn ON cn.concept_id = ca.concept_id
            ORDER BY ca.last_seen DESC
        """).fetchall():
            aliases.append({
                'alias_text': row[0],
                'alias_type': row[1],
                'confidence': self._safe_float(row[2], 0.5),
                'canonical_name': row[3],
            })

        facts = []
        for row in cur.execute("""
            SELECT
                fr.fact_key,
                sc.canonical_name,
                fr.relation_type,
                oc.canonical_name,
                fr.object_text,
                fr.object_kind,
                fr.confidence,
                fr.editable,
                fr.status,
                fr.support_count
            FROM fact_records fr
            LEFT JOIN concept_nodes sc ON sc.concept_id = fr.subject_concept_id
            LEFT JOIN concept_nodes oc ON oc.concept_id = fr.object_concept_id
            ORDER BY fr.support_count DESC, fr.confidence DESC, fr.last_seen DESC
        """).fetchall():
            subject_name = row[1] or ''
            object_name = row[3] if row[5] == 'concept' else row[4]
            facts.append({
                'fact_key': row[0],
                'subject': subject_name,
                'subject_type': 'agent' if subject_name in {'mai', 'current_user'} else 'concept',
                'relation_type': row[2],
                'object': object_name or row[4],
                'object_kind': row[5],
                'confidence': self._safe_float(row[6], 0.5),
                'editable': row[7],
                'status': row[8] or 'active',
                'support_count': int(row[9] or 0),
            })

        evidence = []
        for row in cur.execute("""
            SELECT fact_key, source_type, source_path, source_label, source_timestamp, evidence_text, confidence
            FROM fact_evidence
            ORDER BY evidence_id DESC
            LIMIT 200
        """).fetchall():
            evidence.append({
                'fact_key': row[0],
                'source_type': row[1],
                'source_path': row[2],
                'source_label': row[3],
                'source_timestamp': self._safe_float(row[4], 0.0),
                'evidence_text': row[5],
                'confidence': self._safe_float(row[6], 0.5),
            })

        identity_traits = []
        for row in cur.execute("""
            SELECT trait_key, relation_type, trait_value, confidence, editable, support_count
            FROM identity_traits
            ORDER BY support_count DESC, confidence DESC, last_seen DESC
        """).fetchall():
            identity_traits.append({
                'trait_key': row[0],
                'relation_type': row[1],
                'trait_value': row[2],
                'confidence': self._safe_float(row[3], 0.5),
                'editable': row[4],
                'support_count': int(row[5] or 0),
            })

        return {
            'concepts': concepts,
            'aliases': aliases,
            'facts': facts,
            'evidence': evidence,
            'identity_traits': identity_traits,
        }

    def get_snapshot(self, limit: int = 8) -> dict[str, Any]:
        cur = getattr(self.brain, 'cur', None)
        if cur is None:
            return {'success': False, 'message': 'Knowledge store not available.'}
        concept_count = self._scalar("SELECT COUNT(*) FROM concept_nodes")
        fact_count = self._scalar("SELECT COUNT(*) FROM fact_records")
        evidence_count = self._scalar("SELECT COUNT(*) FROM fact_evidence")
        identity_count = self._scalar("SELECT COUNT(*) FROM identity_traits")
        stable_fact_count = self._scalar("SELECT COUNT(*) FROM fact_records WHERE status = 'stable'")
        provisional_fact_count = self._scalar("SELECT COUNT(*) FROM fact_records WHERE status = 'provisional'")
        top_concepts = self.get_concepts(limit=limit).get('rows', [])
        top_facts = self.get_facts(limit=limit).get('rows', [])
        return {
            'success': True,
            'concept_count': concept_count,
            'fact_count': fact_count,
            'evidence_count': evidence_count,
            'identity_count': identity_count,
            'stable_fact_count': stable_fact_count,
            'provisional_fact_count': provisional_fact_count,
            'top_concepts': top_concepts,
            'top_facts': top_facts,
        }

    def get_concepts(self, query: str = '', limit: int = 25) -> dict[str, Any]:
        cur = getattr(self.brain, 'cur', None)
        if cur is None:
            return {'success': False, 'rows': []}
        rows = []
        clean_query = self._normalize_text(query)
        if clean_query:
            data = cur.execute("""
                SELECT canonical_name, concept_type, confidence, importance, usage_count, last_seen
                FROM concept_nodes
                WHERE canonical_name LIKE ?
                ORDER BY usage_count DESC, confidence DESC, canonical_name ASC
                LIMIT ?
            """, (f'%{clean_query}%', limit)).fetchall()
        else:
            data = cur.execute("""
                SELECT canonical_name, concept_type, confidence, importance, usage_count, last_seen
                FROM concept_nodes
                ORDER BY usage_count DESC, confidence DESC, canonical_name ASC
                LIMIT ?
            """, (limit,)).fetchall()
        for row in data:
            rows.append({
                'canonical_name': row[0],
                'concept_type': row[1],
                'confidence': self._safe_float(row[2], 0.5),
                'importance': self._safe_float(row[3], 0.5),
                'usage_count': int(row[4] or 0),
                'last_seen': self._safe_float(row[5], 0.0),
            })
        return {'success': True, 'rows': rows}

    def get_facts(self, query: str = '', relation_type: str = '', limit: int = 25) -> dict[str, Any]:
        cur = getattr(self.brain, 'cur', None)
        if cur is None:
            return {'success': False, 'rows': []}
        clean_query = self._normalize_text(query)
        clean_relation = self._normalize_text(relation_type)
        clauses = []
        params: list[Any] = []
        if clean_query:
            clauses.append("(sc.canonical_name LIKE ? OR fr.object_text LIKE ? OR oc.canonical_name LIKE ?)")
            params.extend([f'%{clean_query}%', f'%{clean_query}%', f'%{clean_query}%'])
        if clean_relation:
            clauses.append("fr.relation_type = ?")
            params.append(clean_relation)
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        data = cur.execute(f"""
            SELECT
                fr.fact_key,
                sc.canonical_name,
                fr.relation_type,
                oc.canonical_name,
                fr.object_text,
                fr.object_kind,
                fr.confidence,
                fr.editable,
                fr.status,
                fr.support_count,
                fr.contradiction_count,
                fr.source_count,
                fr.first_seen,
                fr.last_seen,
                (SELECT COUNT(*) FROM fact_evidence fe WHERE fe.fact_key = fr.fact_key),
                (SELECT COUNT(DISTINCT fe.source_type) FROM fact_evidence fe WHERE fe.fact_key = fr.fact_key),
                (SELECT fe.source_type FROM fact_evidence fe WHERE fe.fact_key = fr.fact_key ORDER BY fe.evidence_id DESC LIMIT 1),
                (SELECT fe.source_path FROM fact_evidence fe WHERE fe.fact_key = fr.fact_key ORDER BY fe.evidence_id DESC LIMIT 1),
                (SELECT fe.source_label FROM fact_evidence fe WHERE fe.fact_key = fr.fact_key ORDER BY fe.evidence_id DESC LIMIT 1),
                (SELECT fe.source_timestamp FROM fact_evidence fe WHERE fe.fact_key = fr.fact_key ORDER BY fe.evidence_id DESC LIMIT 1)
            FROM fact_records fr
            LEFT JOIN concept_nodes sc ON sc.concept_id = fr.subject_concept_id
            LEFT JOIN concept_nodes oc ON oc.concept_id = fr.object_concept_id
            {where_sql}
            ORDER BY fr.support_count DESC, fr.confidence DESC, fr.last_seen DESC
            LIMIT ?
        """, (*params, limit)).fetchall()
        rows = []
        for row in data:
            fact_row = {
                'fact_key': row[0],
                'subject': row[1],
                'relation_type': row[2],
                'object': row[3] if row[5] == 'concept' else row[4],
                'object_kind': row[5],
                'confidence': self._safe_float(row[6], 0.5),
                'editable': row[7],
                'status': row[8] or 'active',
                'support_count': int(row[9] or 0),
                'contradiction_count': int(row[10] or 0),
                'source_count': int(row[11] or 0),
                'first_seen': self._safe_float(row[12], 0.0),
                'last_seen': self._safe_float(row[13], 0.0),
                'evidence_count': int(row[14] or 0),
                'source_type_count': int(row[15] or 0),
                'latest_source_type': row[16] or '',
                'latest_source_path': row[17] or '',
                'latest_source_label': row[18] or '',
                'latest_source_timestamp': self._safe_float(row[19], 0.0),
            }
            fact_row['summary'] = self._summarize_fact(fact_row['subject'], fact_row['relation_type'], fact_row['object'])
            fact_row['confidence_profile'] = self._build_fact_confidence_profile(fact_row)
            fact_row['provenance'] = self._build_fact_provenance_profile(fact_row)
            rows.append(fact_row)
        return {'success': True, 'rows': rows}

    def get_fact_evidence(self, fact_key: str = '', query: str = '', limit: int = 25) -> dict[str, Any]:
        cur = getattr(self.brain, 'cur', None)
        if cur is None:
            return {'success': False, 'rows': []}
        fact_key = str(fact_key or '').strip()
        clean_query = self._normalize_text(query)
        clauses = []
        params: list[Any] = []
        if fact_key:
            clauses.append("fe.fact_key = ?")
            params.append(fact_key)
        if clean_query:
            clauses.append("""
                fe.fact_key IN (
                    SELECT fr.fact_key
                    FROM fact_records fr
                    LEFT JOIN concept_nodes sc ON sc.concept_id = fr.subject_concept_id
                    LEFT JOIN concept_nodes oc ON oc.concept_id = fr.object_concept_id
                    WHERE sc.canonical_name LIKE ? OR fr.object_text LIKE ? OR oc.canonical_name LIKE ?
                )
            """)
            params.extend([f'%{clean_query}%', f'%{clean_query}%', f'%{clean_query}%'])
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = []
        data = cur.execute(f"""
            SELECT fact_key, source_type, source_path, source_label, source_timestamp, evidence_text, confidence
            FROM fact_evidence fe
            {where_sql}
            ORDER BY evidence_id DESC
            LIMIT ?
        """, (*params, limit)).fetchall()
        for row in data:
            rows.append({
                'fact_key': row[0],
                'source_type': row[1],
                'source_path': row[2] or '',
                'source_label': row[3] or '',
                'source_timestamp': self._safe_float(row[4], 0.0),
                'evidence_text': row[5] or '',
                'confidence': self._safe_float(row[6], 0.5),
            })
        return {'success': True, 'rows': rows}

    def get_identity_traits(self, limit: int = 15) -> dict[str, Any]:
        cur = getattr(self.brain, 'cur', None)
        if cur is None:
            return {'success': False, 'rows': []}
        rows = []
        fetch_limit = max(int(limit or 0), 1) * 3
        data = cur.execute("""
            SELECT trait_key, relation_type, trait_value, confidence, editable, support_count, contradiction_count, source_count, first_seen, last_seen
            FROM identity_traits
            ORDER BY support_count DESC, confidence DESC, last_seen DESC
            LIMIT ?
        """, (fetch_limit,)).fetchall()
        for row in data:
            trait_row = {
                'trait_key': row[0],
                'relation_type': row[1],
                'trait_value': row[2],
                'confidence': self._safe_float(row[3], 0.5),
                'editable': row[4],
                'support_count': int(row[5] or 0),
                'contradiction_count': int(row[6] or 0),
                'source_count': int(row[7] or 0),
                'first_seen': self._safe_float(row[8], 0.0),
                'last_seen': self._safe_float(row[9], 0.0),
                'summary': self._summarize_fact('Mai', row[1], row[2]),
            }
            if not self._is_reasoning_text_usable(trait_row['summary']):
                continue
            trait_row['confidence_profile'] = self._build_identity_trait_confidence_profile(trait_row)
            trait_row['provenance'] = self._build_identity_trait_provenance_profile(trait_row)
            rows.append(trait_row)
            if len(rows) >= max(1, int(limit or 1)):
                break
        return {'success': True, 'rows': rows}

    def consolidate_recent_episodes(self, window: int = 18) -> dict[str, Any]:
        cur = getattr(self.brain, 'cur', None)
        recent_entries = [
            entry for entry in list(getattr(self.brain, 'conversation_memory', []) or [])[-max(2, int(window or 2)):]
            if isinstance(entry, dict)
        ]
        if cur is None or len(recent_entries) < 2:
            return {'success': True, 'promoted_fact_count': 0, 'preference_fact_count': 0}

        timestamps = [self._safe_float(entry.get('timestamp'), 0.0) for entry in recent_entries if entry.get('timestamp')]
        if not timestamps:
            return {'success': True, 'promoted_fact_count': 0, 'preference_fact_count': 0}

        latest_timestamp = max(timestamps)
        window_bucket = max(1, int(latest_timestamp // 1800))

        promoted_fact_count = 0
        candidate_support: dict[str, dict[str, Any]] = {}
        for entry in recent_entries:
            entry_seen = set()
            user_text = str(entry.get('user_input', entry.get('user', '')) or '')
            for candidate in self.extract_fact_candidates(user_text, speaker='user', base_confidence=0.72):
                candidate_key = self._fact_key(candidate['subject'], candidate['relation_type'], candidate['object'], candidate['object_kind'])
                if candidate_key in entry_seen:
                    continue
                entry_seen.add(candidate_key)
                payload = candidate_support.setdefault(candidate_key, {'candidate': dict(candidate), 'count': 0})
                payload['count'] += 1
                payload['candidate']['confidence'] = max(
                    self._safe_float(payload['candidate'].get('confidence'), 0.5),
                    self._safe_float(candidate.get('confidence'), 0.5),
                )

            bot_text = str(entry.get('bot_response', entry.get('bot', '')) or '')
            if not bot_text:
                continue
            bot_quality = self._safe_float(entry.get('quality'), 0.5)
            bot_confidence = min(0.88, max(0.46, 0.4 + (bot_quality * 0.38)))
            for candidate in self.extract_fact_candidates(bot_text, speaker='mai', base_confidence=bot_confidence):
                candidate_key = self._fact_key(candidate['subject'], candidate['relation_type'], candidate['object'], candidate['object_kind'])
                if candidate_key in entry_seen:
                    continue
                entry_seen.add(candidate_key)
                payload = candidate_support.setdefault(candidate_key, {'candidate': dict(candidate), 'count': 0})
                payload['count'] += 1
                payload['candidate']['confidence'] = max(
                    self._safe_float(payload['candidate'].get('confidence'), 0.5),
                    self._safe_float(candidate.get('confidence'), 0.5),
                )

        for payload in candidate_support.values():
            evidence_count = int(payload.get('count', 0) or 0)
            if evidence_count < 2:
                continue
            if self._promote_candidate_from_episodes(payload.get('candidate', {}), evidence_count, window_bucket):
                promoted_fact_count += 1

        preference_fact_count = self._consolidate_user_topic_preferences(recent_entries, window_bucket)
        return {
            'success': True,
            'promoted_fact_count': promoted_fact_count,
            'preference_fact_count': preference_fact_count,
        }

    def get_relevant_concepts(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        normalized_query = self._normalize_text(query)
        query_terms = [term for term in self._normalize_text(query).split() if len(term) > 2]
        intent = self._detect_intent(query)
        concepts = self.get_concepts(limit=max(limit * 4, 10)).get('rows', [])
        scored = []
        for concept in concepts:
            name = str(concept.get('canonical_name', '')).lower()
            overlap = sum(1 for term in query_terms if term in name)
            if not query_terms:
                overlap = 1
            if overlap <= 0:
                continue
            score = overlap + self._safe_float(concept.get('confidence'), 0.5) + self._safe_float(concept.get('importance'), 0.5)
            if not self._is_clean_concept_surface(name):
                score -= 1.15
            if self._query_targets_mai(normalized_query) and name == 'mai':
                score += 1.4
            if intent in {'definition', 'self_description'} and len(name.split()) > 4:
                score -= min(0.9, (len(name.split()) - 4) * 0.18)
            scored.append((score, concept))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [concept for _, concept in scored[:max(1, limit)]]

    def _is_clean_concept_surface(self, concept_name: str) -> bool:
        normalized = self._normalize_text(concept_name)
        if not normalized:
            return False
        if normalized.startswith((
            'what ',
            'how ',
            'why ',
            'who ',
            'where ',
            'when ',
            'tell me ',
            'describe ',
            'summarize ',
        )):
            return False
        if ' interesting question let' in normalized or ' let me think' in normalized:
            return False
        return True

    def _build_query_profile(self, query: str, limit: int = 5) -> dict[str, Any]:
        normalized_query = self._normalize_text(query)
        query_terms = [term for term in normalized_query.split() if len(term) > 2]
        intent = self._detect_intent(query)
        target_concepts = self.get_relevant_concepts(query, limit=max(2, limit))
        concept_names = [str(item.get('canonical_name', '')).strip() for item in target_concepts if item.get('canonical_name')]
        expanded_terms = set(query_terms)
        for concept_name in concept_names:
            expanded_terms.update(token for token in concept_name.split() if len(token) > 2)
        expanded_terms.update(self._get_alias_terms_for_concepts(concept_names))
        relation_hints = {
            RELATION_QUERY_HINTS[token]
            for token in normalized_query.split()
            if token in RELATION_QUERY_HINTS
        }
        return {
            'normalized_query': normalized_query,
            'intent': intent,
            'query_terms': query_terms,
            'expanded_terms': sorted(expanded_terms),
            'target_concepts': target_concepts,
            'concept_names': concept_names,
            'relation_hints': relation_hints,
        }

    def _query_requests_preferences(self, normalized_query: str) -> bool:
        normalized_query = str(normalized_query or '').strip()
        if not normalized_query:
            return False
        preference_markers = (
            'prefer',
            'prefers',
            'like',
            'likes',
            'love',
            'loves',
            'enjoy',
            'enjoys',
            'favorite',
            'favourite',
        )
        return any(marker in normalized_query.split() for marker in preference_markers)

    def _build_query_focus_terms(self, query_profile: dict[str, Any]) -> set[str]:
        stop_terms = {
            'what',
            'when',
            'where',
            'which',
            'while',
            'does',
            'did',
            'just',
            'about',
            'into',
            'from',
            'with',
            'your',
            'you',
            'mai',
            'phoenix',
            'sgm',
            'system',
            'over',
            'time',
        }
        focus_terms = {
            term
            for term in query_profile.get('expanded_terms', [])
            if len(term) > 3 and term not in stop_terms
        }
        for concept_name in query_profile.get('concept_names', [])[:4]:
            normalized_name = self._normalize_text(concept_name)
            if normalized_name in {'mai', 'phoenix', 'sgm', 'sgm system'}:
                continue
            focus_terms.update(
                token for token in normalized_name.split()
                if len(token) > 3 and token not in stop_terms
            )
        return focus_terms

    def _relation_intent_bias(self, relation_type: str, query_profile: dict[str, Any], fact: dict[str, Any]) -> float:
        relation = self._normalize_text(relation_type)
        intent = str(query_profile.get('intent', '') or '').strip().lower()
        normalized_query = str(query_profile.get('normalized_query', '') or '').strip()
        requests_preferences = self._query_requests_preferences(normalized_query)
        subject = self._normalize_text(str(fact.get('subject', '') or ''))

        bias = 0.0
        if intent == 'self_description':
            bias_map = {
                'is_a': 1.25,
                'role': 1.15,
                'part_of': 0.95,
                'has_trait': 0.9,
                'capable_of': 0.5,
                'has_attribute': 0.45,
                'belongs_to': 0.25,
                'prefers': -0.75,
                'avoids': -0.65,
            }
            bias += bias_map.get(relation, 0.0)
            if subject == 'mai':
                bias += 0.25
            if requests_preferences and relation in {'prefers', 'avoids'}:
                bias += 1.0
        elif intent == 'definition':
            bias_map = {
                'is_a': 1.1,
                'role': 0.95,
                'part_of': 0.8,
                'belongs_to': 0.35,
                'has_trait': 0.25,
                'capable_of': 0.2,
                'prefers': -0.8,
                'avoids': -0.75,
            }
            bias += bias_map.get(relation, 0.0)
        elif intent == 'explanation':
            bias_map = {
                'causes': 0.95,
                'part_of': 0.45,
                'belongs_to': 0.35,
                'role': 0.3,
                'capable_of': 0.25,
                'is_a': 0.2,
            }
            bias += bias_map.get(relation, 0.0)
        elif intent == 'self_preference':
            bias_map = {
                'prefers': 1.45,
                'avoids': 0.85,
                'has_trait': 0.2,
                'is_a': -0.35,
                'role': -0.3,
                'part_of': -0.25,
            }
            bias += bias_map.get(relation, 0.0)
            if subject == 'mai':
                bias += 0.2
        return bias

    def _get_alias_terms_for_concepts(self, concept_names: list[str]) -> set[str]:
        cur = getattr(self.brain, 'cur', None)
        names = [self._normalize_text(name) for name in concept_names if self._normalize_text(name)]
        if cur is None or not names:
            return set()
        placeholders = ", ".join(["?"] * len(names))
        rows = cur.execute(f"""
            SELECT ca.alias_text
            FROM concept_aliases ca
            JOIN concept_nodes cn ON cn.concept_id = ca.concept_id
            WHERE cn.canonical_name IN ({placeholders})
        """, tuple(names)).fetchall()
        terms = set()
        for row in rows:
            alias_text = self._normalize_text(row[0] or '')
            if not alias_text:
                continue
            terms.update(token for token in alias_text.split() if len(token) > 2)
        return terms

    def _collect_candidate_facts_for_query(self, query_profile: dict[str, Any], limit: int = 5) -> list[dict[str, Any]]:
        candidate_limit = max(limit * 8, 32)
        facts_by_key: dict[str, dict[str, Any]] = {}
        normalized_query = str(query_profile.get('normalized_query', '') or '').strip()
        if normalized_query:
            for fact in self.get_facts(query=normalized_query, limit=max(candidate_limit, 48)).get('rows', []):
                fact_key = str(fact.get('fact_key', '') or '')
                if fact_key:
                    facts_by_key[fact_key] = fact
        for concept_name in query_profile.get('concept_names', [])[:4]:
            for fact in self.get_facts(query=concept_name, limit=max(limit * 4, 16)).get('rows', []):
                fact_key = str(fact.get('fact_key', '') or '')
                if fact_key:
                    facts_by_key[fact_key] = fact
        if not facts_by_key:
            for fact in self.get_facts(limit=max(candidate_limit, 48)).get('rows', []):
                fact_key = str(fact.get('fact_key', '') or '')
                if fact_key:
                    facts_by_key[fact_key] = fact
        return list(facts_by_key.values())

    def _source_reliability(self, source_type: str) -> float:
        source_type = str(source_type or '').strip().lower()
        table = {
            'training_text': 0.75,
            'episodic_consolidation': 0.72,
            'conversation_user': 0.7,
            'conversation_bot': 0.42,
        }
        return table.get(source_type, 0.5)

    def _editability_score(self, editable: str) -> float:
        editable = str(editable or '').strip().lower()
        table = {
            'low': 0.18,
            'medium': 0.58,
            'high': 0.92,
        }
        return table.get(editable, 0.75)

    def _build_fact_confidence_profile(self, fact: dict[str, Any]) -> dict[str, Any]:
        truth_confidence = self._safe_float(fact.get('confidence'), 0.5)
        support_count = int(fact.get('support_count', 0) or 0)
        contradiction_count = int(fact.get('contradiction_count', 0) or 0)
        source_count = int(fact.get('source_count', 0) or 0)
        source_type_count = int(fact.get('source_type_count', 0) or 0)
        status = str(fact.get('status', '') or '').strip().lower()
        latest_source_type = str(fact.get('latest_source_type', '') or '')

        source_reliability = min(0.98, self._source_reliability(latest_source_type) + min(0.16, max(0, source_type_count - 1) * 0.06))
        status_bonus = {'stable': 0.28, 'active': 0.12, 'provisional': -0.08}.get(status, 0.0)
        stability = min(0.98, max(0.08, 0.28 + (min(5, support_count) * 0.11) + (min(4, source_count) * 0.05) + status_bonus))
        consistency = max(0.05, 1.0 - min(0.92, contradiction_count * 0.22))
        revisability = self._editability_score(fact.get('editable', 'high'))
        overall = min(
            0.99,
            max(
                0.05,
                (truth_confidence * 0.42)
                + (source_reliability * 0.18)
                + (stability * 0.22)
                + (consistency * 0.18),
            ),
        )
        return {
            'overall': overall,
            'truth_confidence': truth_confidence,
            'source_reliability': source_reliability,
            'stability': stability,
            'consistency': consistency,
            'revisability': revisability,
        }

    def _build_fact_provenance_profile(self, fact: dict[str, Any]) -> dict[str, Any]:
        latest_source_type = str(fact.get('latest_source_type', '') or '')
        origin_kind = 'derived'
        if latest_source_type.startswith('conversation_'):
            origin_kind = 'conversation'
        elif latest_source_type == 'training_text':
            origin_kind = 'training'
        elif latest_source_type == 'episodic_consolidation':
            origin_kind = 'consolidated_memory'
        return {
            'origin_kind': origin_kind,
            'latest_source_type': latest_source_type,
            'latest_source_path': str(fact.get('latest_source_path', '') or ''),
            'latest_source_label': str(fact.get('latest_source_label', '') or ''),
            'latest_source_timestamp': self._safe_float(fact.get('latest_source_timestamp'), 0.0),
            'source_count': int(fact.get('source_count', 0) or 0),
            'source_type_count': int(fact.get('source_type_count', 0) or 0),
            'evidence_count': int(fact.get('evidence_count', 0) or 0),
            'first_seen': self._safe_float(fact.get('first_seen'), 0.0),
            'last_seen': self._safe_float(fact.get('last_seen'), 0.0),
            'editable': str(fact.get('editable', 'high') or 'high'),
        }

    def _build_identity_trait_confidence_profile(self, trait: dict[str, Any]) -> dict[str, Any]:
        truth_confidence = self._safe_float(trait.get('confidence'), 0.5)
        support_count = int(trait.get('support_count', 0) or 0)
        contradiction_count = int(trait.get('contradiction_count', 0) or 0)
        source_count = int(trait.get('source_count', 0) or 0)
        stability = min(0.98, max(0.08, 0.3 + (min(5, support_count) * 0.1) + (min(4, source_count) * 0.05)))
        consistency = max(0.05, 1.0 - min(0.92, contradiction_count * 0.2))
        revisability = self._editability_score(trait.get('editable', 'high'))
        overall = min(
            0.99,
            max(
                0.05,
                (truth_confidence * 0.48)
                + (stability * 0.28)
                + (consistency * 0.24),
            ),
        )
        return {
            'overall': overall,
            'truth_confidence': truth_confidence,
            'source_reliability': 0.72,
            'stability': stability,
            'consistency': consistency,
            'revisability': revisability,
        }

    def _build_identity_trait_provenance_profile(self, trait: dict[str, Any]) -> dict[str, Any]:
        return {
            'origin_kind': 'identity_memory',
            'latest_source_type': 'identity_trait',
            'latest_source_path': '',
            'latest_source_label': '',
            'latest_source_timestamp': self._safe_float(trait.get('last_seen'), 0.0),
            'source_count': int(trait.get('source_count', 0) or 0),
            'source_type_count': 1 if int(trait.get('source_count', 0) or 0) > 0 else 0,
            'evidence_count': int(trait.get('support_count', 0) or 0),
            'first_seen': self._safe_float(trait.get('first_seen'), 0.0),
            'last_seen': self._safe_float(trait.get('last_seen'), 0.0),
            'editable': str(trait.get('editable', 'high') or 'high'),
        }

    def _fact_support_strength(self, fact: dict[str, Any]) -> float:
        confidence_profile = fact.get('confidence_profile') if isinstance(fact, dict) else None
        confidence = self._safe_float(
            confidence_profile.get('overall') if isinstance(confidence_profile, dict) else fact.get('confidence'),
            0.5,
        )
        support_count = int(fact.get('support_count', 0) or 0)
        if isinstance(confidence_profile, dict):
            source_reliability = self._safe_float(confidence_profile.get('source_reliability'), self._source_reliability(fact.get('latest_source_type', '')))
            consistency = self._safe_float(confidence_profile.get('consistency'), 1.0)
        else:
            source_reliability = self._source_reliability(fact.get('latest_source_type', ''))
            consistency = max(0.05, 1.0 - (int(fact.get('contradiction_count', 0) or 0) * 0.22))
        status = str(fact.get('status', '') or '').strip().lower()
        status_bonus = {'stable': 0.35, 'active': 0.15, 'provisional': -0.1}.get(status, 0.0)
        return confidence + min(0.45, support_count * 0.08) + (source_reliability * 0.2) + (consistency * 0.1) + status_bonus

    def _build_dynamic_contradiction_pressure(self, facts: list[dict[str, Any]]) -> dict[str, float]:
        pressure: dict[str, float] = {}
        by_subject_relation: dict[tuple[str, str], list[dict[str, Any]]] = {}
        by_subject_object: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for fact in facts:
            subject = str(fact.get('subject', '') or '')
            relation = str(fact.get('relation_type', '') or '')
            obj = str(fact.get('object', '') or '')
            if not subject or not relation or not obj:
                continue
            by_subject_relation.setdefault((subject, relation), []).append(fact)
            by_subject_object.setdefault((subject, obj), []).append(fact)

        for (subject, relation), group in by_subject_relation.items():
            del subject
            unique_objects = {str(item.get('object', '') or '') for item in group if item.get('object')}
            if relation in SOFT_CONFLICT_RELATIONS and len(unique_objects) > 1:
                strongest = max(group, key=self._fact_support_strength)
                for fact in group:
                    if fact.get('fact_key') == strongest.get('fact_key'):
                        continue
                    fact_key = str(fact.get('fact_key', '') or '')
                    pressure[fact_key] = pressure.get(fact_key, 0.0) + (0.18 * max(1, len(unique_objects) - 1))

        for group in by_subject_object.values():
            relations = {str(item.get('relation_type', '') or '') for item in group}
            if not ({'prefers', 'avoids'} <= relations):
                continue
            strongest = max(group, key=self._fact_support_strength)
            for fact in group:
                if fact.get('fact_key') == strongest.get('fact_key'):
                    continue
                fact_key = str(fact.get('fact_key', '') or '')
                pressure[fact_key] = pressure.get(fact_key, 0.0) + 0.35

        return pressure

    def _score_fact_for_query(self, fact: dict[str, Any], query_profile: dict[str, Any], contradiction_pressure: dict[str, float]) -> float:
        subject = self._normalize_text(str(fact.get('subject', '') or ''))
        relation = self._normalize_text(str(fact.get('relation_type', '') or ''))
        obj = self._normalize_text(str(fact.get('object', '') or ''))
        summary = self._normalize_text(str(fact.get('summary', '') or ''))
        fact_terms = {term for term in " ".join([subject, relation, obj, summary]).split() if len(term) > 2}

        query_terms = query_profile.get('query_terms', [])
        expanded_terms = query_profile.get('expanded_terms', [])
        relation_hints = query_profile.get('relation_hints', set())
        concept_names = {self._normalize_text(name) for name in query_profile.get('concept_names', []) if name}

        direct_overlap = sum(1 for term in query_terms if term in fact_terms)
        expanded_overlap = sum(1 for term in expanded_terms if term in fact_terms)
        focus_terms = self._build_query_focus_terms(query_profile)
        fact_focus_terms = {
            term for term in fact_terms
            if term not in {'mai', 'phoenix', 'sgm', 'system'}
        }
        topical_overlap = len(focus_terms.intersection(fact_focus_terms))
        concept_match = 0.0
        if subject in concept_names:
            concept_match += 1.2
        if obj in concept_names:
            concept_match += 1.1
        if query_profile.get('normalized_query') and query_profile['normalized_query'] in summary:
            concept_match += 0.6

        if direct_overlap <= 0 and expanded_overlap <= 0 and concept_match <= 0:
            return float('-inf')

        confidence_profile = fact.get('confidence_profile') if isinstance(fact, dict) else None
        confidence_bonus = self._safe_float(
            confidence_profile.get('overall') if isinstance(confidence_profile, dict) else fact.get('confidence'),
            0.5,
        ) * 0.9
        support_bonus = min(0.65, int(fact.get('support_count', 0) or 0) * 0.09)
        source_bonus = self._safe_float(
            confidence_profile.get('source_reliability') if isinstance(confidence_profile, dict) else self._source_reliability(fact.get('latest_source_type', '')),
            0.5,
        ) * 0.35
        status_bonus = {'stable': 0.55, 'active': 0.2, 'provisional': -0.2}.get(str(fact.get('status', '') or '').strip().lower(), 0.0)
        relation_bonus = 0.55 if relation_hints and relation in relation_hints else 0.0
        relation_intent_bias = self._relation_intent_bias(relation, query_profile, fact)
        topical_bonus = 0.0
        intent = str(query_profile.get('intent', '') or '').strip().lower()
        if focus_terms:
            topical_bonus += min(0.85, topical_overlap * 0.32)
            if intent == 'explanation' and topical_overlap <= 0 and subject in {'mai', 'phoenix'}:
                topical_bonus -= 1.15
            elif intent in {'answer', 'guidance'} and topical_overlap <= 0 and subject in {'mai', 'phoenix'}:
                topical_bonus -= 0.45
        surface_penalty = self._fact_surface_penalty(fact, query_profile)
        consistency_penalty = 0.0
        if isinstance(confidence_profile, dict):
            consistency_penalty = max(0.0, (1.0 - self._safe_float(confidence_profile.get('consistency'), 1.0)) * 0.6)
        contradiction_penalty = (
            (int(fact.get('contradiction_count', 0) or 0) * 0.28)
            + contradiction_pressure.get(str(fact.get('fact_key', '') or ''), 0.0)
            + consistency_penalty
            + surface_penalty
        )

        return (
            (direct_overlap * 1.25)
            + (expanded_overlap * 0.25)
            + concept_match
            + confidence_bonus
            + support_bonus
            + source_bonus
            + status_bonus
            + relation_bonus
            + relation_intent_bias
            + topical_bonus
            - contradiction_penalty
        )

    def _fact_surface_penalty(self, fact: dict[str, Any], query_profile: dict[str, Any]) -> float:
        intent = str(query_profile.get('intent', '') or '').strip().lower()
        relation = self._normalize_text(str(fact.get('relation_type', '') or ''))
        summary = str(fact.get('summary', '') or '').strip()
        subject = self._normalize_text(str(fact.get('subject', '') or ''))
        obj = self._normalize_text(str(fact.get('object', '') or ''))
        object_kind = self._normalize_text(str(fact.get('object_kind', '') or ''))
        latest_source_type = self._normalize_text(str(fact.get('latest_source_type', '') or ''))

        penalty = 0.0
        if not self._is_reasoning_text_usable(summary):
            penalty += 1.45
        if not self._is_clean_concept_surface(subject):
            penalty += 1.35
        if intent in {'definition', 'self_description'} and subject not in {'mai', 'phoenix'} and len(subject.split()) > 4:
            penalty += min(0.8, (len(subject.split()) - 4) * 0.15)
        if intent in {'definition', 'self_description'} and relation in {'is_a', 'part_of', 'role', 'belongs_to'} and object_kind != 'concept':
            penalty += 0.85
        if intent in {'definition', 'self_description'} and object_kind != 'concept' and len(obj.split()) > 6:
            penalty += min(0.7, (len(obj.split()) - 6) * 0.08)
        if latest_source_type == 'conversation_bot' and penalty > 0.0:
            penalty += 0.2
        return penalty

    def _fact_matches_query_focus(self, fact: dict[str, Any], query_profile: dict[str, Any]) -> bool:
        if not isinstance(fact, dict):
            return False
        focus_terms = self._build_query_focus_terms(query_profile)
        if not focus_terms:
            return True
        fact_terms = {
            term for term in self._normalize_text(
                " ".join([
                    str(fact.get('subject', '') or ''),
                    str(fact.get('summary', '') or ''),
                    str(fact.get('object', '') or ''),
                ])
            ).split()
            if term not in {'mai', 'phoenix', 'sgm', 'system'}
        }
        return bool(focus_terms.intersection(fact_terms))

    def build_response_plan(self, query: str, limit: int = 3) -> dict[str, Any]:
        query = str(query or '').strip()
        if not query:
            return {'success': False, 'message': 'No query provided.'}

        intent = self._detect_intent(query)
        query_profile = self._build_query_profile(query, limit=max(2, limit))
        target_concepts = query_profile.get('target_concepts', [])[:max(1, limit)]
        supporting_facts = self.get_relevant_facts(query, limit=max(6, limit * 2))
        if intent in {'answer', 'explanation', 'guidance'}:
            focused_facts = [
                fact for fact in supporting_facts
                if self._fact_matches_query_focus(fact, query_profile)
            ]
            if focused_facts:
                supporting_facts = focused_facts
        stable_supporting_facts = [fact for fact in supporting_facts if fact.get('status') == 'stable']
        preferred_facts = stable_supporting_facts or supporting_facts

        identity_traits: list[dict[str, Any]] = []
        lower_query = self._normalize_text(query)
        if self._query_targets_mai(lower_query):
            identity_traits = self.get_identity_traits(limit=5).get('rows', [])

        structure = self._choose_response_structure(intent, preferred_facts)
        stance = self._choose_response_stance(intent, identity_traits)
        fact_summaries = [str(fact.get('summary', '')).strip() for fact in preferred_facts[:max(1, limit)] if fact.get('summary')]
        identity_summaries = [str(item.get('summary', '')).strip() for item in identity_traits[:2] if item.get('summary')]
        claim_plan = self._build_claim_plan(intent, preferred_facts[:max(1, limit)], identity_traits, structure)
        reasoning_preview = self.get_graph_reasoning_preview(query, limit=max(2, limit), max_depth=2)

        return {
            'success': True,
            'query': query,
            'intent': intent,
            'stance': stance,
            'structure': structure,
            'reply_goal': self._choose_reply_goal(intent, structure),
            'target_concepts': target_concepts,
            'supporting_facts': preferred_facts[:max(1, limit)],
            'identity_traits': identity_traits,
            'fact_summaries': fact_summaries,
            'identity_summaries': identity_summaries,
            'lead_hint': self._build_lead_hint(intent, preferred_facts, identity_traits),
            'main_claim': claim_plan['main_claim'],
            'support_claims': claim_plan['support_claims'],
            'claims': claim_plan['claims'],
            'uncertainties': claim_plan['uncertainties'],
            'evidence_keys': claim_plan['evidence_keys'],
            'plan_confidence': claim_plan['plan_confidence'],
            'reasoning_mode': reasoning_preview.get('mode', 'sparse'),
            'reasoning_summary': reasoning_preview.get('summary', ''),
            'reasoning_paths': reasoning_preview.get('paths', []),
            'reasoning_seed_concepts': reasoning_preview.get('seed_concepts', []),
        }

    def _build_claim_plan(
        self,
        intent: str,
        preferred_facts: list[dict[str, Any]],
        identity_traits: list[dict[str, Any]],
        structure: str,
    ) -> dict[str, Any]:
        main_claim = None
        support_claims: list[dict[str, Any]] = []

        if intent == 'self_preference' and identity_traits:
            preferred_trait = next(
                (
                    trait for trait in identity_traits
                    if str(trait.get('relation_type', '') or '').strip().lower() in {'prefers', 'avoids'}
                ),
                None,
            )
            preferred_fact = next(
                (
                    fact for fact in preferred_facts
                    if str(fact.get('relation_type', '') or '').strip().lower() in {'prefers', 'avoids'}
                ),
                None,
            )
            if preferred_trait is not None:
                main_claim = self._build_claim_from_identity_trait(preferred_trait, role='main')
            elif preferred_fact is not None:
                main_claim = self._build_claim_from_fact(preferred_fact, role='main')
            for trait in identity_traits:
                if trait is preferred_trait:
                    continue
                if str(trait.get('relation_type', '') or '').strip().lower() in {'prefers', 'avoids'}:
                    support_claims.append(self._build_claim_from_identity_trait(trait, role='support'))
                if len(support_claims) >= 2:
                    break
            for fact in preferred_facts:
                if fact is preferred_fact:
                    continue
                if str(fact.get('relation_type', '') or '').strip().lower() in {'prefers', 'avoids'}:
                    support_claims.append(self._build_claim_from_fact(fact, role='support'))
                if len(support_claims) >= 2:
                    break
        elif intent == 'self_description' and identity_traits:
            identity_fact = next(
                (
                    fact for fact in preferred_facts
                    if str(fact.get('relation_type', '') or '').strip().lower() in {'is_a', 'role', 'has_trait'}
                ),
                None,
            )
            if identity_fact is not None:
                main_claim = self._build_claim_from_fact(identity_fact, role='main')
                for fact in preferred_facts:
                    if fact is identity_fact:
                        continue
                    support_claims.append(self._build_claim_from_fact(fact, role='support'))
                    if len(support_claims) >= 2:
                        break
                for trait in identity_traits[:2]:
                    support_claims.append(self._build_claim_from_identity_trait(trait, role='support'))
            else:
                main_claim = self._build_claim_from_identity_trait(identity_traits[0], role='main')
                for trait in identity_traits[1:3]:
                    support_claims.append(self._build_claim_from_identity_trait(trait, role='support'))
                for fact in preferred_facts[:2]:
                    support_claims.append(self._build_claim_from_fact(fact, role='support'))
        else:
            if preferred_facts:
                main_claim = self._build_claim_from_fact(preferred_facts[0], role='main')
            for fact in preferred_facts[1:3]:
                support_claims.append(self._build_claim_from_fact(fact, role='support'))
            if intent == 'self_description':
                for trait in identity_traits[:2]:
                    support_claims.append(self._build_claim_from_identity_trait(trait, role='support'))

        support_claims = [claim for claim in support_claims if claim]
        claims = ([main_claim] if main_claim else []) + support_claims
        evidence_keys = list(dict.fromkeys(
            str(claim.get('fact_key', '')).strip()
            for claim in claims
            if isinstance(claim, dict) and claim.get('fact_key')
        ))
        uncertainties = []
        seen_uncertainties = set()
        for claim in claims:
            for item in claim.get('uncertainties', []) if isinstance(claim, dict) else []:
                key = str(item or '').strip().lower()
                if not key or key in seen_uncertainties:
                    continue
                seen_uncertainties.add(key)
                uncertainties.append(str(item))
        if not claims:
            uncertainties.append('Knowledge support is currently sparse for this prompt.')

        plan_confidence = 0.0
        if main_claim:
            main_profile = main_claim.get('confidence_profile', {})
            plan_confidence = self._safe_float(main_profile.get('overall'), 0.0)
        elif support_claims:
            scores = [self._safe_float(claim.get('confidence_profile', {}).get('overall'), 0.0) for claim in support_claims]
            plan_confidence = sum(scores) / max(1, len(scores))

        return {
            'main_claim': main_claim or {},
            'support_claims': support_claims,
            'claims': claims,
            'uncertainties': uncertainties,
            'evidence_keys': evidence_keys,
            'plan_confidence': min(0.99, max(0.0, plan_confidence)),
            'reply_goal': self._choose_reply_goal(intent, structure),
        }

    def _build_claim_from_fact(self, fact: dict[str, Any], role: str = 'support') -> dict[str, Any]:
        if not isinstance(fact, dict):
            return {}
        summary = str(fact.get('summary', '')).strip()
        if not self._is_reasoning_text_usable(summary):
            return {}
        confidence_profile = fact.get('confidence_profile') if isinstance(fact.get('confidence_profile'), dict) else self._build_fact_confidence_profile(fact)
        provenance = fact.get('provenance') if isinstance(fact.get('provenance'), dict) else self._build_fact_provenance_profile(fact)
        return {
            'claim_type': 'fact',
            'role': role,
            'text': summary,
            'fact_key': str(fact.get('fact_key', '') or ''),
            'relation_type': str(fact.get('relation_type', '') or ''),
            'subject': str(fact.get('subject', '') or ''),
            'object': str(fact.get('object', '') or ''),
            'confidence_profile': confidence_profile,
            'provenance': provenance,
            'uncertainties': self._collect_fact_uncertainties(fact, confidence_profile),
        }

    def _build_claim_from_identity_trait(self, trait: dict[str, Any], role: str = 'support') -> dict[str, Any]:
        if not isinstance(trait, dict):
            return {}
        summary = str(trait.get('summary', '')).strip()
        if not self._is_reasoning_text_usable(summary):
            return {}
        confidence_profile = trait.get('confidence_profile') if isinstance(trait.get('confidence_profile'), dict) else self._build_identity_trait_confidence_profile(trait)
        provenance = trait.get('provenance') if isinstance(trait.get('provenance'), dict) else self._build_identity_trait_provenance_profile(trait)
        return {
            'claim_type': 'identity_trait',
            'role': role,
            'text': summary,
            'fact_key': str(trait.get('trait_key', '') or ''),
            'relation_type': str(trait.get('relation_type', '') or ''),
            'subject': 'mai',
            'object': str(trait.get('trait_value', '') or ''),
            'confidence_profile': confidence_profile,
            'provenance': provenance,
            'uncertainties': self._collect_identity_trait_uncertainties(trait, confidence_profile),
        }

    def _collect_fact_uncertainties(self, fact: dict[str, Any], confidence_profile: dict[str, Any]) -> list[str]:
        notes = []
        if str(fact.get('status', '') or '').strip().lower() == 'provisional':
            notes.append('This fact is still provisional.')
        if int(fact.get('contradiction_count', 0) or 0) > 0:
            notes.append('This fact has competing evidence or conflicting alternatives.')
        if self._safe_float(confidence_profile.get('source_reliability'), 1.0) < 0.55:
            notes.append('The latest supporting source is relatively weak.')
        if int(fact.get('support_count', 0) or 0) <= 1:
            notes.append('This fact has limited direct support so far.')
        return notes

    def _collect_identity_trait_uncertainties(self, trait: dict[str, Any], confidence_profile: dict[str, Any]) -> list[str]:
        notes = []
        if int(trait.get('contradiction_count', 0) or 0) > 0:
            notes.append('This self-trait has conflicting signals.')
        if int(trait.get('support_count', 0) or 0) <= 1:
            notes.append('This self-trait is still lightly supported.')
        if self._safe_float(confidence_profile.get('overall'), 1.0) < 0.62:
            notes.append('This self-trait is still forming.')
        return notes

    def _choose_reply_goal(self, intent: str, structure: str) -> str:
        if intent == 'definition':
            return 'Answer with a concise definition first, then ground it with durable facts.'
        if intent == 'self_preference':
            return 'Answer Mai-related preference questions directly, then support them with one durable preference or habit.'
        if intent == 'self_description':
            return 'Answer as Mai directly, then support the answer with durable self-traits or facts.'
        if intent == 'explanation':
            return 'Explain the core idea first, then add supporting cause, relation, or context claims.'
        if structure == 'direct_answer_then_steps':
            return 'Give a practical answer first, then add the most useful next step.'
        return 'Answer directly and keep the supporting claims coherent.'

    def get_relevant_facts(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        query_profile = self._build_query_profile(query, limit=max(2, limit))
        facts = self._collect_candidate_facts_for_query(query_profile, limit=max(2, limit))
        contradiction_pressure = self._build_dynamic_contradiction_pressure(facts)
        scored = []
        for fact in facts:
            score = self._score_fact_for_query(fact, query_profile, contradiction_pressure)
            if score == float('-inf'):
                continue
            scored.append((score, fact))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [fact for _, fact in scored[:max(1, limit)]]

    def summarize_facts_for_query(self, query: str, limit: int = 2) -> list[str]:
        summaries = []
        for fact in self.get_relevant_facts(query, limit=limit):
            summaries.append(str(fact.get('summary', '')).strip())
        return [summary for summary in summaries if summary]

    def get_graph_reasoning_preview(self, query: str, limit: int = 3, max_depth: int = 2) -> dict[str, Any]:
        query = str(query or '').strip()
        if not query:
            return {'success': False, 'message': 'No query provided.', 'paths': []}

        bounded_limit = max(1, min(6, int(limit or 3)))
        bounded_depth = max(1, min(3, int(max_depth or 2)))
        query_profile = self._build_query_profile(query, limit=max(3, bounded_limit))
        supporting_facts = self.get_relevant_facts(query, limit=max(4, bounded_limit * 2))
        seed_concepts = self._build_reasoning_seed_concepts(query_profile, supporting_facts, limit=max(3, bounded_limit + 1))

        edge_rows: dict[str, dict[str, Any]] = {}
        frontier = list(seed_concepts)
        expanded = set(seed_concepts)
        for _ in range(bounded_depth):
            fetched_edges = self._fetch_graph_edges_for_concepts(frontier, limit=max(12, bounded_limit * 10))
            if not fetched_edges:
                break
            next_frontier: list[str] = []
            for fact in fetched_edges:
                fact_key = str(fact.get('fact_key', '') or '')
                if fact_key:
                    edge_rows[fact_key] = fact
                if str(fact.get('object_kind', '') or '') == 'concept':
                    subject = self._normalize_text(str(fact.get('subject', '') or ''))
                    obj = self._normalize_text(str(fact.get('object', '') or ''))
                    for concept_name in (subject, obj):
                        if concept_name and concept_name not in expanded:
                            expanded.add(concept_name)
                            next_frontier.append(concept_name)
            frontier = next_frontier
            if not frontier:
                break

        candidate_edges = list(edge_rows.values())
        if not candidate_edges:
            candidate_edges = [
                fact for fact in supporting_facts
                if self._is_reasoning_text_usable(str(fact.get('summary', '') or ''))
            ]

        paths = self._build_reasoning_paths(query_profile, seed_concepts, candidate_edges, limit=bounded_limit, max_depth=bounded_depth)
        summary = paths[0].get('explanation', '') if paths else ''
        return {
            'success': True,
            'query': query,
            'mode': 'graph_path' if paths else 'sparse',
            'seed_concepts': seed_concepts,
            'expanded_concepts': sorted(expanded),
            'path_count': len(paths),
            'paths': paths,
            'summary': summary,
            'supporting_fact_keys': list(dict.fromkeys(
                fact_key
                for path in paths
                for fact_key in path.get('fact_keys', [])
                if fact_key
            )),
        }

    def _build_reasoning_seed_concepts(
        self,
        query_profile: dict[str, Any],
        supporting_facts: list[dict[str, Any]],
        limit: int = 4,
    ) -> list[str]:
        seeds: list[str] = []
        seen = set()

        def _add(name: str) -> None:
            normalized = self._normalize_text(name)
            if not normalized or normalized in seen:
                return
            seen.add(normalized)
            seeds.append(normalized)

        for concept_name in query_profile.get('concept_names', [])[:max(2, limit)]:
            _add(str(concept_name))
        for fact in supporting_facts[:max(2, limit * 2)]:
            _add(str(fact.get('subject', '') or ''))
            if str(fact.get('object_kind', '') or '') == 'concept':
                _add(str(fact.get('object', '') or ''))
        return seeds[:max(1, limit)]

    def _is_reasoning_text_usable(self, text: str) -> bool:
        normalized = " ".join(str(text or '').split()).strip()
        if not normalized:
            return False
        lower_text = normalized.lower()
        padded_text = f" {lower_text} "
        if lower_text.startswith((
            'short version is ',
            'the short version is ',
            'the main idea is ',
            'another supporting point is ',
            'another important point is ',
            'a useful way to frame ',
            'a practical way to think about it is ',
            'what stands out most is that ',
        )):
            return False
        if any(fragment in lower_text for fragment in ('interesting question let', 'question let me', ' let me think about that ')):
            return False
        if re.search(r'\b(?:is|are|was|were|helps|help|prefers|avoids)\s+can\b', lower_text):
            return False
        trimmed = lower_text.rstrip('.!?').strip()
        if trimmed.endswith(('part of', 'because of', 'instead of', 'rather than', 'such as')):
            return False
        if padded_text.count(' mai is ') > 1 or padded_text.count(' mai prefers ') > 1:
            return False
        stray_letters = [token for token in re.findall(r'\b[a-zA-Z]\b', lower_text) if token not in {'a', 'i'}]
        if len(stray_letters) > 1:
            return False
        if len(normalized.split()) >= 8:
            prefix = " ".join(normalized.lower().split()[:4])
            if prefix and prefix in " ".join(normalized.lower().split()[4:]):
                return False
        return True

    def _build_fact_row_from_record_tuple(self, row: tuple[Any, ...]) -> dict[str, Any]:
        fact_row = {
            'fact_key': row[0],
            'subject': row[1],
            'relation_type': row[2],
            'object': row[3] if row[5] == 'concept' else row[4],
            'object_kind': row[5],
            'confidence': self._safe_float(row[6], 0.5),
            'editable': row[7],
            'status': row[8] or 'active',
            'support_count': int(row[9] or 0),
            'contradiction_count': int(row[10] or 0),
            'source_count': int(row[11] or 0),
            'first_seen': self._safe_float(row[12], 0.0),
            'last_seen': self._safe_float(row[13], 0.0),
            'evidence_count': int(row[14] or 0),
            'source_type_count': int(row[15] or 0),
            'latest_source_type': row[16] or '',
            'latest_source_path': row[17] or '',
            'latest_source_label': row[18] or '',
            'latest_source_timestamp': self._safe_float(row[19], 0.0),
        }
        fact_row['summary'] = self._summarize_fact(fact_row['subject'], fact_row['relation_type'], fact_row['object'])
        fact_row['confidence_profile'] = self._build_fact_confidence_profile(fact_row)
        fact_row['provenance'] = self._build_fact_provenance_profile(fact_row)
        return fact_row

    def _fetch_graph_edges_for_concepts(self, concept_names: list[str], limit: int = 24) -> list[dict[str, Any]]:
        cur = getattr(self.brain, 'cur', None)
        normalized_names = [self._normalize_text(name) for name in concept_names if self._normalize_text(name)]
        if cur is None or not normalized_names:
            return []

        relation_placeholders = ", ".join("?" for _ in GRAPH_REASONING_RELATIONS)
        concept_placeholders = ", ".join("?" for _ in normalized_names)
        rows = cur.execute(f"""
            SELECT
                fr.fact_key,
                sc.canonical_name,
                fr.relation_type,
                oc.canonical_name,
                fr.object_text,
                fr.object_kind,
                fr.confidence,
                fr.editable,
                fr.status,
                fr.support_count,
                fr.contradiction_count,
                fr.source_count,
                fr.first_seen,
                fr.last_seen,
                (SELECT COUNT(*) FROM fact_evidence fe WHERE fe.fact_key = fr.fact_key),
                (SELECT COUNT(DISTINCT fe.source_type) FROM fact_evidence fe WHERE fe.fact_key = fr.fact_key),
                (SELECT fe.source_type FROM fact_evidence fe WHERE fe.fact_key = fr.fact_key ORDER BY fe.evidence_id DESC LIMIT 1),
                (SELECT fe.source_path FROM fact_evidence fe WHERE fe.fact_key = fr.fact_key ORDER BY fe.evidence_id DESC LIMIT 1),
                (SELECT fe.source_label FROM fact_evidence fe WHERE fe.fact_key = fr.fact_key ORDER BY fe.evidence_id DESC LIMIT 1),
                (SELECT fe.source_timestamp FROM fact_evidence fe WHERE fe.fact_key = fr.fact_key ORDER BY fe.evidence_id DESC LIMIT 1)
            FROM fact_records fr
            LEFT JOIN concept_nodes sc ON sc.concept_id = fr.subject_concept_id
            LEFT JOIN concept_nodes oc ON oc.concept_id = fr.object_concept_id
            WHERE fr.relation_type IN ({relation_placeholders})
              AND (
                sc.canonical_name IN ({concept_placeholders})
                OR oc.canonical_name IN ({concept_placeholders})
              )
            ORDER BY fr.support_count DESC, fr.confidence DESC, fr.last_seen DESC
            LIMIT ?
        """, (*GRAPH_REASONING_RELATIONS, *normalized_names, *normalized_names, max(8, int(limit or 24)))).fetchall()

        facts_by_key: dict[str, dict[str, Any]] = {}
        for row in rows:
            fact_row = self._build_fact_row_from_record_tuple(row)
            if not self._is_reasoning_text_usable(str(fact_row.get('summary', '') or '')):
                continue
            fact_key = str(fact_row.get('fact_key', '') or '')
            if fact_key:
                facts_by_key[fact_key] = fact_row
        return list(facts_by_key.values())

    def _reverse_reasoning_summary(self, fact: dict[str, Any]) -> str:
        subject = str(fact.get('subject', '') or '').strip()
        obj = str(fact.get('object', '') or '').strip()
        relation_type = str(fact.get('relation_type', '') or '').strip()
        reverse_label = GRAPH_REVERSE_RELATION_LABELS.get(relation_type, f"linked_from_{relation_type}")
        summary = self._summarize_fact(obj, reverse_label, subject)
        return summary.replace(' linked_from_', ' ').replace('_', ' ')

    def _score_reasoning_path(self, path_steps: list[dict[str, Any]], query_profile: dict[str, Any]) -> float:
        if not path_steps:
            return 0.0

        step_scores = []
        path_terms = set()
        relation_bonus = 0.0
        for step in path_steps:
            confidence_profile = step.get('confidence_profile', {}) if isinstance(step.get('confidence_profile'), dict) else {}
            surface_bonus = 0.08 if self._is_reasoning_text_usable(str(step.get('summary', '') or '')) else -0.18
            step_score = (
                (self._safe_float(confidence_profile.get('overall'), 0.5) * 0.44)
                + (self._safe_float(confidence_profile.get('source_reliability'), 0.5) * 0.18)
                + (self._safe_float(confidence_profile.get('stability'), 0.5) * 0.16)
                + (self._safe_float(confidence_profile.get('consistency'), 0.5) * 0.12)
                + surface_bonus
            )
            if step.get('direction') == 'forward' and str(step.get('relation_type', '') or '') in {'causes', 'part_of', 'is_a'}:
                relation_bonus += 0.04
            step_scores.append(step_score)
            path_terms.update(term for term in re.findall(r'\b[a-zA-Z0-9_]+\b', str(step.get('summary', '')).lower()) if len(term) > 2)

        expanded_terms = set(query_profile.get('expanded_terms', []) or [])
        overlap = len(path_terms.intersection(expanded_terms)) / max(1, len(expanded_terms)) if expanded_terms else 0.0
        depth_penalty = max(0.0, (len(path_steps) - 1) * 0.04)
        return max(0.0, min(1.0, (sum(step_scores) / max(1, len(step_scores))) + (overlap * 0.22) + relation_bonus - depth_penalty))

    def _build_reasoning_paths(
        self,
        query_profile: dict[str, Any],
        seed_concepts: list[str],
        facts: list[dict[str, Any]],
        limit: int = 3,
        max_depth: int = 2,
    ) -> list[dict[str, Any]]:
        outgoing: dict[str, list[dict[str, Any]]] = {}
        incoming: dict[str, list[dict[str, Any]]] = {}
        for fact in facts:
            subject = self._normalize_text(str(fact.get('subject', '') or ''))
            obj = self._normalize_text(str(fact.get('object', '') or ''))
            if not subject or not obj:
                continue
            step = {
                'fact_key': str(fact.get('fact_key', '') or ''),
                'direction': 'forward',
                'relation_type': str(fact.get('relation_type', '') or ''),
                'from_concept': subject,
                'to_concept': obj,
                'summary': str(fact.get('summary', '') or '').strip(),
                'confidence_profile': fact.get('confidence_profile', {}) if isinstance(fact.get('confidence_profile'), dict) else self._build_fact_confidence_profile(fact),
                'provenance': fact.get('provenance', {}) if isinstance(fact.get('provenance'), dict) else self._build_fact_provenance_profile(fact),
            }
            outgoing.setdefault(subject, []).append(step)
            if str(fact.get('object_kind', '') or '') == 'concept':
                reverse_step = dict(step)
                reverse_step.update({
                    'direction': 'reverse',
                    'from_concept': obj,
                    'to_concept': subject,
                    'summary': self._reverse_reasoning_summary(fact),
                })
                incoming.setdefault(obj, []).append(reverse_step)

        scored_paths: list[tuple[float, dict[str, Any]]] = []
        seen_paths = set()
        for seed in seed_concepts:
            normalized_seed = self._normalize_text(seed)
            if not normalized_seed:
                continue
            queue: deque[tuple[str, list[dict[str, Any]], set[str]]] = deque()
            for step in outgoing.get(normalized_seed, []) + incoming.get(normalized_seed, []):
                queue.append((step.get('to_concept', ''), [step], {normalized_seed, str(step.get('to_concept', '') or '')}))

            while queue:
                current_concept, path_steps, visited = queue.popleft()
                path_key = tuple(f"{step.get('fact_key')}:{step.get('direction')}" for step in path_steps)
                if path_key in seen_paths:
                    continue
                seen_paths.add(path_key)

                score = self._score_reasoning_path(path_steps, query_profile)
                if score > 0:
                    scored_paths.append((score, {
                        'start_concept': normalized_seed,
                        'end_concept': current_concept,
                        'step_count': len(path_steps),
                        'relation_chain': [str(step.get('relation_type', '') or '') for step in path_steps],
                        'fact_keys': [str(step.get('fact_key', '') or '') for step in path_steps if step.get('fact_key')],
                        'step_summaries': [str(step.get('summary', '') or '').strip() for step in path_steps if step.get('summary')],
                        'path_score': round(score, 4),
                        'explanation': " ".join(
                            str(step.get('summary', '') or '').strip()
                            for step in path_steps
                            if str(step.get('summary', '') or '').strip()
                        ).strip(),
                    }))

                if len(path_steps) >= max_depth:
                    continue

                for next_step in outgoing.get(current_concept, []) + incoming.get(current_concept, []):
                    next_concept = str(next_step.get('to_concept', '') or '')
                    if not next_concept or next_concept in visited:
                        continue
                    if any(str(existing.get('fact_key', '') or '') == str(next_step.get('fact_key', '') or '') for existing in path_steps):
                        continue
                    queue.append((next_concept, path_steps + [next_step], set(visited) | {next_concept}))

        scored_paths.sort(key=lambda item: item[0], reverse=True)
        return [path for _, path in scored_paths[:max(1, limit)]]

    def _bootstrap_lexicon(self) -> None:
        cur = getattr(self.brain, 'cur', None)
        if cur is None:
            return
        cur.executemany("""
            INSERT OR IGNORE INTO lexicon_entries (token_text, token_type, canonical_form, confidence, editable, notes)
            VALUES (?, ?, ?, ?, ?, ?)
        """, LEXICON_BOOTSTRAP)

    def _insert_evidence(
        self,
        fact_key: str,
        source_type: str = 'text',
        source_path: str = '',
        source_label: str = '',
        source_timestamp: float | None = None,
        evidence_text: str = '',
        confidence: float = 0.5,
    ) -> None:
        cur = getattr(self.brain, 'cur', None)
        if cur is None or not fact_key:
            return
        evidence_text = str(evidence_text or '')[:400]
        source_type = str(source_type or 'text')
        source_path = str(source_path or '')
        source_label = str(source_label or '')
        timestamp_value = self._safe_float(source_timestamp, time.time())
        existing = cur.execute("""
            SELECT 1
            FROM fact_evidence
            WHERE fact_key = ?
              AND source_type = ?
              AND source_path = ?
              AND source_label = ?
              AND evidence_text = ?
            LIMIT 1
        """, (
            fact_key,
            source_type,
            source_path,
            source_label,
            evidence_text,
        )).fetchone()
        if existing:
            return
        cur.execute("""
            INSERT INTO fact_evidence (
                fact_key, source_type, source_path, source_label, source_timestamp, evidence_text, confidence
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            fact_key,
            source_type,
            source_path,
            source_label,
            timestamp_value,
            evidence_text,
            confidence,
        ))

    def _upsert_concept(self, canonical_name: str, concept_type: str = 'concept', confidence: float = 0.5, importance: float = 0.5) -> int:
        cur = getattr(self.brain, 'cur', None)
        if cur is None:
            return 0
        canonical_name = self._normalize_text(canonical_name)
        if not canonical_name:
            return 0
        now = time.time()
        existing = cur.execute("""
            SELECT concept_id, confidence, importance, usage_count, concept_type
            FROM concept_nodes
            WHERE canonical_name = ?
        """, (canonical_name,)).fetchone()
        if existing:
            concept_id = int(existing[0])
            old_confidence = self._safe_float(existing[1], 0.5)
            old_importance = self._safe_float(existing[2], 0.5)
            usage_count = int(existing[3] or 0) + 1
            merged_confidence = self._merge_confidence(old_confidence, confidence, usage_count)
            merged_importance = max(old_importance, importance)
            merged_type = existing[4] if existing[4] not in ('', 'concept') else concept_type
            cur.execute("""
                UPDATE concept_nodes
                SET concept_type = ?, confidence = ?, importance = ?, last_seen = ?, usage_count = ?
                WHERE concept_id = ?
            """, (merged_type, merged_confidence, merged_importance, now, usage_count, concept_id))
            return concept_id
        cur.execute("""
            INSERT INTO concept_nodes (canonical_name, concept_type, confidence, importance, first_seen, last_seen, usage_count)
            VALUES (?, ?, ?, ?, ?, ?, 1)
        """, (canonical_name, concept_type, confidence, importance, now, now))
        return int(cur.lastrowid or 0)

    def _upsert_alias(self, alias_text: str, concept_id: int, alias_type: str = 'surface', confidence: float = 0.5) -> None:
        cur = getattr(self.brain, 'cur', None)
        alias_text = self._normalize_text(alias_text)
        if cur is None or not alias_text or concept_id <= 0:
            return
        now = time.time()
        existing = cur.execute("""
            SELECT confidence
            FROM concept_aliases
            WHERE alias_text = ?
        """, (alias_text,)).fetchone()
        if existing:
            merged_confidence = self._merge_confidence(self._safe_float(existing[0], 0.5), confidence, 2)
            cur.execute("""
                UPDATE concept_aliases
                SET concept_id = ?, alias_type = ?, confidence = ?, last_seen = ?
                WHERE alias_text = ?
            """, (concept_id, alias_type, merged_confidence, now, alias_text))
            return
        cur.execute("""
            INSERT INTO concept_aliases (alias_text, concept_id, alias_type, confidence, first_seen, last_seen)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (alias_text, concept_id, alias_type, confidence, now, now))

    def _upsert_identity_trait(self, relation_type: str, trait_value: str, confidence: float, editable: str, now: float) -> None:
        cur = getattr(self.brain, 'cur', None)
        if cur is None:
            return
        trait_value = self._normalize_text(trait_value)
        if not trait_value:
            return
        trait_key = f'{relation_type}|{trait_value}'
        existing = cur.execute("""
            SELECT confidence, support_count, editable, source_count
            FROM identity_traits
            WHERE trait_key = ?
        """, (trait_key,)).fetchone()
        if existing:
            support_count = int(existing[1] or 0) + 1
            merged_confidence = self._merge_confidence(self._safe_float(existing[0], 0.5), confidence, support_count)
            merged_editable = self._merge_editability(existing[2], editable)
            source_count = int(existing[3] or 0) + 1
            cur.execute("""
                UPDATE identity_traits
                SET confidence = ?, editable = ?, support_count = ?, last_seen = ?, source_count = ?
                WHERE trait_key = ?
            """, (merged_confidence, merged_editable, support_count, now, source_count, trait_key))
            return
        cur.execute("""
            INSERT INTO identity_traits (
                trait_key, relation_type, trait_value, confidence, editable, support_count,
                contradiction_count, first_seen, last_seen, source_count
            )
            VALUES (?, ?, ?, ?, ?, 1, 0, ?, ?, 1)
        """, (trait_key, relation_type, trait_value, confidence, editable, now, now))

    def _has_evidence_label(self, fact_key: str, source_type: str, source_label: str) -> bool:
        cur = getattr(self.brain, 'cur', None)
        if cur is None or not fact_key:
            return False
        row = cur.execute("""
            SELECT 1
            FROM fact_evidence
            WHERE fact_key = ?
              AND source_type = ?
              AND source_label = ?
            LIMIT 1
        """, (fact_key, source_type, source_label)).fetchone()
        return bool(row)

    def _promote_fact_from_episodes(self, fact_key: str, evidence_count: int, window_bucket: int) -> bool:
        cur = getattr(self.brain, 'cur', None)
        if cur is None or not fact_key or evidence_count < 2:
            return False
        source_label = f'episodic_window:{window_bucket}'
        if self._has_evidence_label(fact_key, 'episodic_consolidation', source_label):
            return False

        row = cur.execute("""
            SELECT
                sc.canonical_name,
                fr.relation_type,
                oc.canonical_name,
                fr.object_text,
                fr.object_kind,
                fr.confidence,
                fr.editable
            FROM fact_records fr
            LEFT JOIN concept_nodes sc ON sc.concept_id = fr.subject_concept_id
            LEFT JOIN concept_nodes oc ON oc.concept_id = fr.object_concept_id
            WHERE fr.fact_key = ?
        """, (fact_key,)).fetchone()
        if not row:
            return False

        subject_name = row[0] or ''
        relation_type = row[1] or ''
        object_kind = row[4] or 'concept'
        object_value = row[2] if object_kind == 'concept' else row[3]
        if not subject_name or not relation_type or not object_value:
            return False

        confidence = min(0.96, max(self._safe_float(row[5], 0.5), self._safe_float(row[5], 0.5) + (0.04 * min(3, evidence_count - 1))))
        evidence_text = f"Repeated across {evidence_count} recent conversation episodes."
        result = self.add_fact(
            subject_name,
            relation_type,
            object_value,
            object_kind=object_kind,
            concept_type='agent' if subject_name in {'mai', 'current_user'} else 'concept',
            source_type='episodic_consolidation',
            source_label=source_label,
            confidence=confidence,
            editable=row[6] or 'medium',
            evidence_text=evidence_text,
        )
        return bool(result.get('success'))

    def _promote_candidate_from_episodes(self, candidate: dict[str, Any], evidence_count: int, window_bucket: int) -> bool:
        if not isinstance(candidate, dict) or evidence_count < 2:
            return False
        subject = str(candidate.get('subject', '') or '').strip()
        relation_type = str(candidate.get('relation_type', '') or '').strip()
        obj = str(candidate.get('object', '') or '').strip()
        object_kind = str(candidate.get('object_kind', 'concept') or 'concept').strip()
        if not subject or not relation_type or not obj:
            return False
        source_label = f'episodic_window:{window_bucket}'
        fact_key = self._fact_key(subject, relation_type, obj, object_kind)
        if self._has_evidence_label(fact_key, 'episodic_consolidation', source_label):
            return False

        base_confidence = self._safe_float(candidate.get('confidence'), 0.55)
        confidence = min(0.96, max(base_confidence, base_confidence + (0.04 * min(3, evidence_count - 1))))
        result = self.add_fact(
            subject,
            relation_type,
            obj,
            object_kind=object_kind,
            concept_type=candidate.get('concept_type', 'concept'),
            source_type='episodic_consolidation',
            source_label=source_label,
            confidence=confidence,
            editable='medium',
            evidence_text=f"Repeated across {evidence_count} recent conversation episodes.",
        )
        return bool(result.get('success'))

    def _consolidate_user_topic_preferences(self, recent_entries: list[dict[str, Any]], window_bucket: int) -> int:
        topic_counts: dict[str, int] = {}
        topic_quality: dict[str, float] = {}
        phrase_counts: dict[str, int] = {}
        phrase_quality: dict[str, float] = {}
        for entry in recent_entries:
            if not isinstance(entry, dict):
                continue
            quality = self._safe_float(entry.get('quality'), 0.5)
            user_text = self._normalize_text(str(entry.get('user_input', entry.get('user', '')) or ''))
            user_tokens = [token for token in user_text.split() if len(token) > 2 and token not in STOP_WORDS]
            seen_phrases = set()
            for width in (3, 2):
                if len(user_tokens) < width:
                    continue
                for idx in range(len(user_tokens) - width + 1):
                    phrase = " ".join(user_tokens[idx:idx + width]).strip()
                    if not phrase or phrase in seen_phrases:
                        continue
                    seen_phrases.add(phrase)
                    phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1
                    phrase_quality[phrase] = max(phrase_quality.get(phrase, 0.0), quality)
            for topic in entry.get('topics', [])[:4]:
                normalized_topic = self._normalize_text(str(topic or ''))
                if not normalized_topic or normalized_topic in {'mai', 'current_user'}:
                    continue
                topic_counts[normalized_topic] = topic_counts.get(normalized_topic, 0) + 1
                topic_quality[normalized_topic] = max(topic_quality.get(normalized_topic, 0.0), quality)

        promoted = 0
        repeated_phrase_topics = {topic for topic, count in phrase_counts.items() if count >= 2}
        for topic, count in sorted(phrase_counts.items(), key=lambda item: (-item[1], -len(item[0]), item[0]))[:4]:
            if count < 2:
                continue
            fact_key = self._fact_key('current_user', 'prefers', topic, 'concept')
            source_label = f'episodic_topic:{topic}:{window_bucket}'
            if self._has_evidence_label(fact_key, 'episodic_consolidation', source_label):
                continue
            confidence = min(0.92, 0.62 + (min(count, 4) * 0.07) + ((phrase_quality.get(topic, 0.5) - 0.5) * 0.18))
            result = self.add_fact(
                'current_user',
                'prefers',
                topic,
                object_kind='concept',
                concept_type='agent',
                source_type='episodic_consolidation',
                source_label=source_label,
                confidence=confidence,
                editable='high',
                evidence_text=f"Repeated recent conversation topic: {topic}.",
            )
            if result.get('success'):
                promoted += 1

        covered_tokens = {token for topic in repeated_phrase_topics for token in topic.split()}
        for topic, count in sorted(topic_counts.items(), key=lambda item: (-item[1], item[0]))[:4]:
            if count < 2:
                continue
            if topic in covered_tokens:
                continue
            fact_key = self._fact_key('current_user', 'prefers', topic, 'concept')
            source_label = f'episodic_topic:{topic}:{window_bucket}'
            if self._has_evidence_label(fact_key, 'episodic_consolidation', source_label):
                continue
            confidence = min(0.9, 0.58 + (min(count, 4) * 0.07) + ((topic_quality.get(topic, 0.5) - 0.5) * 0.18))
            result = self.add_fact(
                'current_user',
                'prefers',
                topic,
                object_kind='concept',
                concept_type='agent',
                source_type='episodic_consolidation',
                source_label=source_label,
                confidence=confidence,
                editable='high',
                evidence_text=f"Repeated recent conversation topic: {topic}.",
            )
            if result.get('success'):
                promoted += 1
        return promoted

    def _apply_fact_conflict_updates(self, subject_name: str, relation_type: str, object_text: str, object_kind: str, fact_key: str) -> None:
        cur = getattr(self.brain, 'cur', None)
        if cur is None or not subject_name or not relation_type or not object_text or not fact_key:
            return

        conflicting_keys = set()
        inverse_relation = INVERSE_RELATION_CONFLICTS.get(relation_type)
        if inverse_relation:
            rows = cur.execute("""
                SELECT fact_key
                FROM fact_records fr
                JOIN concept_nodes sc ON sc.concept_id = fr.subject_concept_id
                WHERE sc.canonical_name = ?
                  AND fr.relation_type = ?
                  AND fr.object_text = ?
                  AND fr.fact_key <> ?
            """, (subject_name, inverse_relation, object_text, fact_key)).fetchall()
            conflicting_keys.update(str(row[0] or '') for row in rows if row and row[0])

        if relation_type in SOFT_CONFLICT_RELATIONS:
            rows = cur.execute("""
                SELECT fact_key
                FROM fact_records fr
                JOIN concept_nodes sc ON sc.concept_id = fr.subject_concept_id
                WHERE sc.canonical_name = ?
                  AND fr.relation_type = ?
                  AND fr.object_kind = ?
                  AND fr.object_text <> ?
                  AND fr.fact_key <> ?
            """, (subject_name, relation_type, object_kind, object_text, fact_key)).fetchall()
            conflicting_keys.update(str(row[0] or '') for row in rows if row and row[0])

        conflicting_keys.discard(fact_key)
        if not conflicting_keys:
            return

        cur.execute("""
            UPDATE fact_records
            SET contradiction_count = contradiction_count + ?
            WHERE fact_key = ?
        """, (len(conflicting_keys), fact_key))
        cur.executemany("""
            UPDATE fact_records
            SET contradiction_count = contradiction_count + 1
            WHERE fact_key = ?
        """, [(conflicting_key,) for conflicting_key in conflicting_keys])

    def _split_sentences(self, text: str) -> list[str]:
        parts = re.split(r'(?<=[.!?])\s+|\n+', text)
        sentences = []
        for part in parts:
            clean = part.strip()
            if not clean:
                continue
            if len(clean.split()) > 28:
                continue
            sentences.append(clean)
        return sentences

    def _strip_hedges(self, sentence: str) -> tuple[str, float]:
        lowered = sentence.strip().lower()
        for prefix in HEDGE_PREFIXES:
            if lowered.startswith(prefix):
                return sentence.strip()[len(prefix):].strip(), 0.08
        return sentence.strip(), 0.0

    def _clean_phrase(self, value: str) -> str:
        value = (value or '').strip().strip(" \"'()[]{}")
        value = re.sub(r'\s+', ' ', value)
        value = value.rstrip('.,;:!')
        return value

    def _canonicalize_concept(self, text: str, speaker: str = 'external') -> str:
        normalized = self._normalize_text(text)
        if normalized in {'i', 'me', 'my', 'myself'}:
            if speaker == 'mai':
                return 'mai'
            if speaker == 'user':
                return 'current_user'
        if normalized in {'you', 'your', 'yourself'}:
            if speaker == 'mai':
                return 'current_user'
            if speaker == 'user':
                return 'mai'
        if normalized in {'mai', 'mai phoenix', 'phoenix'}:
            return 'mai'
        if normalized in {'user', 'the user'}:
            return 'current_user'
        tokens = [token for token in normalized.split() if token not in STOP_WORDS]
        return " ".join(tokens[:6]).strip()

    def _canonicalize_object(self, text: str, speaker: str = 'external', relation_type: str = '') -> str:
        normalized = self._normalize_text(text)
        normalized = re.sub(r'^(?:to|be|being|very)\s+', '', normalized)
        if relation_type in {'has_trait', 'role', 'capable_of', 'has_attribute'}:
            return " ".join(normalized.split()[:8]).strip()
        return self._canonicalize_concept(normalized, speaker=speaker)

    def _normalize_text(self, value: str) -> str:
        value = (value or '').strip().lower()
        value = re.sub(r'[\r\n\t]+', ' ', value)
        value = re.sub(r'[^a-z0-9_ -]+', ' ', value)
        value = re.sub(r'\s+', ' ', value)
        return value.strip()

    def _merge_confidence(self, old_confidence: float, new_confidence: float, support_count: int) -> float:
        if support_count <= 1:
            return min(0.98, max(0.05, new_confidence))
        weighted = ((old_confidence * max(1, support_count - 1)) + new_confidence) / support_count
        return min(0.98, max(0.05, weighted))

    def _merge_editability(self, old_value: str, new_value: str) -> str:
        order = {'low': 0, 'medium': 1, 'high': 2}
        old_rank = order.get(str(old_value or 'high').lower(), 2)
        new_rank = order.get(str(new_value or 'high').lower(), 2)
        return old_value if old_rank <= new_rank else new_value

    def _compute_fact_status(
        self,
        confidence: float,
        support_count: int,
        source_count: int,
        source_type: str,
        current_status: str = 'active',
    ) -> str:
        current_status = str(current_status or 'active')
        source_type = str(source_type or '')
        confidence = self._safe_float(confidence, 0.5)
        support_count = int(support_count or 0)
        source_count = int(source_count or 0)

        if current_status == 'stable':
            return 'stable'
        if source_type == 'conversation_bot' and confidence < 0.62 and support_count < 2:
            return 'provisional'
        if support_count >= 3:
            return 'stable'
        if support_count >= 2 and confidence >= 0.76:
            return 'stable'
        if source_count >= 3 and confidence >= 0.72:
            return 'stable'
        if current_status == 'provisional' and support_count >= 2 and confidence >= 0.65:
            return 'active'
        return 'active'

    def _detect_intent(self, query: str) -> str:
        normalized = self._normalize_text(query)
        if not normalized:
            return 'answer'
        if any(normalized.startswith(prefix) for prefix in ('hi', 'hello', 'hey', 'greetings')):
            return 'greeting'
        if normalized.startswith(('what did you just say', 'what did you say', 'summarize what you said', 'repeat what you said')):
            return 'explanation'
        if normalized.startswith(('why ', 'how ', 'explain ', 'describe ')):
            return 'explanation'
        if normalized.startswith(('can ', 'could ', 'should ', 'would ', 'help me ')):
            return 'guidance'
        if self._query_targets_mai(normalized) and self._query_requests_preferences(normalized):
            return 'self_preference'
        if normalized.startswith(('what is ', 'what are ', 'who is ', 'who are ')):
            if self._query_targets_mai(normalized):
                return 'self_description'
            return 'definition'
        if self._query_targets_mai(normalized):
            return 'self_description'
        return 'answer'

    def _query_targets_mai(self, normalized_query: str) -> bool:
        normalized_query = str(normalized_query or '').strip()
        if not normalized_query:
            return False
        tokens = set(normalized_query.split())
        if 'mai' in tokens or 'phoenix' in tokens:
            return True
        self_reference_phrases = (
            'who are you',
            'what are you',
            'what is you',
            'tell me about yourself',
            'describe yourself',
            'what makes you',
            'how are you different',
            'what kind of system are you',
            'what do you do',
            'what are your',
            'what is your',
        )
        return any(phrase in normalized_query for phrase in self_reference_phrases)

    def _choose_response_structure(self, intent: str, supporting_facts: list[dict[str, Any]]) -> str:
        if intent in {'definition', 'self_description'}:
            return 'definition_then_support'
        if intent == 'self_preference':
            return 'direct_answer_then_support'
        if intent == 'explanation':
            return 'explanation_then_support'
        if intent == 'guidance':
            return 'direct_answer_then_steps'
        if supporting_facts:
            return 'direct_answer_then_support'
        return 'direct_answer'

    def _choose_response_stance(self, intent: str, identity_traits: list[dict[str, Any]]) -> str:
        if intent == 'greeting':
            return 'warm'
        if intent in {'definition', 'explanation'}:
            return 'explanatory'
        if intent == 'guidance':
            return 'practical'
        if intent == 'self_preference':
            return 'self_descriptive'
        if intent == 'self_description' and identity_traits:
            return 'self_descriptive'
        return 'direct'

    def _build_lead_hint(self, intent: str, supporting_facts: list[dict[str, Any]], identity_traits: list[dict[str, Any]]) -> str:
        if intent == 'greeting':
            return 'Answer briefly and naturally.'
        if intent == 'guidance':
            return 'Start with the direct answer, then add one practical supporting point.'
        if intent == 'self_preference':
            return "Answer with Mai's preference directly, then add one short supporting detail if it is durable."
        if intent == 'self_description' and identity_traits:
            return 'Start with a direct self-description, then support it with one or two durable traits or facts.'
        if supporting_facts:
            return 'Start with a direct answer, then support it with one or two high-confidence facts.'
        return 'Answer directly and keep the structure clear.'

    def _fact_key(self, subject_name: str, relation_type: str, object_key: str, object_kind: str) -> str:
        return f"{subject_name}|{relation_type}|{object_kind}|{object_key}"

    def _summarize_fact(self, subject: str, relation_type: str, obj: str) -> str:
        subject_display = self._display_surface_text(subject, relation_type=relation_type, role='subject')
        object_display = self._display_surface_text(obj, relation_type=relation_type, role='object')
        templates = {
            'is_a': f"{subject_display} is a {object_display}",
            'has_trait': f"{subject_display} is {object_display}",
            'has_attribute': f"{subject_display} has {object_display}",
            'causes': f"{subject_display} causes {object_display}",
            'part_of': f"{subject_display} is part of {object_display}",
            'belongs_to': f"{subject_display} belongs to {object_display}",
            'role': f"{subject_display} serves as {object_display}",
            'prefers': f"{subject_display} prefers {object_display}",
            'avoids': f"{subject_display} avoids {object_display}",
            'capable_of': f"{subject_display} can {object_display}",
        }
        summary = templates.get(relation_type, f"{subject_display} {relation_type} {object_display}")
        if not summary.endswith('.'):
            summary += '.'
        return summary[:1].upper() + summary[1:]

    def _display_surface_text(self, value: str, relation_type: str = '', role: str = 'general') -> str:
        normalized = self._normalize_text(value)
        if not normalized:
            return ''
        if normalized == 'mai':
            return 'Mai'
        if normalized == 'current_user':
            return 'Current user'

        acronym_terms = {
            'ai',
            'api',
            'cpu',
            'gpu',
            'hsb',
            'http',
            'json',
            'sgm',
            'sql',
            'ui',
            'vr',
        }
        words = []
        for word in normalized.split():
            if word in acronym_terms:
                words.append(word.upper())
            else:
                words.append(word)
        text = " ".join(words)

        if role == 'object' and relation_type == 'part_of':
            if re.match(r'^[A-Z0-9][A-Z0-9 ]+ system$', text):
                return f"the {text}"
        return text

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _scalar(self, sql: str) -> int:
        cur = getattr(self.brain, 'cur', None)
        if cur is None:
            return 0
        row = cur.execute(sql).fetchone()
        return int(row[0] or 0) if row else 0
