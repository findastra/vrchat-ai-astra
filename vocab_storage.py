"""Compact vocabulary-ID storage for AI Astra n-gram context chains.

Repeated TEXT context tokens dominate SQLite B-tree size. This module keeps a
lexicon of integer word IDs and an ID-keyed chain table so new evidence can be
written compactly without dropping 2/4/6-word backoff behavior.

Modes (via HybridBrain.vocab_id_storage_mode):
- off: text table only (default for existing large brains)
- dual: write text + ID tables; prefer ID reads when a match exists
- ids: write ID table only (safe for fresh/empty brains after migration)
"""

from __future__ import annotations

import sqlite3
from typing import Any, Iterable


PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'
EMPTY_TOKEN = ''


class VocabIdStore:
    """Lexicon + integer-keyed chain table beside the legacy TEXT chain table."""

    def __init__(self, connection: sqlite3.Connection, *, context_size: int = 8):
        self.con = connection
        self.context_size = max(1, int(context_size))
        self._word_to_id: dict[str, int] = {}
        self._id_to_word: dict[int, str] = {}
        self.setup_schema()
        self.reload_cache()

    def setup_schema(self) -> None:
        cur = self.con.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS vocab_lexicon (
                word_id INTEGER PRIMARY KEY,
                word TEXT NOT NULL UNIQUE
            );
            CREATE TABLE IF NOT EXISTS dynamic_word_chain_ids (
                context_len INTEGER NOT NULL,
                id1 INTEGER NOT NULL,
                id2 INTEGER NOT NULL,
                id3 INTEGER NOT NULL,
                id4 INTEGER NOT NULL,
                id5 INTEGER NOT NULL,
                id6 INTEGER NOT NULL,
                id7 INTEGER NOT NULL,
                id8 INTEGER NOT NULL,
                next_id INTEGER NOT NULL,
                priority REAL NOT NULL DEFAULT 1,
                success_rate REAL NOT NULL DEFAULT 0.5,
                usage_count INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (context_len, id1, id2, id3, id4, id5, id6, id7, id8, next_id)
            );
            CREATE INDEX IF NOT EXISTS idx_chain_ids_ctx8
                ON dynamic_word_chain_ids(context_len, id1, id2, id3, id4, id5, id6, id7, id8)
                WHERE context_len = 8;
            CREATE INDEX IF NOT EXISTS idx_chain_ids_ctx4
                ON dynamic_word_chain_ids(context_len, id5, id6, id7, id8)
                WHERE context_len = 4;
            """
        )
        self.con.commit()
        self.ensure_word(UNK_TOKEN)
        self.ensure_word(PAD_TOKEN)
        self.ensure_word(EMPTY_TOKEN)

    def reload_cache(self) -> None:
        rows = self.con.execute('SELECT word_id, word FROM vocab_lexicon').fetchall()
        self._word_to_id = {str(word): int(word_id) for word_id, word in rows}
        self._id_to_word = {int(word_id): str(word) for word_id, word in rows}

    def ensure_word(self, word: str) -> int:
        token = str(word) if word is not None else EMPTY_TOKEN
        cached = self._word_to_id.get(token)
        if cached is not None:
            return cached
        cur = self.con.execute(
            'INSERT OR IGNORE INTO vocab_lexicon(word) VALUES (?)',
            (token,),
        )
        if cur.lastrowid:
            word_id = int(cur.lastrowid)
        else:
            word_id = int(
                self.con.execute(
                    'SELECT word_id FROM vocab_lexicon WHERE word=?',
                    (token,),
                ).fetchone()[0]
            )
        self._word_to_id[token] = word_id
        self._id_to_word[word_id] = token
        return word_id

    def encode_token(self, word: Any) -> int:
        return self.ensure_word(str(word) if word is not None else EMPTY_TOKEN)

    def decode_id(self, word_id: int) -> str:
        return self._id_to_word.get(int(word_id), UNK_TOKEN)

    def encode_context(self, words: Iterable[Any]) -> tuple[int, ...]:
        encoded = [self.encode_token(word) for word in list(words)[: self.context_size]]
        empty_id = self.encode_token(EMPTY_TOKEN)
        while len(encoded) < self.context_size:
            encoded.append(empty_id)
        return tuple(encoded)

    def encode_pattern_row(self, row: tuple[Any, ...]) -> tuple[Any, ...]:
        """Convert a TEXT chain row into an ID chain row."""
        context_len = int(row[0])
        context_words = row[1:1 + self.context_size]
        next_word = row[1 + self.context_size]
        priority = row[2 + self.context_size]
        success_rate = row[3 + self.context_size]
        usage_count = row[4 + self.context_size]
        encoded = self.encode_context(context_words)
        next_id = self.encode_token(next_word)
        return (context_len, *encoded, next_id, priority, success_rate, usage_count)

    def upsert_pattern_rows(self, pattern_data: list[tuple[Any, ...]]) -> int:
        if not pattern_data:
            return 0
        encoded_rows = [self.encode_pattern_row(row) for row in pattern_data]
        self.con.executemany(
            """
            INSERT INTO dynamic_word_chain_ids
            (context_len, id1, id2, id3, id4, id5, id6, id7, id8, next_id, priority, success_rate, usage_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(context_len, id1, id2, id3, id4, id5, id6, id7, id8, next_id) DO UPDATE SET
                priority = dynamic_word_chain_ids.priority + excluded.priority,
                success_rate = CASE
                    WHEN (dynamic_word_chain_ids.usage_count + excluded.usage_count) > 0 THEN
                        ((dynamic_word_chain_ids.success_rate * dynamic_word_chain_ids.usage_count)
                         + (excluded.success_rate * excluded.usage_count))
                        / (dynamic_word_chain_ids.usage_count + excluded.usage_count)
                    ELSE excluded.success_rate
                END,
                usage_count = dynamic_word_chain_ids.usage_count + excluded.usage_count
            """,
            encoded_rows,
        )
        return len(encoded_rows)

    def fetch_next_counts(self, context_len: int, full_context: tuple[Any, ...]) -> list[tuple[str, float]]:
        encoded = self.encode_context(full_context)
        rows = self.con.execute(
            """
            SELECT next_id, SUM(priority) AS cnt
            FROM dynamic_word_chain_ids
            WHERE context_len=? AND id1=? AND id2=? AND id3=? AND id4=?
              AND id5=? AND id6=? AND id7=? AND id8=?
            GROUP BY next_id
            """,
            (int(context_len), *encoded),
        ).fetchall()
        return [(self.decode_id(next_id), float(count or 0.0)) for next_id, count in rows]

    def stats(self) -> dict[str, int]:
        lexicon = self.con.execute('SELECT COUNT(*) FROM vocab_lexicon').fetchone()[0]
        chains = self.con.execute('SELECT COUNT(*) FROM dynamic_word_chain_ids').fetchone()[0]
        text_chains = 0
        try:
            text_chains = self.con.execute('SELECT COUNT(*) FROM dynamic_word_chain').fetchone()[0]
        except sqlite3.Error:
            text_chains = 0
        return {
            'lexicon_size': int(lexicon or 0),
            'id_chain_count': int(chains or 0),
            'text_chain_count': int(text_chains or 0),
        }

    def migrate_text_batch(self, *, limit: int = 5000, offset_rowid: int = 0) -> dict[str, Any]:
        """Copy a batch of TEXT chains into the ID table. Safe to re-run."""
        limit = max(1, int(limit))
        offset_rowid = max(0, int(offset_rowid))
        rows = self.con.execute(
            """
            SELECT rowid, context_len, word1, word2, word3, word4, word5, word6, word7, word8,
                   next_word, priority, success_rate, usage_count
            FROM dynamic_word_chain
            WHERE rowid > ?
            ORDER BY rowid
            LIMIT ?
            """,
            (offset_rowid, limit),
        ).fetchall()
        if not rows:
            return {
                'migrated': 0,
                'next_rowid': offset_rowid,
                'done': True,
                'stats': self.stats(),
            }
        pattern_data = [
            (
                context_len,
                word1,
                word2,
                word3,
                word4,
                word5,
                word6,
                word7,
                word8,
                next_word,
                priority,
                success_rate if success_rate is not None else 0.5,
                usage_count if usage_count is not None else 0,
            )
            for (
                _rowid,
                context_len,
                word1,
                word2,
                word3,
                word4,
                word5,
                word6,
                word7,
                word8,
                next_word,
                priority,
                success_rate,
                usage_count,
            ) in rows
        ]
        written = self.upsert_pattern_rows(pattern_data)
        self.con.commit()
        next_rowid = int(rows[-1][0])
        return {
            'migrated': written,
            'next_rowid': next_rowid,
            'done': len(rows) < limit,
            'stats': self.stats(),
        }
