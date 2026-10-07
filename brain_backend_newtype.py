"""
Bridge: use Newtype's high-speed storage as the pattern/association backend for Main.
Import from main app: set HybridBrain to use this backend for a faster, single-file (.hsb) experience.

Usage (once Main is refactored to use a backend):
  from brain_backend_newtype import NewtypeBrainBackend
  backend = NewtypeBrainBackend(base_path=".", hsb_file="mai_phoenix_brain.hsb")
  # Then pass backend to HybridBrain instead of raw SQLite.
"""

import os
import sys
import math
from collections import defaultdict

# Newtype: prefer package import, else run from Newtype/ or add to path
_here = os.path.dirname(os.path.abspath(__file__))
_newtype_dir = os.path.join(_here, "Newtype")
if _newtype_dir not in sys.path and os.path.isdir(_newtype_dir):
    sys.path.insert(0, _newtype_dir)
try:
    from Newtype.storage_engine import HighSpeedStorageEngine
    from Newtype.hsb_format import create_hsb_brain_from_data, read_hsb_brain
except ImportError:
    from storage_engine import HighSpeedStorageEngine  # type: ignore[reportMissingImports]
    from hsb_format import create_hsb_brain_from_data, read_hsb_brain  # type: ignore[reportMissingImports]

MAX_CONTEXT_SIZE = 8


class NewtypeBrainBackend:
    """
    Implements the core pattern/association storage used by Main's HybridBrain,
    using Newtype's HighSpeedStorageEngine and optional HSB persistence.
    """

    def __init__(self, base_path: str = ".", hsb_file: str = None, use_hsb_persistence: bool = True):
        self.base_path = os.path.abspath(base_path or ".")
        self.hsb_file = hsb_file or os.path.join(self.base_path, "mai_phoenix_brain.hsb")
        self.use_hsb_persistence = use_hsb_persistence
        self.engine = HighSpeedStorageEngine(self.base_path)
        self._word_associations_cache = None  # built on demand for coherence scoring
        self._load_from_hsb_if_exists()

    def _load_from_hsb_if_exists(self):
        if not self.use_hsb_persistence or not getattr(self, 'hsb_file', None) or not isinstance(self.hsb_file, str):
            return
        path = self.hsb_file.strip()
        if not path or not os.path.exists(path):
            return
        try:
            self.engine.load_from_disk(path)
        except Exception as e:
            print(f"Newtype backend: could not load HSB {path}: {e}")

    def save_to_hsb(self, path: str = None):
        """Persist current state to an HSB file."""
        path = path or self.hsb_file
        if not path or not isinstance(path, str):
            print("Newtype backend: save_to_hsb skipped (no path)")
            return
        try:
            self.engine.save_to_disk(path)
        except Exception as e:
            print(f"Newtype backend: could not save HSB {path}: {e}")

    # ---- Pattern lookup (replacement for _kn_next_probs / dynamic_word_chain) ----

    def get_next_word_probs(self, last_words: list, D: float = 0.75) -> dict:
        """
        Kneser–Ney style next-word probabilities from context.
        Returns dict: word -> probability (sums to 1).
        """
        try:
            D = float(D) if D is not None else 0.75
        except (TypeError, ValueError):
            D = 0.75
        if last_words is None:
            return self._unigram_probs()
        if not last_words:
            return self._unigram_probs()
        last_words = [str(w).strip() for w in last_words if w is not None and str(w).strip()]
        if not last_words:
            return self._unigram_probs()
        for L in range(min(MAX_CONTEXT_SIZE, len(last_words)), 0, -1):
            ctx = last_words[-L:]
            results = self.engine.query_patterns(ctx, L)
            if results:
                try:
                    total = sum(float(p) for _, p in results)
                except (TypeError, ValueError):
                    continue
                if total <= 0:
                    continue
                probs = {}
                for w, cnt in results:
                    try:
                        c = float(cnt)
                        key = str(w).strip() if w is not None else ""
                        probs[key] = max(c - D, 0) / total
                    except (TypeError, ValueError):
                        continue
                lam = (D * len(results)) / total
                lower = self.get_next_word_probs(last_words[-(L - 1) :], D) if L > 1 else self._unigram_probs()
                for w, q in lower.items():
                    probs[w] = probs.get(w, 0.0) + lam * q
                return probs
        return self._unigram_probs()

    def _unigram_probs(self) -> dict:
        """All next words and their total priority, normalized (from columnar next_words/priorities)."""
        col = getattr(self.engine, 'columnar', None)
        if col is None:
            return {}
        n = min(len(col.next_words), len(col.priorities))
        if n == 0:
            return {}
        d = defaultdict(int)
        for i in range(n):
            try:
                w = col.next_words[i]
                w = str(w) if w is not None else ""
                d[w] += int(col.priorities[i])
            except (TypeError, ValueError, KeyError):
                continue
        tot = sum(d.values()) or 1
        return {w: c / tot for w, c in d.items()}

    # ---- Association lookup (for coherence: word_associations[source][next] = strength) ----

    def get_word_associations_dict(self) -> dict:
        """
        Returns { source_word: { next_word: priority_or_strength } } for coherence scoring.
        Cached and rebuilt from engine columnar data when needed.
        """
        if self._word_associations_cache is not None:
            return self._word_associations_cache
        col = getattr(self.engine, 'columnar', None)
        if col is None:
            return {}
        n = min(len(col.source_words), len(col.assoc_next_words), len(col.assoc_priorities))
        d = defaultdict(lambda: defaultdict(int))
        for i in range(n):
            try:
                p = int(col.assoc_priorities[i])
            except (TypeError, ValueError):
                p = 0
            s = col.source_words[i]
            nw = col.assoc_next_words[i]
            s = str(s) if s is not None else ""
            nw = str(nw) if nw is not None else ""
            d[s][nw] += p
        self._word_associations_cache = {k: dict(v) for k, v in d.items()}
        return self._word_associations_cache

    def invalidate_word_associations_cache(self):
        self._word_associations_cache = None

    # ---- Stats and sample for UI ----
    def get_pattern_count(self) -> int:
        col = getattr(self.engine, 'columnar', None)
        if col is None:
            return 0
        nw = getattr(col, 'next_words', None)
        return len(nw) if nw is not None else 0

    def get_association_count(self) -> int:
        col = getattr(self.engine, 'columnar', None)
        if col is None:
            return 0
        sw = getattr(col, 'source_words', None)
        return len(sw) if sw is not None else 0

    def get_unique_next_words_count(self) -> int:
        col = getattr(self.engine, 'columnar', None)
        if col is None or not getattr(col, 'next_words', None):
            return 0
        try:
            return len(set(str(w) for w in col.next_words if w is not None and str(w).strip() != ""))
        except (TypeError, ValueError):
            return 0

    def get_patterns_sample(self, limit: int = 100):
        """Return list of (context_len, w1, w2, w3, w4, w5, w6, w7, w8, next_word, priority, success_rate, usage_count) sorted by priority desc."""
        try:
            limit = max(0, int(limit)) if limit is not None else 100
        except (TypeError, ValueError):
            limit = 100
        col = getattr(self.engine, 'columnar', None)
        if col is None:
            return []
        wc = getattr(col, 'word_columns', None)
        if not wc or not isinstance(wc, (list, tuple)) or len(wc) < 8:
            return []
        n = min(len(col.context_lens), len(col.next_words), *[len(wc[j]) for j in range(8)])
        if n == 0:
            return []
        success_rates = getattr(col, 'success_rates', None)
        usage_counts = getattr(col, 'usage_counts', None)
        rows = []
        for i in range(n):
            try:
                ctx_len = int(col.context_lens[i])
            except (TypeError, ValueError):
                ctx_len = 0
            words = [str(wc[j][i]) if i < len(wc[j]) and wc[j][i] is not None else "" for j in range(8)]
            next_w = col.next_words[i]
            next_w = str(next_w) if next_w is not None else ""
            try:
                pri = int(col.priorities[i])
            except (TypeError, ValueError):
                pri = 0
            try:
                sr = float(success_rates[i]) if success_rates is not None and i < len(success_rates) else 0.5
            except (TypeError, ValueError):
                sr = 0.5
            try:
                uc = int(usage_counts[i]) if usage_counts is not None and i < len(usage_counts) else 0
            except (TypeError, ValueError):
                uc = 0
            rows.append((ctx_len, *words, next_w, pri, sr, uc))
        rows.sort(key=lambda r: r[10], reverse=True)  # priority at index 10
        return rows[:limit]

    # ---- Writes (replacement for batch_operations chain/assoc and _flush_batch_operations) ----

    def add_pattern_batch(self, context_len: int, padded_context: tuple, next_word: str, priority: int, success_rate: float = 0.5, usage_count: int = 0):
        """Add one pattern. padded_context is (word1..word8) with <PAD> for unused slots."""
        try:
            padded_context = tuple(padded_context) if padded_context is not None and hasattr(padded_context, '__iter__') and not isinstance(padded_context, str) else ()
        except (TypeError, ValueError):
            padded_context = ()
        next_word = (next_word if next_word is not None else "") or ""
        context_len = max(0, min(8, int(context_len) if isinstance(context_len, (int, float)) else 0))
        words = tuple(str(w).strip() for w in padded_context if w is not None and str(w).strip() and str(w).strip() != "<PAD>")
        if len(words) > context_len:
            words = words[:context_len]
        self.engine.add_pattern(context_len, words, next_word, priority, success_rate, usage_count)
        self.invalidate_word_associations_cache()

    def add_association_batch(self, source_word: str, next_word: str, priority: int, success_rate: float = 0.5, usage_count: int = 0):
        source_word = (source_word if source_word is not None else "") or ""
        next_word = (next_word if next_word is not None else "") or ""
        self.engine.add_association(source_word, next_word, priority, success_rate, usage_count)
        self.invalidate_word_associations_cache()

    def flush(self):
        """No-op for in-memory engine; call save_to_hsb() to persist."""
        pass

    # ---- Optional: reinforce (priority update) ----

    def reinforce_pattern(self, context_len: int, padded_context: tuple, next_word: str, factor: float):
        """Increase effective priority for this pattern (e.g. by re-adding with higher priority)."""
        padded_context = padded_context or ()
        try:
            context_len = max(0, min(MAX_CONTEXT_SIZE, int(context_len)))
        except (TypeError, ValueError):
            return
        context_words = [w for w in padded_context[:context_len] if w and w != "<PAD>"]
        if len(context_words) != context_len:
            return
        if not context_words and context_len > 0:
            return
        next_word_str = str(next_word).strip() if next_word is not None else ""
        results = self.engine.query_patterns(context_words, context_len)
        for w, p in results:
            w_str = str(w).strip() if w is not None else ""
            if w_str == next_word_str:
                try:
                    new_pri = max(1, int(float(p) * factor))
                except (TypeError, ValueError):
                    new_pri = 1
                self.engine.add_pattern(
                    context_len,
                    tuple(context_words),
                    next_word,
                    new_pri,
                    0.5,
                    0,
                )
                self.invalidate_word_associations_cache()
                break

    # ---- IDF-style penalty (used by Main for scoring) ----

    def get_idf_penalty(self, word: str) -> float:
        """1.0 / (1.0 + log1p(total_priority for word))."""
        if word is None or not isinstance(word, str):
            return 1.0
        results = self.engine.query_associations(word)
        total = 0
        if results:
            for _, p in results:
                try:
                    total += float(p)
                except (TypeError, ValueError):
                    pass
        # Also count as next_word in patterns for full IDF
        col = getattr(self.engine, 'columnar', None)
        if col:
            nw = getattr(col, 'next_words', None)
            pr = getattr(col, 'priorities', None)
            if nw is not None and pr is not None:
                n = min(len(nw), len(pr))
                for i in range(n):
                    nw_str = str(nw[i]).strip() if nw[i] is not None else ""
                    if nw_str == (str(word).strip() if word else ""):
                        try:
                            total += int(pr[i])
                        except (TypeError, ValueError):
                            pass
        f = total or 1
        return 1.0 / (1.0 + math.log1p(f))


# Example: how Main would switch backends (after refactor)
#
# In mai_phoenix_desktop.py, HybridBrain.__init__ could do:
#
#   use_newtype = os.environ.get("MAI_USE_HSB", "").lower() in ("1", "true", "yes")
#   if use_newtype:
#       from brain_backend_newtype import NewtypeBrainBackend
#       self._storage = NewtypeBrainBackend(base_path=".", hsb_file=DB_FILE.replace(".db", ".hsb"))
#       self.cur = None
#       self.con = None
#   else:
#       self._storage = None
#       self.con = sqlite3.connect(db_file, ...)
#       self.cur = self.con.cursor()
#
# Then in _kn_next_probs: use self._storage.get_next_word_probs(last_words) if self._storage else current SQL.
# And in _flush_batch_operations: if self._storage, call self._storage.add_pattern_batch / add_association_batch; else current executemany.
