"""
High-Performance Storage Engine for AI Pattern Data
Replaces SQLite with optimized memory-mapped and columnar storage
"""

import os
import mmap
import struct
import threading
import time
import json
import ast
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from dataclasses import dataclass
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import pickle
import zlib

@dataclass
class PatternRecord:
    """Optimized pattern record structure"""
    context_len: int
    words: Tuple[str, ...]  # Up to 8 words
    next_word: str
    priority: int
    success_rate: float
    usage_count: int

@dataclass
class AssociationRecord:
    """Optimized association record structure"""
    source_word: str
    next_word: str
    priority: int
    success_rate: float
    usage_count: int

class MemoryMappedIndex:
    """High-speed memory-mapped index for pattern lookups"""
    
    def __init__(self, file_path: str, record_size: int):
        self.file_path = file_path
        self.record_size = record_size
        self.file = None
        self.mmap = None
        self.index_map = {}  # Hash -> file offset mapping
        self.lock = threading.RLock()
        
    def initialize(self):
        """Initialize memory-mapped file"""
        if not os.path.exists(self.file_path):
            # Create file with initial size
            with open(self.file_path, 'wb') as f:
                f.write(b'\x00' * (1024 * 1024))  # 1MB initial size
        
        self.file = open(self.file_path, 'r+b')
        self.mmap = mmap.mmap(self.file.fileno(), 0)
        
    def get_hash(self, key: Tuple) -> int:
        """Generate hash for key"""
        try:
            return hash(key) & 0x7FFFFFFF  # Ensure positive
        except (TypeError, ValueError):
            return hash(str(key)) & 0x7FFFFFFF
        
    def find_offset(self, key: Tuple) -> Optional[int]:
        """Find file offset for key"""
        with self.lock:
            return self.index_map.get(self.get_hash(key))
            
    def add_record(self, key: Tuple, data: bytes) -> int:
        """Add record and return offset"""
        with self.lock:
            offset = len(self.mmap)
            self.mmap.resize(len(self.mmap) + len(data))
            self.mmap[offset:offset + len(data)] = data
            self.index_map[self.get_hash(key)] = offset
            return offset
            
    def get_record(self, key: Tuple) -> Optional[bytes]:
        """Get record data by key"""
        offset = self.find_offset(key)
        if offset is None:
            return None
        return self.mmap[offset:offset + self.record_size]
        
    def close(self):
        """Close memory-mapped file"""
        if self.mmap:
            self.mmap.close()
        if self.file:
            self.file.close()

class ColumnarStorage:
    """Columnar storage optimized for AI pattern queries"""
    
    def __init__(self, base_path: str):
        self.base_path = base_path or "."
        self.patterns_file = os.path.join(self.base_path, "patterns.mmap")
        self.associations_file = os.path.join(self.base_path, "associations.mmap")
        
        # Columnar data structures
        self.context_lens = np.array([], dtype=np.int32)
        self.word_columns = [np.array([], dtype='U50') for _ in range(8)]  # 8 word columns
        self.next_words = np.array([], dtype='U50')
        self.priorities = np.array([], dtype=np.int32)
        self.success_rates = np.array([], dtype=np.float32)
        self.usage_counts = np.array([], dtype=np.int32)
        
        # Association data
        self.source_words = np.array([], dtype='U50')
        self.assoc_next_words = np.array([], dtype='U50')
        self.assoc_priorities = np.array([], dtype=np.int32)
        self.assoc_success_rates = np.array([], dtype=np.float32)
        self.assoc_usage_counts = np.array([], dtype=np.int32)
        
        self.lock = threading.RLock()
        
    def add_pattern(self, pattern: PatternRecord):
        """Add pattern to columnar storage"""
        words = pattern.words if pattern.words is not None else ()
        with self.lock:
            self.context_lens = np.append(self.context_lens, pattern.context_len)
            self.next_words = np.append(self.next_words, pattern.next_word)
            self.priorities = np.append(self.priorities, pattern.priority)
            self.success_rates = np.append(self.success_rates, pattern.success_rate)
            self.usage_counts = np.append(self.usage_counts, pattern.usage_count)
            for i, word in enumerate(words):
                if i < 8:
                    self.word_columns[i] = np.append(self.word_columns[i], str(word) if word is not None else "")
                else:
                    break
            for i in range(len(words), 8):
                self.word_columns[i] = np.append(self.word_columns[i], "")
                
    def add_association(self, assoc: AssociationRecord):
        """Add association to columnar storage"""
        with self.lock:
            self.source_words = np.append(self.source_words, assoc.source_word)
            self.assoc_next_words = np.append(self.assoc_next_words, assoc.next_word)
            self.assoc_priorities = np.append(self.assoc_priorities, assoc.priority)
            self.assoc_success_rates = np.append(self.assoc_success_rates, assoc.success_rate)
            self.assoc_usage_counts = np.append(self.assoc_usage_counts, assoc.usage_count)
            
    def query_patterns(self, context_words: List[str], context_len: int) -> List[Tuple[str, int]]:
        """High-speed pattern query using vectorized operations"""
        if context_words is None or not context_words or context_len is None or context_len <= 0:
            return []
        try:
            context_words = [str(w) if w is not None else "" for w in list(context_words)[:8]]
        except (TypeError, ValueError):
            return []
        with self.lock:
            # Create boolean mask for context length
            length_mask = self.context_lens == context_len

            # Create boolean mask for word matches
            word_mask = np.ones(len(self.context_lens), dtype=bool)

            for i, word in enumerate(context_words):
                if i < 8:
                    word_mask &= (self.word_columns[i] == word)
                    
            # Combine masks
            final_mask = length_mask & word_mask
            
            if not np.any(final_mask):
                return []
                
            # Extract results
            next_words = self.next_words[final_mask]
            priorities = self.priorities[final_mask]
            
            return list(zip(next_words, priorities))
            
    def query_associations(self, source_word: str) -> List[Tuple[str, int]]:
        """High-speed association query"""
        if source_word is None or not isinstance(source_word, str):
            return []
        source_word = str(source_word)
        with self.lock:
            mask = self.source_words == source_word
            if not np.any(mask):
                return []
                
            next_words = self.assoc_next_words[mask]
            priorities = self.assoc_priorities[mask]
            
            return list(zip(next_words, priorities))

class LockFreeCache:
    """Lock-free cache for hot data"""
    
    def __init__(self, max_size: int = 100000):
        self.max_size = max_size
        self.cache = {}
        self.access_times = {}
        self.hit_count = 0
        self.miss_count = 0
        
    def get(self, key: Tuple) -> Optional[Any]:
        """Get value from cache"""
        if key in self.cache:
            self.access_times[key] = time.time()
            self.hit_count += 1
            return self.cache[key]
        self.miss_count += 1
        return None
        
    def put(self, key: Tuple, value: Any):
        """Put value in cache"""
        if len(self.cache) >= self.max_size and self.access_times:
            # Remove least recently used; only consider keys present in both to avoid KeyError
            common = [k for k in self.access_times if k in self.cache]
            if common:
                lru_key = min(common, key=self.access_times.get)
                self.cache.pop(lru_key, None)
                self.access_times.pop(lru_key, None)
            else:
                self.cache.clear()
                self.access_times.clear()
            
        self.cache[key] = value
        self.access_times[key] = time.time()
        
    def get_stats(self) -> Dict[str, float]:
        """Get cache statistics"""
        total = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total if total > 0 else 0
        return {
            'hit_rate': hit_rate,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'size': len(self.cache)
        }

class SemanticClusterStorage:
    """High-speed storage for semantic clusters and statistical relationships"""
    
    def __init__(self, base_path: str):
        self.base_path = base_path or "."
        self.clusters_file = os.path.join(self.base_path, 'semantic_clusters.json')
        
        # Semantic cluster data structures
        self.word_cooccurrence = {}
        self.clusters = {}
        self.word_to_cluster = {}
        self.cluster_strength = {}
        self.cluster_coherence = {}
        self.next_cluster_id = 0
        self.last_rebuild = 0
        self.rebuild_threshold = 500
        
        # Load existing clusters
        self.load_clusters()
        
    def update_cooccurrence(self, words, force_minimal=False):
        """Update word cooccurrence statistics"""
        if words is None:
            return
        if force_minimal:
            words = words[::5]
        meaningful_words = [w for w in words if isinstance(w, str) and len(w) > 2 and w.isalpha()]
        if len(meaningful_words) > 100:
            meaningful_words = meaningful_words[::2]
        
        for i, word1 in enumerate(meaningful_words):
            window_end = min(i + 11, len(meaningful_words))
            for j in range(i + 1, window_end):
                word2 = meaningful_words[j]
                key = tuple(sorted([word1, word2]))
                self.word_cooccurrence[key] = self.word_cooccurrence.get(key, 0) + 1
        
        if len(self.word_cooccurrence) - self.last_rebuild > self.rebuild_threshold:
            self.rebuild_clusters()
            self.last_rebuild = len(self.word_cooccurrence)
    
    def rebuild_clusters(self):
        """Rebuild semantic clusters from cooccurrence data"""
        print("Rebuilding semantic clusters...")
        
        # Simple clustering algorithm based on cooccurrence strength
        self.clusters = {}
        self.word_to_cluster = {}
        self.cluster_strength = {}
        self.cluster_coherence = {}
        self.next_cluster_id = 0
        
        # Sort word pairs by cooccurrence strength
        sorted_pairs = sorted(self.word_cooccurrence.items(), key=lambda x: x[1], reverse=True)
        
        for pair, strength in sorted_pairs:
            try:
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    continue
                word1, word2 = pair[0], pair[1]
            except (TypeError, ValueError, IndexError):
                continue
            # Find existing clusters for these words
            cluster1 = self.word_to_cluster.get(word1)
            cluster2 = self.word_to_cluster.get(word2)
            
            if cluster1 is None and cluster2 is None:
                # Create new cluster
                cluster_id = self.next_cluster_id
                self.next_cluster_id += 1
                self.clusters[cluster_id] = {word1, word2}
                self.word_to_cluster[word1] = cluster_id
                self.word_to_cluster[word2] = cluster_id
                self.cluster_strength[cluster_id] = strength
                self.cluster_coherence[cluster_id] = 1.0
                
            elif cluster1 is not None and cluster2 is None:
                # Add word2 to existing cluster
                self.clusters[cluster1].add(word2)
                self.word_to_cluster[word2] = cluster1
                self.cluster_strength[cluster1] += strength
                
            elif cluster1 is None and cluster2 is not None:
                # Add word1 to existing cluster
                self.clusters[cluster2].add(word1)
                self.word_to_cluster[word1] = cluster2
                self.cluster_strength[cluster2] += strength
                
            elif cluster1 != cluster2:
                # Merge clusters
                self._merge_clusters(cluster1, cluster2, strength)
    
    def _merge_clusters(self, cluster1, cluster2, strength):
        """Merge two clusters"""
        if cluster1 not in self.clusters or cluster2 not in self.clusters:
            return
        for word in self.clusters[cluster2]:
            self.word_to_cluster[word] = cluster1
            self.clusters[cluster1].add(word)
        
        # Update cluster strength
        self.cluster_strength[cluster1] += self.cluster_strength[cluster2] + strength
        
        # Remove cluster2
        del self.clusters[cluster2]
        del self.cluster_strength[cluster2]
        self.cluster_coherence.pop(cluster2, None)
    
    def get_cluster_context_score(self, candidate_word, conversation_context):
        """Get cluster-based context score"""
        if not conversation_context:
            return 1.0
        if not hasattr(conversation_context, '__iter__') or isinstance(conversation_context, str):
            return 1.0
        try:
            if candidate_word is None or candidate_word not in self.word_to_cluster:
                return 1.0
        except (TypeError, ValueError):
            return 1.0
        candidate_cluster = self.word_to_cluster[candidate_word]
        cluster_matches = 0
        for context_word in conversation_context:
            if context_word in self.word_to_cluster:
                if self.word_to_cluster[context_word] == candidate_cluster:
                    cluster_matches += 1
        
        return 1.0 + (cluster_matches * 0.3)
    
    def save_clusters(self):
        """Save clusters to disk"""
        if not getattr(self, 'clusters_file', None):
            return
        word_cooccurrence_serializable = {}
        for key, value in self.word_cooccurrence.items():
            try:
                word_cooccurrence_serializable[str(key)] = value
            except (TypeError, ValueError):
                continue
        clusters_ser = {}
        for k, v in self.clusters.items():
            try:
                clusters_ser[str(k)] = list(v) if isinstance(v, (list, set, tuple)) else [v]
            except (TypeError, ValueError):
                continue
        data = {
            'word_cooccurrence': word_cooccurrence_serializable,
            'clusters': clusters_ser,
            'word_to_cluster': self.word_to_cluster,
            'cluster_strength': self.cluster_strength,
            'cluster_coherence': self.cluster_coherence,
            'next_cluster_id': self.next_cluster_id
        }
        try:
            with open(self.clusters_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except (OSError, IOError, TypeError) as e:
            print(f"Error saving semantic clusters: {e}")
    
    def load_clusters(self):
        """Load clusters from disk"""
        clusters_file = getattr(self, 'clusters_file', None)
        if not clusters_file or not os.path.exists(clusters_file):
            return
        if not os.path.isfile(clusters_file):
            return
        try:
            with open(clusters_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if not isinstance(data, dict):
                data = {}

            self.word_cooccurrence = {}
            for key, value in data.get('word_cooccurrence', {}).items():
                try:
                    tup = ast.literal_eval(key)
                    if isinstance(tup, tuple) and len(tup) == 2:
                        self.word_cooccurrence[tup] = value
                except Exception:
                    continue
            self.clusters = {}
            for k, v in data.get('clusters', {}).items():
                try:
                    self.clusters[int(k)] = set(v) if isinstance(v, (list, set)) else {v}
                except (TypeError, ValueError):
                    continue
            raw_w2c = data.get('word_to_cluster', {})
            self.word_to_cluster = {}
            for k, v in raw_w2c.items():
                try:
                    self.word_to_cluster[k] = int(v) if not isinstance(v, int) else v
                except (TypeError, ValueError):
                    self.word_to_cluster[k] = v
            self.cluster_strength = {}
            for k, v in data.get('cluster_strength', {}).items():
                try:
                    self.cluster_strength[int(k)] = v
                except (TypeError, ValueError):
                    continue
            self.cluster_coherence = {}
            for k, v in data.get('cluster_coherence', {}).items():
                try:
                    self.cluster_coherence[int(k)] = v
                except (TypeError, ValueError):
                    continue
            try:
                raw_id = data.get('next_cluster_id', 0)
                self.next_cluster_id = int(raw_id) if raw_id is not None else 0
            except (TypeError, ValueError):
                self.next_cluster_id = 0
            print(f"Loaded {len(self.clusters)} semantic clusters from {clusters_file}")
        except Exception as e:
            print(f"Error loading semantic clusters: {e}")
            self.word_cooccurrence = {}
            self.clusters = {}
            self.word_to_cluster = {}
            self.cluster_strength = {}
            self.cluster_coherence = {}
            self.next_cluster_id = 0

class HighSpeedStorageEngine:
    """Main high-speed storage engine"""
    
    def __init__(self, base_path: str):
        self.base_path = base_path or "."
        try:
            os.makedirs(self.base_path, exist_ok=True)
        except OSError as e:
            print(f"Warning: Could not create base path {self.base_path}: {e}")
        # Initialize components
        self.columnar = ColumnarStorage(self.base_path)
        self.semantic_clusters = SemanticClusterStorage(self.base_path)
        self.pattern_cache = LockFreeCache(max_size=50000)
        self.assoc_cache = LockFreeCache(max_size=25000)
        
        # Performance tracking
        self.query_count = 0
        self.query_time = 0
        self.insert_count = 0
        self.insert_time = 0
        
    def add_pattern(self, context_len: int, words: Tuple[str, ...], 
                   next_word: str, priority: int, success_rate: float = 0.5, 
                   usage_count: int = 0):
        """Add pattern with high performance"""
        if words is None:
            words = ()
        next_word = (next_word if next_word is not None else "") or ""
        start_time = time.time()
        pattern = PatternRecord(
            context_len=context_len,
            words=words,
            next_word=next_word,
            priority=priority,
            success_rate=success_rate,
            usage_count=usage_count
        )
        
        self.columnar.add_pattern(pattern)
        
        # Cache hot data
        cache_key = (context_len,) + words
        self.pattern_cache.put(cache_key, (next_word, priority))
        
        self.insert_count += 1
        self.insert_time += time.time() - start_time
        
    def add_association(self, source_word: str, next_word: str, 
                       priority: int, success_rate: float = 0.5, 
                       usage_count: int = 0):
        """Add association with high performance"""
        source_word = (source_word if source_word is not None else "") or ""
        next_word = (next_word if next_word is not None else "") or ""
        start_time = time.time()
        
        assoc = AssociationRecord(
            source_word=source_word,
            next_word=next_word,
            priority=priority,
            success_rate=success_rate,
            usage_count=usage_count
        )
        
        self.columnar.add_association(assoc)
        
        # Cache hot data
        self.assoc_cache.put((source_word,), (next_word, priority))
        
        self.insert_count += 1
        self.insert_time += time.time() - start_time
        
    def query_patterns(self, context_words: List[str], context_len: int) -> List[Tuple[str, int]]:
        """High-speed pattern query"""
        start_time = time.time()
        if context_words is None:
            context_words = []
        try:
            context_words = list(context_words)[:8] if hasattr(context_words, '__iter__') and not isinstance(context_words, str) else []
        except (TypeError, ValueError):
            context_words = []
        # Check cache first
        cache_key = (context_len,) + tuple(context_words[:context_len]) if context_words else (context_len,)
        cached = self.pattern_cache.get(cache_key)
        if cached:
            self.query_count += 1
            self.query_time += time.time() - start_time
            return [cached]
            
        # Query columnar storage
        results = self.columnar.query_patterns(context_words, context_len)
        
        # Cache results
        if results:
            self.pattern_cache.put(cache_key, results[0])
            
        self.query_count += 1
        self.query_time += time.time() - start_time
        return results
        
    def query_associations(self, source_word: str) -> List[Tuple[str, int]]:
        """High-speed association query"""
        start_time = time.time()
        if source_word is None:
            self.query_count += 1
            self.query_time += time.time() - start_time
            return []
        # Check cache first
        cached = self.assoc_cache.get((source_word,))
        if cached:
            self.query_count += 1
            self.query_time += time.time() - start_time
            return [cached]
            
        # Query columnar storage
        results = self.columnar.query_associations(source_word)
        
        # Cache results
        if results:
            self.assoc_cache.put((source_word,), results[0])
            
        self.query_count += 1
        self.query_time += time.time() - start_time
        return results
        
    def batch_insert_patterns(self, patterns: List[Tuple]):
        """Batch insert patterns for maximum performance"""
        patterns = list(patterns) if patterns is not None else []
        if not patterns:
            return
        print(f"Storage engine: Starting batch insert of {len(patterns)} patterns")
        start_time = time.time()
        for i, pattern_data in enumerate(patterns):
            if i % 1000 == 0:
                print(f"Storage engine: Inserted {i}/{len(patterns)} patterns")
            try:
                if not pattern_data or len(pattern_data) < 6:
                    continue
                context_len, words, next_word, priority, success_rate, usage_count = pattern_data[0], pattern_data[1], pattern_data[2], pattern_data[3], pattern_data[4], pattern_data[5]
                self.add_pattern(context_len, words, next_word, priority, success_rate, usage_count)
            except (TypeError, ValueError, IndexError):
                continue
            
        self.insert_time += time.time() - start_time
        print(f"Storage engine: Completed batch insert of {len(patterns)} patterns")
        
    def batch_insert_associations(self, associations: List[Tuple]):
        """Batch insert associations for maximum performance"""
        associations = list(associations) if associations is not None else []
        if not associations:
            return
        print(f"Storage engine: Starting batch insert of {len(associations)} associations")
        start_time = time.time()
        for i, assoc_data in enumerate(associations):
            if i % 1000 == 0:
                print(f"Storage engine: Inserted {i}/{len(associations)} associations")
            try:
                if not assoc_data or len(assoc_data) < 5:
                    continue
                source_word, next_word, priority, success_rate, usage_count = assoc_data[0], assoc_data[1], assoc_data[2], assoc_data[3], assoc_data[4]
                self.add_association(source_word, next_word, priority, success_rate, usage_count)
            except (TypeError, ValueError, IndexError):
                continue
            
        self.insert_time += time.time() - start_time
        print(f"Storage engine: Completed batch insert of {len(associations)} associations")
        
    def _safe_cache_stats(self, cache) -> Dict[str, Any]:
        """Return cache get_stats() or empty dict if missing/fails."""
        if cache is None:
            return {}
        try:
            return cache.get_stats() if hasattr(cache, 'get_stats') else {}
        except Exception:
            return {}

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        avg_query_time = self.query_time / self.query_count if self.query_count > 0 else 0
        avg_insert_time = self.insert_time / self.insert_count if self.insert_count > 0 else 0
        
        return {
            'query_stats': {
                'count': self.query_count,
                'total_time': self.query_time,
                'avg_time': avg_query_time,
                'queries_per_second': 1.0 / avg_query_time if avg_query_time > 0 else 0
            },
            'insert_stats': {
                'count': self.insert_count,
                'total_time': self.insert_time,
                'avg_time': avg_insert_time,
                'inserts_per_second': 1.0 / avg_insert_time if avg_insert_time > 0 else 0
            },
            'cache_stats': {
                'pattern_cache': self._safe_cache_stats(getattr(self, 'pattern_cache', None)),
                'assoc_cache': self._safe_cache_stats(getattr(self, 'assoc_cache', None))
            }
        }
        
    def save_to_disk(self, file_path: str = None):
        """Save data to disk using HSB format"""
        if file_path is None:
            base = getattr(self, 'base_path', None) or '.'
            file_path = os.path.join(base, 'mai_phoenix_brain.hsb')
        if not file_path or not isinstance(file_path, str):
            raise ValueError("save_to_disk: file_path must be a non-empty string")
        print(f"Storage engine: Starting save to disk: {file_path}")
        
        # Convert columnar data to patterns and associations
        print("Converting columnar data to patterns...")
        patterns = []
        associations = []
        
        # Convert patterns
        n_patterns = len(self.columnar.context_lens)
        for i in range(n_patterns):
            try:
                context_len = int(self.columnar.context_lens[i])
                context_len = max(0, min(8, context_len))
            except (TypeError, ValueError, IndexError):
                continue
            try:
                words_list = [self.columnar.word_columns[j][i] for j in range(min(8, context_len)) if j < len(self.columnar.word_columns)]
                while words_list and words_list[-1] == "":
                    words_list.pop()
                words = tuple(words_list)
                next_word = self.columnar.next_words[i] if i < len(self.columnar.next_words) else ""
                priority = int(self.columnar.priorities[i]) if i < len(self.columnar.priorities) else 0
                success_rate = float(self.columnar.success_rates[i]) if i < len(self.columnar.success_rates) else 0.5
                usage_count = int(self.columnar.usage_counts[i]) if i < len(self.columnar.usage_counts) else 0
            except (TypeError, ValueError, IndexError):
                continue
            
            patterns.append((context_len, words, next_word, priority, success_rate, usage_count))
            
        print(f"Converted {len(patterns)} patterns")
        
        # Convert associations
        print("Converting columnar data to associations...")
        n_assoc = len(self.columnar.source_words)
        for i in range(n_assoc):
            try:
                source_word = self.columnar.source_words[i] if i < len(self.columnar.source_words) else ""
                next_word = self.columnar.assoc_next_words[i] if i < len(self.columnar.assoc_next_words) else ""
                priority = int(self.columnar.assoc_priorities[i]) if i < len(self.columnar.assoc_priorities) else 0
                success_rate = float(self.columnar.assoc_success_rates[i]) if i < len(self.columnar.assoc_success_rates) else 0.5
                usage_count = int(self.columnar.assoc_usage_counts[i]) if i < len(self.columnar.assoc_usage_counts) else 0
                associations.append((source_word, next_word, priority, success_rate, usage_count))
            except (TypeError, ValueError, IndexError):
                continue
            
        print(f"Converted {len(associations)} associations")
        
        # Prepare semantic clusters data
        print("Preparing semantic clusters data...")
        clusters_data = {
            'word_cooccurrence': {str(k): v for k, v in self.semantic_clusters.word_cooccurrence.items()},
            'clusters': {str(k): list(v) for k, v in self.semantic_clusters.clusters.items()},
            'word_to_cluster': self.semantic_clusters.word_to_cluster,
            'cluster_strength': self.semantic_clusters.cluster_strength,
            'cluster_coherence': self.semantic_clusters.cluster_coherence,
            'next_cluster_id': self.semantic_clusters.next_cluster_id
        }
        
        print("Creating HSB brain file...")
        # Create HSB brain file
        from hsb_format import create_hsb_brain_from_data
        create_hsb_brain_from_data(file_path, patterns, associations, clusters_data)
        
        print(f"Saved HSB brain: {file_path}")
            
    def load_from_disk(self, file_path: str = None):
        """Load data from disk using HSB format"""
        if file_path is None:
            file_path = os.path.join(self.base_path, 'mai_phoenix_brain.hsb')
            
        if not os.path.exists(file_path):
            print(f"No HSB brain file found: {file_path}")
            return
            
        # Load HSB brain file
        from hsb_format import read_hsb_brain
        reader = read_hsb_brain(file_path)
        
        # Load patterns and convert to columnar format
        patterns = reader.get_patterns() if reader else []
        if not isinstance(patterns, list):
            patterns = []
        for pattern in patterns:
            try:
                if not pattern or len(pattern) < 6:
                    continue
                context_len, words, next_word, priority, success_rate, usage_count = pattern[0], pattern[1], pattern[2], pattern[3], pattern[4], pattern[5]
                words = words if words is not None else ()
                self.add_pattern(context_len, words, next_word, priority, success_rate, usage_count)
            except (TypeError, ValueError, IndexError):
                continue
        # Load associations and convert to columnar format
        associations = reader.get_associations() if reader else []
        if not isinstance(associations, list):
            associations = []
        for assoc in associations:
            try:
                if not assoc or len(assoc) < 5:
                    continue
                source_word, next_word, priority = assoc[0], assoc[1], assoc[2]
                success_rate = assoc[3] if len(assoc) > 3 else 0.5
                usage_count = assoc[4] if len(assoc) > 4 else 0
                self.add_association(source_word, next_word, priority, success_rate, usage_count)
            except (TypeError, ValueError, IndexError):
                continue
            
        # Load semantic clusters
        clusters_data = reader.get_semantic_clusters() if reader else {}
        if clusters_data and isinstance(clusters_data, dict):
            # Restore semantic cluster data
            self.semantic_clusters.word_cooccurrence = {}
            for key, value in clusters_data.get('word_cooccurrence', {}).items():
                try:
                    tup = ast.literal_eval(key)
                    if isinstance(tup, tuple) and len(tup) == 2:
                        self.semantic_clusters.word_cooccurrence[tup] = value
                except Exception:
                    continue
                    
            self.semantic_clusters.clusters = {int(k): set(v) for k, v in clusters_data.get('clusters', {}).items()}
            self.semantic_clusters.word_to_cluster = clusters_data.get('word_to_cluster', {})
            self.semantic_clusters.cluster_strength = {int(k): v for k, v in clusters_data.get('cluster_strength', {}).items()}
            self.semantic_clusters.cluster_coherence = {int(k): v for k, v in clusters_data.get('cluster_coherence', {}).items()}
            self.semantic_clusters.next_cluster_id = clusters_data.get('next_cluster_id', 0)
            
        reader.close()
        print(f"Loaded HSB brain: {file_path}")
        
    def update_semantic_cooccurrence(self, words, force_minimal=False):
        """Update semantic cooccurrence statistics"""
        self.semantic_clusters.update_cooccurrence(words, force_minimal)
        
    def get_cluster_context_score(self, candidate_word, conversation_context):
        """Get cluster-based context score"""
        return self.semantic_clusters.get_cluster_context_score(candidate_word, conversation_context)
        
    def get_semantic_cluster_stats(self):
        """Get semantic cluster statistics"""
        return {
            'total_clusters': len(self.semantic_clusters.clusters),
            'total_words': len(self.semantic_clusters.word_to_cluster),
            'cooccurrence_pairs': len(self.semantic_clusters.word_cooccurrence),
            'next_cluster_id': self.semantic_clusters.next_cluster_id
        }
