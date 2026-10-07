"""
Comprehensive Benchmark Tests for High-Speed Storage Engine
Compares performance against SQLite for AI pattern operations
"""

import time
import random
import sqlite3
import os
import tempfile
import statistics
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt
import numpy as np
from storage_engine import HighSpeedStorageEngine, PatternRecord, AssociationRecord

class SQLiteBenchmark:
    """SQLite benchmark for comparison"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        if not db_path:
            raise ValueError("SQLiteBenchmark: db_path cannot be empty")
        self.con = sqlite3.connect(db_path)
        self.cur = self.con.cursor()
        self.setup_database()
        
    def setup_database(self):
        """Setup SQLite database with same schema as original"""
        self.cur.execute('''CREATE TABLE IF NOT EXISTS dynamic_word_chain (
                                context_len INTEGER, word1 TEXT, word2 TEXT, word3 TEXT, word4 TEXT,
                                word5 TEXT, word6 TEXT, word7 TEXT, word8 TEXT, next_word TEXT,
                                priority INTEGER, success_rate REAL DEFAULT 0.5, usage_count INTEGER DEFAULT 0,
                                PRIMARY KEY (context_len, word1, word2, word3, word4, word5, word6, word7, word8, next_word)
                            )''')
        self.cur.execute('''CREATE TABLE IF NOT EXISTS word_associations (
                            source_word TEXT, next_word TEXT, priority INTEGER,
                            success_rate REAL DEFAULT 0.5, usage_count INTEGER DEFAULT 0,
                            PRIMARY KEY (source_word, next_word)
                        )''')
        
        # Create indexes
        self.cur.execute('''CREATE INDEX IF NOT EXISTS idx_dynamic_context_8 ON dynamic_word_chain(context_len, word1, word2, word3, word4, word5, word6, word7, word8) WHERE context_len = 8''')
        self.cur.execute('''CREATE INDEX IF NOT EXISTS idx_dynamic_context_4 ON dynamic_word_chain(context_len, word5, word6, word7, word8) WHERE context_len = 4''')
        self.cur.execute('''CREATE INDEX IF NOT EXISTS idx_associations_source ON word_associations(source_word)''')
        self.con.commit()
        
    def insert_pattern(self, context_len: int, words: Tuple[str, ...], 
                      next_word: str, priority: int, success_rate: float = 0.5, 
                      usage_count: int = 0):
        """Insert pattern into SQLite"""
        words = tuple(words) if words is not None and isinstance(words, (list, tuple)) else ()
        padded_words = list(words)[:8] + [''] * (8 - min(8, len(words)))
        self.cur.execute('''INSERT OR REPLACE INTO dynamic_word_chain 
                            (context_len, word1, word2, word3, word4, word5, word6, word7, word8, next_word, priority, success_rate, usage_count)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                        (context_len, *padded_words, next_word, priority, success_rate, usage_count))
        
    def insert_association(self, source_word: str, next_word: str, 
                          priority: int, success_rate: float = 0.5, 
                          usage_count: int = 0):
        """Insert association into SQLite"""
        self.cur.execute('''INSERT OR REPLACE INTO word_associations 
                            (source_word, next_word, priority, success_rate, usage_count)
                            VALUES (?, ?, ?, ?, ?)''',
                        (source_word, next_word, priority, success_rate, usage_count))
        
    def query_patterns(self, context_words: List[str], context_len: int) -> List[Tuple[str, int]]:
        """Query patterns from SQLite"""
        if context_words is None or not context_words or context_len is None or context_len <= 0:
            return []
        context_words = [str(w) if w is not None else '' for w in context_words[:8]]
        # Build WHERE clause
        conditions = ["context_len = ?"]
        params = [context_len]
        for i, word in enumerate(context_words):
            if i < 8:
                conditions.append(f"word{i+1} = ?")
                params.append(word)
                
        where_clause = " AND ".join(conditions)
        sql = f"SELECT next_word, SUM(priority) FROM dynamic_word_chain WHERE {where_clause} GROUP BY next_word"
        
        rows = self.cur.execute(sql, params).fetchall()
        return rows
        
    def query_associations(self, source_word: str) -> List[Tuple[str, int]]:
        """Query associations from SQLite"""
        if source_word is None or not isinstance(source_word, str):
            return []
        rows = self.cur.execute("SELECT next_word, priority FROM word_associations WHERE source_word = ?", 
                              (str(source_word),)).fetchall()
        return rows
        
    def batch_insert_patterns(self, patterns: List[Tuple]):
        """Batch insert patterns"""
        data = []
        for pattern_data in patterns:
            if not pattern_data or len(pattern_data) < 6:
                continue
            try:
                context_len, words, next_word, priority, success_rate, usage_count = pattern_data[:6]
                if words is None or not isinstance(words, (list, tuple)):
                    words = ()
                padded_words = list(words)[:8] + [''] * (8 - min(8, len(words)))
                data.append((context_len, *padded_words, next_word, priority, success_rate, usage_count))
            except (TypeError, ValueError):
                continue
            
        if data:
            self.cur.executemany('''INSERT OR REPLACE INTO dynamic_word_chain 
                                    (context_len, word1, word2, word3, word4, word5, word6, word7, word8, next_word, priority, success_rate, usage_count)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', data)
            try:
                self.con.commit()
            except Exception as e:
                print(f"Warning: batch_insert_patterns commit failed: {e}")
        
    def batch_insert_associations(self, associations: List[Tuple]):
        """Batch insert associations"""
        if not associations:
            return
        data = []
        for assoc in associations:
            if not assoc or len(assoc) < 5:
                continue
            try:
                data.append((assoc[0], assoc[1], assoc[2], assoc[3], assoc[4]))
            except (TypeError, IndexError):
                continue
        if data:
            try:
                self.cur.executemany('''INSERT OR REPLACE INTO word_associations 
                                        (source_word, next_word, priority, success_rate, usage_count)
                                        VALUES (?, ?, ?, ?, ?)''', data)
                self.con.commit()
            except Exception as e:
                print(f"Warning: batch_insert_associations commit failed: {e}")
        
    def close(self):
        """Close SQLite connection"""
        try:
            if self.con is not None:
                self.con.close()
        except Exception:
            pass

class BenchmarkSuite:
    """Comprehensive benchmark suite"""
    
    def __init__(self):
        self.results = {}
        
    def generate_test_data(self, num_patterns: int, num_associations: int) -> Tuple[List, List]:
        """Generate realistic test data"""
        if num_patterns < 0:
            num_patterns = 0
        if num_associations < 0:
            num_associations = 0
        words = ['the', 'and', 'is', 'in', 'to', 'of', 'a', 'that', 'it', 'with', 'for', 'as', 'was', 'on', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'man', 'men', 'put', 'say', 'she', 'too', 'use']
        patterns = []
        associations = []
        for _ in range(num_patterns):
            context_len = random.randint(1, 8)
            context_words = tuple(random.choices(words, k=context_len))
            next_word = random.choice(words)
            priority = random.randint(1, 100)
            success_rate = random.uniform(0.1, 1.0)
            usage_count = random.randint(1, 1000)
            
            patterns.append((context_len, context_words, next_word, priority, success_rate, usage_count))
            
        # Generate associations
        for _ in range(num_associations):
            source_word = random.choice(words)
            next_word = random.choice(words)
            priority = random.randint(1, 100)
            success_rate = random.uniform(0.1, 1.0)
            usage_count = random.randint(1, 1000)
            
            associations.append((source_word, next_word, priority, success_rate, usage_count))
            
        return patterns, associations
        
    def benchmark_insert_performance(self, num_patterns: int, num_associations: int, num_runs: int = 5):
        """Benchmark insert performance"""
        num_runs = max(1, int(num_runs))
        print(f"\n=== INSERT PERFORMANCE BENCHMARK ===")
        print(f"Patterns: {num_patterns:,}, Associations: {num_associations:,}, Runs: {num_runs}")
        
        patterns, associations = self.generate_test_data(num_patterns, num_associations)
        
        # SQLite benchmark
        sqlite_times = []
        for run in range(num_runs):
            tmp_name = None
            try:
                tmp = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
                tmp_name = tmp.name
                tmp.close()
            except OSError:
                continue
            try:
                sqlite_bench = SQLiteBenchmark(tmp_name)
                start_time = time.time()
                sqlite_bench.batch_insert_patterns(patterns)
                sqlite_bench.batch_insert_associations(associations)
                end_time = time.time()
                sqlite_times.append(end_time - start_time)
                sqlite_bench.close()
            finally:
                if tmp_name:
                    try:
                        os.unlink(tmp_name)
                    except OSError:
                        pass
                
        # High-speed engine benchmark
        engine_times = []
        for run in range(num_runs):
            try:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    engine = HighSpeedStorageEngine(tmp_dir)
                    start_time = time.time()
                    engine.batch_insert_patterns(patterns)
                    engine.batch_insert_associations(associations)
                    end_time = time.time()
                    engine_times.append(end_time - start_time)
            except Exception as e:
                print(f"Warning: engine run failed: {e}")
        # Calculate statistics
        sqlite_avg = statistics.mean(sqlite_times) if sqlite_times else 0.0
        sqlite_std = statistics.stdev(sqlite_times) if len(sqlite_times) >= 2 else 0.0
        engine_avg = statistics.mean(engine_times) if engine_times else 0.0
        engine_std = statistics.stdev(engine_times) if len(engine_times) >= 2 else 0.0
        speedup = sqlite_avg / engine_avg if engine_avg > 0 else 0.0
        print(f"SQLite:     {sqlite_avg:.4f}s ± {sqlite_std:.4f}s")
        print(f"Engine:     {engine_avg:.4f}s ± {engine_std:.4f}s")
        print(f"Speedup:    {speedup:.2f}x faster")
        self.results['insert'] = {
            'sqlite_avg': sqlite_avg,
            'sqlite_std': sqlite_std,
            'engine_avg': engine_avg,
            'engine_std': engine_std,
            'speedup': speedup
        }
        
    def benchmark_query_performance(self, num_patterns: int, num_associations: int, num_queries: int = 1000):
        """Benchmark query performance"""
        num_queries = max(1, int(num_queries))
        print(f"\n=== QUERY PERFORMANCE BENCHMARK ===")
        print(f"Patterns: {num_patterns:,}, Associations: {num_associations:,}, Queries: {num_queries:,}")
        
        patterns, associations = self.generate_test_data(num_patterns, num_associations)
        
        # Setup SQLite
        tmp_name = None
        try:
            tmp = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
            tmp_name = tmp.name
            tmp.close()
        except OSError as e:
            print(f"Warning: Could not create temp DB: {e}")
            return
        try:
            sqlite_bench = SQLiteBenchmark(tmp_name)
            sqlite_bench.batch_insert_patterns(patterns)
            sqlite_bench.batch_insert_associations(associations)
            query_patterns = []
            query_associations = []
            for _ in range(num_queries // 2):
                context_len = random.randint(1, 8)
                context_words = [random.choice(['the', 'and', 'is', 'in', 'to', 'of', 'a', 'that', 'it', 'with']) for _ in range(context_len)]
                query_patterns.append((context_words, context_len))
            for _ in range(num_queries // 2):
                source_word = random.choice(['the', 'and', 'is', 'in', 'to', 'of', 'a', 'that', 'it', 'with'])
                query_associations.append(source_word)
            sqlite_times = []
            for context_words, context_len in query_patterns:
                start_time = time.time()
                sqlite_bench.query_patterns(context_words or [], context_len)
                sqlite_times.append(time.time() - start_time)
            for source_word in query_associations:
                start_time = time.time()
                sqlite_bench.query_associations(source_word)
                sqlite_times.append(time.time() - start_time)
            sqlite_bench.close()
        finally:
            if tmp_name:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
        # Setup High-speed engine
        with tempfile.TemporaryDirectory() as tmp_dir:
            engine = HighSpeedStorageEngine(tmp_dir)
            engine.batch_insert_patterns(patterns)
            engine.batch_insert_associations(associations)
            
            # Engine queries
            engine_times = []
            for context_words, context_len in query_patterns:
                start_time = time.time()
                engine.query_patterns(context_words, context_len)
                engine_times.append(time.time() - start_time)
                
            for source_word in query_associations:
                start_time = time.time()
                engine.query_associations(source_word)
                engine_times.append(time.time() - start_time)
                
        # Calculate statistics
        sqlite_avg = statistics.mean(sqlite_times) if sqlite_times else 0.0
        sqlite_std = statistics.stdev(sqlite_times) if len(sqlite_times) >= 2 else 0.0
        engine_avg = statistics.mean(engine_times) if engine_times else 0.0
        engine_std = statistics.stdev(engine_times) if len(engine_times) >= 2 else 0.0
        speedup = sqlite_avg / engine_avg if engine_avg > 0 else 0.0
        print(f"SQLite:     {sqlite_avg*1000:.4f}ms ± {sqlite_std*1000:.4f}ms")
        print(f"Engine:     {engine_avg*1000:.4f}ms ± {engine_std*1000:.4f}ms")
        print(f"Speedup:    {speedup:.2f}x faster")
        self.results['query'] = {
            'sqlite_avg': sqlite_avg,
            'sqlite_std': sqlite_std,
            'engine_avg': engine_avg,
            'engine_std': engine_std,
            'speedup': speedup
        }
        
    def benchmark_concurrent_performance(self, num_patterns: int, num_associations: int, num_threads: int = 4):
        """Benchmark concurrent access performance"""
        print(f"\n=== CONCURRENT ACCESS BENCHMARK ===")
        print(f"Patterns: {num_patterns:,}, Associations: {num_associations:,}, Threads: {num_threads}")
        
        patterns, associations = self.generate_test_data(num_patterns, num_associations)
        
        # SQLite concurrent benchmark
        sqlite_times = []
        for _ in range(3):  # 3 runs
            tmp_name = None
            try:
                tmp = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
                tmp_name = tmp.name
                tmp.close()
            except OSError:
                continue
            try:
                sqlite_bench = SQLiteBenchmark(tmp_name)
                sqlite_bench.batch_insert_patterns(patterns)
                sqlite_bench.batch_insert_associations(associations)
                def sqlite_worker():
                    start_time = time.time()
                    for _ in range(100):
                        context_words = [random.choice(['the', 'and', 'is', 'in', 'to']) for _ in range(4)]
                        sqlite_bench.query_patterns(context_words, 4)
                    return time.time() - start_time
                start_time = time.time()
                with ThreadPoolExecutor(max_workers=num_threads) as executor:
                    futures = [executor.submit(sqlite_worker) for _ in range(num_threads)]
                    thread_times = [f.result() for f in futures]
                end_time = time.time()
                sqlite_times.append(end_time - start_time)
                sqlite_bench.close()
            finally:
                if tmp_name:
                    try:
                        os.unlink(tmp_name)
                    except OSError:
                        pass
                
        # High-speed engine concurrent benchmark
        engine_times = []
        for _ in range(3):  # 3 runs
            with tempfile.TemporaryDirectory() as tmp_dir:
                engine = HighSpeedStorageEngine(tmp_dir)
                engine.batch_insert_patterns(patterns)
                engine.batch_insert_associations(associations)
                
                def engine_worker():
                    start_time = time.time()
                    for _ in range(100):  # 100 queries per thread
                        context_words = [random.choice(['the', 'and', 'is', 'in', 'to']) for _ in range(4)]
                        engine.query_patterns(context_words, 4)
                    return time.time() - start_time
                
                start_time = time.time()
                with ThreadPoolExecutor(max_workers=num_threads) as executor:
                    futures = [executor.submit(engine_worker) for _ in range(num_threads)]
                    thread_times = [f.result() for f in futures]
                end_time = time.time()
                
                engine_times.append(end_time - start_time)
                
        # Calculate statistics
        sqlite_avg = statistics.mean(sqlite_times) if sqlite_times else 0.0
        sqlite_std = statistics.stdev(sqlite_times) if len(sqlite_times) >= 2 else 0.0
        engine_avg = statistics.mean(engine_times) if engine_times else 0.0
        engine_std = statistics.stdev(engine_times) if len(engine_times) >= 2 else 0.0
        speedup = sqlite_avg / engine_avg if engine_avg > 0 else 0.0
        print(f"SQLite:     {sqlite_avg:.4f}s ± {sqlite_std:.4f}s")
        print(f"Engine:     {engine_avg:.4f}s ± {engine_std:.4f}s")
        print(f"Speedup:    {speedup:.2f}x faster")
        self.results['concurrent'] = {
            'sqlite_avg': sqlite_avg,
            'sqlite_std': sqlite_std,
            'engine_avg': engine_avg,
            'engine_std': engine_std,
            'speedup': speedup
        }
        
    def benchmark_memory_usage(self, num_patterns: int, num_associations: int):
        """Benchmark memory usage"""
        print(f"\n=== MEMORY USAGE BENCHMARK ===")
        print(f"Patterns: {num_patterns:,}, Associations: {num_associations:,}")
        
        patterns, associations = self.generate_test_data(num_patterns, num_associations)
        
        import psutil
        import gc
        
        # SQLite memory usage
        try:
            process = psutil.Process()
        except Exception:
            print("Warning: psutil.Process() failed, skipping memory benchmark.")
            return
        gc.collect()
        try:
            sqlite_memory_before = process.memory_info().rss / 1024 / 1024  # MB
        except (AttributeError, OSError):
            sqlite_memory_before = 0.0
        tmp_name = None
        try:
            tmp = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
            tmp_name = tmp.name
            tmp.close()
        except OSError as e:
            print(f"Warning: Could not create temp DB for memory benchmark: {e}")
            return
        try:
            sqlite_bench = SQLiteBenchmark(tmp_name)
            sqlite_bench.batch_insert_patterns(patterns)
            sqlite_bench.batch_insert_associations(associations)
            sqlite_memory_after = process.memory_info().rss / 1024 / 1024  # MB
            sqlite_memory_used = sqlite_memory_after - sqlite_memory_before
            sqlite_bench.close()
        finally:
            if tmp_name:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
            
        # High-speed engine memory usage
        gc.collect()
        try:
            engine_memory_before = process.memory_info().rss / 1024 / 1024  # MB
        except (AttributeError, OSError):
            engine_memory_before = 0.0
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            engine = HighSpeedStorageEngine(tmp_dir)
            engine.batch_insert_patterns(patterns)
            engine.batch_insert_associations(associations)
            
            engine_memory_after = process.memory_info().rss / 1024 / 1024  # MB
            engine_memory_used = engine_memory_after - engine_memory_before
            
        print(f"SQLite:     {sqlite_memory_used:.2f} MB")
        print(f"Engine:     {engine_memory_used:.2f} MB")
        efficiency = sqlite_memory_used / engine_memory_used if engine_memory_used > 0 else 0.0
        print(f"Memory efficiency: {efficiency:.2f}x")
        
        self.results['memory'] = {
            'sqlite_mb': sqlite_memory_used,
            'engine_mb': engine_memory_used,
            'efficiency': efficiency
        }
        
    def generate_report(self):
        """Generate comprehensive benchmark report"""
        print(f"\n{'='*60}")
        print(f"COMPREHENSIVE BENCHMARK REPORT")
        print(f"{'='*60}")
        
        if 'insert' in self.results:
            insert = self.results['insert']
            print(f"\nINSERT PERFORMANCE:")
            print(f"  SQLite: {insert.get('sqlite_avg', 0):.4f}s ± {insert.get('sqlite_std', 0):.4f}s")
            print(f"  Engine: {insert.get('engine_avg', 0):.4f}s ± {insert.get('engine_std', 0):.4f}s")
            print(f"  Speedup: {insert.get('speedup', 0):.2f}x faster")
        if 'query' in self.results:
            query = self.results['query']
            print(f"\nQUERY PERFORMANCE:")
            print(f"  SQLite: {query.get('sqlite_avg', 0)*1000:.4f}ms ± {query.get('sqlite_std', 0)*1000:.4f}ms")
            print(f"  Engine: {query.get('engine_avg', 0)*1000:.4f}ms ± {query.get('engine_std', 0)*1000:.4f}ms")
            print(f"  Speedup: {query.get('speedup', 0):.2f}x faster")
        if 'concurrent' in self.results:
            concurrent = self.results['concurrent']
            print(f"\nCONCURRENT ACCESS:")
            print(f"  SQLite: {concurrent.get('sqlite_avg', 0):.4f}s ± {concurrent.get('sqlite_std', 0):.4f}s")
            print(f"  Engine: {concurrent.get('engine_avg', 0):.4f}s ± {concurrent.get('engine_std', 0):.4f}s")
            print(f"  Speedup: {concurrent.get('speedup', 0):.2f}x faster")
        if 'memory' in self.results:
            memory = self.results['memory']
            print(f"\nMEMORY USAGE:")
            print(f"  SQLite: {memory.get('sqlite_mb', 0):.2f} MB")
            print(f"  Engine: {memory.get('engine_mb', 0):.2f} MB")
            print(f"  Efficiency: {memory.get('efficiency', 0):.2f}x")
            
        # Overall assessment
        speedups = []
        if 'insert' in self.results:
            speedups.append(self.results['insert']['speedup'])
        if 'query' in self.results:
            speedups.append(self.results['query']['speedup'])
        if 'concurrent' in self.results:
            speedups.append(self.results['concurrent']['speedup'])
            
        if speedups:
            avg_speedup = statistics.mean(speedups)
            print(f"\nOVERALL PERFORMANCE:")
            print(f"  Average Speedup: {avg_speedup:.2f}x")
            print(f"  Performance Improvement: {((avg_speedup - 1) * 100):.1f}%")
            
    def run_full_benchmark_suite(self):
        """Run complete benchmark suite"""
        print("Starting comprehensive benchmark suite...")
        
        # Small dataset test
        print("\n" + "="*60)
        print("SMALL DATASET TEST (1K patterns, 500 associations)")
        print("="*60)
        self.benchmark_insert_performance(1000, 500, 3)
        self.benchmark_query_performance(1000, 500, 500)
        self.benchmark_concurrent_performance(1000, 500, 2)
        self.benchmark_memory_usage(1000, 500)
        
        # Medium dataset test
        print("\n" + "="*60)
        print("MEDIUM DATASET TEST (10K patterns, 5K associations)")
        print("="*60)
        self.benchmark_insert_performance(10000, 5000, 3)
        self.benchmark_query_performance(10000, 5000, 1000)
        self.benchmark_concurrent_performance(10000, 5000, 4)
        self.benchmark_memory_usage(10000, 5000)
        
        # Large dataset test
        print("\n" + "="*60)
        print("LARGE DATASET TEST (100K patterns, 50K associations)")
        print("="*60)
        self.benchmark_insert_performance(100000, 50000, 2)
        self.benchmark_query_performance(100000, 50000, 2000)
        self.benchmark_concurrent_performance(100000, 50000, 8)
        self.benchmark_memory_usage(100000, 50000)
        
        # Generate final report
        self.generate_report()

if __name__ == "__main__":
    benchmark = BenchmarkSuite()
    benchmark.run_full_benchmark_suite()
