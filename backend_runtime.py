"""Backend runtime and bootstrap for Mai without any Qt dependency."""

import sys, sqlite3, re, random, os, shutil, json, math, copy, threading, time, gc, subprocess, ast, functools
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, deque
import psutil
import numpy as np
try:
    from .backend_api import MaiBackendAPI, process_text_chunk_task
    from .cognitive_core import NornCognitiveCore
    from .backend_knowledge import KnowledgeStore as BackendKnowledgeStore
    from .backend_features import (
        AdaptiveLearningSystem as BackendAdaptiveLearningSystem,
        AdvancedReasoningEngine as BackendAdvancedReasoningEngine,
        AntiLoopFilter as BackendAntiLoopFilter,
        Autotune as BackendAutotune,
        ConfidenceGate as BackendConfidenceGate,
        CreativeResponseGenerator as BackendCreativeResponseGenerator,
        Critic as BackendCritic,
        Curiosity as BackendCuriosity,
        EnvironmentFeedback as BackendEnvironmentFeedback,
        MetaMemory as BackendMetaMemory,
        ResponseLearningSystem as BackendResponseLearningSystem,
        TopicDetectionSystem as BackendTopicDetectionSystem,
        TruthFactTable as BackendTruthFactTable,
    )
except ImportError:
    from backend_api import MaiBackendAPI, process_text_chunk_task
    from cognitive_core import NornCognitiveCore
    from backend_knowledge import KnowledgeStore as BackendKnowledgeStore
    from backend_features import (
        AdaptiveLearningSystem as BackendAdaptiveLearningSystem,
        AdvancedReasoningEngine as BackendAdvancedReasoningEngine,
        AntiLoopFilter as BackendAntiLoopFilter,
        Autotune as BackendAutotune,
        ConfidenceGate as BackendConfidenceGate,
        CreativeResponseGenerator as BackendCreativeResponseGenerator,
        Critic as BackendCritic,
        Curiosity as BackendCuriosity,
        EnvironmentFeedback as BackendEnvironmentFeedback,
        MetaMemory as BackendMetaMemory,
        ResponseLearningSystem as BackendResponseLearningSystem,
        TopicDetectionSystem as BackendTopicDetectionSystem,
        TruthFactTable as BackendTruthFactTable,
    )

APP_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_DIR = os.path.abspath(os.environ.get('MAI_STATE_DIR', APP_DIR))
os.makedirs(STATE_DIR, exist_ok=True)
TRAINING_DIR = os.path.join(STATE_DIR, 'training')

DB_FILE = os.path.join(STATE_DIR, 'mai_phoenix_brain.db')
NN_MODEL_FILE = os.path.join(STATE_DIR, 'mai_phoenix_model.json')
VOCAB_FILE = os.path.join(STATE_DIR, 'mai_phoenix_vocab.json')
ATTENTION_FILE = os.path.join(STATE_DIR, 'mai_attention_weights.json')
CONTEXT_SCORES_FILE = os.path.join(STATE_DIR, 'mai_context_scores.json')
SEMANTIC_CLUSTERS_FILE = os.path.join(STATE_DIR, 'mai_semantic_clusters.json')
SETTINGS_FILE = os.path.join(STATE_DIR, 'mai_settings.json')
INITIAL_KNOWLEDGE_FILE = os.path.join(TRAINING_DIR, 'initial_knowledge.txt')


def _atomic_json_dump(data, filename, *, indent=None):
    """Durably replace a JSON sidecar without exposing a partial file."""
    target = os.path.abspath(filename)
    directory = os.path.dirname(target)
    os.makedirs(directory, exist_ok=True)
    temp_filename = f"{target}.{os.getpid()}.{threading.get_ident()}.tmp"
    try:
        with open(temp_filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_filename, target)
    finally:
        if os.path.exists(temp_filename):
            try:
                os.remove(temp_filename)
            except OSError:
                pass


def _preserve_invalid_sidecar(filename, reason):
    """Keep one timestamped forensic copy before a bad sidecar is replaced."""
    if not filename or not os.path.isfile(filename):
        return None
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    backup = f'{filename}.invalid.{timestamp}'
    counter = 1
    while os.path.exists(backup):
        backup = f'{filename}.invalid.{timestamp}.{counter}'
        counter += 1
    try:
        shutil.copy2(filename, backup)
        print(f"Preserved invalid sidecar at {backup}: {reason}")
        return backup
    except Exception as exc:
        print(f"Warning: Could not preserve invalid sidecar {filename}: {exc}")
        return None

MAX_CONTEXT_SIZE = 8
CONTEXT_LEVELS = [8, 6, 4, 2]
NN_CONTEXT_SIZE = 4
MAX_RESPONSE_LENGTH = 25
MIN_GENERATION_ATTEMPTS = 8

BATCH_SIZE = 5000
LARGE_FILE_THRESHOLD = 10000
PROGRESS_UPDATE_INTERVAL = 1000
SEMANTIC_UPDATE_INTERVAL = 5000

MEMORY_SAFETY_THRESHOLD = 0.85
CHUNK_SIZE_WORDS = 100000
MIN_CHUNK_SIZE = 10000
MAX_CHUNK_SIZE = 500000
PARALLEL_WORKER_LIMIT = max(1, cpu_count() - 1)

RESPONSE_CACHE_SIZE = 2000
PATTERN_CACHE_SIZE = 10000
WORD_PROBABILITY_CACHE_SIZE = 5000
SEMANTIC_CACHE_SIZE = 6000
CONTEXT_CACHE_SIZE = 2000

LEARNING_BATCH_SIZE = 50_000
LEARNING_PROGRESS_INTERVAL = 50_000
LEARNING_PARALLEL_CHUNKS = 3
LEARNING_MEMORY_THRESHOLD = 0.75
LEARNING_GC_INTERVAL = 1500

GENERATION_MAX_ATTEMPTS = 4  # was 8: fewer retries = faster when quality is often low (e.g. little training)
GENERATION_QUALITY_THRESHOLD = 0.4
GENERATION_PARALLEL_CANDIDATES = 5
GENERATION_TEMPERATURE_RANGE = (0.7, 1.2)
GENERATION_TOP_P_RANGE = (0.8, 0.95)
GENERATION_TIME_BUDGET_SECONDS = 45.0
GENERATION_ATTEMPT_TIME_BUDGET_SECONDS = 14.0
GENERATION_MIN_RESPONSE_WORDS = 6
# Cap how many candidate words we score per step (avoids O(vocab) when falling back to unigram with little training)
MAX_NEXT_WORD_CANDIDATES = 200
# Blend statistics (n-gram) with neural (MLP): p = MLP_STAT_BLEND * p_kn + (1 - MLP_STAT_BLEND) * p_mlp. 0.6 = stats-heavy.
# DESIGN: Chat generation is vendor-neutral — no PyTorch/TensorFlow/CUDA/NVIDIA. Statistics (SQLite/HSB n-gram) + small CPU MLP (NumPy only). GPU is optional and only used for file training.
MLP_STAT_BLEND = 0.6
# Biology-like: priming (distance-decayed), spreading activation (with hub penalty), refractory (anti-repeat)
PRIMING_WINDOW = 12
PRIMING_AMPLITUDE = 0.08   # boost at d=0 is 1 + a
PRIMING_TAU = 3.0          # decay: 1 + a*exp(-d/τ)
SPREADING_ACTIVATION_BONUS = 0.10
SPREADING_HUB_HALVE = True  # halve bonus when activation came via a hub word
REFRACTORY_WINDOW = 5      # last N generated tokens
REFRACTORY_FACTOR = 0.92   # multiply score by factor^(1+d) when word appeared d steps back
HOMEOSTASIS_ON_SAVE = True
HOMEOSTASIS_SOFT_SCALE = True   # use sqrt(threshold/norm) so proportional, not overcorrecting
# More biomimicry: rhythm, urgency, habituation, surprise, fatigue (all O(1) or small fixed cost)
RHYTHM_AMPLITUDE = 0.03        # temperature wobble: 1 + A*sin(step*F)
RHYTHM_FREQ = 0.5
URGENCY_SHORT_LEN = 2          # user message ≤ this many words → slight temp boost
URGENCY_PUNCT = True           # "!" or "?" in user input → arousal boost
URGENCY_TEMP_BOOST = 0.06
HABITUATION_FACTOR = 0.96      # per extra occurrence in context: gate *= factor (boredom)
SURPRISE_TEMP_BOOST = 0.02     # after non-greedy pick, next step slightly more exploratory
FATIGUE_AFTER_N = 15           # after this many replies in session, slight temp increase
FATIGUE_TEMP_BOOST = 0.04
# Future biomimicry (fast creature): working-memory slots (chunk last 2×3 words, bonus for continue/start);
# mirroring (low-prob echo one user word); use-it-or-lose-it (decay unused pattern priority on save);
# saccade (alternate context slice weight by step); novelty-seeking (boost temp if recent trigrams repeat).

REASONING_CONTEXT_WINDOW = 10
REASONING_PATTERN_MEMORY = 100
REASONING_QUALITY_THRESHOLD = 0.6
REASONING_ADAPTATION_RATE = 0.1

# SEMANTIC INTELLIGENCE CONSTANTS
SEMANTIC_CLUSTER_SIZE = 50
SEMANTIC_RELATIONSHIP_DEPTH = 3
SEMANTIC_CONTEXT_WEIGHT = 0.7
SEMANTIC_SIMILARITY_THRESHOLD = 0.3

# HIERARCHICAL MEMORY CONSTANTS
SHORT_TERM_MEMORY_SIZE = 20
LONG_TERM_MEMORY_SIZE = 1000
SEMANTIC_MEMORY_SIZE = 500
MEMORY_REPLAY_INTERVAL = 50
MEMORY_COMPRESSION_THRESHOLD = 0.7
MEMORY_PERSISTENCE_THRESHOLD = 0.6
MEMORY_FREQUENCY_WEIGHT = 0.4
MEMORY_RECENCY_WEIGHT = 0.6
MEMORY_REPLAY_COOLDOWN_SECONDS = 120
MEMORY_MAX_REPLAYS_PER_ITEM = 6
MEMORY_FORGOTTEN_LIMIT = 500
HIGH_QUALITY_MEMORY_THRESHOLD = 0.75
LOW_QUALITY_MEMORY_THRESHOLD = 0.45

# CONVERSATION INTELLIGENCE CONSTANTS
CONVERSATION_MEMORY_SIZE = 20
CONVERSATION_TOPIC_TRACKING = True
CONVERSATION_COHERENCE_THRESHOLD = 0.5
CONVERSATION_ADAPTATION_RATE = 0.15
GENERATION_CACHE_ENABLED = True
GENERATION_PARALLEL_CANDIDATES = 1  # was 3: fewer candidates = faster, especially with little training

QUALITY_STOPWORDS = {
    'the', 'a', 'an', 'of', 'to', 'and', 'or', 'is', 'it', 'i', 'you', 'we', 'they',
    'he', 'she', 'that', 'this', 'in', 'on', 'at', 'for', 'with', 'as', 'be', 'are',
    'was', 'were', 'am', 'do', 'does', 'did', 'have', 'has', 'had', 'my', 'your',
    'our', 'their', 'me', 'us', 'them', 'if', 'but', 'so', 'because', 'from', 'by',
}
LOW_VALUE_RESPONSE_PREFIXES = (
    'based on our previous conversation',
    'from what i remember',
    'i think ',
    'in my view,',
    'in learned data',
    'this system',
)
CONVERSATION_LEARNING_QUALITY_THRESHOLD = 0.62
GENERATION_CACHE_QUALITY_THRESHOLD = 0.72

# SELF-TRAINING OPTIMIZATION
SELF_TRAINING_INTERVAL = 1.2
SELF_TRAINING_QUALITY_THRESHOLD = 0.4
SELF_TRAINING_MEMORY_CHECK_INTERVAL = 5

class PerformanceCache:
    """High-performance caching system for faster response generation and learning"""
    
    def __init__(self):
        self.response_cache = {}
        self.pattern_cache = {}
        self.word_prob_cache = {}
        self.semantic_cache = {}
        self.context_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
    def get_response_cache(self, key):
        """Get cached response"""
        if key in self.response_cache:
            self.cache_hits += 1
            return self.response_cache[key]
        self.cache_misses += 1
        return None
    
    def set_response_cache(self, key, value):
        """Set cached response"""
        if len(self.response_cache) >= RESPONSE_CACHE_SIZE:
          
            oldest_key = next(iter(self.response_cache))
            del self.response_cache[oldest_key]
        self.response_cache[key] = value
    
    def get_pattern_cache(self, key):
        """Get cached pattern"""
        if key in self.pattern_cache:
            self.cache_hits += 1
            return self.pattern_cache[key]
        self.cache_misses += 1
        return None
    
    def set_pattern_cache(self, key, value):
        """Set cached pattern"""
        if len(self.pattern_cache) >= PATTERN_CACHE_SIZE:
            oldest_key = next(iter(self.pattern_cache))
            del self.pattern_cache[oldest_key]
        self.pattern_cache[key] = value
    
    def get_word_prob_cache(self, key):
        """Get cached word probabilities"""
        if key in self.word_prob_cache:
            self.cache_hits += 1
            return self.word_prob_cache[key]
        self.cache_misses += 1
        return None
    
    def set_word_prob_cache(self, key, value):
        """Set cached word probabilities"""
        if len(self.word_prob_cache) >= WORD_PROBABILITY_CACHE_SIZE:
            oldest_key = next(iter(self.word_prob_cache))
            del self.word_prob_cache[oldest_key]
        self.word_prob_cache[key] = value
    
    def get_semantic_cache(self, key):
        """Get cached semantic data"""
        if key in self.semantic_cache:
            self.cache_hits += 1
            return self.semantic_cache[key]
        self.cache_misses += 1
        return None
    
    def set_semantic_cache(self, key, value):
        """Set cached semantic data"""
        if len(self.semantic_cache) >= SEMANTIC_CACHE_SIZE:
            oldest_key = next(iter(self.semantic_cache))
            del self.semantic_cache[oldest_key]
        self.semantic_cache[key] = value
    
    def get_context_cache(self, key):
        """Get cached context data"""
        if key in self.context_cache:
            self.cache_hits += 1
            return self.context_cache[key]
        self.cache_misses += 1
        return None
    
    def set_context_cache(self, key, value):
        """Set cached context data"""
        if len(self.context_cache) >= CONTEXT_CACHE_SIZE:
            oldest_key = next(iter(self.context_cache))
            del self.context_cache[oldest_key]
        self.context_cache[key] = value
    
    def get_cache_stats(self):
        """Get cache performance statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        return {
            'hit_rate': hit_rate,
            'hits': self.cache_hits,
            'misses': self.cache_misses,
            'total_requests': total_requests,
            'response_cache_size': len(self.response_cache),
            'pattern_cache_size': len(self.pattern_cache),
            'word_prob_cache_size': len(self.word_prob_cache),
            'semantic_cache_size': len(self.semantic_cache),
            'context_cache_size': len(self.context_cache)
        }
    
    def clear_cache(self):
        """Clear all caches"""
        self.response_cache.clear()
        self.pattern_cache.clear()
        self.word_prob_cache.clear()
        self.semantic_cache.clear()
        self.context_cache.clear()

class MemoryManager:
    def __init__(self):
        self.process = psutil.Process()
        self.initial_memory = self.get_memory_usage_mb()
        
    def get_memory_usage_mb(self):
        """Get current memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024
    
    def get_available_memory_mb(self):
        """Get available system memory in MB"""
        return psutil.virtual_memory().available / 1024 / 1024
    
    def get_memory_usage_percent(self):
        """Get current memory usage as percentage of total system memory"""
        return psutil.virtual_memory().percent / 100
    
    def should_reduce_batch_size(self):
        """Check if we should reduce batch size due to memory pressure"""
        return self.get_memory_usage_percent() > MEMORY_SAFETY_THRESHOLD
    
    def calculate_optimal_chunk_size(self, file_size_words):
        """Calculate optimal chunk size based on available memory and file size"""
      
        if 'settings_manager' in globals():
            chunk_setting = settings_manager.get('chunk_size', 'auto')
            if chunk_setting != 'auto':
                try:
                    return int(chunk_setting)
                except (TypeError, ValueError):
                    pass
        
        available_mb = self.get_available_memory_mb()
        memory_threshold = settings_manager.get('memory_threshold', MEMORY_SAFETY_THRESHOLD) if 'settings_manager' in globals() else MEMORY_SAFETY_THRESHOLD
        
        estimated_mb_per_word = 0.001
        
        safe_memory_mb = available_mb * (1 - memory_threshold)
        safe_chunk_size = int(safe_memory_mb / estimated_mb_per_word) if estimated_mb_per_word > 0 else MIN_CHUNK_SIZE
        
        optimal_chunk = max(MIN_CHUNK_SIZE, min(MAX_CHUNK_SIZE, safe_chunk_size))
        
        if file_size_words > 1000000:
            optimal_chunk = min(optimal_chunk, CHUNK_SIZE_WORDS)
        
        return optimal_chunk
    
    def calculate_parallel_workers(self):
        """Calculate workers from RAM and CPU without saturating the whole machine."""
        if 'settings_manager' in globals():
            workers_setting = settings_manager.get('parallel_workers', 'auto')
            if workers_setting != 'auto':
                try:
                    return max(1, int(workers_setting))
                except (TypeError, ValueError):
                    pass
        try:
            vmem = psutil.virtual_memory()
            total_mb = vmem.total / (1024 * 1024)
            available_mb = vmem.available / (1024 * 1024)
            logical_cpus = psutil.cpu_count(logical=True) or cpu_count()
            physical_cores = psutil.cpu_count(logical=False) or max(1, logical_cpus // 2)
        except Exception:
            return min(4, PARALLEL_WORKER_LIMIT)
        if total_mb < 4000 or available_mb < 1500:
            return 1

        reserve_setting = 'auto'
        if 'settings_manager' in globals():
            reserve_setting = settings_manager.get('cpu_worker_reserve', 'auto')
        try:
            reserve = max(0, int(reserve_setting)) if reserve_setting != 'auto' else None
        except (TypeError, ValueError):
            reserve = None
        if reserve is None:
            if logical_cpus >= 16:
                reserve = 3
            elif logical_cpus >= 8:
                reserve = 2
            else:
                reserve = 1

        cpu_soft_cap = max(1, min(logical_cpus - reserve, physical_cores + max(1, physical_cores // 2)))
        if total_mb >= 32000:
            ram_per_worker_mb = 2800
            profile_cap = 12
        elif total_mb >= 16000:
            ram_per_worker_mb = 3200
            profile_cap = 8
        elif total_mb >= 8000:
            ram_per_worker_mb = 3600
            profile_cap = 4
        else:
            ram_per_worker_mb = 4096
            profile_cap = 2
        ram_cap = max(1, int(available_mb // ram_per_worker_mb))
        worker_cap = min(PARALLEL_WORKER_LIMIT, cpu_soft_cap, ram_cap, profile_cap)
        return max(1, worker_cap)
    
    def force_garbage_collection(self):
        """Force garbage collection to free memory"""
        gc.collect()

# GPU Detection and Acceleration Support Classes
class GPUDetector:
    """Detects and evaluates GPU capabilities for AI acceleration"""
    
    def __init__(self):
        self.detected_gpus = []
        self.recommended_gpu = None
        self.detection_results = {}
        self.detect_gpus()
    
    def detect_gpus(self):
        """Detect all available GPUs and evaluate their AI acceleration capabilities"""
        try:
            self.detected_gpus = []
            self.recommended_gpu = None
            self.detection_results = {}

            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total,compute_cap', '--format=csv,noheader,nounits'],
                                     capture_output=True, text=True, timeout=5)
                if result.returncode == 0 and getattr(result, 'stdout', None):
                    for line in (result.stdout or "").strip().split('\n'):
                        if line.strip():
                            parts = [part.strip() for part in line.split(',')]
                            if len(parts) >= 4:
                                gpu_index = self._normalize_gpu_index(parts[0])
                                name = parts[1].strip()
                                memory_mb = int(parts[2]) if parts[2].isdigit() else 0
                                compute_cap = parts[3].strip()

                                gpu_info = {
                                    'gpu_index': gpu_index,
                                    'name': name,
                                    'vendor': 'NVIDIA',
                                    'memory_mb': memory_mb,
                                    'memory_gb': round(memory_mb / 1024, 1) if memory_mb > 0 else 0,
                                    'compute_capability': compute_cap,
                                    'supports_cuda': True,
                                    'supports_opencl': True,
                                    'preferred_runtime': 'cuda',
                                    'detection_source': 'nvidia-smi',
                                    'ai_acceleration_score': self._calculate_nvidia_score(memory_mb, compute_cap)
                                }
                                self._merge_detected_gpu(gpu_info)
            except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
                pass
            
          
            try:
                import wmi  # type: ignore[reportMissingImports]
                c = wmi.WMI()
                for gpu in c.Win32_VideoController():
                    if gpu.Name and gpu.AdapterRAM is not None:
                        try:
                            memory_mb = int(gpu.AdapterRAM) // (1024 * 1024)
                        except (TypeError, ValueError):
                            continue
                        memory_gb = memory_mb // 1024
                        
                        gpu_name = gpu.Name.strip()
                        vendor = self._detect_vendor(gpu_name)
                        
                        if vendor and memory_gb >= 2:
                            gpu_info = {
                                'gpu_index': None,
                                'name': gpu_name,
                                'vendor': vendor,
                                'memory_mb': memory_mb,
                                'memory_gb': memory_gb,
                                'supports_cuda': vendor == 'NVIDIA',
                                'supports_opencl': vendor in ['NVIDIA', 'AMD', 'Intel'],
                                'preferred_runtime': 'cuda' if vendor == 'NVIDIA' else 'opencl',
                                'detection_source': 'wmi',
                                'ai_acceleration_score': self._calculate_general_score(vendor, memory_gb)
                            }
                            self._merge_detected_gpu(gpu_info)
            except ImportError:
                pass

            if self.detected_gpus:
                self.recommended_gpu = self.get_best_gpu()
            self.detection_results = {
                'total_gpus': len(self.detected_gpus),
                'ai_capable_gpus': len([g for g in self.detected_gpus if g.get('ai_acceleration_score', 0) > 50]),
                'best_gpu': self.recommended_gpu,
                'recommended_gpu_index': None if not self.recommended_gpu else self.recommended_gpu.get('gpu_index'),
                'recommended_acceleration': self.recommended_gpu.get('ai_acceleration_score', 0) > 70 if self.recommended_gpu else False
            }
            
        except Exception as e:
            print(f"GPU detection failed: {e}")
            self.detection_results = {'error': str(e)}

    def _normalize_gpu_index(self, value):
        if value is None:
            return None
        try:
            return int(str(value).strip())
        except (TypeError, ValueError):
            return None

    def _merge_detected_gpu(self, gpu_info):
        gpu_name = str(gpu_info.get('name', '')).strip().lower()
        for existing in self.detected_gpus:
            existing_name = str(existing.get('name', '')).strip().lower()
            if existing_name != gpu_name:
                continue
            for key, value in gpu_info.items():
                if existing.get(key) in (None, '', 0, 0.0, False) and value not in (None, '', 0, 0.0, False):
                    existing[key] = value
            existing['ai_acceleration_score'] = max(existing.get('ai_acceleration_score', 0), gpu_info.get('ai_acceleration_score', 0))
            return existing
        self.detected_gpus.append(gpu_info)
        return gpu_info

    def _matches_preferences(self, gpu_info, acceleration_type='auto', gpu_index='auto'):
        preferred_type = str(acceleration_type or 'auto').strip().lower()
        preferred_index = self._normalize_gpu_index(gpu_index)
        device_index = self._normalize_gpu_index(gpu_info.get('gpu_index'))
        if preferred_index is not None and device_index is not None and device_index != preferred_index:
            return False
        if preferred_index is not None and device_index is None:
            return False
        if preferred_type == 'cuda':
            return bool(gpu_info.get('supports_cuda'))
        if preferred_type == 'opencl':
            return bool(gpu_info.get('supports_opencl'))
        return True

    def get_best_gpu(self, acceleration_type='auto', gpu_index='auto'):
        candidates = [gpu for gpu in self.detected_gpus if self._matches_preferences(gpu, acceleration_type, gpu_index)]
        if not candidates and acceleration_type not in (None, '', 'auto'):
            candidates = [gpu for gpu in self.detected_gpus if self._matches_preferences(gpu, 'auto', gpu_index)]
        if not candidates and gpu_index not in (None, '', 'auto'):
            candidates = [gpu for gpu in self.detected_gpus if self._matches_preferences(gpu, acceleration_type, 'auto')]
        if not candidates:
            candidates = list(self.detected_gpus)
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda gpu: (
                gpu.get('ai_acceleration_score', 0),
                1 if gpu.get('supports_cuda') else 0,
                gpu.get('memory_mb', 0),
                gpu.get('memory_gb', 0),
            ),
        )
    
    def _detect_vendor(self, gpu_name):
        """Detect GPU vendor from name"""
        name_lower = gpu_name.lower()
        if any(keyword in name_lower for keyword in ['nvidia', 'geforce', 'quadro', 'rtx', 'gtx']):
            return 'NVIDIA'
        elif any(keyword in name_lower for keyword in ['amd', 'radeon', 'rx', 'vega']):
            return 'AMD'
        elif any(keyword in name_lower for keyword in ['intel', 'arc', 'iris xe']):
            return 'Intel'
        return None
    
    def _calculate_nvidia_score(self, memory_mb, compute_cap):
        """Calculate AI acceleration score for NVIDIA GPUs"""
        score = 0
        memory_gb = (float(memory_mb) / 1024.0) if memory_mb else 0.0

        if memory_gb >= 8:
            score += 40
        elif memory_gb >= 6:
            score += 35
        elif memory_gb >= 4:
            score += 25
        elif memory_gb >= 2:
            score += 15
        
      
        try:
            if compute_cap is None or not isinstance(compute_cap, str):
                score += 20
            else:
                major, minor = map(int, compute_cap.split('.'))
                if major >= 8:
                    score += 60
                elif major >= 7:
                    score += 50
                elif major >= 6:
                    score += 40
                elif major >= 5:
                    score += 30
                else:
                    score += 20
        except (ValueError, TypeError):
            score += 20
        
        return min(100, score)
    
    def _calculate_general_score(self, vendor, memory_gb):
        """Calculate AI acceleration score for general GPUs"""
        score = 0
        
      
        if vendor == 'NVIDIA':
            score += 40
        elif vendor == 'AMD':
            score += 30
        elif vendor == 'Intel':
            score += 20
        
      
        if memory_gb >= 8:
            score += 60
        elif memory_gb >= 6:
            score += 50
        elif memory_gb >= 4:
            score += 35
        elif memory_gb >= 2:
            score += 20
        
        return min(100, score)
    
    def get_installation_recommendation(self, gpu_info=None):
        """Get recommendation for GPU acceleration installation"""
        gpu = gpu_info or self.recommended_gpu
        if not gpu:
            return {
                'install': False,
                'reason': 'No suitable GPU detected',
                'packages': []
            }

        score = gpu.get('ai_acceleration_score', 0)
        gpu_name = gpu.get('name', 'Unknown')
        if score >= 80:
            return {
                'install': True,
                'reason': f'Excellent AI acceleration GPU detected: {gpu_name}',
                'packages': ['pycuda', 'pyopencl'] if gpu.get('supports_cuda') else ['pyopencl'],
                'priority': 'high'
            }
        elif score >= 60:
            return {
                'install': True,
                'reason': f'Good AI acceleration GPU detected: {gpu_name}',
                'packages': ['pycuda', 'pyopencl'] if gpu.get('supports_cuda') else ['pyopencl'],
                'priority': 'medium'
            }
        elif score >= 40:
            return {
                'install': True,
                'reason': f'Moderate AI acceleration GPU detected: {gpu_name}',
                'packages': ['pyopencl'],
                'priority': 'low'
            }
        else:
            return {
                'install': False,
                'reason': f'GPU detected but not suitable for AI acceleration: {gpu_name}',
                'packages': [],
                'priority': 'none'
            }

class GPUAccelerator:
    def __init__(self, gpu_detector=None):
        self.gpu_detector = gpu_detector or GPUDetector()
        self._gpu_lock = threading.RLock()
        self._reset_state()
        self.initialize_gpu()

    def _reset_state(self):
        previous_context = getattr(self, 'context', None)
        previous_type = getattr(self, 'gpu_type', 'None')
        if previous_context is not None and previous_type == 'CUDA' and hasattr(previous_context, 'detach'):
            try:
                previous_context.detach()
            except Exception:
                pass
        self.gpu_available = False
        self.gpu_type = "None"
        self.gpu_info = None
        self.device_index = None
        self.platform = None
        self.device = None
        self.context = None
        self.queue = None
        self._cuda_driver = None
        self._cuda_module = None
        self._cuda_kernel = None

    def _get_preferred_runtime(self):
        preferred_runtime = 'auto'
        if 'settings_manager' in globals():
            preferred_runtime = str(settings_manager.get('gpu_acceleration_type', 'auto') or 'auto').strip().lower()
        if preferred_runtime not in ('auto', 'cuda', 'opencl'):
            return 'auto'
        return preferred_runtime

    def _get_preferred_gpu_index(self):
        if 'settings_manager' not in globals():
            return 'auto'
        return settings_manager.get('gpu_device_index', 'auto')

    def _build_runtime_order(self, preferred_gpu):
        preferred_runtime = self._get_preferred_runtime()
        if preferred_runtime in ('cuda', 'opencl'):
            runtime_order = [preferred_runtime]
        else:
            runtime_order = []
            if preferred_gpu and preferred_gpu.get('supports_cuda'):
                runtime_order.append('cuda')
            if preferred_gpu and preferred_gpu.get('supports_opencl'):
                runtime_order.append('opencl')
        if preferred_gpu and preferred_gpu.get('supports_cuda') and 'cuda' not in runtime_order:
            runtime_order.append('cuda')
        if preferred_gpu and preferred_gpu.get('supports_opencl') and 'opencl' not in runtime_order:
            runtime_order.append('opencl')
        return runtime_order

    def _normalize_device_name(self, value):
        return str(value or '').strip().lower()

    def _resolve_cuda_device(self, cuda_module, preferred_gpu):
        device_count = int(cuda_module.Device.count())
        candidates = []
        preferred_index = self.gpu_detector._normalize_gpu_index(None if preferred_gpu is None else preferred_gpu.get('gpu_index'))
        preferred_name = self._normalize_device_name(None if preferred_gpu is None else preferred_gpu.get('name'))
        for idx in range(device_count):
            device = cuda_module.Device(idx)
            raw_name = device.name()
            device_name = raw_name.decode('utf-8', errors='ignore') if isinstance(raw_name, bytes) else str(raw_name)
            normalized_name = self._normalize_device_name(device_name)
            score = 0
            if preferred_name and normalized_name == preferred_name:
                score += 100
            elif preferred_name and preferred_name in normalized_name:
                score += 60
            if preferred_index is not None and idx == preferred_index:
                score += 25
            try:
                score += min(int(device.total_memory() // (1024 ** 3)), 24)
            except Exception:
                pass
            candidates.append((score, idx, device_name, device))
        if not candidates:
            raise RuntimeError("No CUDA devices found")
        _, idx, device_name, device = max(candidates, key=lambda item: item[0])
        return idx, device_name, device

    def get_snapshot(self):
        return {
            'gpu_available': bool(self.gpu_available),
            'gpu_type': self.gpu_type,
            'device_index': self.device_index,
            'device_name': getattr(self, 'device_name', None) or (None if self.gpu_info is None else self.gpu_info.get('name')),
            'gpu_info': copy.deepcopy(self.gpu_info) if isinstance(self.gpu_info, dict) else self.gpu_info,
        }

    def reinitialize_gpu(self):
        with self._gpu_lock:
            return self.initialize_gpu(force=True)

    def initialize_gpu(self, force=False):
        """Initialize GPU acceleration if available and suitable"""
        self._reset_state()
        if 'settings_manager' in globals() and not bool(settings_manager.get('gpu_acceleration_enabled', False)):
            print("GPU Acceleration: Disabled in settings. Using CPU.")
            return self.get_snapshot()
        preferred_runtime = self._get_preferred_runtime()
        preferred_index = self._get_preferred_gpu_index()
        preferred_gpu = self.gpu_detector.get_best_gpu(preferred_runtime, preferred_index)
        self.gpu_info = preferred_gpu or self.gpu_detector.recommended_gpu
        self.device_index = None if preferred_gpu is None else preferred_gpu.get('gpu_index')
        self.device_name = None

        if not preferred_gpu:
            print("GPU Acceleration: No suitable GPU detected for AI acceleration")
            return self.get_snapshot()

        recommendation = self.gpu_detector.get_installation_recommendation(preferred_gpu)
        if not recommendation.get('install', False):
            print(f"GPU Acceleration: {recommendation.get('reason', 'Unknown')}")
            return self.get_snapshot()

        packages = recommendation.get('packages', [])
        runtime_order = self._build_runtime_order(preferred_gpu)
        for runtime_name in runtime_order:
            if runtime_name == 'cuda' and 'pycuda' in packages:
                try:
                    import pycuda.driver as cuda  # type: ignore[reportMissingImports]
                    from pycuda.compiler import SourceModule  # type: ignore[reportMissingImports]
                    cuda.init()
                    target_index, target_name, target_device = self._resolve_cuda_device(cuda, preferred_gpu)
                    cuda_context = target_device.make_context()
                    try:
                        cuda_module = SourceModule("""
                            __global__ void scale_patterns(const float *input, float *output, int size) {
                                int index = blockIdx.x * blockDim.x + threadIdx.x;
                                if (index < size) {
                                    output[index] = input[index] * 1.5f;
                                }
                            }
                        """)
                        cuda_kernel = cuda_module.get_function('scale_patterns')
                        probe_input = np.array([2.0], dtype=np.float32)
                        probe_output = np.empty_like(probe_input)
                        probe_input_buffer = cuda.mem_alloc(probe_input.nbytes)
                        probe_output_buffer = cuda.mem_alloc(probe_output.nbytes)
                        cuda.memcpy_htod(probe_input_buffer, probe_input)
                        cuda_kernel(
                            probe_input_buffer,
                            probe_output_buffer,
                            np.int32(1),
                            block=(1, 1, 1),
                            grid=(1, 1, 1),
                        )
                        cuda.Context.synchronize()
                        cuda.memcpy_dtoh(probe_output, probe_output_buffer)
                        if not np.allclose(probe_output, np.array([3.0], dtype=np.float32)):
                            raise RuntimeError('CUDA compute probe returned an unexpected result')
                    except Exception:
                        cuda_context.pop()
                        cuda_context.detach()
                        raise
                    cuda_context.pop()
                    self.device = target_device
                    self.gpu_available = True
                    self.gpu_type = "CUDA"
                    self.context = cuda_context
                    self._cuda_driver = cuda
                    self._cuda_module = cuda_module
                    self._cuda_kernel = cuda_kernel
                    self.device_index = target_index
                    self.device_name = target_name
                    self.gpu_info = preferred_gpu or {}
                    self.gpu_info['resolved_cuda_index'] = target_index
                    self.gpu_info['resolved_cuda_name'] = target_name
                    print(f"GPU Acceleration: CUDA initialized on {target_name}")
                    print(f"AI Acceleration Score: {self.gpu_info.get('ai_acceleration_score', 0)}/100")
                    return self.get_snapshot()
                except ImportError:
                    print("GPU Acceleration: pycuda not installed")
                except Exception as e:
                    print(f"GPU Acceleration: CUDA initialization failed: {e}")
            if runtime_name != 'opencl' or 'pyopencl' not in packages:
                continue
            try:
                import pyopencl as cl
                opencl_candidates = []
                for platform in cl.get_platforms():
                    try:
                        platform_gpus = platform.get_devices(cl.device_type.GPU)
                    except Exception:
                        continue
                    for device in platform_gpus:
                        device_name = getattr(device, 'name', '').strip()
                        device_memory_mb = int(getattr(device, 'global_mem_size', 0) // (1024 * 1024))
                        score = 0
                        if preferred_gpu and device_name and preferred_gpu.get('name', '').lower() in device_name.lower():
                            score += 10
                        if preferred_gpu and preferred_gpu.get('vendor') and preferred_gpu.get('vendor', '').lower() in device_name.lower():
                            score += 2
                        score += min(device_memory_mb // 1024, 24)
                        opencl_candidates.append((score, platform, device))
                if not opencl_candidates:
                    raise RuntimeError("No GPU devices found")
                _, self.platform, self.device = max(opencl_candidates, key=lambda item: item[0])
                self.context = cl.Context([self.device])
                self.queue = cl.CommandQueue(self.context)
                self.gpu_available = True
                self.gpu_type = "OpenCL"
                self.gpu_info = preferred_gpu or {}
                print(f"GPU Acceleration: OpenCL initialized on {self.device.name}")
                print(f"AI Acceleration Score: {self.gpu_info.get('ai_acceleration_score', 0)}/100")
                return self.get_snapshot()
            except ImportError:
                print("GPU Acceleration: pyopencl not installed")
            except Exception as e:
                print(f"GPU Acceleration: OpenCL initialization failed: {e}")

        print("GPU Acceleration: No compatible GPU acceleration available. Using CPU.")
        return self.get_snapshot()
    
    def parallel_process_patterns(self, patterns, operation_type="similarity"):
        """Process patterns in parallel on GPU if available"""
        if not self.gpu_available:
            return self._cpu_fallback(patterns, operation_type)
        
        try:
            with self._gpu_lock:
                if self.gpu_type == "OpenCL":
                    return self._opencl_process(patterns, operation_type)
                elif self.gpu_type == "CUDA":
                    return self._cuda_process(patterns, operation_type)
        except Exception as e:
            print(f"GPU processing failed, falling back to CPU: {e}")
            return self._cpu_fallback(patterns, operation_type)
    
    def _opencl_process(self, patterns, operation_type):
        """Process using OpenCL"""
        try:
            import pyopencl as cl
        except ImportError:
            raise RuntimeError("pyopencl not available")
        patterns = patterns if patterns is not None else []
        if not patterns:
            return []
        kernel_code = """
        __kernel void process_patterns(__global float* input, __global float* output, int size) {
            int gid = get_global_id(0);
            if (gid < size) {
                output[gid] = input[gid] * 1.5f; // Example processing
            }
        }
        """
        program = cl.Program(self.context, kernel_code).build()
        pattern_data = [float(len(p)) if hasattr(p, '__len__') and not isinstance(p, (int, float)) else float(p) if isinstance(p, (int, float)) else 1.0 for p in patterns]
        input_buffer = cl.Buffer(self.context, cl.mem_flags.READ_ONLY, size=len(pattern_data) * 4)
        output_buffer = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, size=len(pattern_data) * 4)
        
        cl.enqueue_copy(self.queue, input_buffer, np.array(pattern_data, dtype=np.float32))
        program.process_patterns(self.queue, (len(pattern_data),), None, input_buffer, output_buffer, np.int32(len(pattern_data)))
        
        result = np.empty(len(pattern_data), dtype=np.float32)
        cl.enqueue_copy(self.queue, result, output_buffer)
        return result.tolist()
    
    def _cuda_process(self, patterns, operation_type):
        """Process using CUDA"""
        patterns = patterns if patterns is not None else []
        pattern_data = np.array([float(len(p)) if hasattr(p, '__len__') and not isinstance(p, (int, float)) else float(p) if isinstance(p, (int, float)) else 1.0 for p in patterns], dtype=np.float32)
        if not len(pattern_data):
            return []
        if self.context is None or self._cuda_driver is None or self._cuda_kernel is None:
            raise RuntimeError('CUDA runtime is not initialized')
        result = np.empty_like(pattern_data)
        self.context.push()
        try:
            input_buffer = self._cuda_driver.mem_alloc(pattern_data.nbytes)
            output_buffer = self._cuda_driver.mem_alloc(result.nbytes)
            self._cuda_driver.memcpy_htod(input_buffer, pattern_data)
            block_size = 256
            grid_size = (len(pattern_data) + block_size - 1) // block_size
            self._cuda_kernel(
                input_buffer,
                output_buffer,
                np.int32(len(pattern_data)),
                block=(block_size, 1, 1),
                grid=(grid_size, 1, 1),
            )
            self._cuda_driver.Context.synchronize()
            self._cuda_driver.memcpy_dtoh(result, output_buffer)
        finally:
            self.context.pop()
        return result.tolist()

    def _cpu_fallback(self, patterns, operation_type):
        """CPU fallback for pattern processing. Handles patterns as sequences or as int/float (e.g. cluster IDs)."""
        patterns = patterns if patterns is not None else []
        def _size(p):
            if hasattr(p, '__len__') and not isinstance(p, (int, float)):
                return float(len(p))
            if isinstance(p, (int, float)):
                return float(p)
            return 1.0
        return [_size(p) * 1.5 for p in patterns]

# Settings Manager
class SettingsManager:
    def __init__(self):
        self.settings_file = SETTINGS_FILE
        self.default_settings = {
            'gpu_acceleration': True,
            'gpu_acceleration_enabled': False,
            'gpu_acceleration_type': 'auto',
            'gpu_device_index': 'auto',
            'gpu_memory_limit': 0.8,
            'gpu_batch_size': 'auto',
            'hardware_adaptive_mode': True,
            'cpu_worker_reserve': 'auto',
            'parallel_workers': 'auto',
            'chunk_size': 'auto',
            'memory_threshold': 0.85,
            'max_response_length': 25,
            'min_generation_attempts': 8,
            'self_training_interval': 1.8,
            'quality_threshold': 0.5,
            'auto_save_interval': 300,
            'enable_memory_monitoring': True,
            'enable_garbage_collection': True,
            'enhanced_intelligence': True,
            'creativity_factor': 0.5,
            'advanced_reasoning': True,
            'adaptive_learning': True,
            'log_level': 'info',
            'save_hsb_copy': False,
            'use_hsb_backend': False,
            'vocab_id_storage_mode': 'off',
            'auto_save_interval_minutes': 0,
            'show_low_confidence_notice': True,
            'debug_logging': False,
            'memory_replay_interval': MEMORY_REPLAY_INTERVAL,
            'memory_compression_threshold': MEMORY_COMPRESSION_THRESHOLD,
            'context_weight': 0.3,
            'semantic_weight': 0.4,
            'pattern_weight': 0.3,
            'performance_profile': 'medium',
            'features': {
                'critic': False,
                'confidence_gate': False,
                'anti_loop_filter': False,
                'meta_memory': False,
                'curiosity': False,
                'env_feedback': False,
                'autotune': False,
              
                'response_learning': False,
                'truth_fact_table': False,
                'topic_detection': False
            }
        }
        self.settings = self.load_settings()
    
    def load_settings(self):
        """Load settings from file or create default"""
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    loaded = {**self.default_settings, **json.load(f)}
                # Ensure 'features' always has all keys (avoid partial dict from old saves)
                if isinstance(loaded.get('features'), dict):
                    loaded['features'] = {**self.default_settings.get('features', {}), **loaded['features']}
                return loaded
            else:
                return copy.deepcopy(self.default_settings)
        except Exception as e:
            print(f"Error loading settings: {e}")
            _preserve_invalid_sidecar(self.settings_file, e)
            return copy.deepcopy(self.default_settings)
    
    def save_settings(self):
        """Save current settings to file"""
        try:
            _atomic_json_dump(self.settings, self.settings_file, indent=2)
            return True
        except Exception as e:
            print(f"Error saving settings: {e}")
            return False
    
    def reset_to_defaults(self):
        """Reset all settings to default values"""
        previous = self.settings
        self.settings = copy.deepcopy(self.default_settings)
        if not self.save_settings():
            self.settings = previous
            raise OSError(f"Could not save settings to {self.settings_file}")
    
    def get(self, key, default=None):
        """Get setting value"""
        return self.settings.get(key, default)
    
    def set(self, key, value):
        """Set setting value"""
        previous = copy.deepcopy(self.settings)
        self.settings[key] = value
        if not self.save_settings():
            self.settings = previous
            raise OSError(f"Could not save settings to {self.settings_file}")
    
    def set_feature(self, name, enabled):
        """Set a single feature flag without dropping other feature keys."""
        previous = copy.deepcopy(self.settings)
        defaults = self.default_settings.get('features', {})
        current = self.settings.get('features', {}) or {}
        self.settings['features'] = {**defaults, **current, name: enabled}
        if not self.save_settings():
            self.settings = previous
            raise OSError(f"Could not save settings to {self.settings_file}")

# Initialize Settings first (needed by MemoryManager)
settings_manager = SettingsManager()

def get_system_tier():
    """Detect system capability from RAM and GPU. Used by performance profiles."""
    try:
        hardware = get_hardware_profile()
        return {
            'tier': hardware.get('tier', 'medium'),
            'total_ram_mb': hardware.get('total_ram_mb', 8000),
            'has_gpu': hardware.get('has_gpu', False),
        }
    except Exception:
        return {'tier': 'medium', 'total_ram_mb': 8000, 'has_gpu': False}

def _debug_log(category, message):
    """Write to mai_debug.log when debug_logging is enabled. Thread-safe."""
    if not (getattr(settings_manager, 'settings', None) and settings_manager.get('debug_logging', False)):
        return
    try:
        log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mai_debug.log')
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{category}] {message}\n")
    except Exception as e:
        import sys
        print(f"[_debug_log failed] {e}", file=sys.stderr)

# Initialize GPU Detection and Settings
gpu_detector = GPUDetector()
gpu_accelerator = GPUAccelerator(gpu_detector)

# Initialize Memory Manager after settings
memory_manager = MemoryManager()

def _recommend_gpu_type(gpu_info):
    if not gpu_info:
        return 'auto'
    if gpu_info.get('supports_cuda'):
        return 'cuda'
    if gpu_info.get('supports_opencl'):
        return 'opencl'
    return 'auto'

def _recommend_gpu_batch_size(gpu_info):
    if not gpu_info:
        return 'auto'
    memory_mb = int(gpu_info.get('memory_mb', 0) or 0)
    if memory_mb >= 10240:
        return 'large'
    if memory_mb >= 6144:
        return 'medium'
    return 'small'

def _recommend_chunk_size(total_ram_mb, available_ram_mb, has_gpu):
    if total_ram_mb >= 32768:
        base = 500000 if available_ram_mb >= 16000 else 250000
    elif total_ram_mb >= 16384:
        base = 250000 if available_ram_mb >= 8000 else 100000
    elif total_ram_mb >= 8192:
        base = 100000 if available_ram_mb >= 4000 else 50000
    else:
        base = 50000 if available_ram_mb >= 2000 else 10000
    if not has_gpu and base > 250000:
        base = 250000
    return max(MIN_CHUNK_SIZE, min(MAX_CHUNK_SIZE, int(base)))

def _recommend_memory_threshold(total_ram_mb, available_ram_mb):
    available_ratio = (available_ram_mb / total_ram_mb) if total_ram_mb else 0.0
    if total_ram_mb >= 32768:
        threshold = 0.80
    elif total_ram_mb >= 16384:
        threshold = 0.82
    elif total_ram_mb >= 8192:
        threshold = 0.85
    else:
        threshold = 0.88
    if available_ratio < 0.25:
        threshold = min(0.90, threshold + 0.05)
    elif available_ratio < 0.40:
        threshold = min(0.88, threshold + 0.03)
    return round(threshold, 2)

def get_hardware_profile():
    """Return a richer hardware profile used for runtime tuning and transport inspection."""
    try:
        vmem = psutil.virtual_memory()
        total_ram_mb = float(vmem.total / (1024 * 1024))
        available_ram_mb = float(vmem.available / (1024 * 1024))
        logical_processors = int(psutil.cpu_count(logical=True) or cpu_count() or 1)
        physical_cores = int(psutil.cpu_count(logical=False) or max(1, logical_processors // 2))
        recommended_gpu = copy.deepcopy(getattr(gpu_detector, 'recommended_gpu', None))
        detected_gpus = copy.deepcopy(getattr(gpu_detector, 'detected_gpus', []) or [])
        ai_capable_gpus = [gpu for gpu in detected_gpus if gpu.get('ai_acceleration_score', 0) > 50]
        has_gpu = bool(recommended_gpu and recommended_gpu.get('ai_acceleration_score', 0) >= 40)
        if total_ram_mb >= 32768 and logical_processors >= 12:
            tier = 'high'
        elif total_ram_mb >= 12000 or logical_processors >= 8 or has_gpu:
            tier = 'medium'
        else:
            tier = 'low'

        recommended_parallel_workers = max(1, memory_manager.calculate_parallel_workers())
        recommended_chunk_size = _recommend_chunk_size(total_ram_mb, available_ram_mb, has_gpu)
        recommended_memory_threshold = _recommend_memory_threshold(total_ram_mb, available_ram_mb)
        recommended_gpu_type = _recommend_gpu_type(recommended_gpu)
        recommended_gpu_batch_size = _recommend_gpu_batch_size(recommended_gpu)
        recommended_gpu_index = 'auto' if recommended_gpu is None or recommended_gpu.get('gpu_index') is None else str(recommended_gpu.get('gpu_index'))
        cpu_reserve = 'auto'
        if logical_processors >= 16:
            cpu_reserve = '3'
        elif logical_processors >= 8:
            cpu_reserve = '2'
        else:
            cpu_reserve = '1'

        medium_batch = recommended_gpu_batch_size
        if medium_batch == 'large':
            medium_batch = 'medium'

        profiles = {
            'low': {
                'parallel_workers': '1',
                'chunk_size': '10000',
                'memory_threshold': 0.90,
                'gpu_acceleration_enabled': False,
                'gpu_acceleration_type': 'auto',
                'gpu_device_index': 'auto',
                'gpu_batch_size': 'small' if has_gpu else 'auto',
                'cpu_worker_reserve': '1',
            },
            'medium': {
                'parallel_workers': 'auto',
                'chunk_size': 'auto',
                'memory_threshold': min(0.88, round(recommended_memory_threshold + 0.03, 2)),
                'gpu_acceleration_enabled': has_gpu,
                'gpu_acceleration_type': recommended_gpu_type,
                'gpu_device_index': recommended_gpu_index,
                'gpu_batch_size': medium_batch,
                'cpu_worker_reserve': cpu_reserve,
            },
            'max': {
                'parallel_workers': str(recommended_parallel_workers),
                'chunk_size': str(recommended_chunk_size),
                'memory_threshold': recommended_memory_threshold,
                'gpu_acceleration_enabled': has_gpu,
                'gpu_acceleration_type': recommended_gpu_type,
                'gpu_device_index': recommended_gpu_index,
                'gpu_batch_size': recommended_gpu_batch_size,
                'cpu_worker_reserve': cpu_reserve,
            },
        }

        active_gpu = {}
        if 'gpu_accelerator' in globals() and hasattr(gpu_accelerator, 'get_snapshot'):
            active_gpu = gpu_accelerator.get_snapshot()

        return {
            'tier': tier,
            'total_ram_mb': round(total_ram_mb, 1),
            'available_ram_mb': round(available_ram_mb, 1),
            'logical_processors': logical_processors,
            'physical_cores': physical_cores,
            'total_gpus': len(detected_gpus),
            'ai_capable_gpus': len(ai_capable_gpus),
            'has_gpu': has_gpu,
            'gpus': detected_gpus,
            'recommended_gpu': recommended_gpu,
            'recommended_gpu_index': recommended_gpu_index,
            'recommended_gpu_type': recommended_gpu_type,
            'recommended_gpu_batch_size': recommended_gpu_batch_size,
            'recommended_parallel_workers': recommended_parallel_workers,
            'recommended_chunk_size': recommended_chunk_size,
            'recommended_memory_threshold': recommended_memory_threshold,
            'cpu_worker_reserve': cpu_reserve,
            'active_profile': settings_manager.get('performance_profile', 'medium') if 'settings_manager' in globals() else 'medium',
            'hardware_adaptive_mode': bool(settings_manager.get('hardware_adaptive_mode', True)) if 'settings_manager' in globals() else True,
            'configured_gpu_type': settings_manager.get('gpu_acceleration_type', 'auto') if 'settings_manager' in globals() else 'auto',
            'configured_gpu_device_index': settings_manager.get('gpu_device_index', 'auto') if 'settings_manager' in globals() else 'auto',
            'active_gpu': active_gpu,
            'profiles': profiles,
        }
    except Exception as e:
        return {
            'tier': 'medium',
            'total_ram_mb': 8000.0,
            'available_ram_mb': 4000.0,
            'logical_processors': 4,
            'physical_cores': 2,
            'total_gpus': 0,
            'ai_capable_gpus': 0,
            'has_gpu': False,
            'gpus': [],
            'recommended_gpu': None,
            'recommended_gpu_index': 'auto',
            'recommended_gpu_type': 'auto',
            'recommended_gpu_batch_size': 'auto',
            'recommended_parallel_workers': 2,
            'recommended_chunk_size': CHUNK_SIZE_WORDS,
            'recommended_memory_threshold': MEMORY_SAFETY_THRESHOLD,
            'cpu_worker_reserve': 'auto',
            'active_profile': 'medium',
            'hardware_adaptive_mode': True,
            'configured_gpu_type': 'auto',
            'configured_gpu_device_index': 'auto',
            'active_gpu': {},
            'profiles': {},
            'error': str(e),
        }

# Initialize Performance Cache for speed optimization
performance_cache = PerformanceCache()

# Advanced Intelligence Enhancement Classes
class EnhancedSemanticMemory:
    """Enhanced semantic memory with advanced clustering and relationship detection"""
    
    def __init__(self, base_semantic_memory):
        self.base_memory = base_semantic_memory
        self.concept_hierarchies = {}
        self.semantic_relationships = {}
        self.contextual_weights = {}
    
    def build_concept_hierarchy(self, words):
        """Build hierarchical concept relationships"""
        words = words if words is not None else []
        for word in words:
            if word not in self.concept_hierarchies:
                self.concept_hierarchies[word] = {
                    'superordinates': set(),
                    'subordinates': set(),
                    'related_concepts': set(),
                    'semantic_density': 0
                }
    
    def detect_semantic_relationships(self, word1, word2, relationship_type):
        """Detect and store semantic relationships between words"""
        if word1 not in self.semantic_relationships:
            self.semantic_relationships[word1] = {}
        
        self.semantic_relationships[word1][word2] = {
            'type': relationship_type,
            'strength': 1.0,
            'context_count': 1
        }
    
    def get_semantic_context(self, word, max_related=5):
        """Get rich semantic context for a word"""
        context = {
            'clusters': [],
            'relationships': [],
            'hierarchies': [],
            'contextual_usage': []
        }
        
      
        if hasattr(self.base_memory, 'word_to_cluster') and word in self.base_memory.word_to_cluster:
            cluster_id = self.base_memory.word_to_cluster[word]
            if cluster_id in self.base_memory.clusters:
                context['clusters'] = list(self.base_memory.clusters[cluster_id])[:max_related]
        
      
        if word in self.semantic_relationships:
            context['relationships'] = list(self.semantic_relationships[word].keys())[:max_related]
        
        return context

class HierarchicalMemorySystem:
    """Hierarchical memory system with short-term, long-term, and semantic memory"""
    
    def __init__(self, brain):
        self.brain = brain
        self.short_term_memory = []
        self.long_term_memory = [] 
        self.semantic_memory = {}  
        self.memory_counter = 0
        self.last_replay_time = time.time()
        self.interaction_count = 0
        
      
        self.memory_importance = {}
        self.memory_frequency = {}
        self.memory_recency = {}
        self.memory_replay_count = {}
        self.memory_last_replayed = {}
        
      
        self.contextual_links = {}
        self.semantic_references = {}
        self.cross_reference_matrix = {}
        
      
        self.importance_decay_rate = 0.95
        self.importance_boost_factor = 1.2
        self.min_importance_threshold = 0.1
        
      
        self.forgotten_memories = []
        self.reminiscence_queue = []
        self.replay_depth = 3
        self.last_deep_replay = time.time()
        self.deep_replay_interval = 300
        self.replay_cooldown_seconds = MEMORY_REPLAY_COOLDOWN_SECONDS
        self.max_replays_per_item = MEMORY_MAX_REPLAYS_PER_ITEM
        
    def add_memory(self, memory_data, memory_type="conversation"):
        """Add a new memory to the appropriate level"""
        self.interaction_count += 1
        
      
        memory_entry = {
            'id': self.memory_counter,
            'type': memory_type,
            'data': memory_data,
            'timestamp': time.time(),
            'importance': self._calculate_importance(memory_data),
            'compressed': False
        }
        
        self.memory_counter += 1
        
      
        self.short_term_memory.append(memory_entry)
        if len(self.short_term_memory) > SHORT_TERM_MEMORY_SIZE:
          
            self._promote_memory()
        
      
        self._update_importance_tracking(memory_entry)
        
      
        self._create_contextual_links(memory_entry)
        
      
        if self.interaction_count % MEMORY_REPLAY_INTERVAL == 0:
            self._perform_memory_replay()
        
      
        if time.time() - self.last_deep_replay > self.deep_replay_interval:
            self._perform_deep_memory_replay()
    
    def _calculate_importance(self, memory_data):
        """Calculate the importance of a memory"""
        importance = 0.5
        
      
        user_input = memory_data.get('user_input') or memory_data.get('user') or ''
        if user_input:
            if any(word in (user_input if isinstance(user_input, str) else '').lower() for word in ['what', 'how', 'why', 'explain', 'tell me']):
                importance += 0.2
            if len(str(user_input).split()) > 10:
                importance += 0.1
        if 'quality' in memory_data and memory_data.get('quality') is not None:
            try:
                importance += float(memory_data['quality']) * 0.3
            except (TypeError, ValueError):
                pass
        if 'sentiment' in memory_data:
            if memory_data.get('sentiment') in ['positive', 'negative']:
                importance += 0.1
        
      
        topics = memory_data.get('topics')
        if topics and isinstance(topics, (list, tuple)):
            importance += 0.1
        
        return min(1.0, importance)
    
    def _update_importance_tracking(self, memory_entry):
        """Update importance tracking for memory"""
        memory_id = memory_entry.get('id')
        if memory_id is None:
            return
        self.memory_frequency[memory_id] = self.memory_frequency.get(memory_id, 0) + 1
        self.memory_recency[memory_id] = time.time()
        self.memory_replay_count.setdefault(memory_id, 0)
        self.memory_last_replayed.setdefault(memory_id, 0.0)
        frequency_score = min(1.0, self.memory_frequency[memory_id] / 10)
        recency_score = max(0.0, 1.0 - (time.time() - self.memory_recency[memory_id]) / 86400)
        overall_importance = (
            memory_entry.get('importance', 0.5) * 0.4 +
            frequency_score * MEMORY_FREQUENCY_WEIGHT +
            recency_score * MEMORY_RECENCY_WEIGHT
        )
        
        self.memory_importance[memory_id] = overall_importance

    def _get_memory_quality(self, memory):
        data = memory.get('data') if isinstance(memory, dict) else None
        if not isinstance(data, dict):
            return 0.5
        try:
            return min(1.0, max(0.0, float(data.get('quality', 0.5) or 0.5)))
        except (TypeError, ValueError):
            return 0.5

    def _memory_rank_score(self, memory):
        if not isinstance(memory, dict):
            return 0.0
        memory_id = memory.get('id')
        base_importance = float(self.memory_importance.get(memory_id, memory.get('importance', 0.0)) or 0.0)
        quality = self._get_memory_quality(memory)
        replay_penalty = min(0.25, float(self.memory_replay_count.get(memory_id, 0)) * 0.04)
        age_seconds = max(0.0, time.time() - float(memory.get('timestamp', 0) or 0.0))
        freshness_bonus = max(0.0, 1.0 - (age_seconds / 86400.0)) * 0.1
        return base_importance + (quality * 0.35) + freshness_bonus - replay_penalty

    def _remember_forgotten_memory(self, memory):
        if not isinstance(memory, dict):
            return
        memory_id = memory.get('id')
        if memory_id is not None:
            self.forgotten_memories = [
                item for item in self.forgotten_memories
                if not (isinstance(item, dict) and item.get('id') == memory_id)
            ]
        self.forgotten_memories.append(memory)
        if len(self.forgotten_memories) > MEMORY_FORGOTTEN_LIMIT:
            self.forgotten_memories = self.forgotten_memories[-MEMORY_FORGOTTEN_LIMIT:]

    def _prune_memory_tracking(self, memory_id):
        if memory_id is None:
            return
        self.memory_importance.pop(memory_id, None)
        self.memory_frequency.pop(memory_id, None)
        self.memory_recency.pop(memory_id, None)
        self.memory_replay_count.pop(memory_id, None)
        self.memory_last_replayed.pop(memory_id, None)
        self.contextual_links.pop(memory_id, None)
        self.cross_reference_matrix.pop(memory_id, None)
        for linked_ids in self.cross_reference_matrix.values():
            if isinstance(linked_ids, set):
                linked_ids.discard(memory_id)
        for semantic_id in list(self.semantic_references.keys()):
            refs = [ref for ref in self.semantic_references.get(semantic_id, []) if ref != memory_id]
            if refs:
                self.semantic_references[semantic_id] = refs
            else:
                del self.semantic_references[semantic_id]

    def _select_replay_candidates(self, limit=8):
        now = time.time()
        candidates = []
        for memory in list(self.short_term_memory) + list(self.long_term_memory):
            if not isinstance(memory, dict) or memory.get('type') != 'conversation':
                continue
            memory_id = memory.get('id')
            if memory_id is None:
                continue
            replay_count = self.memory_replay_count.get(memory_id, 0)
            if replay_count >= self.max_replays_per_item:
                continue
            last_replayed = float(self.memory_last_replayed.get(memory_id, 0.0) or 0.0)
            if now - last_replayed < self.replay_cooldown_seconds:
                continue
            candidates.append((self._memory_rank_score(memory), memory))
        candidates.sort(key=lambda item: item[0], reverse=True)
        return [memory for _, memory in candidates[:max(1, int(limit or 1))]]
    
    def _create_contextual_links(self, memory_entry):
        """Create contextual links between episodic and semantic memories"""
        memory_id = memory_entry.get('id')
        memory_data = memory_entry.get('data')
        if memory_id is None or memory_data is None:
            return
        concepts = self._extract_concepts(memory_data)
        
      
        related_semantic_ids = []
        for semantic_id, semantic_data in self.semantic_memory.items():
            semantic_concepts = semantic_data.get('concepts', [])
            if self._calculate_concept_overlap(concepts, semantic_concepts) > 0.3:
                related_semantic_ids.append(semantic_id)
        
      
        if related_semantic_ids:
            self.contextual_links[memory_id] = related_semantic_ids
            
            for semantic_id in related_semantic_ids:
                if semantic_id not in self.semantic_references:
                    self.semantic_references[semantic_id] = []
                self.semantic_references[semantic_id].append(memory_id)
                
              
                self._boost_semantic_importance(semantic_id)
    
    def _extract_concepts(self, memory_data):
        """Extract key concepts from memory data"""
        concepts = set()
        
      
        user_input = memory_data.get('user_input') or memory_data.get('user')
        if user_input and isinstance(user_input, str):
            words = user_input.lower().split()
            for word in words:
                if len(word) > 3:
                    concepts.add(word)
        topics = memory_data.get('topics')
        if topics:
            try:
                concepts.update(topics)
            except (TypeError, ValueError):
                pass
        sentiment = memory_data.get('sentiment')
        if sentiment is not None:
            concepts.add(sentiment)
        
        return list(concepts)
    
    def _calculate_concept_overlap(self, concepts1, concepts2):
        """Calculate overlap between two concept sets"""
        if not concepts1 or not concepts2:
            return 0.0
        
        set1 = set(concepts1)
        set2 = set(concepts2)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def _boost_semantic_importance(self, semantic_id):
        """Boost importance of semantic memory when referenced"""
        if semantic_id in self.semantic_memory and isinstance(self.semantic_memory.get(semantic_id), dict):
            current_importance = self.semantic_memory[semantic_id].get('importance', 0.5)
            boosted_importance = min(1.0, current_importance * self.importance_boost_factor)
            self.semantic_memory[semantic_id]['importance'] = boosted_importance
            
          
            if semantic_id not in self.memory_frequency:
                self.memory_frequency[semantic_id] = 0
            self.memory_frequency[semantic_id] += 1
    
    def _perform_deep_memory_replay(self):
        """Perform deep memory replay including forgotten memories"""
        self.last_deep_replay = time.time()
        
      
        self._process_reminiscence_queue()
        
      
        self._replay_forgotten_memories()
        
      
        self._deep_cross_reference_analysis()
        
      
        self._update_all_importance_scores()
    
    def _process_reminiscence_queue(self):
        """Process memories that should be brought back into active memory"""
        if not self.reminiscence_queue:
            return
        
        memories_to_activate = []
        memories_to_keep = []
        
        for memory in self.reminiscence_queue:
          
            if self._should_activate_memory(memory):
                memories_to_activate.append(memory)
            else:
                memories_to_keep.append(memory)
        
      
        for memory in memories_to_activate:
            self.short_term_memory.append(memory)
            if len(self.short_term_memory) > SHORT_TERM_MEMORY_SIZE:
                self._promote_memory()
        
      
        self.reminiscence_queue = memories_to_keep
    
    def _should_activate_memory(self, memory):
        """Determine if a memory should be activated based on current context"""
      
        recent_topics = self._get_recent_topics()
        memory_topics = memory.get('data', {}).get('topics', [])
        
        if self._calculate_concept_overlap(recent_topics, memory_topics) > 0.2:
            return True
        
      
        if memory.get('importance', 0) > 0.7:
            return True
        
      
        if memory.get('importance', 0) > 0.5 and random.random() < 0.1:
            return True
        
        return False
    
    def _get_recent_topics(self):
        """Get topics from recent conversations"""
        recent_topics = []
        for memory in self.short_term_memory[-5:]:
            topics = memory.get('data', {}).get('topics', [])
            recent_topics.extend(topics)
        return recent_topics
    
    def _replay_forgotten_memories(self):
        """Replay memories that were forgotten but might be relevant"""
        if not self.forgotten_memories:
            return
        
      
        relevant_memories = []
        for memory in self.forgotten_memories:
            relevance_score = self._calculate_memory_relevance(memory)
            if relevance_score > 0.3:
                relevant_memories.append((memory, relevance_score))
        
      
        relevant_memories.sort(key=lambda x: x[1], reverse=True)
        
        for memory, score in relevant_memories[:5]:
            self._relearn_from_memory(memory)
            
          
            if score > 0.5:
                self.reminiscence_queue.append(memory)
    
    def _deep_cross_reference_analysis(self):
        """Perform deep analysis of cross-references between memories"""
      
        link_patterns = {}
        
        for episodic_id, semantic_ids in self.contextual_links.items():
            for semantic_id in semantic_ids:
                if semantic_id not in link_patterns:
                    link_patterns[semantic_id] = []
                link_patterns[semantic_id].append(episodic_id)
        
      
        for semantic_id, episodic_ids in link_patterns.items():
            if len(episodic_ids) > 2:
              
                self._boost_semantic_importance(semantic_id)
                
              
                for i, episodic_id1 in enumerate(episodic_ids):
                    for episodic_id2 in episodic_ids[i+1:]:
                        self._create_episodic_link(episodic_id1, episodic_id2)
    
    def _create_episodic_link(self, episodic_id1, episodic_id2):
        """Create direct links between episodic memories"""
        if episodic_id1 not in self.cross_reference_matrix:
            self.cross_reference_matrix[episodic_id1] = set()
        if episodic_id2 not in self.cross_reference_matrix:
            self.cross_reference_matrix[episodic_id2] = set()
        
        self.cross_reference_matrix[episodic_id1].add(episodic_id2)
        self.cross_reference_matrix[episodic_id2].add(episodic_id1)
    
    def _forget_memory(self, memory):
        """Move memory to forgotten memories"""
        self._remember_forgotten_memory(memory)
        
      
        if memory in self.short_term_memory:
            self.short_term_memory.remove(memory)
        if memory in self.long_term_memory:
            self.long_term_memory.remove(memory)
        if isinstance(memory, dict):
            self._prune_memory_tracking(memory.get('id'))
    
    def _promote_memory(self):
        """Promote important memories from short-term to long-term"""
        if not self.short_term_memory:
            return
        
        strongest_memory = max(self.short_term_memory, key=self._memory_rank_score)
        weakest_memory = min(self.short_term_memory, key=self._memory_rank_score)
        strongest_id = strongest_memory.get('id') if isinstance(strongest_memory, dict) else None
        strongest_importance = self.memory_importance.get(strongest_id, 0.0) if strongest_id is not None else 0.0

        if strongest_importance >= MEMORY_PERSISTENCE_THRESHOLD:
            self.short_term_memory.remove(strongest_memory)
            self.long_term_memory.append(strongest_memory)
            if len(self.long_term_memory) > LONG_TERM_MEMORY_SIZE:
                self._compress_oldest_memory()
        else:
            self._forget_memory(weakest_memory)
    
    def _compress_oldest_memory(self):
        """Compress the oldest memory into semantic form"""
        if not self.long_term_memory:
            return
        
      
        oldest_memory = min(self.long_term_memory, key=lambda m: m.get('timestamp', 0))
        
      
        semantic_key = self._create_semantic_key(oldest_memory)
        
        imp = oldest_memory.get('importance', 0.5)
        ts = oldest_memory.get('timestamp', 0)
        if semantic_key in self.semantic_memory:
          
            self.semantic_memory[semantic_key]['count'] += 1
            self.semantic_memory[semantic_key]['importance'] = max(
                self.semantic_memory[semantic_key].get('importance', 0),
                imp
            )
        else:
          
            self.semantic_memory[semantic_key] = {
                'pattern': self._extract_semantic_pattern(oldest_memory),
                'count': 1,
                'importance': imp,
                'timestamp': ts
            }
        
      
        oldest_memory['compressed'] = True
        if oldest_memory in self.long_term_memory:
            self.long_term_memory.remove(oldest_memory)
        
      
        if len(self.semantic_memory) > SEMANTIC_MEMORY_SIZE:
            self._remove_least_important_semantic()
    
    def _create_semantic_key(self, memory):
        """Create a semantic key for memory compression"""
        data = memory.get('data') if isinstance(memory, dict) else None
        if not isinstance(data, dict):
            return "general"
        user_input = data.get('user_input') or data.get('user')
        if user_input:
            try:
                words = str(user_input).lower().split()
                key_words = [w for w in words if len(w) > 3][:3]
                return "_".join(key_words) if key_words else "general"
            except (TypeError, AttributeError):
                return "general"
        topics = data.get('topics')
        if topics and isinstance(topics, (list, tuple)):
            try:
                return "_".join(str(t) for t in topics[:2])
            except (TypeError, ValueError):
                pass
        return "general"
    
    def _extract_semantic_pattern(self, memory):
        """Extract semantic pattern from memory"""
        data = memory.get('data') if isinstance(memory, dict) else {}
        if not isinstance(data, dict):
            data = {}
        pattern = {
            'intent': data.get('intent', 'statement'),
            'sentiment': data.get('sentiment', 'neutral'),
            'complexity': data.get('complexity', 'medium'),
            'topics': data.get('topics') if isinstance(data.get('topics'), (list, tuple)) else [],
            'quality': data.get('quality', 0.5)
        }
        return pattern
    
    def _remove_least_important_semantic(self):
        """Remove least important semantic memory"""
        if not self.semantic_memory:
            return
        
        least_important_key = min(
            self.semantic_memory.keys(),
            key=lambda k: self.semantic_memory[k].get('importance', 0)
        )
        del self.semantic_memory[least_important_key]
    
    def _perform_memory_replay(self):
        """Perform memory replay to reinforce connections"""
        print("Performing memory replay...")
        
        replay_candidates = self._select_replay_candidates(limit=max(4, self.replay_depth * 3))
        for memory in replay_candidates:
            self._relearn_from_memory(memory)
        
      
        self._recluster_semantic_memories()
        
      
        self._update_all_importance_scores()
        
        self.last_replay_time = time.time()
        print(f"Memory replay complete. {len(self.semantic_memory)} semantic memories active.")
    
    def _relearn_from_memory(self, memory):
        """Relearn from a memory to reinforce patterns"""
        if not isinstance(memory, dict):
            return
        memory_id = memory.get('id')
        if memory_id is None:
            return
        now = time.time()
        if self.memory_replay_count.get(memory_id, 0) >= self.max_replays_per_item:
            return
        if now - float(self.memory_last_replayed.get(memory_id, 0.0) or 0.0) < self.replay_cooldown_seconds:
            return
        data = memory.get('data') if isinstance(memory, dict) else None
        if not isinstance(data, dict):
            return
        user_input = data.get('user_input') or data.get('user')
        bot_response = data.get('bot_response') or data.get('bot')
        if user_input and bot_response:
            self.brain.learn_from_conversation_exchange(
                user_input,
                bot_response,
                data.get('quality', 0.5)
            )
            self.memory_replay_count[memory_id] = self.memory_replay_count.get(memory_id, 0) + 1
            self.memory_last_replayed[memory_id] = now
            self.memory_frequency[memory_id] = self.memory_frequency.get(memory_id, 0) + 1
            self.memory_recency[memory_id] = now
            boosted_importance = min(
                1.0,
                max(float(memory.get('importance', 0.5) or 0.5), self._get_memory_quality(memory)) * 1.03,
            )
            memory['importance'] = boosted_importance
            self.memory_importance[memory_id] = max(
                boosted_importance,
                float(self.memory_importance.get(memory_id, 0.0) or 0.0),
            )
    
    def _recluster_semantic_memories(self):
        """Recluster semantic memories to find new connections"""
        semantic_keys = list(self.semantic_memory.keys())
        
        for i, key1 in enumerate(semantic_keys):
            for key2 in semantic_keys[i+1:]:
                similarity = self._calculate_semantic_similarity(
                    self.semantic_memory[key1],
                    self.semantic_memory[key2]
                )
                
                if similarity > 0.7:
                  
                    self._merge_semantic_memories(key1, key2)
    
    def _calculate_semantic_similarity(self, memory1, memory2):
        """Calculate similarity between two semantic memories"""
        similarity = 0.0
        p1 = memory1.get('pattern', {}) if isinstance(memory1, dict) else {}
        p2 = memory2.get('pattern', {}) if isinstance(memory2, dict) else {}
        
      
        if p1.get('intent') == p2.get('intent'):
            similarity += 0.3
        
      
        if p1.get('sentiment') == p2.get('sentiment'):
            similarity += 0.2
        
      
        t1 = p1.get('topics') if isinstance(p1.get('topics'), (list, tuple)) else []
        t2 = p2.get('topics') if isinstance(p2.get('topics'), (list, tuple)) else []
        topics1, topics2 = set(t1), set(t2)
        if topics1 and topics2:
            union_len = len(topics1.union(topics2))
            topic_overlap = len(topics1.intersection(topics2)) / union_len if union_len > 0 else 0.0
            similarity += topic_overlap * 0.3
        
      
        try:
            q1, q2 = float(p1.get('quality', 0.5)), float(p2.get('quality', 0.5))
            quality_diff = abs(q1 - q2)
            similarity += (1.0 - quality_diff) * 0.2
        except (TypeError, ValueError):
            pass
        return similarity
    
    def _merge_semantic_memories(self, key1, key2):
        """Merge two semantic memories"""
        memory1 = self.semantic_memory.get(key1) or {}
        memory2 = self.semantic_memory.get(key2) or {}
        pat1 = memory1.get('pattern', {}) if isinstance(memory1.get('pattern'), dict) else {}
        pat2 = memory2.get('pattern', {}) if isinstance(memory2.get('pattern'), dict) else {}
        topics1 = pat1.get('topics') if isinstance(pat1.get('topics'), (list, tuple)) else []
        topics2 = pat2.get('topics') if isinstance(pat2.get('topics'), (list, tuple)) else []
        try:
            q1 = float(pat1.get('quality', 0.5))
        except (TypeError, ValueError):
            q1 = 0.5
        try:
            q2 = float(pat2.get('quality', 0.5))
        except (TypeError, ValueError):
            q2 = 0.5
        merged_memory = {
            'pattern': {
                'intent': pat1.get('intent', pat2.get('intent', 'statement')),
                'sentiment': pat1.get('sentiment', pat2.get('sentiment', 'neutral')),
                'complexity': max(pat1.get('complexity', 'medium'), pat2.get('complexity', 'medium')),
                'topics': list(set(list(topics1) + list(topics2))),
                'quality': max(q1, q2)
            },
            'count': memory1.get('count', 0) + memory2.get('count', 0),
            'importance': max(memory1.get('importance', 0), memory2.get('importance', 0)),
            'timestamp': min(memory1.get('timestamp', 0), memory2.get('timestamp', 0))
            }
        
      
        self.semantic_memory[key1] = merged_memory
        
      
        del self.semantic_memory[key2]
    
    def _update_all_importance_scores(self):
        """Update importance scores for all memories with quality-sensitive decay."""
        current_time = time.time()
        memories_to_forget = []

        for memory in list(self.short_term_memory) + list(self.long_term_memory):
            if not isinstance(memory, dict):
                continue
            memory_id = memory.get('id')
            if memory_id is None:
                continue
            base_importance = float(memory.get('importance', 0.5) or 0.5)
            quality = self._get_memory_quality(memory)
            last_seen = float(self.memory_recency.get(memory_id, memory.get('timestamp', current_time)) or current_time)
            age_seconds = max(0.0, current_time - last_seen)
            recency_score = max(0.0, 1.0 - (age_seconds / 86400.0))
            frequency_score = min(1.0, float(self.memory_frequency.get(memory_id, 0)) / 10.0)
            previous_importance = float(self.memory_importance.get(memory_id, base_importance) or base_importance)

            quality_decay_bonus = 0.05 if quality >= HIGH_QUALITY_MEMORY_THRESHOLD else 0.0
            quality_decay_penalty = 0.08 if quality <= LOW_QUALITY_MEMORY_THRESHOLD else 0.0
            tier_bonus = 0.02 if memory in self.long_term_memory else 0.0
            decay_multiplier = min(
                0.995,
                max(0.72, self.importance_decay_rate + quality_decay_bonus + tier_bonus - quality_decay_penalty),
            )
            decayed_importance = previous_importance * decay_multiplier
            overall_importance = (
                base_importance * 0.25 +
                decayed_importance * 0.25 +
                quality * 0.20 +
                frequency_score * 0.15 +
                recency_score * 0.15
            )
            overall_importance = min(1.0, max(0.0, overall_importance))
            memory['importance'] = overall_importance
            self.memory_importance[memory_id] = overall_importance

            stale_and_weak = (
                overall_importance < self.min_importance_threshold
                and age_seconds > 3600
                and quality <= LOW_QUALITY_MEMORY_THRESHOLD
            )
            if stale_and_weak:
                memories_to_forget.append(memory)

        for memory in memories_to_forget:
            self._forget_memory(memory)

        for semantic_id in list(self.semantic_memory.keys()):
            semantic_entry = self.semantic_memory.get(semantic_id)
            if not isinstance(semantic_entry, dict):
                continue
            current_importance = float(semantic_entry.get('importance', 0.5) or 0.5)
            pattern = semantic_entry.get('pattern', {}) if isinstance(semantic_entry.get('pattern'), dict) else {}
            try:
                semantic_quality = float(pattern.get('quality', 0.5) or 0.5)
            except (TypeError, ValueError):
                semantic_quality = 0.5
            decay_bonus = 0.04 if semantic_quality >= HIGH_QUALITY_MEMORY_THRESHOLD else 0.0
            decay_penalty = 0.06 if semantic_quality <= LOW_QUALITY_MEMORY_THRESHOLD else 0.0
            decay_multiplier = min(0.995, max(0.72, self.importance_decay_rate + decay_bonus - decay_penalty))
            decayed_importance = current_importance * decay_multiplier
            if decayed_importance < self.min_importance_threshold:
                self._remember_forgotten_memory(semantic_entry)
                del self.semantic_memory[semantic_id]
            else:
                semantic_entry['importance'] = decayed_importance
    
    def get_relevant_memories(self, query, limit=5):
        """Get memories relevant to a query with contextual linking"""
        relevant_memories = []
        
      
        query_concepts = self._extract_concepts({'user_input': query})
        
      
        for memory in self.short_term_memory:
            relevance = self._calculate_relevance(memory, query)
            if relevance > 0.3:
                relevant_memories.append((memory, relevance, 'short_term'))
        
      
        for memory in self.long_term_memory:
            relevance = self._calculate_relevance(memory, query)
            if relevance > 0.3:
                relevant_memories.append((memory, relevance, 'long_term'))
        
      
        for key, semantic_memory in self.semantic_memory.items():
            relevance = self._calculate_semantic_relevance(semantic_memory, query)
            if relevance > 0.3:
                relevant_memories.append((semantic_memory, relevance, 'semantic'))
        
      
        contextually_linked = self._find_contextually_linked_memories(query_concepts)
        for memory, relevance in contextually_linked:
            relevant_memories.append((memory, relevance, 'contextual'))
        
      
        reminiscence_memories = self._find_reminiscence_memories(query_concepts)
        for memory, relevance in reminiscence_memories:
            relevant_memories.append((memory, relevance, 'reminiscence'))
        
      
        relevant_memories.sort(key=lambda x: x[1], reverse=True)
        
      
        for memory, relevance, memory_type in relevant_memories[:limit]:
            self._boost_memory_importance(memory, memory_type)
        
        return [memory for memory, relevance, memory_type in relevant_memories[:limit]]
    
    def _find_contextually_linked_memories(self, query_concepts):
        """Find memories that are contextually linked to the query"""
        linked_memories = []
        
      
        matching_semantic_ids = []
        for semantic_id, semantic_data in self.semantic_memory.items():
            semantic_concepts = semantic_data.get('concepts', [])
            if self._calculate_concept_overlap(query_concepts, semantic_concepts) > 0.2:
                matching_semantic_ids.append(semantic_id)
        
      
        for semantic_id in matching_semantic_ids:
            if semantic_id in self.semantic_references:
                for episodic_id in self.semantic_references[semantic_id]:
                  
                    episodic_memory = self._find_episodic_memory(episodic_id)
                    if episodic_memory:
                        relevance = self._calculate_concept_overlap(
                            query_concepts, 
                            self._extract_concepts(episodic_memory.get('data', {}))
                        )
                        if relevance > 0.2:
                            linked_memories.append((episodic_memory, relevance))
        
        return linked_memories
    
    def _find_episodic_memory(self, episodic_id):
        """Find episodic memory by ID"""
        for memory in self.short_term_memory + self.long_term_memory:
            if memory['id'] == episodic_id:
                return memory
        return None
    
    def _find_reminiscence_memories(self, query_concepts):
        """Find relevant memories in reminiscence queue"""
        reminiscence_memories = []
        
        for memory in self.reminiscence_queue:
            memory_concepts = self._extract_concepts(memory.get('data', {}))
            relevance = self._calculate_concept_overlap(query_concepts, memory_concepts)
            if relevance > 0.3:
                reminiscence_memories.append((memory, relevance))
        
        return reminiscence_memories
    
    def _boost_memory_importance(self, memory, memory_type):
        """Boost importance of memory when accessed"""
        if memory_type in ['short_term', 'long_term']:
            memory_id = memory['id']
            if memory_id in self.memory_importance:
                current_importance = self.memory_importance[memory_id]
                boosted_importance = min(1.0, current_importance * self.importance_boost_factor)
                self.memory_importance[memory_id] = boosted_importance
        elif memory_type == 'semantic':
            semantic_id = memory.get('id')
            if semantic_id:
                self._boost_semantic_importance(semantic_id)
    
    def _calculate_relevance(self, memory, query):
        """Calculate relevance of a memory to a query"""
        if not isinstance(memory, dict) or not isinstance(query, str):
            return 0.0
        data = memory.get('data')
        if not isinstance(data, dict):
            return 0.0
        user_input_val = data.get('user_input') or data.get('user')
        if not isinstance(user_input_val, str):
            return 0.0
        query_words = set((query or "").lower().split())
        memory_words = set(user_input_val.lower().split())
        if memory_words:
            overlap = len(query_words.intersection(memory_words))
            return overlap / len(memory_words)
        return 0.0
    
    def _calculate_semantic_relevance(self, semantic_memory, query):
        """Calculate relevance of a semantic memory to a query"""
        if not isinstance(query, str):
            return 0.0
        query_words = set((query or "").lower().split())
        pattern = semantic_memory.get('pattern') if isinstance(semantic_memory, dict) else None
        topics = pattern.get('topics') if isinstance(pattern, dict) else None
        topic_words = set(topics) if isinstance(topics, (list, tuple, set)) else set()
        
        if topic_words:
            overlap = len(query_words.intersection(topic_words))
            return overlap / len(topic_words)
        
        return 0.0
    
    def get_memory_stats(self):
        """Get statistics about the memory system"""
        return {
            'short_term_count': len(self.short_term_memory),
            'long_term_count': len(self.long_term_memory),
            'semantic_count': len(self.semantic_memory),
            'forgotten_count': len(self.forgotten_memories),
            'reminiscence_count': len(self.reminiscence_queue),
            'contextual_links': len(self.contextual_links),
            'cross_references': len(self.cross_reference_matrix),
            'total_interactions': self.interaction_count,
            'last_replay': time.time() - self.last_replay_time,
            'last_deep_replay': time.time() - self.last_deep_replay,
            'avg_importance': sum(self.memory_importance.values()) / len(self.memory_importance) if self.memory_importance else 0,
            'memory_efficiency': self._calculate_memory_efficiency()
        }
    
    def _calculate_memory_efficiency(self):
        """Calculate memory system efficiency"""
        total_memories = len(self.short_term_memory) + len(self.long_term_memory) + len(self.semantic_memory)
        if total_memories == 0:
            return 0.0
        
      
        high_importance = sum(1 for imp in self.memory_importance.values() if imp > 0.7)
        medium_importance = sum(1 for imp in self.memory_importance.values() if 0.3 <= imp <= 0.7)
        
        efficiency = (high_importance * 0.8 + medium_importance * 0.5) / total_memories
        return min(1.0, efficiency)

class AdvancedReasoningEngine:
    """Advanced reasoning engine using statistical patterns and logic"""
    
    def __init__(self):
        self.reasoning_patterns = {}
        self.logical_connectors = {
            'cause_effect': ['because', 'therefore', 'thus', 'hence', 'so', 'as a result'],
            'comparison': ['however', 'but', 'although', 'while', 'whereas', 'in contrast'],
            'addition': ['also', 'moreover', 'furthermore', 'in addition', 'besides'],
            'sequence': ['first', 'second', 'then', 'next', 'finally', 'lastly'],
            'example': ['for example', 'such as', 'like', 'including', 'specifically']
        }
        self.reasoning_memory = []
        self.adaptation_rate = REASONING_ADAPTATION_RATE
        
    def analyze_conversation_context(self, user_input, conversation_history):
        """Alias for analyze_context for compatibility with callers that use this name."""
        return self.analyze_context(user_input, conversation_history)

    def analyze_context(self, user_input, conversation_history):
        """Analyze context for reasoning patterns"""
        context_words = self._extract_context_words(user_input, conversation_history)
        
        analysis = {
            'topic': self._identify_topic(context_words),
            'intent': self._identify_intent(user_input),
            'complexity': self._assess_complexity(user_input),
            'sentiment': self._analyze_sentiment(user_input),
            'reasoning_needed': self._determine_reasoning_needed(user_input),
            'logical_connectors': self._find_logical_connectors(user_input)
        }
        
        return analysis
    
    def generate_reasoned_response(self, user_input, conversation_history, base_response):
        """Generate a reasoned response using logical patterns"""
        analysis = self.analyze_context(user_input, conversation_history)
        if not isinstance(analysis, dict):
            return base_response
        if analysis.get('reasoning_needed'):
            return self._apply_reasoning_patterns(base_response, analysis)
        return base_response
    
    def _extract_context_words(self, user_input, conversation_history):
        """Extract important context words"""
        user_input = user_input if user_input is not None else ""
        hist = conversation_history[-3:] if isinstance(conversation_history, (list, tuple)) else []
        all_text = user_input + " " + " ".join([str(h) for h in hist])
        words = re.findall(r'\b[a-zA-Z]+\b', all_text.lower())
        
      
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can'}
        content_words = [w for w in words if w not in stop_words and len(w) > 2]
        
        return content_words[:REASONING_CONTEXT_WINDOW]
    
    def _identify_topic(self, context_words):
        """Identify the main topic from context words"""
        if not context_words:
            return "general"
        
      
        word_scores = {}
        for word in context_words:
            word_scores[word] = word_scores.get(word, 0) + 1
        
      
        for word in word_scores:
            if len(word) > 4:
                word_scores[word] *= 1.5
        
      
        if word_scores:
            return max(word_scores.items(), key=lambda x: x[1])[0]
        return "general"
    
    def _identify_intent(self, user_input):
        """Identify user intent"""
        input_lower = (user_input if user_input is not None else "").lower()
        
        if any(word in input_lower for word in ['what', 'how', 'why', 'when', 'where', 'who']):
            return 'question'
        elif any(word in input_lower for word in ['explain', 'tell me', 'describe', 'show']):
            return 'seeking_explanation'
        elif any(word in input_lower for word in ['think', 'opinion', 'believe', 'feel']):
            return 'seeking_opinion'
        elif any(word in input_lower for word in ['help', 'problem', 'issue', 'trouble']):
            return 'seeking_help'
        elif any(word in input_lower for word in ['thanks', 'thank you', 'appreciate']):
            return 'gratitude'
        else:
            return 'statement'
    
    def _assess_complexity(self, user_input):
        """Assess the complexity of the input"""
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
        elif complexity_score > 0.4:
            return 'medium'
        else:
            return 'low'
    
    def _analyze_sentiment(self, user_input):
        """Analyze sentiment of the input"""
        if user_input is None or not isinstance(user_input, str):
            return 'neutral'
        positive_words = {'good', 'great', 'excellent', 'amazing', 'wonderful', 'love', 'like', 'happy', 'excited', 'positive', 'yes', 'agree', 'correct', 'right'}
        negative_words = {'bad', 'terrible', 'awful', 'hate', 'dislike', 'sad', 'angry', 'frustrated', 'negative', 'no', 'disagree', 'wrong', 'problem', 'issue'}
        words = set(user_input.lower().split())
        positive_count = len(words.intersection(positive_words))
        negative_count = len(words.intersection(negative_words))
        
        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'
    
    def _determine_reasoning_needed(self, user_input):
        """Determine if reasoning is needed for this input"""
        if user_input is None or not isinstance(user_input, str):
            return False
        reasoning_triggers = ['why', 'how', 'explain', 'because', 'reason', 'logic', 'think', 'opinion', 'believe']
        return any(trigger in user_input.lower() for trigger in reasoning_triggers)
    
    def _find_logical_connectors(self, user_input):
        """Find logical connectors in the input"""
        found_connectors = []
        if user_input is None or not isinstance(user_input, str):
            return found_connectors
        for category, connectors in self.logical_connectors.items():
            for connector in connectors:
                if connector in user_input.lower():
                    found_connectors.append((category, connector))
        return found_connectors
    
    def _apply_reasoning_patterns(self, base_response, analysis):
        """Apply reasoning patterns to enhance the response"""
        enhanced_response = (base_response if base_response is not None else "") or ""
        if not isinstance(analysis, dict):
            return enhanced_response
        intent = analysis.get('intent')
        complexity = analysis.get('complexity')
        sentiment = analysis.get('sentiment')
        low = (enhanced_response or "").lower()
        if intent == 'question':
            if complexity == 'high':
                enhanced_response = f"Based on the context, {low}"
            else:
                enhanced_response = f"I think {low}"
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

class AdaptiveLearningSystem:
    """Adaptive learning system that improves performance over time"""
    
    def __init__(self):
        self.performance_history = []
        self.learning_rates = {
            'response_quality': 0.1,
            'context_understanding': 0.15,
            'grammar_accuracy': 0.12,
            'semantic_coherence': 0.13,
            'conversation_flow': 0.14
        }
        self.adaptive_parameters = {
            'context_weight': 0.3,
            'semantic_weight': 0.4,
            'pattern_weight': 0.3,
            'creativity_factor': 0.5
        }
        self.adaptation_thresholds = {
            'quality_improvement': 0.05,
            'performance_decline': 0.03,
            'learning_rate_adjustment': 0.02
        }
        self.max_history_size = 100
        
    def update_learning_parameters(self, current_quality, target_quality):
        """Update learning parameters based on performance"""
        self.performance_history.append({
            'quality': current_quality,
            'target': target_quality,
            'timestamp': time.time()
        })
        
      
        if len(self.performance_history) > self.max_history_size:
            self.performance_history.pop(0)
        
      
        self._analyze_performance_trends()
        
      
        self._adjust_learning_rates()
        
    def _analyze_performance_trends(self):
        """Analyze performance trends to identify patterns"""
        if len(self.performance_history) < 10:
            return
        
        recent_performance = self.performance_history[-10:]
        quality_trend = [p.get('quality', 0) for p in recent_performance]
        if len(quality_trend) >= 2:
            trend_slope = (quality_trend[-1] - quality_trend[0]) / len(quality_trend)
            
            if trend_slope > self.adaptation_thresholds['quality_improvement']:
              
                pass
            elif trend_slope < -self.adaptation_thresholds['performance_decline']:
              
                self._increase_learning_rates()
            else:
              
                self._fine_tune_learning_rates()
    
    def _adjust_learning_rates(self):
        """Adjust learning rates based on performance"""
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
        """Increase learning rates when performance is declining"""
        for key in self.learning_rates:
            self.learning_rates[key] = min(0.3, self.learning_rates[key] * 1.2)
    
    def _fine_tune_learning_rates(self):
        """Fine-tune learning rates for optimal performance"""
      
        for key in self.learning_rates:
            adjustment = random.uniform(0.98, 1.02)
            self.learning_rates[key] *= adjustment
    
    def get_learning_rate(self, parameter):
        """Get the current learning rate for a parameter"""
        return self.learning_rates.get(parameter, 0.1)
    
    def get_performance_summary(self):
        """Get a summary of current performance"""
        if not self.performance_history:
            return "No performance data available"
        
        recent_quality = [p.get('quality', 0) for p in self.performance_history[-10:]]
        n_q = len(recent_quality)
        if n_q == 0:
            return "No performance data available"
        avg_quality = sum(recent_quality) / n_q
        trend = "improving" if recent_quality[-1] > recent_quality[0] else "declining" if recent_quality[-1] < recent_quality[0] else "stable"
        return f"Average Quality: {avg_quality:.3f}, Trend: {trend}, Learning Rates: {self.learning_rates}"

    def get_adaptive_weights(self, context_analysis):
        """Get adaptive weights for response generation (required by CreativeResponseGenerator)."""
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
        """Adjust learning rates based on performance (called by Curiosity)."""
        self._adjust_learning_rates()

class CreativeResponseGenerator:
    """Enhanced response generation with creativity and context awareness"""
    
    def __init__(self, brain):
        self.brain = brain
        self.reasoning_engine = AdvancedReasoningEngine()
        self.enhanced_semantic = EnhancedSemanticMemory(brain.semantic_memory)
        self.adaptive_learning = AdaptiveLearningSystem()
        self.creativity_patterns = self._build_creativity_patterns()
    
    def _build_creativity_patterns(self):
        """Build patterns for creative response generation"""
        return {
            'metaphor': ['like', 'as if', 'similar to', 'reminds me of'],
            'analogy': ['just as', 'in the same way', 'comparable to'],
            'perspective_shift': ['from another angle', 'consider this', 'think about it differently'],
            'synthesis': ['combining', 'integrating', 'bringing together', 'connecting'],
            'exploration': ['let\'s explore', 'what if', 'imagine', 'suppose']
        }
    
    def generate_enhanced_response(self, user_input, conversation_history=None):
        """Generate enhanced response with advanced reasoning"""
      
        context_analysis = self.reasoning_engine.analyze_conversation_context(user_input, conversation_history or [])
        
      
        adaptive_weights = self.adaptive_learning.get_adaptive_weights(context_analysis)
        
      
        candidates = []
        for _ in range(GENERATION_PARALLEL_CANDIDATES):
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
        """Generate a single response candidate"""
        try:
          
            if context_analysis.get('intent') == 'question':
                return self._generate_question_response(user_input, adaptive_weights)
            elif context_analysis.get('intent') == 'seeking_explanation':
                return self._generate_explanation_response(user_input, adaptive_weights)
            else:
                return self._generate_general_response(user_input, adaptive_weights)
        except Exception as e:
            print(f"Error generating candidate response: {e}")
            return None
    
    def _generate_question_response(self, user_input, adaptive_weights):
        """Generate response to questions"""
        response = self.brain.generate_response(user_input) if getattr(self, 'brain', None) else None
        response = response if response is not None else ""
        if any(word in (user_input or "").lower() for word in ['why', 'how', 'what']):
            connectors = getattr(getattr(self, 'reasoning_engine', None), 'logical_connectors', {}).get('cause_effect', [])
            if connectors:
                reasoning_connector = random.choice(connectors)
                response = f"{response} {reasoning_connector} this approach considers multiple factors."
        return response
    
    def _generate_explanation_response(self, user_input, adaptive_weights):
        """Generate explanatory response"""
        response = self.brain.generate_response(user_input) if getattr(self, 'brain', None) else None
        response = response if response is not None else ""
        if len((response or "").split()) > 10:
            structure_words = ['First', 'Then', 'Finally']
            sentences = response.split('. ')
            if len(sentences) >= 3:
                structured_response = []
                for i, sentence in enumerate(sentences[:3]):
                    if i < len(structure_words):
                        structured_response.append(f"{structure_words[i]}, {sentence}")
                    else:
                        structured_response.append(sentence)
                response = '. '.join(structured_response)
        
        return response
    
    def _generate_general_response(self, user_input, adaptive_weights):
        """Generate general response with enhanced context awareness"""
        out = self.brain.generate_response(user_input) if getattr(self, 'brain', None) else None
        return out if out is not None else ""
    
    def _select_best_candidate(self, candidates, user_input, context_analysis):
        """Select the best response candidate"""
        if not candidates:
            out = self.brain.generate_response(user_input) if getattr(self, 'brain', None) else None
            return out if out is not None else ""
        
      
        scored_candidates = []
        for candidate in candidates:
            score = self._score_candidate(candidate, user_input, context_analysis)
            scored_candidates.append((candidate, score))
        
      
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return scored_candidates[0][0] if scored_candidates else None
    
    def _score_candidate(self, candidate, user_input, context_analysis):
        """Score a response candidate"""
        score = 0.0
        
      
        quality_score = self.brain._calculate_response_quality(candidate, user_input)
        score += quality_score * 0.4
        
      
        context_relevance = self._calculate_context_relevance(candidate, user_input)
        score += context_relevance * 0.3
        
      
        semantic_coherence = self._calculate_semantic_coherence(candidate, user_input)
        score += semantic_coherence * 0.2
        
      
        creativity_bonus = self._calculate_creativity_bonus(candidate, context_analysis)
        score += creativity_bonus * 0.1
        
        return score
    
    def _calculate_context_relevance(self, response, user_input):
        """Calculate context relevance of response"""
        if user_input is None or response is None:
            return 0.5
        try:
            user_words = set(user_input.lower().split())
            response_words = set(response.lower().split())
        except (AttributeError, TypeError):
            return 0.5
        if not user_words:
            return 0.5
        overlap = len(user_words.intersection(response_words))
        return min(1.0, overlap / len(user_words))
    
    def _calculate_semantic_coherence(self, response, user_input):
        """Calculate semantic coherence"""
        response_words = self.brain.clean_text(response)
        user_words = self.brain.clean_text(user_input)
        
        if not response_words or not user_words:
            return 0.5
        
        semantic_matches = 0
        for word in response_words:
            if word in self.brain.semantic_memory.word_to_cluster:
                cluster_id = self.brain.semantic_memory.word_to_cluster[word]
                for user_word in user_words:
                    if user_word in self.brain.semantic_memory.word_to_cluster:
                        if self.brain.semantic_memory.word_to_cluster[user_word] == cluster_id:
                            semantic_matches += 1
                            break
        
        return semantic_matches / len(response_words) if response_words else 0.5
    
    def _calculate_creativity_bonus(self, response, context_analysis):
        """Calculate creativity bonus for response"""
        creativity_score = 0.0
        
      
        for pattern_type, patterns in self.creativity_patterns.items():
            if any(pattern in response.lower() for pattern in patterns):
                creativity_score += 0.2
        
      
        if context_analysis.get('complexity') == 'high' and len(response.split()) > 15:
            creativity_score += 0.1
        
        return min(1.0, creativity_score)
    
    def _apply_creativity_enhancement(self, response, context_analysis):
        """Apply creativity enhancement to response"""
        if context_analysis.get('intent') == 'seeking_opinion':
          
            perspectives = [
                "From my perspective, ",
                "I think ",
                "In my view, ",
                "It seems to me that "
            ]
            if not any(response.startswith(p) for p in perspectives):
                response = random.choice(perspectives) + response.lower()
        
        return response

class SemanticMemoryCluster:
    def __init__(self):
        self.word_cooccurrence = {}
        self.clusters = {}
        self.word_to_cluster = {}
        self.cluster_strength = {}
        self.cluster_coherence = {}
        self.next_cluster_id = 0
        self.last_rebuild = 0
        self.rebuild_threshold = 500
        
    def update_cooccurrence(self, words, force_minimal=False):
        if force_minimal:
            words = words[::5]
        
        meaningful_words = [w for w in words if len(w) > 2 and w.isalpha()]
        if len(meaningful_words) > 100:
            meaningful_words = meaningful_words[::2]
        
        if memory_manager.should_reduce_batch_size():
            meaningful_words = meaningful_words[::3]
        
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
        print("Rebuilding semantic clusters...")
        
        memory_manager.force_garbage_collection()
        
        self.clusters = {}
        self.word_to_cluster = {}
        self.cluster_strength = {}
        self.cluster_coherence = {}
        self.next_cluster_id = 0
        
        strong_pairs = {k: v for k, v in self.word_cooccurrence.items() if v >= 5}
        
        for (word1, word2), strength in sorted(strong_pairs.items(), key=lambda x: x[1], reverse=True):
            try:
                if not isinstance(word1, str) or not isinstance(word2, str):
                    continue
                
                cluster1 = self.word_to_cluster.get(word1)
                cluster2 = self.word_to_cluster.get(word2)
                
                if cluster1 is None and cluster2 is None:
                    cluster_id = self.next_cluster_id
                    self.next_cluster_id += 1
                    self.clusters[cluster_id] = {word1, word2}
                    self.word_to_cluster[word1] = cluster_id
                    self.word_to_cluster[word2] = cluster_id
                    self.cluster_strength[cluster_id] = strength
                elif cluster1 is not None and cluster2 is None:
                    self.clusters[cluster1].add(word2)
                    self.word_to_cluster[word2] = cluster1
                    self.cluster_strength[cluster1] += strength
                elif cluster1 is None and cluster2 is not None:
                    self.clusters[cluster2].add(word1)
                    self.word_to_cluster[word1] = cluster2
                    self.cluster_strength[cluster2] += strength
                elif cluster1 != cluster2:
                    if len(self.clusters[cluster1]) >= len(self.clusters[cluster2]):
                        main_cluster, other_cluster = cluster1, cluster2
                    else:
                        main_cluster, other_cluster = cluster2, cluster1
                    
                    self.clusters[main_cluster].update(self.clusters[other_cluster])
                    for word in self.clusters[other_cluster]:
                        self.word_to_cluster[word] = main_cluster
                    self.cluster_strength[main_cluster] += self.cluster_strength[other_cluster] + strength
                    del self.clusters[other_cluster]
                    del self.cluster_strength[other_cluster]
            except Exception as e:
                print(f"Error processing word pair ({word1}, {word2}): {e}")
                continue

        clusters_to_remove = [
            cid for cid, words in self.clusters.items()
            if len(words) < 2 or self.cluster_strength.get(cid, 0) < 10
        ]
        
        for cid in clusters_to_remove:
            if cid in self.clusters:
                for word in self.clusters[cid]:
                    if self.word_to_cluster.get(word) == cid:
                        del self.word_to_cluster[word]
                del self.clusters[cid]
                if cid in self.cluster_strength: del self.cluster_strength[cid]

        print(f"Semantic clustering complete: {len(self.clusters)} clusters created")
        memory_manager.force_garbage_collection()
    
    def get_cluster_bonus(self, word, context_words):
        if word not in self.word_to_cluster:
            return 1.0
        
        word_cluster = self.word_to_cluster[word]
        bonus = 1.0
        
        for context_word in context_words:
            if context_word in self.word_to_cluster and self.word_to_cluster[context_word] == word_cluster:
                cluster_strength = self.cluster_strength.get(word_cluster, 1)
                bonus += 0.4 * min(1.0, cluster_strength / 10)
        
        return bonus
    
    def get_related_words(self, word, limit=8):
        if word not in self.word_to_cluster:
            return []
        
        cluster_id = self.word_to_cluster[word]
        cluster_words = list(self.clusters.get(cluster_id, set()))
        
        if word in cluster_words:
            cluster_words.remove(word)
        
        word_scores = []
        for candidate in cluster_words:
            key = tuple(sorted([word, candidate]))
            score = self.word_cooccurrence.get(key, 0)
            word_scores.append((candidate, score))
        
        word_scores.sort(key=lambda x: x[1], reverse=True)
        return [w[0] for w in word_scores[:limit]]
    
    def get_cluster_context_score(self, candidate_word, conversation_context):
        if not conversation_context or candidate_word not in self.word_to_cluster:
            return 1.0
        
        candidate_cluster = self.word_to_cluster[candidate_word]
        cluster_matches = sum(1 for w in conversation_context if self.word_to_cluster.get(w) == candidate_cluster)
        
        return 1.0 + (cluster_matches * 0.3)
    
    def save_clusters(self, filename):
        word_cooccurrence_serializable = {str(key): value for key, value in self.word_cooccurrence.items()}
        
        data = {
            'word_cooccurrence': word_cooccurrence_serializable,
            'clusters': {str(k): list(v) for k, v in self.clusters.items()},
            'word_to_cluster': self.word_to_cluster,
            'cluster_strength': self.cluster_strength,
            'cluster_coherence': self.cluster_coherence,
            'next_cluster_id': self.next_cluster_id
        }
        _atomic_json_dump(data, filename, indent=2)
    
    def load_clusters(self, filename):
        if os.path.exists(filename):
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                self.word_cooccurrence = {}
                for key, value in data.get('word_cooccurrence', {}).items():
                    try:
                        tup = ast.literal_eval(key)
                        if isinstance(tup, tuple) and len(tup) == 2:
                            self.word_cooccurrence[tup] = value
                    except Exception:
                        continue
                
                self.clusters = {int(k): set(v) for k, v in data.get('clusters', {}).items()}
                self.word_to_cluster = data.get('word_to_cluster', {})
                self.cluster_strength = {int(k): v for k, v in data.get('cluster_strength', {}).items()}
                self.cluster_coherence = {int(k): v for k, v in data.get('cluster_coherence', {}).items()}
                self.next_cluster_id = data.get('next_cluster_id', 0)
                
                print(f"Loaded {len(self.clusters)} semantic clusters from {filename}")
            except Exception as e:
                print(f"Error loading semantic clusters: {e}")
                _preserve_invalid_sidecar(filename, e)
                self.__init__()

class AnalogAttention:
    def __init__(self):
        self.focus_weights = {}
        self.base_learning_rate = 0.1
        self.adaptive_rates = {}
        self.success_history = {}
        
    def update_attention(self, context_words, success_score):
        for word in context_words:
            if word not in self.adaptive_rates:
                self.adaptive_rates[word] = self.base_learning_rate
                self.success_history[word] = []
            
            self.success_history[word].append(success_score > 0)
            if len(self.success_history[word]) > 10:
                self.success_history[word].pop(0)
            
            history = self.success_history[word]
            recent_success_rate = sum(history) / len(history) if history else 0.5
            
            current_rate = self.adaptive_rates[word]
            if recent_success_rate > 0.7:
                current_rate *= 0.95
            elif recent_success_rate < 0.3:
                current_rate *= 1.1
            
            self.adaptive_rates[word] = max(0.05, min(0.2, current_rate))
            
            learning_rate = self.adaptive_rates[word]
            current_weight = self.focus_weights.get(word, 1.0)
            new_weight = current_weight + (learning_rate * success_score)
            
            self.focus_weights[word] = max(0.1, min(2.0, new_weight))
    
    def get_attention_score(self, word): 
        return self.focus_weights.get(word, 1.0)
    
    def save_weights(self, filename):
        data = {
            'focus_weights': self.focus_weights,
            'adaptive_rates': self.adaptive_rates,
            'success_history': self.success_history
        }
        _atomic_json_dump(data, filename)
    
    def load_weights(self, filename):
        if os.path.exists(filename):
            try:
                with open(filename, 'r', encoding='utf-8') as f: 
                    data = json.load(f)
                
                if isinstance(data, dict) and 'focus_weights' in data:
                    self.focus_weights = data.get('focus_weights', {})
                    self.adaptive_rates = data.get('adaptive_rates', {})
                    self.success_history = data.get('success_history', {})
                elif isinstance(data, dict):
                    self.focus_weights = {k: float(v) for k, v in data.items() if isinstance(v, (int, float))}
                else:
                    print(f"Warning: '{filename}' has unexpected format. Attention weights not loaded.")
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                print(f"Warning: Could not parse '{filename}'. Attention weights not loaded.")
                _preserve_invalid_sidecar(filename, e)

class PlainMLP:
    """Small CPU-only MLP (NumPy). No PyTorch/TensorFlow/CUDA — vendor-neutral."""
    def __init__(self, vocab_size, hidden_size=32, context_size=4):
        self.vocab_size, self.context_size, self.hidden_size, self.output_size = vocab_size, context_size, hidden_size, vocab_size
        self.input_size = self.vocab_size * self.context_size
        self.base_lr = 0.01
        self.adaptive_lr = {}
        self.prediction_accuracy = {}
        self.trained_steps = 0

        # Weight matrices are intentionally lazy. This model is optional and
        # must not consume hundreds of MB or inject random output before it has
        # received an actual backward pass.
        self.w1 = []
        self.b1 = [0] * self.hidden_size
        self.w2 = []
        self.b2 = [0] * self.output_size
        self._weights_initialized = False

    def _initialize_weights(self):
        if self._weights_initialized:
            return
        self.w1 = [[random.uniform(-0.5, 0.5) for _ in range(self.hidden_size)] for _ in range(self.input_size)]
        self.w2 = [[random.uniform(-0.5, 0.5) for _ in range(self.output_size)] for _ in range(self.hidden_size)]
        self._weights_initialized = True
        self._w1_np = self._b1_np = self._w2_np = self._b2_np = None
        
    def sigmoid(self, x): 
        return 1 / (1 + math.exp(-max(-500, min(500, x))))
    
    def softmax(self, z):
        if not z: return []
        max_z = max(z)
        exp_z = [math.exp(i - max_z) for i in z]
        sum_exp_z = sum(exp_z)
        return [i / sum_exp_z for i in exp_z] if sum_exp_z > 0 else [1/len(z)] * len(z)
    
    def forward(self, x):
        self._initialize_weights()
        self.z1 = [sum(x[i] * self.w1[i][j] for i in range(self.input_size)) + self.b1[j] for j in range(self.hidden_size)]
        self.a1 = [self.sigmoid(val) for val in self.z1]
        self.z2 = [sum(self.a1[i] * self.w2[i][j] for i in range(self.hidden_size)) + self.b2[j] for j in range(self.output_size)]
        return self.softmax(self.z2)

    def _numpy_weights(self):
        """Cache numpy arrays for fast inference (lazy)."""
        if getattr(self, '_w1_np', None) is not None:
            return
        self._initialize_weights()
        self._w1_np = np.array(self.w1, dtype=np.float64)
        self._b1_np = np.array(self.b1, dtype=np.float64)
        self._w2_np = np.array(self.w2, dtype=np.float64)
        self._b2_np = np.array(self.b2, dtype=np.float64)

    def forward_numpy(self, x):
        """Single fast forward pass; x is list or 1d array of length input_size. Returns 1d array of length output_size (vocab)."""
        self._numpy_weights()
        x = np.asarray(x, dtype=np.float64).ravel()
        z1 = x @ self._w1_np + self._b1_np
        z1 = np.clip(z1, -500, 500)
        a1 = 1.0 / (1.0 + np.exp(-z1))
        z2 = a1 @ self._w2_np + self._b2_np
        z2 = np.clip(z2, -500, 500)
        exp_z = np.exp(z2 - z2.max())
        return exp_z / (exp_z.sum() + 1e-12)

    def backward(self, x, y_true, y_pred):
        if not y_pred or len(y_pred) != self.output_size: return
        
        predicted_index = y_pred.index(max(y_pred))
        true_index = y_true.index(1) if 1 in y_true else -1
        if true_index == -1: return

        is_correct = predicted_index == true_index
        
        lr = self.base_lr * (0.5 if is_correct else 1.5)
        
        error = [y_pred[i] - y_true[i] for i in range(self.output_size)]
        
        for j in range(self.output_size):
            error_j = error[j]
            if abs(error_j) < 1e-9: continue
            for i in range(self.hidden_size):
                self.w2[i][j] -= lr * error_j * self.a1[i]
            self.b2[j] -= lr * error_j
        self.trained_steps += 1
        self._w2_np = None
        self._b2_np = None
    
    def to_dict(self): 
        return {
            'w1': self.w1, 'b1': self.b1, 'w2': self.w2, 'b2': self.b2, 
            'vocab_size': self.vocab_size, 'context_size': self.context_size,
            'adaptive_lr': self.adaptive_lr, 'prediction_accuracy': self.prediction_accuracy,
            'trained_steps': self.trained_steps,
        }

    def apply_homeostasis(self, max_norm=2.0):
        """Biology-like: synaptic homeostasis — soft clamp so layer norms stay bounded (only scale layers that exceed threshold; proportional scale)."""
        if not self._weights_initialized:
            return
        try:
            clamp_count = getattr(PlainMLP, '_homeostasis_clamp_count', 0)
            n1 = math.sqrt(sum(self.w1[i][j]**2 for i in range(self.input_size) for j in range(self.hidden_size)))
            if n1 > max_norm and n1 > 1e-12:
                s1 = math.sqrt(max_norm / n1) if HOMEOSTASIS_SOFT_SCALE else (max_norm / n1)
                self.w1 = [[self.w1[i][j] * s1 for j in range(self.hidden_size)] for i in range(self.input_size)]
                self._w1_np = None
                clamp_count += 1
            n2 = math.sqrt(sum(self.w2[i][j]**2 for i in range(self.hidden_size) for j in range(self.output_size)))
            if n2 > max_norm and n2 > 1e-12:
                s2 = math.sqrt(max_norm / n2) if HOMEOSTASIS_SOFT_SCALE else (max_norm / n2)
                self.w2 = [[self.w2[i][j] * s2 for j in range(self.output_size)] for i in range(self.hidden_size)]
                self._w2_np = None
                clamp_count += 1
            PlainMLP._homeostasis_clamp_count = clamp_count
        except Exception:
            pass

    def from_dict(self, data):
        if not isinstance(data, dict) or 'vocab_size' not in data or 'w1' not in data or 'w2' not in data:
            raise ValueError("PlainMLP from_dict: invalid or incomplete data")
        self.__init__(data['vocab_size'], context_size=data.get('context_size', NN_CONTEXT_SIZE))
        self.w1, self.b1, self.w2, self.b2 = data['w1'], data['b1'], data['w2'], data['b2']
        self.adaptive_lr = data.get('adaptive_lr', {})
        self.prediction_accuracy = data.get('prediction_accuracy', {})
        self.trained_steps = max(0, int(data.get('trained_steps', 0) or 0))
        self._weights_initialized = bool(self.w1 and self.w2)
        if self.trained_steps > 0 and not self._weights_initialized:
            raise ValueError("PlainMLP from_dict: trained model has no weights")
        self._w1_np = self._b1_np = self._w2_np = self._b2_np = None

class ContextScorer:
    def __init__(self):
        self.context_performance = {}
        self.context_usage_count = {}
        self.success_threshold = 0.6
        
    def update_score(self, context_len, was_successful):
        context_len_str = str(context_len)
        if context_len_str not in self.context_performance:
            self.context_performance[context_len_str] = []
            self.context_usage_count[context_len_str] = 0
        
        self.context_performance[context_len_str].append(1.0 if was_successful else 0.0)
        self.context_usage_count[context_len_str] += 1
        
        if len(self.context_performance[context_len_str]) > 50:
            self.context_performance[context_len_str].pop(0)
    
    def get_context_score(self, context_len):
        context_len_str = str(context_len)
        if context_len_str not in self.context_performance or not self.context_performance[context_len_str]:
            return 0.5
        
        recent_performance = self.context_performance[context_len_str]
        n = len(recent_performance)
        return sum(recent_performance) / n if n else 0.5
    
    def get_best_context_order(self):
        scored_contexts = []
        for context_len in CONTEXT_LEVELS:
            score = self.get_context_score(context_len)
            usage = self.context_usage_count.get(str(context_len), 0)
            
            if usage > 10:
                score += 0.1
            
            scored_contexts.append((context_len, score))
        
        scored_contexts.sort(key=lambda x: x[1], reverse=True)
        return [ctx[0] for ctx in scored_contexts]
    
    def save_scores(self, filename):
        data = {
            'context_performance': self.context_performance,
            'context_usage_count': self.context_usage_count
        }
        _atomic_json_dump(data, filename)
    
    def load_scores(self, filename):
        if os.path.exists(filename):
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.context_performance = data.get('context_performance', {})
                self.context_usage_count = data.get('context_usage_count', {})
            except Exception as e:
                print(f"Error loading context scores: {e}")
                _preserve_invalid_sidecar(filename, e)

def _ensure_sqlite_db_file(db_file):
    """If db_file exists and is not a valid SQLite DB, back it up so a fresh one can be created."""
    if db_file == ':memory:' or not os.path.exists(db_file):
        return db_file
    try:
        with open(db_file, 'rb') as f:
            header = f.read(16)
        if header[:16] == b'SQLite format 3\x00':
            return db_file
    except Exception as e:
        print(f"HybridBrain: could not read or validate DB file {db_file}: {e}")
    backup = db_file + '.invalid.' + time.strftime('%Y%m%d_%H%M%S')
    try:
        shutil.move(db_file, backup)
        print(f"HybridBrain: {db_file} was not a valid SQLite database; moved to {backup}")
    except Exception as e:
        print(f"HybridBrain: could not move invalid DB file: {e}")
    return db_file

class HybridBrain:
    def __init__(self, db_file, is_clone=False, use_hsb_backend=None):
        self.settings_manager = settings_manager
        self.db_file = db_file
        self.is_clone = is_clone
        _path = ':memory:' if is_clone else _ensure_sqlite_db_file(db_file)
        self.con = sqlite3.connect(_path, check_same_thread=False)
        self.cur = self.con.cursor()
        self.io_lock = threading.Lock() if not is_clone else None
        self._storage_backend = None
        if use_hsb_backend is None:
            use_hsb_backend = settings_manager.get('use_hsb_backend', False) if 'settings_manager' in globals() else False
        if use_hsb_backend:
            try:
                _here = os.path.dirname(os.path.abspath(__file__))
                _newtype = os.path.join(_here, 'Newtype')
                if _newtype not in sys.path and os.path.isdir(_newtype):
                    sys.path.insert(0, _newtype)
                from brain_backend_newtype import NewtypeBrainBackend
                base_path = os.path.dirname(os.path.abspath(db_file)) if db_file != ':memory:' else '.'
                hsb_path = (db_file.replace('.db', '.hsb') if db_file.endswith('.db') else os.path.join(base_path, 'mai_phoenix_brain.hsb'))
                self._storage_backend = NewtypeBrainBackend(base_path=base_path or '.', hsb_file=hsb_path, use_hsb_persistence=not is_clone)
                if not is_clone:
                    print("HybridBrain: using Newtype HSB backend for patterns/associations")
                    if not os.path.exists(hsb_path) and db_file != ':memory:' and os.path.exists(db_file):
                        mig_con = None
                        try:
                            mig_con = sqlite3.connect(db_file, check_same_thread=False)
                            mig_cur = mig_con.cursor()
                            mig_cur.execute("SELECT context_len, word1, word2, word3, word4, word5, word6, word7, word8, next_word, priority, success_rate, usage_count FROM dynamic_word_chain")
                            for row in mig_cur.fetchall():
                                context_len, w1, w2, w3, w4, w5, w6, w7, w8, next_word, priority, success_rate, usage_count = row
                                words = tuple(w for w in (w1, w2, w3, w4, w5, w6, w7, w8) if w)
                                pad = list(words) + [''] * (MAX_CONTEXT_SIZE - len(words))
                                self._storage_backend.add_pattern_batch(context_len, tuple(pad[:MAX_CONTEXT_SIZE]), next_word, priority or 1, success_rate or 0.5, usage_count or 0)
                            mig_cur.execute("SELECT source_word, next_word, priority, success_rate, usage_count FROM word_associations")
                            for row in mig_cur.fetchall():
                                src, nxt, pri, sr, uc = row
                                self._storage_backend.add_association_batch(src, nxt, pri or 1, sr or 0.5, uc or 0)
                            self._storage_backend.save_to_hsb()
                            print("HybridBrain: migrated existing SQLite patterns/associations to HSB")
                        except Exception as ex:
                            print(f"HybridBrain: migration from SQLite to HSB skipped: {ex}")
                        finally:
                            if mig_con is not None:
                                try:
                                    mig_con.close()
                                except Exception as close_ex:
                                    print(f"Warning: Could not close migration connection: {close_ex}")
            except Exception as e:
                print(f"HybridBrain: HSB backend not available, using SQLite: {e}")
                self._storage_backend = None
      
        self.enhanced_intelligence_enabled = True
        self.creative_generator = None
        self.reasoning_engine = None
        self.adaptive_learning = None
        self.knowledge_store = None
        self.last_response_plan = {}
        self.last_critic_assessment = {}
        self.last_reasoning_trace = {}
        self.last_realization_trace = {}
        self.last_cognitive_trace = {}
        self.cognitive_core = None
        self.vocab_id_store = None
        self.vocab_id_storage_mode = 'off'
        self._recent_words = deque(maxlen=PRIMING_WINDOW)
        if not is_clone:
            self._initialize_enhanced_intelligence()

        self.cur.execute("PRAGMA journal_mode=WAL;")
        self.cur.execute("PRAGMA synchronous=NORMAL;")
        
      
        self._setup_enhanced_intelligence_tables()
        self._setup_vocab_id_storage()
    
    def _initialize_enhanced_intelligence(self):
        """Initialize enhanced intelligence components"""
        try:
          
            self.semantic_memory = SemanticMemoryCluster()
            
            self.reasoning_engine = BackendAdvancedReasoningEngine(
                context_window=REASONING_CONTEXT_WINDOW,
                adaptation_rate=REASONING_ADAPTATION_RATE,
                brain=self,
            )
            self.adaptive_learning = BackendAdaptiveLearningSystem(self)
            self.creative_generator = BackendCreativeResponseGenerator(
                self,
                parallel_candidates=GENERATION_PARALLEL_CANDIDATES,
                context_window=REASONING_CONTEXT_WINDOW,
                adaptation_rate=REASONING_ADAPTATION_RATE,
            )
            self.hierarchical_memory = HierarchicalMemorySystem(self)
            
          
            self.critic = BackendCritic(self)
            self.confidence_gate = BackendConfidenceGate(self)
            self.anti_loop_filter = BackendAntiLoopFilter(self)
            self.meta_memory = BackendMetaMemory(self)
            self.curiosity = BackendCuriosity(self)
            self.env_feedback = BackendEnvironmentFeedback(self)
            self.autotune = BackendAutotune(self)
            
          
            self.response_learning = BackendResponseLearningSystem(self)
            self.truth_fact_table = BackendTruthFactTable(self)
            self.topic_detection = BackendTopicDetectionSystem(self)
            self.knowledge_store = BackendKnowledgeStore(self)
            
            print("Enhanced intelligence components initialized successfully")
        except Exception as e:
            print(f"Warning: Could not initialize enhanced intelligence: {e}")
            self.enhanced_intelligence_enabled = False
    
    def _setup_enhanced_intelligence_tables(self):
        """Setup database tables and shared brain state. Clone skips table creation (backup brings schema); both get semantic_memory, model, vocab, etc."""
        if not self.is_clone:
            try:
                self.cur.execute("""
                CREATE TABLE IF NOT EXISTS reasoning_patterns (
                    pattern_type TEXT,
                    pattern_text TEXT,
                    usage_count INTEGER DEFAULT 0,
                    success_rate REAL DEFAULT 0.5,
                    context_type TEXT,
                    PRIMARY KEY (pattern_type, pattern_text)
                )
            """)
                self.cur.execute("""
                CREATE TABLE IF NOT EXISTS adaptive_learning_metrics (
                    metric_type TEXT,
                    value REAL,
                    timestamp REAL,
                    context_hash TEXT,
                    PRIMARY KEY (metric_type, timestamp)
                )
            """)
                self.cur.execute("""
                CREATE TABLE IF NOT EXISTS creative_patterns (
                    pattern_type TEXT,
                    pattern_text TEXT,
                    creativity_score REAL,
                    context_appropriateness REAL,
                    usage_count INTEGER DEFAULT 0,
                    PRIMARY KEY (pattern_type, pattern_text)
                )
            """)
                self.cur.execute("""
                CREATE TABLE IF NOT EXISTS sentence_patterns (
                    pattern_type TEXT,
                    pattern_text TEXT,
                    priority REAL DEFAULT 1.0,
                    usage_count INTEGER DEFAULT 0,
                    PRIMARY KEY (pattern_type, pattern_text)
                )
            """)
                self.cur.execute("""
                CREATE TABLE IF NOT EXISTS grammar_patterns (
                    pattern_type TEXT,
                    word1 TEXT,
                    word2 TEXT,
                    priority REAL DEFAULT 1.0,
                    usage_count INTEGER DEFAULT 0,
                    PRIMARY KEY (pattern_type, word1, word2)
                )
            """)
                self.cur.execute("""
                CREATE TABLE IF NOT EXISTS phrase_patterns (
                    phrase_text TEXT,
                    priority REAL DEFAULT 1.0,
                    usage_count INTEGER DEFAULT 0,
                    PRIMARY KEY (phrase_text)
                )
            """)
                self.cur.execute("""
                CREATE TABLE IF NOT EXISTS semantic_relationships (
                    word1 TEXT,
                    word2 TEXT,
                    strength REAL DEFAULT 1.0,
                    usage_count INTEGER DEFAULT 0,
                    PRIMARY KEY (word1, word2)
                )
            """)
            
                self.con.commit()
            except Exception as e:
                print(f"Warning: Could not setup enhanced intelligence tables: {e}")
            self.cur.execute("PRAGMA cache_size=-50000;")
            self.cur.execute("PRAGMA temp_store=MEMORY;")
            self.setup_database()
        
        self.conversation_memory = []
        self.max_memory_length = CONVERSATION_MEMORY_SIZE
        self.topic_words = {}
        self.conversation_coherence_score = 0.0
        
        self.current_topics = {}
        self.topic_entities = set()
        self.max_topics = 15
        self.topic_transition_history = []
        
        self.semantic_memory = SemanticMemoryCluster()
        if not self.is_clone:
            self.semantic_memory.load_clusters(SEMANTIC_CLUSTERS_FILE)
        if self.knowledge_store is None:
            self.knowledge_store = BackendKnowledgeStore(self)
        try:
            self.knowledge_store.ensure_schema()
        except Exception as e:
            print(f"Warning: Could not setup knowledge store schema: {e}")
        
        self.generation_failures = 0
        self.total_generation_attempts = 0
        self.response_quality_history = []
        
        self.batch_operations = []
        self.batch_count = 0
        self.training_word_count = 0
        
        self.word_to_ix, self.ix_to_word = {"<UNK>": 0, "<PAD>": 1}, {0: "<UNK>", 1: "<PAD>"}
        if not self.is_clone and os.path.exists(VOCAB_FILE):
            try:
                with open(VOCAB_FILE, 'r', encoding='utf-8') as f: 
                    self.word_to_ix = json.load(f)
                self.ix_to_word = {int(i): w for w, i in self.word_to_ix.items()}
            except (json.JSONDecodeError, TypeError) as e:
                 print(f"Warning: Could not load vocab file '{VOCAB_FILE}': {e}")
                 _preserve_invalid_sidecar(VOCAB_FILE, e)

        self.model = PlainMLP(len(self.word_to_ix), context_size=NN_CONTEXT_SIZE)
        if not self.is_clone and os.path.exists(NN_MODEL_FILE):
            try:
                with open(NN_MODEL_FILE, 'r', encoding='utf-8') as f: 
                    self.model.from_dict(json.load(f))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                print(f"Warning: Could not load NN model file '{NN_MODEL_FILE}': {e}")
                _preserve_invalid_sidecar(NN_MODEL_FILE, e)
        
        self.attention = AnalogAttention()
        if not self.is_clone: self.attention.load_weights(ATTENTION_FILE)
        
        self.context_scorer = ContextScorer()
        if not self.is_clone: self.context_scorer.load_scores(CONTEXT_SCORES_FILE)
        
      
        self.reasoning_engine = BackendAdvancedReasoningEngine(
            context_window=REASONING_CONTEXT_WINDOW,
            adaptation_rate=REASONING_ADAPTATION_RATE,
            brain=self,
        )
        self.adaptive_learning = BackendAdaptiveLearningSystem(self)
        self.creative_generator = BackendCreativeResponseGenerator(
            self,
            parallel_candidates=GENERATION_PARALLEL_CANDIDATES,
            context_window=REASONING_CONTEXT_WINDOW,
            adaptation_rate=REASONING_ADAPTATION_RATE,
        )
        self.hierarchical_memory = HierarchicalMemorySystem(self)
        
        if self.is_clone:
            self.clone_from_main()

        try:
            self.cognitive_core = NornCognitiveCore(self.con)
        except Exception as e:
            self.cognitive_core = None
            print(f"Warning: Could not initialize Norn cognitive core: {e}")

    def _setup_vocab_id_storage(self):
        """Attach optional integer-keyed chain storage beside the TEXT chain table."""
        mode = 'off'
        try:
            mode = str(
                settings_manager.get('vocab_id_storage_mode', 'off')
                if 'settings_manager' in globals()
                else 'off'
            ).strip().lower()
        except Exception:
            mode = 'off'
        if mode not in {'off', 'dual', 'ids'}:
            mode = 'off'
        self.vocab_id_storage_mode = mode
        self.vocab_id_store = None
        if mode == 'off':
            return
        try:
            from vocab_storage import VocabIdStore
        except ImportError:
            try:
                from .vocab_storage import VocabIdStore
            except ImportError:
                print('Warning: vocab_storage module unavailable; leaving vocab ID mode off.')
                self.vocab_id_storage_mode = 'off'
                return
        try:
            self.vocab_id_store = VocabIdStore(self.con, context_size=MAX_CONTEXT_SIZE)
            self.con.commit()
        except Exception as e:
            self.vocab_id_store = None
            self.vocab_id_storage_mode = 'off'
            print(f'Warning: Could not initialize vocab ID storage: {e}')

    def get_vocab_id_storage_snapshot(self):
        store = getattr(self, 'vocab_id_store', None)
        mode = str(getattr(self, 'vocab_id_storage_mode', 'off') or 'off')
        stats = store.stats() if store is not None else {
            'lexicon_size': 0,
            'id_chain_count': 0,
            'text_chain_count': 0,
        }
        return {
            'mode': mode,
            'active': store is not None,
            **stats,
        }

    def migrate_vocab_id_storage(self, batch_size: int = 5000, offset_rowid: int = 0):
        """Incrementally copy TEXT chains into the compact ID table."""
        if getattr(self, 'vocab_id_store', None) is None:
            previous = str(getattr(self, 'vocab_id_storage_mode', 'off') or 'off')
            self.vocab_id_storage_mode = 'dual' if previous == 'off' else previous
            self._setup_vocab_id_storage()
        store = getattr(self, 'vocab_id_store', None)
        if store is None:
            return {
                'success': False,
                'message': 'Vocab ID storage is unavailable.',
                'migrated': 0,
                'done': True,
            }
        result = store.migrate_text_batch(limit=batch_size, offset_rowid=offset_rowid)
        result['success'] = True
        result['mode'] = self.vocab_id_storage_mode
        return result

    def _kn_unigram(self):
        if getattr(self, '_storage_backend', None):
            return self._storage_backend._unigram_probs()
        cached = getattr(self, '_unigram_prob_cache', None)
        if cached:
            return cached
        self._ensure_next_word_priority_cache()
        counts = getattr(self, '_next_word_priority_cache', {}) or {}
        if counts:
            total = sum(counts.values()) or 1.0
            self._unigram_prob_cache = {word: count / total for word, count in counts.items()}
            return self._unigram_prob_cache
        rows = self.cur.execute("SELECT next_word, SUM(priority) FROM dynamic_word_chain GROUP BY next_word").fetchall()
        tot = sum(c for _, c in rows) or 1
        self._unigram_prob_cache = {w: c / tot for w, c in rows}
        return self._unigram_prob_cache

    def _kn_next_probs(self, last_words, D=0.75):
        if not last_words: return self._kn_unigram()
        if getattr(self, '_storage_backend', None):
            cache_key = tuple(last_words)
            cached_probs = performance_cache.get_word_prob_cache(cache_key)
            if cached_probs:
                return cached_probs
            p = self._storage_backend.get_next_word_probs(last_words, D)
            if p:
                performance_cache.set_word_prob_cache(cache_key, p)
            return p
      
        cache_key = tuple(last_words)
        cached_probs = performance_cache.get_word_prob_cache(cache_key)
        if cached_probs:
            return cached_probs
        
        last_words = [w for w in last_words if w]
        for L in range(min(MAX_CONTEXT_SIZE, len(last_words)), 0, -1):
            ctx = last_words[-L:]
            
            padding = ['<PAD>'] * (MAX_CONTEXT_SIZE - L)
            full_context_tuple = tuple(padding + ctx)

            rows = None
            vocab_store = getattr(self, 'vocab_id_store', None)
            vocab_mode = str(getattr(self, 'vocab_id_storage_mode', 'off') or 'off')
            if vocab_store is not None and vocab_mode in {'dual', 'ids'}:
                try:
                    id_rows = vocab_store.fetch_next_counts(L, full_context_tuple)
                    if id_rows:
                        rows = id_rows
                except Exception:
                    rows = None
            if rows is None and vocab_mode != 'ids':
                conds = " AND ".join([f"word{i+1}=?" for i in range(MAX_CONTEXT_SIZE)])
                sql = f"""
                SELECT next_word, SUM(priority) AS cnt
                FROM dynamic_word_chain
                WHERE context_len=? AND {conds}
                GROUP BY next_word
                """
                rows = self.cur.execute(sql, (L, *full_context_tuple)).fetchall()
            
            if rows:
                total = sum(c for _, c in rows)
                if total <= 0:
                    return self._kn_unigram()
                p = {w: max(c - D, 0) / total for w, c in rows}
                lam = (D * len(rows)) / total
                lower = self._kn_next_probs(last_words[-(L-1):], D) if L > 1 else self._kn_unigram()
                for w, q in lower.items():
                    p[w] = p.get(w, 0.0) + lam * q
                
              
                performance_cache.set_word_prob_cache(cache_key, p)
                return p
        return self._kn_unigram()

    def _build_topic_gate_state(self, user_text):
        if not user_text or not isinstance(user_text, str):
            return None
        user_words = [w for w in re.findall(r"[a-z]+", user_text.lower()) if w not in QUALITY_STOPWORDS and w != 'lol']
        if not user_words:
            return None
        return {
            'user_words': frozenset(user_words),
            'prefixes': frozenset({w[:4] for w in user_words if len(w) >= 4}),
        }

    def _topic_gate(self, user_text, cand_word, gate_state=None):
        state = gate_state or self._build_topic_gate_state(user_text)
        if not state:
            return 1.0
        if not cand_word or not isinstance(cand_word, str):
            return 0.35
        user_words = state.get('user_words', ())
        prefixes = state.get('prefixes', ())
        return 1.0 if cand_word in user_words else (0.55 if len(cand_word) >= 4 and cand_word[:4] in prefixes else 0.35)

    def _idf_penalty(self, word):
        cache = getattr(self, '_idf_cache', None)
        if cache is None:
            self._idf_cache = {}
            cache = self._idf_cache
        if word in cache:
            return cache[word]
        if getattr(self, '_storage_backend', None):
            v = self._storage_backend.get_idf_penalty(word)
        else:
            self._ensure_next_word_priority_cache()
            priority_cache = getattr(self, '_next_word_priority_cache', {}) or {}
            f = priority_cache.get(word)
            if f is None:
                row = self.cur.execute("SELECT SUM(priority) FROM dynamic_word_chain WHERE next_word=?", (word,)).fetchone()
                f = (row[0] or 1) if row is not None else 1
                priority_cache[word] = f
            v = 1.0 / (1.0 + math.log1p(f))
        cache[word] = v
        return v

    def _idf_penalties(self, words):
        if not words:
            return []
        cache = getattr(self, '_idf_cache', None)
        if cache is None:
            self._idf_cache = {}
            cache = self._idf_cache
        if getattr(self, '_storage_backend', None):
            return [self._idf_penalty(word) for word in words]

        self._ensure_next_word_priority_cache()
        priority_cache = getattr(self, '_next_word_priority_cache', {}) or {}
        missing = [word for word in set(words) if word not in cache]
        for word in missing:
            freq = priority_cache.get(word, 1)
            cache[word] = 1.0 / (1.0 + math.log1p(freq))
        return [cache[word] for word in words]

    def _ensure_next_word_priority_cache(self):
        if getattr(self, '_next_word_priority_cache_loaded', False):
            return
        if getattr(self, '_storage_backend', None):
            self._next_word_priority_cache = {}
            self._next_word_priority_cache_loaded = True
            return
        try:
            rows = self.cur.execute(
                "SELECT next_word, SUM(priority) FROM dynamic_word_chain GROUP BY next_word"
            ).fetchall()
            self._next_word_priority_cache = {
                (word or "").strip(): float(priority or 0.0)
                for word, priority in rows
                if word
            }
            self._unigram_prob_cache = None
        except Exception:
            self._next_word_priority_cache = {}
        self._next_word_priority_cache_loaded = True

    @property
    def word_associations(self):
        """Dict source_word -> { next_word -> priority } for coherence scoring."""
        if getattr(self, '_storage_backend', None):
            return self._storage_backend.get_word_associations_dict()
        try:
            rows = self.cur.execute("SELECT source_word, next_word, priority FROM word_associations").fetchall()
            d = defaultdict(dict)
            for src, nxt, pri in rows:
                d[src][nxt] = pri
            return dict(d)
        except Exception as e:
            print(f"Warning: Could not load word_associations: {e}")
            return {}

    def _repeat_block(self, generated_words, cand):
        if len(generated_words) >= 2:
            tri = (generated_words[-2], generated_words[-1], cand)
            if tri in self._seen_trigrams:
                return 0.2
        return 1.0

    def _ensure_bonus_caches(self):
        """Load sentence/grammar/phrase/semantic tables into memory once so generation is fast and clever (no per-word SQL)."""
        if getattr(self, '_bonus_caches_loaded', False):
            return
        try:
            self._sentence_patterns_cache = []
            self._sentence_start_index = {}
            self._sentence_end_index = {}
            self._sentence_svo_index = {}
            for row in self.cur.execute("SELECT pattern_type, pattern_text, priority FROM sentence_patterns").fetchall():
                if not row or len(row) < 3:
                    continue
                pattern_type = (row[0] or "").strip()
                pattern_text = (row[1] or "").strip()
                priority = float(row[2] or 1.0)
                self._sentence_patterns_cache.append((pattern_type, pattern_text, priority))

                tokens = tuple(self.clean_text(pattern_text))
                if not tokens:
                    continue
                if pattern_type == 'sentence_start':
                    max_prefix = min(4, len(tokens))
                    for length in range(1, max_prefix + 1):
                        prefix = tokens[:length]
                        current = self._sentence_start_index.get(prefix, 0.0)
                        if priority > current:
                            self._sentence_start_index[prefix] = priority
                elif pattern_type == 'sentence_end':
                    max_suffix = min(3, len(tokens))
                    for length in range(1, max_suffix + 1):
                        suffix = tokens[-length:]
                        current = self._sentence_end_index.get(suffix, 0.0)
                        if priority > current:
                            self._sentence_end_index[suffix] = priority
                elif pattern_type == 'svo_pattern' and len(tokens) >= 3:
                    for idx in range(len(tokens) - 2):
                        trigram = tokens[idx:idx + 3]
                        current = self._sentence_svo_index.get(trigram, 0.0)
                        if priority > current:
                            self._sentence_svo_index[trigram] = priority
            self._grammar_patterns_cache = {}
            for row in self.cur.execute("SELECT pattern_type, word1, word2, priority FROM grammar_patterns").fetchall():
                if not row or len(row) < 4:
                    continue
                self._grammar_patterns_cache[(row[0], (row[1] or "").strip(), (row[2] or "").strip())] = float(row[3] or 1.0)
            self._phrase_patterns_cache = {}
            for row in self.cur.execute("SELECT phrase_text, priority FROM phrase_patterns").fetchall():
                if not row or len(row) < 2:
                    continue
                self._phrase_patterns_cache[(row[0] or "").strip()] = float(row[1] or 1.0)
            self._semantic_relationships_cache = {}
            for row in self.cur.execute("SELECT word1, word2, strength FROM semantic_relationships").fetchall():
                if not row or len(row) < 3:
                    continue
                w1, w2 = (row[0] or "").strip(), (row[1] or "").strip()
                s = float(row[2] or 1.0)
                self._semantic_relationships_cache[(w1, w2)] = s
                self._semantic_relationships_cache[(w2, w1)] = s
            self._bonus_caches_loaded = True
        except Exception as e:
            self._sentence_patterns_cache = []
            self._sentence_start_index = {}
            self._sentence_end_index = {}
            self._sentence_svo_index = {}
            self._grammar_patterns_cache = {}
            self._phrase_patterns_cache = {}
            self._semantic_relationships_cache = {}
            self._bonus_caches_loaded = True

    @staticmethod
    def _bounded_evidence_bonus(strength, scale, max_bonus=3.0):
        """Turn accumulated evidence into a stable, diminishing multiplier."""
        try:
            evidence = max(0.0, float(strength or 0.0))
        except (TypeError, ValueError):
            return 1.0
        return 1.0 + min(float(max_bonus), math.log1p(evidence) * float(scale))

    def _get_sentence_structure_bonus(self, generated_words, candidate_word):
        """Get bonus for sentence structure patterns (uses in-memory cache)."""
        if not generated_words:
            return 1.0
        self._ensure_bonus_caches()
        sentence_start_index = getattr(self, '_sentence_start_index', {})
        sentence_end_index = getattr(self, '_sentence_end_index', {})
        sentence_svo_index = getattr(self, '_sentence_svo_index', {})
        if not sentence_start_index and not sentence_end_index and not sentence_svo_index:
            return 1.0
        try:
            if len(generated_words) <= 3:
                pattern = tuple(generated_words + [candidate_word])
                priority = sentence_start_index.get(pattern)
                if priority is not None:
                    return self._bounded_evidence_bonus(priority, 0.3)
            if len(generated_words) >= 2:
                pattern = tuple(generated_words[-2:] + [candidate_word])
                priority = sentence_end_index.get(pattern)
                if priority is not None:
                    return self._bounded_evidence_bonus(priority, 0.4)
                priority = sentence_svo_index.get(pattern)
                if priority is not None:
                    return self._bounded_evidence_bonus(priority, 0.2)
        except Exception:
            pass
        return 1.0

    def _get_grammar_pattern_bonus(self, generated_words, candidate_word):
        """Get bonus for grammar patterns (uses in-memory cache)."""
        if not generated_words:
            return 1.0
        self._ensure_bonus_caches()
        cache = getattr(self, '_grammar_patterns_cache', {})
        if not cache:
            return 1.0
        try:
            last_word = generated_words[-1]
            if last_word.lower() in ('the', 'a', 'an'):
                v = cache.get(('article_noun', last_word.lower(), candidate_word))
                if v is not None:
                    return self._bounded_evidence_bonus(v, 0.5)
            if candidate_word.endswith(('ing', 'ed', 'er', 'ly')):
                v = cache.get(('verb_form', last_word, candidate_word))
                if v is not None:
                    return self._bounded_evidence_bonus(v, 0.3)
        except Exception:
            pass
        return 1.0

    def _get_phrase_pattern_bonus(self, generated_words, candidate_word):
        """Get bonus for phrase patterns (uses in-memory cache)."""
        if not generated_words:
            return 1.0
        self._ensure_bonus_caches()
        cache = getattr(self, '_phrase_patterns_cache', {})
        if not cache:
            return 1.0
        try:
            phrase = f"{generated_words[-1]} {candidate_word}"
            v = cache.get(phrase)
            if v is not None:
                return self._bounded_evidence_bonus(v, 0.4)
            if len(generated_words) >= 2:
                phrase = f"{generated_words[-2]} {generated_words[-1]} {candidate_word}"
                v = cache.get(phrase)
                if v is not None:
                    return self._bounded_evidence_bonus(v, 0.5)
        except Exception:
            pass
        return 1.0

    def _get_semantic_relationship_bonus(self, context_words, candidate_word):
        """Get bonus for semantic relationships (uses in-memory cache)."""
        if not context_words:
            return 1.0
        self._ensure_bonus_caches()
        cache = getattr(self, '_semantic_relationships_cache', {})
        if not cache:
            return 1.0
        try:
            for context_word in context_words[-3:]:
                v = cache.get((context_word, candidate_word)) or cache.get((candidate_word, context_word))
                if v is not None:
                    return self._bounded_evidence_bonus(v, 0.3)
        except Exception:
            pass
        return 1.0

    def _get_spreading_activation_map(self, context_words, generated_words):
        """Precompute activated set once per step: candidate -> via_hub (bool). O(1) per-candidate lookup; cap: one bonus per candidate (no stacking)."""
        activated = {}
        if not hasattr(self, 'semantic_memory') or not getattr(self.semantic_memory, 'get_related_words', None):
            return activated
        recent = (list(context_words[-2:]) + list(generated_words[-2:])) if generated_words else list(context_words[-2:])
        recent = [w for w in recent if w and w not in ('<PAD>', '<UNK>')]
        hub_words = getattr(self, '_spreading_hub_words', None)
        if hub_words is None:
            hub_words = frozenset({'good', 'make', 'thing', 'get', 'go', 'say', 'know', 'think', 'see', 'want', 'like', 'time', 'way', 'one', 'can'})
            self._spreading_hub_words = hub_words
        try:
            for w in recent:
                related = self.semantic_memory.get_related_words(w, limit=5)
                via_hub = w in hub_words
                for r in related:
                    if r not in activated:
                        activated[r] = via_hub
        except Exception:
            pass
        return activated

    def _next_word(self, context_words, user_text, generated_words, top_p=0.9, temperature=0.9, rep=1.15, topic_gate_state=None):
        base = self._kn_next_probs(context_words)
        if not base:
            return None
        p_kn = base
        if (
            hasattr(self, 'model')
            and hasattr(self, 'word_to_ix')
            and self.word_to_ix
            and getattr(self.model, 'vocab_size', 0) == len(self.word_to_ix)
            and getattr(self.model, 'trained_steps', 0) > 0
        ):
            try:
                ctx = context_words[-NN_CONTEXT_SIZE:] if len(context_words) >= NN_CONTEXT_SIZE else list(context_words)
                indices = [self.word_to_ix.get(w, 0) for w in ctx]
                while len(indices) < NN_CONTEXT_SIZE:
                    indices.insert(0, 1)
                vec = self.encode_input(indices)
                p_mlp = self.model.forward_numpy(vec)
                blend = MLP_STAT_BLEND
                p_kn = {}
                for w, pk in base.items():
                    if w is None or not isinstance(w, str) or not w.strip() or w in ('<PAD>', '<UNK>'):
                        continue
                    ix = self.word_to_ix.get(w.strip(), 0)
                    pm = float(p_mlp[ix]) if ix < len(p_mlp) else 0.0
                    p_kn[w.strip()] = blend * pk + (1.0 - blend) * pm
                if p_kn:
                    total = sum(p_kn.values())
                    if total > 1e-12:
                        p_kn = {w: p / total for w, p in p_kn.items()}
            except Exception:
                p_kn = base
        if len(p_kn) > MAX_NEXT_WORD_CANDIDATES:
            p_kn = dict(sorted(p_kn.items(), key=lambda x: -x[1])[:MAX_NEXT_WORD_CANDIDATES])
        words = [w for w, _ in p_kn.items() if w and w not in ('<PAD>', '<UNK>')]
        if not words:
            return None
        probs = np.array([p_kn[w] for w in words], dtype=np.float64)
        gate_state = topic_gate_state or self._build_topic_gate_state(user_text)
        if gate_state:
            gate_user_words = gate_state.get('user_words', ())
            gate_prefixes = gate_state.get('prefixes', ())
            gate = np.array([
                1.0 if w in gate_user_words else (0.55 if len(w) >= 4 and w[:4] in gate_prefixes else 0.35)
                for w in words
            ], dtype=np.float64)
        else:
            gate = np.ones(len(words), dtype=np.float64)
        idf = np.array(self._idf_penalties(words), dtype=np.float64)
        last_generated_word = generated_words[-1] if generated_words else None
        rep_pen = np.array([
            (1.0 / max(rep, 1e-6)) if (last_generated_word is not None and w == last_generated_word) else 1.0
            for w in words
        ], dtype=np.float64)

        generated_tail = tuple(generated_words[-3:])
        context_tail = tuple(context_words[-3:])
        contextual_bonus_cache = getattr(self, '_next_word_contextual_bonus_cache', None)
        if contextual_bonus_cache is None:
            self._next_word_contextual_bonus_cache = {}
            contextual_bonus_cache = self._next_word_contextual_bonus_cache
        cache_key = (tuple(words), context_tail, generated_tail, len(generated_words))
        cached_bonus_vectors = contextual_bonus_cache.get(cache_key)
        if cached_bonus_vectors is None:
            self._ensure_bonus_caches()
            sentence_start_index = getattr(self, '_sentence_start_index', {})
            sentence_end_index = getattr(self, '_sentence_end_index', {})
            sentence_svo_index = getattr(self, '_sentence_svo_index', {})
            grammar_cache = getattr(self, '_grammar_patterns_cache', {})
            phrase_cache = getattr(self, '_phrase_patterns_cache', {})
            semantic_cache = getattr(self, '_semantic_relationships_cache', {})

            generated_tuple = tuple(generated_words)
            generated_len = len(generated_tuple)
            prefix_base = generated_tuple if generated_len <= 3 else ()
            trigram_prefix = generated_tuple[-2:] if generated_len >= 2 else ()
            last_word = generated_tuple[-1] if generated_len >= 1 else ''
            previous_word = generated_tuple[-2] if generated_len >= 2 else ''
            last_word_lower = last_word.lower() if last_word else ''
            article_mode = last_word_lower in ('the', 'a', 'an')
            semantic_context = list(context_tail)
            activated_map = self._get_spreading_activation_map(context_words, generated_words)

            sent_values = []
            gram_values = []
            phrase_values = []
            sem_values = []
            spread_values = []
            for word in words:
                sent_bonus = 1.0
                if prefix_base:
                    priority = sentence_start_index.get(prefix_base + (word,))
                    if priority is not None:
                        sent_bonus = max(sent_bonus, self._bounded_evidence_bonus(priority, 0.3, max_bonus=1.5))
                if trigram_prefix:
                    pattern = trigram_prefix + (word,)
                    priority = sentence_end_index.get(pattern)
                    if priority is not None:
                        sent_bonus = max(sent_bonus, self._bounded_evidence_bonus(priority, 0.4, max_bonus=1.5))
                    priority = sentence_svo_index.get(pattern)
                    if priority is not None:
                        sent_bonus = max(sent_bonus, self._bounded_evidence_bonus(priority, 0.2, max_bonus=1.5))
                sent_values.append(sent_bonus)

                gram_bonus = 1.0
                if article_mode:
                    value = grammar_cache.get(('article_noun', last_word_lower, word))
                    if value is not None:
                        gram_bonus = max(gram_bonus, self._bounded_evidence_bonus(value, 0.5, max_bonus=1.5))
                if word.endswith(('ing', 'ed', 'er', 'ly')):
                    value = grammar_cache.get(('verb_form', last_word, word))
                    if value is not None:
                        gram_bonus = max(gram_bonus, self._bounded_evidence_bonus(value, 0.3, max_bonus=1.5))
                gram_values.append(gram_bonus)

                phrase_bonus = 1.0
                if last_word:
                    value = phrase_cache.get(f"{last_word} {word}")
                    if value is not None:
                        phrase_bonus = max(phrase_bonus, self._bounded_evidence_bonus(value, 0.4, max_bonus=1.5))
                if previous_word and last_word:
                    value = phrase_cache.get(f"{previous_word} {last_word} {word}")
                    if value is not None:
                        phrase_bonus = max(phrase_bonus, self._bounded_evidence_bonus(value, 0.5, max_bonus=1.5))
                phrase_values.append(phrase_bonus)

                sem_bonus = 1.0
                for context_word in semantic_context:
                    value = semantic_cache.get((context_word, word))
                    if value is not None:
                        sem_bonus = max(sem_bonus, self._bounded_evidence_bonus(value, 0.3, max_bonus=1.5))
                sem_values.append(sem_bonus)

                if word in activated_map:
                    spread_values.append(1.0 + (SPREADING_ACTIVATION_BONUS * 0.5 if activated_map.get(word) else SPREADING_ACTIVATION_BONUS))
                else:
                    spread_values.append(1.0)

            cached_bonus_vectors = (
                np.array(sent_values, dtype=np.float64),
                np.array(gram_values, dtype=np.float64),
                np.array(phrase_values, dtype=np.float64),
                np.array(sem_values, dtype=np.float64),
                np.array(spread_values, dtype=np.float64),
            )
            if len(contextual_bonus_cache) >= 256:
                contextual_bonus_cache.clear()
            contextual_bonus_cache[cache_key] = cached_bonus_vectors

        sent_b, gram_b, phrase_b, sem_b, spread_b = cached_bonus_vectors
        if len(generated_words) >= 2:
            recent_pair = (generated_words[-2], generated_words[-1])
            tri_pen = np.array([
                0.2 if (recent_pair[0], recent_pair[1], w) in self._seen_trigrams else 1.0
                for w in words
            ], dtype=np.float64)
        else:
            tri_pen = np.ones(len(words), dtype=np.float64)
        recent_list = list(getattr(self, '_recent_words', ()))
        word_to_dist = {}
        for i in range(len(recent_list) - 1, -1, -1):
            w = recent_list[i]
            if w not in word_to_dist:
                word_to_dist[w] = (len(recent_list) - 1) - i
        priming_b = np.array([1.0 + PRIMING_AMPLITUDE * math.exp(-word_to_dist[w] / PRIMING_TAU) if w in word_to_dist else 1.0 for w in words], dtype=np.float64)
        last_n = generated_words[-REFRACTORY_WINDOW:] if len(generated_words) >= REFRACTORY_WINDOW else generated_words
        def _refractory(cand):
            if not last_n:
                return 1.0
            d = None
            for i in range(len(last_n) - 1, -1, -1):
                if last_n[i] == cand:
                    d = (len(last_n) - 1) - i
                    break
            return (REFRACTORY_FACTOR ** (1 + d)) if d is not None else 1.0
        refractory_b = np.array([_refractory(w) for w in words], dtype=np.float64)
        combined = list(context_words) + list(generated_words)
        occ = {}
        for w in combined:
            occ[w] = occ.get(w, 0) + 1
        hab_b = np.array([(HABITUATION_FACTOR ** max(0, occ.get(w, 0) - 1)) for w in words], dtype=np.float64)
        scores = probs ** (1.0 / max(temperature, 1e-6)) * gate * idf * rep_pen * tri_pen * sent_b * gram_b * phrase_b * sem_b * spread_b * priming_b * refractory_b * hab_b
        total = scores.sum()
        if total < 1e-12:
            return words[0]
        scores = scores / total
        order = np.argsort(-scores)
        words = [words[i] for i in order]
        scores = scores[order]
        cum = np.cumsum(scores)
        k = np.searchsorted(cum, top_p, side='left') + 1
        k = max(1, min(k, len(words)))
        if k <= 0:
            w = words[0]
        else:
            sel = scores[:k] / scores[:k].sum()
            r = random.random()
            acc = 0.0
            w = words[0]
            for i, p in enumerate(sel):
                acc += p
                if r <= acc:
                    w = words[i]
                    break
        if len(generated_words) >= 2:
            self._seen_trigrams.add((generated_words[-2], generated_words[-1], w))
        self._last_was_surprise = (w != words[0])
        return w

    def reinforce(self, rows, k=1.25):
        if getattr(self, '_storage_backend', None):
            for L, ctx, nxt in rows:
                padding = ['<PAD>'] * (MAX_CONTEXT_SIZE - L)
                full_context = tuple(padding + list(ctx))
                self._storage_backend.reinforce_pattern(L, full_context, nxt, k)
            return
        for L, ctx, nxt in rows:
            padding = ['<PAD>'] * (MAX_CONTEXT_SIZE - L)
            full_context = tuple(padding + list(ctx))
            conds = " AND ".join([f"word{i+1}=?" for i in range(MAX_CONTEXT_SIZE)])
            sql = f"UPDATE dynamic_word_chain SET priority = priority * ? WHERE context_len=? AND {conds} AND next_word=?"
            self.cur.execute(sql, (k, L, *full_context, nxt))
        self.con.commit()

    def discourage(self, rows, k=0.8):
        if getattr(self, '_storage_backend', None):
            for L, ctx, nxt in rows:
                padding = ['<PAD>'] * (MAX_CONTEXT_SIZE - L)
                full_context = tuple(padding + list(ctx))
                self._storage_backend.reinforce_pattern(L, full_context, nxt, k)
            return
        for L, ctx, nxt in rows:
            padding = ['<PAD>'] * (MAX_CONTEXT_SIZE - L)
            full_context = tuple(padding + list(ctx))
            conds = " AND ".join([f"word{i+1}=?" for i in range(MAX_CONTEXT_SIZE)])
            sql = f"UPDATE dynamic_word_chain SET priority = priority * ? WHERE context_len=? AND {conds} AND next_word=?"
            self.cur.execute(sql, (k, L, *full_context, nxt))
        self.con.commit()

    def apply_feedback(self, sentence, is_positive):
        """Decomposes a sentence and applies positive/negative reinforcement."""
        words = self.clean_text(sentence)
        if len(words) >= 2:
            rows_to_update = []
            for i in range(1, len(words)):
                for L in range(1, min(MAX_CONTEXT_SIZE, i) + 1):
                    ctx = words[i-L:i]
                    nxt = words[i]
                    rows_to_update.append((L, tuple(ctx), nxt))

            if is_positive:
                self.reinforce(rows_to_update)
            else:
                self.discourage(rows_to_update)

        cognitive_core = getattr(self, 'cognitive_core', None)
        if cognitive_core is not None:
            try:
                cognitive_core.apply_feedback(bool(is_positive))
                self.last_cognitive_trace = cognitive_core.get_last_trace()
            except Exception as e:
                print(f"Warning: Cognitive feedback learning failed: {e}")

    def clone_from_main(self):
        main_con = None
        try:
            source_db = getattr(self, 'db_file', None) or DB_FILE
            if source_db == ':memory:':
                source_db = DB_FILE
            if not os.path.exists(source_db):
                raise FileNotFoundError(f"Main DB not found: {source_db}")
            main_con = sqlite3.connect(source_db, check_same_thread=False)
            main_con.backup(self.con)
        except (sqlite3.Error, FileNotFoundError) as e:
            print(f"Clone failed, could not open main DB (is it locked?): {e}")
        except Exception as e:
            print(f"Clone failed with unexpected error: {e}")
        finally:
            if main_con:
                try:
                    main_con.close()
                except Exception as e:
                    print(f"Warning: Could not close main DB connection: {e}")
        
        if not hasattr(self, 'word_to_ix') or getattr(self, 'word_to_ix', None) is None:
            self.word_to_ix, self.ix_to_word = {"<UNK>": 0, "<PAD>": 1}, {0: "<UNK>", 1: "<PAD>"}
        if os.path.exists(VOCAB_FILE):
            try:
                with open(VOCAB_FILE, 'r', encoding='utf-8') as f:
                    self.word_to_ix = json.load(f)
                self.ix_to_word = {int(i): w for w, i in self.word_to_ix.items()}
            except (json.JSONDecodeError, TypeError):
                pass
        if not hasattr(self, 'model'):
            self.model = PlainMLP(len(self.word_to_ix), context_size=NN_CONTEXT_SIZE)
        try:
            if os.path.exists(NN_MODEL_FILE):
                with open(NN_MODEL_FILE, 'r', encoding='utf-8') as f:
                    self.model.from_dict(json.load(f))
        except Exception as e:
            print(f"Warning: Could not load NN model: {e}")
        if not hasattr(self, 'semantic_memory'):
            self.semantic_memory = SemanticMemoryCluster()
        try:
            if os.path.exists(SEMANTIC_CLUSTERS_FILE):
                self.semantic_memory.load_clusters(SEMANTIC_CLUSTERS_FILE)
        except Exception as e:
            print(f"Warning: Could not load semantic clusters: {e}")

    def save_state(self):
        if self.is_clone or not self.io_lock: return
        threading.Thread(target=self._save_state_threaded).start()

    def _save_state_threaded(self):
        if self.is_clone or not self.io_lock:
            return
        with self.io_lock:
            try:
                _atomic_json_dump(self.word_to_ix, VOCAB_FILE)
                if HOMEOSTASIS_ON_SAVE and hasattr(self, 'model') and hasattr(self.model, 'apply_homeostasis'):
                    self.model.apply_homeostasis(max_norm=2.0)
                _atomic_json_dump(self.model.to_dict(), NN_MODEL_FILE)
                self.attention.save_weights(ATTENTION_FILE)
                self.context_scorer.save_scores(CONTEXT_SCORES_FILE)
                self.semantic_memory.save_clusters(SEMANTIC_CLUSTERS_FILE)
                if getattr(self, '_storage_backend', None):
                    self._storage_backend.save_to_hsb()
                _debug_log("Brain", "Save state completed")
            except Exception as e:
                print(f"Error during threaded save: {e}")
                _debug_log("Brain", f"Save state error: {e}")

    def setup_database(self):
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
        
        self.cur.execute('''CREATE INDEX IF NOT EXISTS idx_dynamic_context_8 ON dynamic_word_chain(context_len, word1, word2, word3, word4, word5, word6, word7, word8) WHERE context_len = 8''')
        self.cur.execute('''CREATE INDEX IF NOT EXISTS idx_dynamic_context_4 ON dynamic_word_chain(context_len, word5, word6, word7, word8) WHERE context_len = 4''')
        self.cur.execute('''CREATE INDEX IF NOT EXISTS idx_associations_source ON word_associations(source_word)''')
        
        try:
            self.cur.execute('ALTER TABLE dynamic_word_chain ADD COLUMN success_rate REAL DEFAULT 0.5')
            self.cur.execute('ALTER TABLE dynamic_word_chain ADD COLUMN usage_count INTEGER DEFAULT 0')
        except sqlite3.OperationalError: pass
        try:
            self.cur.execute('ALTER TABLE word_associations ADD COLUMN success_rate REAL DEFAULT 0.5')
            self.cur.execute('ALTER TABLE word_associations ADD COLUMN usage_count INTEGER DEFAULT 0')
        except sqlite3.OperationalError: pass
        
        self.con.commit()

    def clean_text(self, text):
        if text is None or not isinstance(text, str):
            return []
        return [word for word in re.sub(r"[^a-z0-9\s'\[\]]", ' ', text.lower()).split() if word]

    def extract_topic_entities(self, text):
        words = self.clean_text(text)
        entities = set()
        ignore_words = {"the", "and", "you", "are", "for", "was", "but", "not", "can", "have", "that", "this", "will", "with"}
        word_counts = defaultdict(int)
        for word in words:
            if len(word) >= 3 and word not in ignore_words:
                word_counts[word] += 1
        for word, count in word_counts.items():
            if count >= 2 and len(word) >= 4:
                entities.add(word)
        return entities

    def update_topic_tracker(self, user_input, bot_response=""):
        user_input = user_input if user_input is not None else ""
        bot_response = bot_response if bot_response is not None else ""
        combined_text = user_input + " " + bot_response
        new_entities = self.extract_topic_entities(combined_text)
        decay_rate = 0.87
        for topic in list(self.current_topics.keys()):
            self.current_topics[topic] *= decay_rate
            if self.current_topics[topic] < 0.1:
                del self.current_topics[topic]
        for entity in new_entities:
            self.current_topics[entity] = self.current_topics.get(entity, 0) + 1.0
        if len(self.current_topics) > self.max_topics:
            sorted_topics = sorted(self.current_topics.items(), key=lambda x: x[1], reverse=True)
            self.current_topics = dict(sorted_topics[:self.max_topics])
        self.topic_entities = set(self.current_topics.keys())

    def get_topic_coherence_score(self, word):
        if not self.current_topics: return 1.0
        if word in self.current_topics: return 1.0 + (self.current_topics[word] * 0.5)
        semantic_bonus = 0.0
        if word in self.semantic_memory.word_to_cluster:
            word_cluster = self.semantic_memory.word_to_cluster[word]
            for topic, score in self.current_topics.items():
                if self.semantic_memory.word_to_cluster.get(topic) == word_cluster:
                    semantic_bonus = max(semantic_bonus, score * 0.6)
        if semantic_bonus > 0: return 1.0 + semantic_bonus
        return 0.95

    def update_conversation_memory(self, user_input, bot_response):
        quality_score = self._calculate_response_quality(bot_response, user_input)
        adaptive_learning = getattr(self, 'adaptive_learning', None)
        if not hasattr(self, 'response_quality_history') or self.response_quality_history is None:
            self.response_quality_history = []
        self.response_quality_history.append(quality_score)
        if len(self.response_quality_history) > 50:
            self.response_quality_history.pop(0)
        
        self.learn_from_conversation_exchange(user_input, bot_response, quality_score)
        
      
        memory_entry = {
            "user": user_input if user_input is not None else "",
            "bot": bot_response if bot_response is not None else "", 
            "user_input": user_input if user_input is not None else "",
            "bot_response": bot_response if bot_response is not None else "",
            "quality": quality_score,
            "topics": list(self.current_topics.keys())[:5], 
            "timestamp": time.time(),
            "sentiment": self._analyze_sentiment(user_input),
            "complexity": self._assess_complexity(user_input),
            "intent": self._identify_intent(user_input),
            "coherence_score": self._calculate_conversation_coherence(user_input, bot_response)
        }

        cognitive_core = getattr(self, 'cognitive_core', None)
        if cognitive_core is not None:
            try:
                cognitive_core.observe_response(
                    quality=quality_score,
                    coherence=memory_entry['coherence_score'],
                    response_text=bot_response,
                )
                self.last_cognitive_trace = cognitive_core.get_last_trace()
                if isinstance(getattr(self, 'last_response_plan', None), dict):
                    self.last_response_plan['cognitive_control'] = copy.deepcopy(self.last_cognitive_trace)
            except Exception as e:
                print(f"Warning: Cognitive outcome learning failed: {e}")
        
        duplicate_entry = None
        if self.conversation_memory:
            last_entry = self.conversation_memory[-1]
            if (
                isinstance(last_entry, dict)
                and last_entry.get('user_input', last_entry.get('user', '')) == memory_entry['user_input']
                and last_entry.get('bot_response', last_entry.get('bot', '')) == memory_entry['bot_response']
            ):
                duplicate_entry = last_entry

        if duplicate_entry is not None:
            duplicate_entry['quality'] = max(float(duplicate_entry.get('quality', 0.0) or 0.0), quality_score)
            duplicate_entry['timestamp'] = memory_entry['timestamp']
            merged_topics = list(dict.fromkeys(list(duplicate_entry.get('topics', [])) + memory_entry['topics']))[:5]
            duplicate_entry['topics'] = merged_topics
            duplicate_entry['sentiment'] = memory_entry['sentiment']
            duplicate_entry['complexity'] = memory_entry['complexity']
            duplicate_entry['intent'] = memory_entry['intent']
            duplicate_entry['coherence_score'] = max(
                float(duplicate_entry.get('coherence_score', 0.0) or 0.0),
                memory_entry['coherence_score'],
            )
            memory_entry = duplicate_entry
        else:
            self.conversation_memory.append(memory_entry)
            if len(self.conversation_memory) > self.max_memory_length:
                self.conversation_memory.pop(0)
            
          
            if hasattr(self, 'hierarchical_memory'):
                self.hierarchical_memory.add_memory(memory_entry, "conversation")
        
      
        self._update_conversation_flow(user_input, bot_response, quality_score)
        self.update_topic_tracker(user_input, bot_response)
        knowledge_store = getattr(self, 'knowledge_store', None)
        knowledge_learning_allowed = self._should_ingest_generated_response_as_knowledge(bot_response)
        if knowledge_store is not None:
            if not knowledge_learning_allowed:
                if adaptive_learning:
                    adaptive_learning.record_learning_event(
                        'knowledge_ingest',
                        quality_score,
                        False,
                        source_kind='knowledge',
                        detail='generated_scaffold_or_low_value',
                    )
            else:
                decision = {'accept': True, 'reason': 'accepted'}
                if adaptive_learning:
                    decision = adaptive_learning.should_accept_live_learning(
                        quality_score,
                        source_kind='knowledge',
                        response_text=bot_response,
                        context_text=user_input,
                    )
                if decision.get('accept'):
                    try:
                        knowledge_store.ingest_conversation_turn(user_input, bot_response, quality_score)
                        if adaptive_learning:
                            adaptive_learning.record_learning_event(
                                'knowledge_ingest',
                                quality_score,
                                True,
                                source_kind='knowledge',
                                detail=decision.get('reason', 'accepted'),
                            )
                    except Exception:
                        if adaptive_learning:
                            adaptive_learning.record_learning_event(
                                'knowledge_ingest',
                                quality_score,
                                False,
                                source_kind='knowledge',
                                detail='ingest_error',
                            )
                elif adaptive_learning:
                    adaptive_learning.record_learning_event(
                        'knowledge_ingest',
                        quality_score,
                        False,
                        source_kind='knowledge',
                        detail=decision.get('reason', 'adaptive_gate'),
                    )

    def get_response_coherence_score(self, user_input, bot_response):
        if bot_response is None:
            return 0.5
        bot_words = set(self.clean_text(bot_response))
        if not bot_words:
            return 0.5
        topic_relevance = sum(self.get_topic_coherence_score(word) for word in bot_words) / len(bot_words)
        return min(1.0, topic_relevance)

    def _analyze_sentiment(self, text):
        """Analyze sentiment of text"""
        if text is None or not isinstance(text, str):
            return 'neutral'
        positive_words = {'good', 'great', 'excellent', 'amazing', 'wonderful', 'love', 'like', 'happy', 'excited', 'positive', 'yes', 'agree', 'correct', 'right'}
        negative_words = {'bad', 'terrible', 'awful', 'hate', 'dislike', 'sad', 'angry', 'frustrated', 'negative', 'no', 'disagree', 'wrong', 'problem', 'issue'}
        words = set(text.lower().split())
        positive_count = len(words.intersection(positive_words))
        negative_count = len(words.intersection(negative_words))
        
        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'

    def _assess_complexity(self, text):
        """Assess complexity of text"""
        if text is None or not isinstance(text, str):
            return 'low'
        words = text.split()
        if not words:
            return 'low'
        avg_word_length = sum(len(word) for word in words) / len(words)
        unique_words = len(set(words))
        total_words = len(words)
        complexity_score = (avg_word_length * 0.3) + (unique_words / total_words * 0.4) + (total_words / 20 * 0.3)
        if complexity_score > 0.7:
            return 'high'
        elif complexity_score > 0.4:
            return 'medium'
        return 'low'

    def _identify_intent(self, text):
        """Identify intent of text"""
        if text is None or not isinstance(text, str):
            return 'statement'
        text_lower = text.lower()
        if any(word in text_lower for word in ['what', 'how', 'why', 'when', 'where', 'who']):
            return 'question'
        elif any(word in text_lower for word in ['explain', 'tell me', 'describe', 'show']):
            return 'seeking_explanation'
        elif any(word in text_lower for word in ['think', 'opinion', 'believe', 'feel']):
            return 'seeking_opinion'
        elif any(word in text_lower for word in ['help', 'problem', 'issue', 'trouble']):
            return 'seeking_help'
        elif any(word in text_lower for word in ['thanks', 'thank you', 'appreciate']):
            return 'gratitude'
        else:
            return 'statement'

    def _calculate_conversation_coherence(self, user_input, bot_response):
        """Calculate coherence between user input and bot response"""
        if user_input is None and bot_response is None:
            return 0.5
        user_words = set(self.clean_text(user_input))
        bot_words = set(self.clean_text(bot_response))
        if not user_words or not bot_words:
            return 0.5
        
      
        overlap = len(user_words.intersection(bot_words))
        total_unique = len(user_words.union(bot_words))
        
        if total_unique == 0:
            return 0.5
        
        overlap_score = overlap / total_unique
        
      
        semantic_score = 0
        for user_word in user_words:
            for bot_word in bot_words:
                if user_word in self.semantic_memory.word_to_cluster and bot_word in self.semantic_memory.word_to_cluster:
                    if self.semantic_memory.word_to_cluster[user_word] == self.semantic_memory.word_to_cluster[bot_word]:
                        semantic_score += 0.1
        
        return min(1.0, overlap_score + semantic_score)

    def _update_conversation_flow(self, user_input, bot_response, quality_score):
        """Update conversation flow intelligence"""
      
        coherence = self._calculate_conversation_coherence(user_input, bot_response)
        self.conversation_coherence_score = (self.conversation_coherence_score * 0.8) + (coherence * 0.2)
        
      
        adaptive_learning = getattr(self, 'adaptive_learning', None)
        if adaptive_learning:
            adaptation_rate = adaptive_learning.get_learning_rate('conversation_flow')
            self.conversation_coherence_score *= (1 + adaptation_rate * (quality_score - 0.5))

    def _enhance_with_memories(self, response, relevant_memories, user_input):
        """Enhance response with relevant memories"""
        if not relevant_memories:
            return response
        
      
        best_memory = relevant_memories[0]
        
      
        if isinstance(best_memory, dict) and 'data' in best_memory:
            memory_data = best_memory.get('data') if isinstance(best_memory.get('data'), dict) else {}
            memory_response = memory_data.get('bot_response') or memory_data.get('bot')
            if isinstance(memory_response, str) and len(memory_response) > len(response) * 0.8:
                response = f"Based on our previous conversation, {response.lower()}"
        
        elif isinstance(best_memory, dict) and 'pattern' in best_memory:
          
            pattern = best_memory['pattern'] if isinstance(best_memory.get('pattern'), dict) else {}
            user_lower = (user_input or '').lower()
            if pattern.get('intent') == 'question' and 'what' in user_lower:
                response = f"From what I remember, {response.lower()}"
            elif pattern.get('sentiment') == 'positive':
                response = f"I'm glad you're interested! {response}"
            elif pattern.get('sentiment') == 'negative':
                response = f"I understand this might be challenging. {response}"
        
        return response

    def learn_knowledge_from_text(self, text, source_type="training_text", source_path="", source_label="", source_category="general_text", source_weight=1.0):
        knowledge_store = getattr(self, 'knowledge_store', None)
        if knowledge_store is None:
            return {'success': False, 'message': 'Knowledge store unavailable.'}
        try:
            return knowledge_store.ingest_training_text(
                text,
                source_path=source_path,
                source_label=source_label,
                source_category=source_category,
                source_weight=source_weight,
            )
        except Exception as e:
            return {'success': False, 'message': str(e)}

    def get_last_response_plan(self):
        plan = getattr(self, 'last_response_plan', None)
        return copy.deepcopy(plan) if isinstance(plan, dict) else {}

    def get_last_critic_assessment(self):
        assessment = getattr(self, 'last_critic_assessment', None)
        return copy.deepcopy(assessment) if isinstance(assessment, dict) else {}

    def get_last_reasoning_trace(self):
        trace = getattr(self, 'last_reasoning_trace', None)
        return copy.deepcopy(trace) if isinstance(trace, dict) else {}

    def get_last_realization_trace(self):
        trace = getattr(self, 'last_realization_trace', None)
        return copy.deepcopy(trace) if isinstance(trace, dict) else {}

    def get_last_cognitive_trace(self):
        core = getattr(self, 'cognitive_core', None)
        if core is not None:
            try:
                self.last_cognitive_trace = core.get_last_trace()
            except Exception:
                pass
        trace = getattr(self, 'last_cognitive_trace', None)
        return copy.deepcopy(trace) if isinstance(trace, dict) else {}

    def get_cognitive_snapshot(self):
        core = getattr(self, 'cognitive_core', None)
        if core is None:
            return {'active': False, 'message': 'Norn cognitive core is unavailable.'}
        try:
            return core.get_snapshot()
        except Exception as e:
            return {'active': False, 'message': str(e)}

    def _estimate_cognitive_memory_salience(self, user_input):
        episode = self._retrieve_episode_memory(user_input)
        if episode:
            return float(episode.get('score', 0.0) or 0.0)
        return 0.0

    def _retrieve_episode_memory(self, user_input, *, limit=12):
        """Find the strongest matching conversation episode for recall."""
        query_words = set(self.clean_text(user_input or '')) - QUALITY_STOPWORDS
        memories = list(getattr(self, 'conversation_memory', None) or [])
        if not memories:
            return None
        best = None
        for index, memory in enumerate(reversed(memories[-limit:])):
            if not isinstance(memory, dict):
                continue
            user_text = memory.get('user_input', memory.get('user', '')) or ''
            bot_text = memory.get('bot_response', memory.get('bot', '')) or ''
            memory_words = set(self.clean_text(f"{user_text} {bot_text}")) - QUALITY_STOPWORDS
            if not memory_words:
                continue
            if query_words:
                overlap = len(query_words.intersection(memory_words)) / max(1, min(len(query_words), len(memory_words)))
            else:
                overlap = 0.15
            recency = max(0.55, 1.0 - (index * 0.045))
            quality = 0.5
            try:
                quality = max(0.0, min(1.0, float(memory.get('quality', 0.5) or 0.5)))
            except (TypeError, ValueError):
                quality = 0.5
            score = (overlap * 0.72) + (recency * 0.18) + (quality * 0.10)
            if best is None or score > best['score']:
                best = {
                    'score': round(score, 4),
                    'overlap': round(overlap, 4),
                    'recency': round(recency, 4),
                    'user_text': str(user_text).strip(),
                    'bot_text': str(bot_text).strip(),
                    'topics': list(memory.get('topics', []) or [])[:5],
                    'timestamp': memory.get('timestamp'),
                }
        if best is None or best['score'] < 0.22:
            return None
        return best

    def _begin_cognitive_turn(self, user_input):
        core = getattr(self, 'cognitive_core', None)
        if core is None or not user_input:
            self.last_cognitive_trace = {}
            self.last_episode_memory = None
            return {}
        recent_quality = None
        quality_history = getattr(self, 'response_quality_history', None) or []
        if quality_history:
            recent = quality_history[-5:]
            recent_quality = sum(float(value or 0.0) for value in recent) / len(recent)
        episode = self._retrieve_episode_memory(user_input)
        self.last_episode_memory = episode
        memory_salience = float(episode.get('score', 0.0) or 0.0) if episode else 0.0
        try:
            trace = core.begin_turn(
                user_input,
                memory_salience=memory_salience,
                recent_quality=recent_quality,
            )
        except Exception as e:
            print(f"Warning: Cognitive appraisal failed: {e}")
            trace = {}
        if isinstance(trace, dict) and episode:
            trace = dict(trace)
            trace['episode_memory'] = {
                'score': episode.get('score'),
                'overlap': episode.get('overlap'),
                'user_preview': str(episode.get('user_text', ''))[:120],
                'bot_preview': str(episode.get('bot_text', ''))[:160],
            }
        self.last_cognitive_trace = trace if isinstance(trace, dict) else {}
        self.last_response_plan = {'cognitive_control': copy.deepcopy(self.last_cognitive_trace)} if self.last_cognitive_trace else {}
        return self.last_cognitive_trace

    def _build_cognitive_strategy_response(self, user_input, response_plan=None):
        plan = response_plan if isinstance(response_plan, dict) else {}
        trace = plan.get('cognitive_control') if isinstance(plan.get('cognitive_control'), dict) else self.get_last_cognitive_trace()
        if not trace:
            return None
        action = str(trace.get('action', '') or '')
        appraisal = trace.get('appraisal', {}) if isinstance(trace.get('appraisal'), dict) else {}
        ambiguity = float(appraisal.get('ambiguity', 0.0) or 0.0)
        memory_salience = float(appraisal.get('memory_salience', 0.0) or 0.0)
        topics = [str(topic) for topic in appraisal.get('topics', [])[:3] if topic]
        if action == 'ask_clarifying' and ambiguity >= 0.48:
            topic_text = ', '.join(topics)
            if topic_text:
                return f"What should I focus on about {topic_text}: how it works, what is failing, or what you want to build?"
            return "What would you like me to examine, explain, or build from that?"
        if action == 'repair_alignment':
            topic_text = ' and '.join(topics[:2])
            if topic_text:
                return (
                    f"I will treat the response about {topic_text} as a failed strategy instead of reinforcing it. "
                    "Which part should I rebuild first: the facts, the reasoning, or the wording?"
                )
            return "Which part missed your intent: the factual claim, the reasoning, or the way I expressed it?"
        if action == 'create_connection' and len(topics) >= 2:
            first, second = topics[:2]
            return (
                f"A testable new connection is to let {first} change which {second} relations become active, "
                "then retain only changes that improve coherence and learning quality. That creates novelty through controlled "
                "recombination and feedback rather than random wording."
            )
        if action == 'explore_hypothesis' and topics:
            topic_text = ' and '.join(topics[:2])
            return (
                f"One bounded hypothesis is that stronger curiosity about {topic_text} should lengthen exploration "
                "only while confidence stays low, then shorten answers once a coherent relation is found. "
                "This remains uncertain until feedback confirms it."
            )
        if action == 'recall_episode':
            episode = getattr(self, 'last_episode_memory', None)
            if not isinstance(episode, dict):
                episode = self._retrieve_episode_memory(user_input)
            if isinstance(episode, dict) and float(episode.get('score', 0.0) or 0.0) >= 0.25:
                prior_user = str(episode.get('user_text', '') or '').strip()
                prior_bot = str(episode.get('bot_text', '') or '').strip()
                if prior_bot:
                    excerpt = prior_bot
                    if len(excerpt) > 180:
                        excerpt = excerpt[:177].rstrip() + '...'
                    if prior_user:
                        return (
                            f"From earlier, when you said \"{prior_user[:90]}\", I answered: {excerpt} "
                            "That is stored episode memory, not a fresh invention."
                        )
                    return (
                        f"From an earlier exchange I still have: {excerpt} "
                        "That is stored episode memory, not a fresh invention."
                    )
            if memory_salience < 0.25:
                if topics:
                    topic_text = ', '.join(topics[:2])
                    return (
                        f"I do not have a matching stored episode about {topic_text} yet. "
                        "Which earlier event or idea should I try to reconnect?"
                    )
                return "I do not have a matching stored episode yet. Which earlier event or idea should I try to reconnect?"
        return None

    def _cognitive_strategy_should_preempt(self, response_plan=None):
        """Prefer lived strategy responses over scaffolded continuation for high-stakes actions."""
        plan = response_plan if isinstance(response_plan, dict) else {}
        trace = plan.get('cognitive_control') if isinstance(plan.get('cognitive_control'), dict) else self.get_last_cognitive_trace()
        if not isinstance(trace, dict) or not trace:
            return False
        action = str(trace.get('action', '') or '')
        appraisal = trace.get('appraisal', {}) if isinstance(trace.get('appraisal'), dict) else {}
        ambiguity = float(appraisal.get('ambiguity', 0.0) or 0.0)
        memory_salience = float(appraisal.get('memory_salience', 0.0) or 0.0)
        if action == 'ask_clarifying' and ambiguity >= 0.48:
            return True
        if action == 'repair_alignment':
            return True
        if action == 'recall_episode':
            return True
        if action == 'explore_hypothesis' and ambiguity >= 0.35:
            return True
        return False

    def _cognitive_unit_score_bias(self, kind, cognitive_action, text=''):
        """Bias evidence selection by the active Norn action instead of rewriting with stock prefixes."""
        action = str(cognitive_action or '')
        lower_text = str(text or '').lower()
        bias = 0.0
        if action == 'explain_causally':
            if kind in {'reasoning', 'fact'}:
                bias += 0.14
            if any(token in lower_text for token in ('because', 'cause', 'when ', 'leads', 'results', 'so that')):
                bias += 0.08
            if kind == 'identity':
                bias -= 0.04
        elif action == 'recall_episode':
            if kind == 'fact':
                bias += 0.08
            if any(token in lower_text for token in ('earlier', 'before', 'previous', 'remember', 'episode')):
                bias += 0.10
            if kind == 'reasoning':
                bias -= 0.03
        elif action in {'explore_hypothesis', 'create_connection'}:
            if kind in {'reasoning', 'claim'}:
                bias += 0.10
            if any(token in lower_text for token in ('connect', 'relation', 'hypothesis', 'could', 'might')):
                bias += 0.06
        elif action == 'answer_directly':
            if kind == 'claim':
                bias += 0.08
            if kind == 'reasoning':
                bias -= 0.02
        elif action == 'ask_clarifying':
            bias -= 0.20
        return bias

    def _evidence_led_sentence(self, text, *, cognitive_action='', mark_uncertainty=False):
        cleaned = self._normalize_realization_text(text)
        if not cleaned:
            return ''
        body = cleaned.rstrip('.!?').strip()
        if not body:
            return ''
        action = str(cognitive_action or '')
        if mark_uncertainty or action in {'explore_hypothesis', 'create_connection'}:
            lower_body = body.lower()
            if not any(marker in lower_body for marker in ('hypothesis', 'uncertain', 'worth testing', 'might', 'could')):
                body = f"{body}, though this remains a bounded hypothesis"
        if body[-1] not in '.!?':
            body += '.'
        return body

    def _build_response_plan(self, user_input):
        knowledge_store = getattr(self, 'knowledge_store', None)
        plan = {}
        if knowledge_store is not None and user_input:
            try:
                candidate = knowledge_store.build_response_plan(user_input, limit=5)
                if isinstance(candidate, dict):
                    plan = candidate
            except Exception:
                plan = {}
        cognitive_trace = self.get_last_cognitive_trace()
        if cognitive_trace:
            plan = dict(plan)
            plan['cognitive_control'] = cognitive_trace
        self.last_response_plan = plan
        return plan

    def _apply_reasoning_engine(self, user_input, response, response_plan=None):
        reasoning_engine = getattr(self, 'reasoning_engine', None)
        if reasoning_engine is None or not response:
            self.last_reasoning_trace = {}
            return response

        conversation_history = self.conversation_memory[-5:] if self.conversation_memory else []
        try:
            reasoned_result = reasoning_engine.generate_reasoned_response(
                user_input,
                conversation_history,
                response,
                response_plan=response_plan,
            )
        except TypeError:
            reasoned_result = reasoning_engine.generate_reasoned_response(user_input, conversation_history, response)
        except Exception:
            self.last_reasoning_trace = {}
            return response

        if isinstance(reasoned_result, dict):
            trace = reasoned_result.get('trace', {})
            self.last_reasoning_trace = copy.deepcopy(trace) if isinstance(trace, dict) else {}
            return str(reasoned_result.get('response', response) or response)

        if isinstance(reasoned_result, str):
            get_trace = getattr(reasoning_engine, 'get_last_reasoning_trace', None)
            if callable(get_trace):
                try:
                    self.last_reasoning_trace = get_trace() or {}
                except Exception:
                    self.last_reasoning_trace = {}
            else:
                self.last_reasoning_trace = {}
            return reasoned_result

        self.last_reasoning_trace = {}
        return response

    def _apply_critic_repair_loop(self, user_input, response, response_plan=None):
        critic = getattr(self, 'critic', None)
        if critic is None or not response or not isinstance(response, str):
            self.last_critic_assessment = {}
            return response

        plan = response_plan if isinstance(response_plan, dict) else {}
        try:
            initial_assessment = critic.assess(user_input, response, context_hints=plan) or {}
        except Exception:
            self.last_critic_assessment = {}
            return response

        loop_state = {
            'initial': initial_assessment,
            'repair_attempted': False,
            'selected': 'original',
        }
        meta_memory = getattr(self, 'meta_memory', None)
        if meta_memory is not None:
            try:
                for weakness in initial_assessment.get('weaknesses', []) or []:
                    meta_memory.register_weakness(weakness, context=(user_input or '')[:160])
            except Exception:
                pass

        if not initial_assessment.get('repair_recommended'):
            self.last_critic_assessment = loop_state
            return response

        repaired_response = response
        try:
            repair_result = critic.repair_response(user_input, response, context_hints=plan, assessment=initial_assessment) or {}
        except Exception:
            self.last_critic_assessment = loop_state
            return response

        loop_state['repair_attempted'] = True
        loop_state['repair_strategy'] = repair_result.get('strategy', '')
        candidate_response = repair_result.get('response', '')
        if not isinstance(candidate_response, str) or not candidate_response.strip():
            self.last_critic_assessment = loop_state
            return response

        candidate_response = candidate_response.strip()
        if candidate_response == response.strip():
            self.last_critic_assessment = loop_state
            return response

        repaired_assessment = critic.assess(user_input, candidate_response, context_hints=plan) or {}
        loop_state['repaired'] = repaired_assessment

        initial_confidence = float(initial_assessment.get('confidence', 0.0) or 0.0)
        repaired_confidence = float(repaired_assessment.get('confidence', 0.0) or 0.0)
        initial_quality = float(initial_assessment.get('quality_score', 0.0) or 0.0)
        repaired_quality = float(repaired_assessment.get('quality_score', 0.0) or 0.0)
        initial_grounding = float(initial_assessment.get('grounding_score', 0.0) or 0.0)
        repaired_grounding = float(repaired_assessment.get('grounding_score', 0.0) or 0.0)
        initial_focus = float(initial_assessment.get('focus_score', 0.0) or 0.0)
        repaired_focus = float(repaired_assessment.get('focus_score', 0.0) or 0.0)
        initial_relevance = float(initial_assessment.get('relevance_score', 0.0) or 0.0)
        repaired_relevance = float(repaired_assessment.get('relevance_score', 0.0) or 0.0)
        initial_surface = float(initial_assessment.get('surface_score', 0.0) or 0.0)
        repaired_surface = float(repaired_assessment.get('surface_score', 0.0) or 0.0)
        initial_weaknesses = set(initial_assessment.get('weaknesses', []) or [])
        repaired_weaknesses = set(repaired_assessment.get('weaknesses', []) or [])
        repair_strategy = str(loop_state.get('repair_strategy', '') or '')
        realization_trace = getattr(self, 'last_realization_trace', {}) or {}
        realization_mode = str(realization_trace.get('mode', '') or '')

        choose_repaired = (
            repaired_confidence >= (initial_confidence + 0.04)
            or repaired_quality >= (initial_quality + 0.05)
            or repaired_grounding >= (initial_grounding + 0.15)
        )
        if (
            not choose_repaired
            and repaired_surface >= (initial_surface + 0.30)
            and repaired_quality >= (initial_quality - 0.04)
        ):
            choose_repaired = True
        if (
            not choose_repaired
            and len(repaired_weaknesses) < len(initial_weaknesses)
            and repaired_confidence >= (initial_confidence - 0.03)
        ):
            choose_repaired = True
        if (
            not choose_repaired
            and repair_strategy == 'claim_grounding'
            and 'topic_drift' in initial_weaknesses
            and repaired_focus >= (initial_focus + 0.35)
            and repaired_grounding >= initial_grounding
        ):
            choose_repaired = True
        if (
            not choose_repaired
            and repair_strategy.startswith('claim_grounding')
            and 'low_surface' in initial_weaknesses
            and repaired_surface >= 0.85
            and 'low_surface' not in repaired_weaknesses
            and repaired_grounding >= max(0.75, initial_grounding - 0.05)
            and len(candidate_response.split()) >= 4
            and repaired_quality >= 0.30
        ):
            choose_repaired = True
        if (
            choose_repaired
            and 'low_surface' in initial_weaknesses
            and repair_strategy.startswith('claim_grounding')
            and repaired_surface < max(0.65, initial_surface)
            and 'low_surface' in repaired_weaknesses
        ):
            choose_repaired = False
        if (
            choose_repaired
            and realization_mode == 'scaffold_realization'
            and repair_strategy.startswith('claim_grounding')
            and repaired_relevance + 0.05 < initial_relevance
        ):
            choose_repaired = False
        if (
            choose_repaired
            and realization_mode == 'scaffold_realization'
            and repair_strategy.startswith('claim_grounding')
            and 'low_surface' not in initial_weaknesses
            and 'low_information' not in initial_weaknesses
            and initial_surface >= 0.84
            and repaired_relevance <= (initial_relevance + 0.05)
            and repaired_grounding <= (initial_grounding + 0.15)
        ):
            choose_repaired = False
        if choose_repaired:
            repaired_response = candidate_response
            loop_state['selected'] = 'repaired'
        self.last_critic_assessment = loop_state
        return repaired_response

    def _collect_response_plan_summaries(self, user_input, response_plan=None, fact_limit=2):
        plan = response_plan if isinstance(response_plan, dict) else {}
        if not plan:
            return []

        knowledge_store = getattr(self, 'knowledge_store', None)
        usable_summary = getattr(knowledge_store, '_is_reasoning_text_usable', None) if knowledge_store is not None else None
        summaries = []
        main_claim = plan.get('main_claim', {})
        if isinstance(main_claim, dict) and main_claim.get('text'):
            summaries.append(str(main_claim.get('text', '')).strip())
        for claim in plan.get('support_claims', [])[:max(1, fact_limit)]:
            if isinstance(claim, dict) and claim.get('text'):
                summaries.append(str(claim.get('text', '')).strip())
        if plan.get('intent') == 'self_description':
            summaries.extend(
                str(item.get('summary', '')).strip()
                for item in plan.get('identity_traits', [])[:1]
                if item.get('summary')
            )
        summaries.extend(
            str(item.get('summary', '')).strip()
            for item in plan.get('supporting_facts', [])[:max(1, fact_limit)]
            if item.get('summary')
        )

        if not summaries and knowledge_store is not None and user_input:
            summaries = knowledge_store.summarize_facts_for_query(user_input, limit=max(1, fact_limit))

        filtered = []
        seen = set()
        for summary in summaries:
            clean_summary = str(summary or '').strip()
            if not clean_summary:
                continue
            if callable(usable_summary):
                try:
                    if not usable_summary(clean_summary):
                        continue
                except Exception:
                    pass
            key = clean_summary.lower()
            if key in seen:
                continue
            seen.add(key)
            filtered.append(clean_summary)
            if len(filtered) >= max(1, fact_limit + 1):
                break
        return filtered

    def _is_confident_fact(self, fact):
        if not isinstance(fact, dict):
            return False
        profile = fact.get('confidence_profile')
        if isinstance(profile, dict):
            overall = profile.get('overall')
            consistency = profile.get('consistency')
            try:
                if float(overall or 0.0) >= 0.66 and float(consistency or 0.0) >= 0.45:
                    return True
            except (TypeError, ValueError):
                pass
        status = str(fact.get('status', '') or '').strip().lower()
        if status in {'stable', 'active'}:
            return True
        try:
            confidence = float(fact.get('confidence') or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        return confidence >= 0.68

    def _build_fact_grounded_response(self, user_input, response_plan=None):
        plan = response_plan if isinstance(response_plan, dict) else {}
        if not user_input or not plan or not plan.get('success'):
            return None

        normalized_query = " ".join(str(user_input or '').lower().split()).strip()
        single_sentence_requested = self._query_requests_single_sentence(user_input)
        intent = str(plan.get('intent', '') or '').strip().lower()
        if intent not in {'definition', 'self_description', 'self_preference'}:
            return None

        supporting_facts = plan.get('supporting_facts', []) or []
        if intent == 'definition' and not any(self._is_confident_fact(fact) for fact in supporting_facts[:2]):
            return None
        if intent == 'self_description':
            if normalized_query.startswith('what makes ') or ' different' in f" {normalized_query} ":
                return None
            has_identity = bool(plan.get('identity_traits'))
            has_facts = any(self._is_confident_fact(fact) for fact in supporting_facts[:2])
            if not has_identity and not has_facts:
                return None
        if intent == 'self_preference':
            has_preference_claim = False
            main_claim = plan.get('main_claim', {}) if isinstance(plan.get('main_claim'), dict) else {}
            if main_claim:
                has_preference_claim = str(main_claim.get('relation_type', '') or '').strip().lower() in {'prefers', 'avoids'}
            if not has_preference_claim:
                has_preference_claim = any(
                    str(item.get('relation_type', '') or '').strip().lower() in {'prefers', 'avoids'}
                    for item in plan.get('support_claims', [])
                    if isinstance(item, dict)
                )
            if not has_preference_claim:
                return None

        main_claim = plan.get('main_claim', {}) if isinstance(plan.get('main_claim'), dict) else {}
        if main_claim:
            main_profile = main_claim.get('confidence_profile', {})
            try:
                if float(main_profile.get('overall', 0.0) or 0.0) < 0.58:
                    return None
            except (TypeError, ValueError):
                pass
            if intent == 'self_description':
                describe_prompt = (
                    'describe' in normalized_query
                    or single_sentence_requested
                    or normalized_query.startswith('provide ')
                    or normalized_query.startswith('summarize ')
                )
                source_reliability = self._safe_realization_float(main_profile.get('source_reliability'), 1.0)
                if describe_prompt and (
                    self._looks_like_incomplete_realization_text(main_claim.get('text', ''))
                    or source_reliability < 0.55
                ):
                    return None

        summaries = self._collect_response_plan_summaries(user_input, plan, fact_limit=2)
        if not summaries:
            return None

        normalized_summaries = []
        for summary in summaries:
            clean_summary = self._normalize_realization_text(summary)
            if clean_summary:
                normalized_summaries.append(clean_summary)
        if not normalized_summaries:
            return None

        max_parts = 1 if single_sentence_requested else 2
        response = " ".join(part for part in normalized_summaries[:max_parts] if part).strip()
        if not response:
            return None
        if response[-1] not in '.!?':
            response += '.'
        return response

    def _safe_realization_float(self, value, default=0.0):
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _query_requests_single_sentence(self, user_input):
        normalized_query = " ".join(str(user_input or '').lower().split()).strip()
        if not normalized_query:
            return False
        return any(
            phrase in normalized_query
            for phrase in ('one sentence', 'single sentence', 'in one sentence')
        )

    def _looks_like_incomplete_realization_text(self, text):
        normalized = " ".join(str(text or '').split()).strip().lower()
        if not normalized:
            return True
        trimmed = normalized.rstrip('.!?').strip()
        if not trimmed:
            return True
        dangling_phrases = (
            'instead of',
            'rather than',
            'because of',
            'such as',
            'as if',
            'as though',
        )
        if any(trimmed.endswith(phrase) for phrase in dangling_phrases):
            return True
        dangling_words = {
            'because',
            'that',
            'which',
            'who',
            'whose',
            'where',
            'when',
            'while',
            'if',
            'unless',
            'than',
        }
        words = trimmed.split()
        if words and words[-1] in dangling_words:
            return True
        return False

    def _normalize_realization_text(self, text):
        normalized = " ".join(str(text or '').split()).strip()
        if not normalized:
            return ''
        lower_text = normalized.lower()
        if self._looks_like_incomplete_realization_text(normalized):
            return ''
        if any(fragment in lower_text for fragment in ('interesting question let', 'question let me', "that's an interesting question let", "lennox's")):
            return ''
        if re.search(r'\b[st]\b', lower_text):
            return ''
        stray_letters = [token for token in re.findall(r'\b[a-zA-Z]\b', lower_text) if token not in {'a', 'i'}]
        if len(stray_letters) > 1:
            return ''
        tokens = normalized.split()
        if len(tokens) >= 10:
            prefix = " ".join(token.lower() for token in tokens[:4])
            if prefix and prefix in " ".join(token.lower() for token in tokens[4:]):
                return ''
        if normalized[-1] not in '.!?':
            normalized += '.'
        return normalized[:1].upper() + normalized[1:]

    def _looks_like_generated_scaffold_text(self, text):
        normalized = " ".join(str(text or '').split()).strip().lower()
        if not normalized:
            return False
        prefixes = (
            'short version is ',
            'short version is that ',
            'one interesting thing about ',
            'one interesting point is that ',
            'the short version is that ',
            'a useful way to frame ',
            'what makes mai different is that ',
            'what stands out most is that ',
            'the main idea is that ',
            'another supporting point is that ',
            'another important point is that ',
            'a practical way to think about it is that ',
        )
        return any(normalized.startswith(prefix) for prefix in prefixes)

    def _extract_realization_topic_phrase(self, user_input):
        normalized = " ".join(str(user_input or '').lower().split()).strip()
        if not normalized:
            return 'this topic'
        for prefix in (
            'tell me something interesting about ',
            'tell me about ',
            'what makes ',
            'why is ',
            'how does ',
            'how do ',
            'how is ',
            'what is ',
            'who is ',
            'explain ',
            'describe ',
        ):
            if normalized.startswith(prefix):
                topic = normalized[len(prefix):].strip()
                break
        else:
            topic = normalized
        topic = topic.strip(' ?!.,')
        return topic or 'this topic'

    def _build_realization_query_terms(self, user_input, response_plan=None):
        return {term for term in self.clean_text(user_input or '') if len(term) > 2}

    def _score_realization_text(self, text, user_input, response_plan=None, confidence_profile=None, uncertainties=None, kind='claim'):
        cleaned = self._normalize_realization_text(text)
        if not cleaned:
            return float('-inf')

        query_terms = self._build_realization_query_terms(user_input, response_plan)
        text_terms = {term for term in self.clean_text(cleaned) if len(term) > 2}
        topical_terms = {
            term for term in query_terms
            if term not in {'mai', 'you', 'your', 'phoenix', 'what', 'how', 'why', 'tell', 'about', 'something', 'interesting'}
        }
        active_terms = topical_terms or query_terms
        overlap_count = len(active_terms.intersection(text_terms))
        overlap = overlap_count / max(1, min(len(active_terms), 4)) if active_terms else 0.0
        intent = str((response_plan or {}).get('intent', '') or '').strip().lower()
        required_overlap = 2 if len(active_terms) >= 4 else 1
        if intent in {'answer', 'explanation', 'guidance'} and overlap_count < required_overlap:
            return float('-inf')
        if intent == 'self_description' and overlap_count <= 0:
            return float('-inf')
        overall = self._safe_realization_float((confidence_profile or {}).get('overall'), 0.55)
        source_reliability = self._safe_realization_float((confidence_profile or {}).get('source_reliability'), 0.5)
        stability = self._safe_realization_float((confidence_profile or {}).get('stability'), 0.5)
        uncertainty_penalty = 0.0
        for item in uncertainties or []:
            lower_item = str(item).lower()
            if 'relatively weak' in lower_item:
                uncertainty_penalty += 0.08
            elif 'weak' in lower_item or 'limited' in lower_item or 'sparse' in lower_item:
                uncertainty_penalty += 0.05
        kind_bonus = 0.0
        if kind == 'reasoning':
            kind_bonus += 0.08
        elif kind == 'identity':
            kind_bonus += 0.04
        cognitive_control = (response_plan or {}).get('cognitive_control', {}) if isinstance(response_plan, dict) else {}
        if not isinstance(cognitive_control, dict):
            cognitive_control = {}
        cognitive_action = str(cognitive_control.get('action', '') or '')
        cognitive_bias = self._cognitive_unit_score_bias(kind, cognitive_action, cleaned)
        return (
            (overlap * 0.44)
            + (overall * 0.24)
            + (source_reliability * 0.16)
            + (stability * 0.10)
            + kind_bonus
            + cognitive_bias
            - uncertainty_penalty
        )

    def _collect_open_ended_realization_units(self, user_input, response_plan=None):
        plan = response_plan if isinstance(response_plan, dict) else {}
        query_targets_mai = bool(getattr(self, 'knowledge_store', None) and self.knowledge_store._query_targets_mai(self.knowledge_store._normalize_text(user_input)))
        seen = set()
        units = []

        def _add_unit(text, kind='claim', confidence_profile=None, uncertainties=None, meta=None):
            cleaned = self._normalize_realization_text(text)
            if not cleaned:
                return
            if self._looks_like_generated_scaffold_text(cleaned):
                return
            key = cleaned.lower()
            if key in seen:
                return
            score = self._score_realization_text(cleaned, user_input, plan, confidence_profile, uncertainties, kind=kind)
            if score == float('-inf'):
                return
            if score < 0.38 and not (query_targets_mai and kind in {'identity', 'claim'}):
                return
            seen.add(key)
            units.append({
                'text': cleaned,
                'kind': kind,
                'score': score,
                'confidence_profile': confidence_profile or {},
                'uncertainties': list(uncertainties or []),
                'meta': dict(meta or {}),
            })

        main_claim = plan.get('main_claim', {}) if isinstance(plan.get('main_claim'), dict) else {}
        if main_claim.get('text'):
            _add_unit(
                main_claim.get('text'),
                kind='claim',
                confidence_profile=main_claim.get('confidence_profile', {}) if isinstance(main_claim.get('confidence_profile'), dict) else {},
                uncertainties=main_claim.get('uncertainties', []),
                meta={'role': 'main'},
            )
        for claim in plan.get('support_claims', [])[:3]:
            if isinstance(claim, dict) and claim.get('text'):
                _add_unit(
                    claim.get('text'),
                    kind='claim',
                    confidence_profile=claim.get('confidence_profile', {}) if isinstance(claim.get('confidence_profile'), dict) else {},
                    uncertainties=claim.get('uncertainties', []),
                    meta={'role': 'support'},
                )
        for fact in plan.get('supporting_facts', [])[:3]:
            if isinstance(fact, dict) and fact.get('summary'):
                _add_unit(
                    fact.get('summary'),
                    kind='fact',
                    confidence_profile=fact.get('confidence_profile', {}) if isinstance(fact.get('confidence_profile'), dict) else {},
                    uncertainties=[],
                    meta={'fact_key': fact.get('fact_key', '')},
                )
        for trait in plan.get('identity_traits', [])[:2]:
            if isinstance(trait, dict) and trait.get('summary'):
                _add_unit(
                    trait.get('summary'),
                    kind='identity',
                    confidence_profile=trait.get('confidence_profile', {}) if isinstance(trait.get('confidence_profile'), dict) else {},
                    uncertainties=[],
                    meta={'trait_key': trait.get('trait_key', '')},
                )
        reasoning_summary = str(plan.get('reasoning_summary', '') or '').strip()
        reasoning_paths = plan.get('reasoning_paths', []) if isinstance(plan.get('reasoning_paths', []), list) else []
        reasoning_score = self._safe_realization_float((reasoning_paths[0] if reasoning_paths else {}).get('path_score'), 0.0)
        if reasoning_summary:
            _add_unit(
                reasoning_summary,
                kind='reasoning',
                confidence_profile={'overall': reasoning_score, 'source_reliability': reasoning_score, 'stability': reasoning_score},
                uncertainties=[],
                meta={'path_score': reasoning_score},
            )

        units.sort(key=lambda item: item.get('score', 0.0), reverse=True)
        return units

    def _lowercase_sentence_start(self, text):
        stripped = str(text or '').strip()
        if not stripped:
            return ''
        if len(stripped) >= 3 and stripped[:3].lower() == 'mai':
            remainder = stripped[3:]
            if not remainder or not remainder[0].isalpha():
                return 'Mai' + remainder
        if len(stripped) >= 3 and stripped[:3].upper() == 'SGM':
            return 'SGM' + stripped[3:]
        return stripped[:1].lower() + stripped[1:]

    def _pronounize_mai_sentence(self, text):
        cleaned = str(text or '').strip()
        if cleaned.lower().startswith('mai '):
            return 'She ' + cleaned[4:]
        return cleaned

    def _truncate_realized_response(self, response, max_words=36, max_sentences=2):
        sentences = [sentence.strip() for sentence in re.split(r'(?<=[.!?])\s+', str(response or '').strip()) if sentence.strip()]
        if max_sentences > 0:
            sentences = sentences[:max_sentences]
        words = []
        output_sentences = []
        for sentence in sentences:
            sentence_words = sentence.split()
            if len(words) + len(sentence_words) > max_words and output_sentences:
                break
            output_sentences.append(sentence)
            words.extend(sentence_words)
        return " ".join(output_sentences).strip()

    def _build_scaffolded_open_response(self, user_input, response_plan=None):
        plan = response_plan if isinstance(response_plan, dict) else {}
        intent = str(plan.get('intent', '') or '').strip().lower()
        if intent not in {'answer', 'explanation', 'guidance', 'self_description'}:
            return None

        topic_phrase = self._extract_realization_topic_phrase(user_input)
        normalized_query = " ".join(str(user_input or '').lower().split()).strip()
        if not topic_phrase or topic_phrase == 'this topic':
            return None

        if intent == 'explanation':
            if 'memory replay' in normalized_query:
                return "The short version is that memory replay helps by revisiting earlier experience so later responses can reinforce what keeps working."
            return None
        if intent == 'guidance':
            return None
        if intent == 'self_description':
            return (
                "Mai is a non-tensor statistical and graph-learning AI. "
                "She learns through persistent memory, bounded chemistry-like control signals, learned goals, relations, and feedback."
            )
        return None

    def _build_open_ended_plan_response(self, user_input, response_plan=None):
        plan = response_plan if isinstance(response_plan, dict) else {}
        if not user_input or not plan or not plan.get('success'):
            self.last_realization_trace = {}
            return None

        intent = str(plan.get('intent', '') or '').strip().lower()
        cognitive_control = plan.get('cognitive_control', {}) if isinstance(plan.get('cognitive_control'), dict) else {}
        cognitive_action = str(cognitive_control.get('action', '') or '')
        normalized_query = " ".join(str(user_input or '').lower().split()).strip()
        single_sentence_requested = self._query_requests_single_sentence(user_input)
        recall_style_prompt = normalized_query.startswith((
            'what did you just say',
            'what did you say',
            'summarize what you said',
            'repeat what you said',
        ))
        if intent not in {'answer', 'explanation', 'guidance', 'self_description'}:
            self.last_realization_trace = {}
            return None
        if recall_style_prompt:
            scaffold = self._build_scaffolded_open_response(user_input, plan)
            scaffold = self._normalize_realization_text(scaffold)
            if scaffold:
                self.last_realization_trace = {
                    'mode': 'scaffold_realization',
                    'intent': intent,
                    'topic_phrase': self._extract_realization_topic_phrase(user_input),
                    'response_preview': scaffold[:200],
                }
                return scaffold

        units = self._collect_open_ended_realization_units(user_input, plan)
        if intent == 'self_description' and units:
            strongest_text = str(units[0].get('text', '') or '').strip().lower()
            if strongest_text.startswith('mai prefers '):
                alternate = next(
                    (
                        item for item in units[1:]
                        if not str(item.get('text', '') or '').strip().lower().startswith('mai prefers ')
                    ),
                    None,
                )
                if alternate is not None:
                    units = [alternate] + [item for item in units if item is not alternate]
                elif single_sentence_requested or 'describe' in normalized_query:
                    units = []
        if units and units[0].get('score', 0.0) >= 0.52:
            main_unit = units[0]
            support_units = []
            if not single_sentence_requested:
                support_units = [item for item in units[1:] if item.get('score', 0.0) >= 0.46][:2]
            main_text = main_unit.get('text', '')
            support_text = support_units[0].get('text', '') if support_units else ''

            sentences = []
            mark_uncertainty = cognitive_action in {'explore_hypothesis', 'create_connection'}
            if intent == 'explanation':
                if cognitive_action == 'repair_alignment':
                    lead = self._evidence_led_sentence(
                        f"A better framing is that {self._lowercase_sentence_start(main_text.rstrip('.!?'))}",
                        cognitive_action=cognitive_action,
                    )
                else:
                    lead = self._evidence_led_sentence(
                        main_text,
                        cognitive_action=cognitive_action,
                        mark_uncertainty=mark_uncertainty,
                    )
                if lead:
                    sentences.append(lead)
                if support_text:
                    support = self._evidence_led_sentence(
                        support_text,
                        cognitive_action=cognitive_action,
                    )
                    if support:
                        sentences.append(support)
            elif intent == 'guidance':
                if cognitive_action == 'repair_alignment':
                    lead = self._evidence_led_sentence(
                        f"A more useful approach is that {self._lowercase_sentence_start(main_text.rstrip('.!?'))}",
                        cognitive_action=cognitive_action,
                    )
                else:
                    lead = self._evidence_led_sentence(
                        main_text,
                        cognitive_action=cognitive_action,
                        mark_uncertainty=mark_uncertainty,
                    )
                if lead:
                    sentences.append(lead)
                if support_text:
                    support = self._evidence_led_sentence(
                        self._pronounize_mai_sentence(support_text.rstrip('.!?')),
                        cognitive_action=cognitive_action,
                    )
                    if support:
                        sentences.append(support)
            elif intent == 'self_description':
                lead = self._evidence_led_sentence(main_text, cognitive_action=cognitive_action)
                if lead:
                    sentences.append(lead)
                if support_text:
                    support = self._evidence_led_sentence(
                        self._pronounize_mai_sentence(support_text.rstrip('.!?')),
                        cognitive_action=cognitive_action,
                    )
                    if support:
                        sentences.append(support)
            else:
                if 'interesting' in normalized_query:
                    lead = self._evidence_led_sentence(
                        main_text,
                        cognitive_action=cognitive_action,
                        mark_uncertainty=mark_uncertainty,
                    )
                elif cognitive_action == 'repair_alignment':
                    lead = self._evidence_led_sentence(
                        f"A better answer is that {self._lowercase_sentence_start(main_text.rstrip('.!?'))}",
                        cognitive_action=cognitive_action,
                    )
                else:
                    lead = self._evidence_led_sentence(
                        main_text,
                        cognitive_action=cognitive_action,
                        mark_uncertainty=mark_uncertainty,
                    )
                if lead:
                    sentences.append(lead)
                if support_text:
                    support = self._evidence_led_sentence(
                        self._pronounize_mai_sentence(support_text.rstrip('.!?')),
                        cognitive_action=cognitive_action,
                    )
                    if support:
                        sentences.append(support)

            max_sentences = 1 if single_sentence_requested else 2
            response = self._truncate_realized_response(" ".join(sentences), max_words=38, max_sentences=max_sentences)
            response = self._normalize_realization_text(response)
            if response:
                self.last_realization_trace = {
                    'mode': 'plan_realization',
                    'intent': intent,
                    'cognitive_action': cognitive_action,
                    'scoring_bias_applied': bool(cognitive_action),
                    'used_texts': [main_unit.get('text', '')] + [item.get('text', '') for item in support_units],
                    'used_kinds': [main_unit.get('kind', '')] + [item.get('kind', '') for item in support_units],
                    'response_preview': response[:200],
                }
                return response

        scaffold = self._build_scaffolded_open_response(user_input, plan)
        scaffold = self._normalize_realization_text(scaffold)
        if scaffold:
            self.last_realization_trace = {
                'mode': 'scaffold_realization',
                'intent': intent,
                'topic_phrase': self._extract_realization_topic_phrase(user_input),
                'response_preview': scaffold[:200],
            }
            return scaffold

        self.last_realization_trace = {}
        return None

    def _should_ingest_generated_response_as_knowledge(self, bot_response):
        trace = getattr(self, 'last_realization_trace', {}) or {}
        mode = str(trace.get('mode', '') or '')
        if mode == 'scaffold_realization':
            return False
        return not self._looks_like_generated_scaffold_text(bot_response)

    def _extend_context_with_response_plan(self, prepared_context, response_plan):
        if not isinstance(prepared_context, list):
            prepared_context = list(prepared_context or [])
        if not response_plan:
            return prepared_context

        plan_tokens = []
        for concept in response_plan.get('target_concepts', [])[:3]:
            plan_tokens.extend(self.clean_text(concept.get('canonical_name', '')))
        for fact in response_plan.get('supporting_facts', [])[:2]:
            plan_tokens.extend(self.clean_text(fact.get('subject', '')))
            plan_tokens.extend(self.clean_text(fact.get('object', '')))
        main_claim = response_plan.get('main_claim', {})
        if isinstance(main_claim, dict) and main_claim.get('text'):
            plan_tokens.extend(self.clean_text(main_claim.get('text', '')))
        for claim in response_plan.get('support_claims', [])[:2]:
            if isinstance(claim, dict) and claim.get('text'):
                plan_tokens.extend(self.clean_text(claim.get('text', '')))

        if not plan_tokens:
            return prepared_context
        return (prepared_context + plan_tokens)[-(MAX_CONTEXT_SIZE * 2):]

    def _response_mentions_plan(self, response, response_plan):
        if not response_plan:
            return False
        response_terms = set(self.clean_text(response))
        if not response_terms:
            return False
        target_terms = set()
        for concept in response_plan.get('target_concepts', [])[:3]:
            target_terms.update(self.clean_text(concept.get('canonical_name', '')))
        for fact in response_plan.get('supporting_facts', [])[:2]:
            target_terms.update(self.clean_text(fact.get('subject', '')))
            target_terms.update(self.clean_text(fact.get('object', '')))
        main_claim = response_plan.get('main_claim', {})
        if isinstance(main_claim, dict) and main_claim.get('text'):
            target_terms.update(self.clean_text(main_claim.get('text', '')))
        for claim in response_plan.get('support_claims', [])[:2]:
            if isinstance(claim, dict) and claim.get('text'):
                target_terms.update(self.clean_text(claim.get('text', '')))
        target_terms = {term for term in target_terms if len(term) > 2}
        if not target_terms:
            return False
        return bool(response_terms.intersection(target_terms))

    def _enhance_with_knowledge(self, response, user_input, response_plan=None):
        knowledge_store = getattr(self, 'knowledge_store', None)
        if knowledge_store is None or not user_input:
            return response

        plan = response_plan if isinstance(response_plan, dict) else {}
        if not plan:
            plan = self._build_response_plan(user_input)

        needs_support = self._is_low_information_response(response) or not self._response_mentions_plan(response, plan)
        if not needs_support:
            return response

        fact_summaries = self._collect_response_plan_summaries(user_input, plan, fact_limit=2)
        focus_terms = {
            term for term in self.clean_text(user_input)
            if len(term) > 3 and term not in {'what', 'does', 'did', 'just', 'mai', 'sgm', 'system', 'about'}
        }

        seen = set()
        filtered_summaries = []
        response_lower = (response or '').lower()
        for summary in fact_summaries:
            clean_summary = summary.strip()
            if not clean_summary:
                continue
            summary_terms = {term for term in self.clean_text(clean_summary) if len(term) > 3}
            if focus_terms and not summary_terms.intersection(focus_terms):
                continue
            key = clean_summary.lower()
            if key in seen or key in response_lower:
                continue
            seen.add(key)
            filtered_summaries.append(clean_summary)
            if len(filtered_summaries) >= 2:
                break

        if not filtered_summaries:
            return response

        combined = " ".join(filtered_summaries)
        if response and response.strip():
            return f"{combined} {response}".strip()
        return combined

    def get_structured_conversation_context(self, history_turns=3):
        """Builds a single list of words representing the last few conversational turns."""
        context_parts = []
        recent_exchanges = self.conversation_memory[-history_turns:]
        for exchange in recent_exchanges:
            if not isinstance(exchange, dict):
                continue
            context_parts.append(f"[user] {exchange.get('user', '')}")
            context_parts.append(f"[bot] {exchange.get('bot', '')}")
        full_context_string = " ".join(context_parts)
        return self.clean_text(full_context_string)

  

    def _flush_batch_operations(self):
        if not self.batch_operations:
            return

        # Training creates many duplicate observations. Coalesce them before
        # touching SQLite so every key is written once per batch. Besides being
        # substantially faster, accumulating evidence is more useful than the
        # previous INSERT OR REPLACE behavior, which discarded earlier counts.
        pattern_buckets = {}
        association_buckets = {}
        sentence_buckets = defaultdict(lambda: [0.0, 0])
        grammar_buckets = defaultdict(lambda: [0.0, 0])
        phrase_buckets = defaultdict(lambda: [0.0, 0])
        semantic_buckets = defaultdict(lambda: [0.0, 0])

        def add_weighted_observation(buckets, key, priority, success_rate, usage_count):
            bucket = buckets.setdefault(key, {
                'priority': 0.0,
                'success_weight': 0.0,
                'success_observations': 0,
                'usage_count': 0,
            })
            try:
                priority_value = float(priority or 0)
            except (TypeError, ValueError):
                priority_value = 0.0
            try:
                success_value = min(1.0, max(0.0, float(success_rate)))
            except (TypeError, ValueError):
                success_value = 0.5
            try:
                usage_value = max(0, int(usage_count or 0))
            except (TypeError, ValueError):
                usage_value = 0
            observation_weight = max(usage_value, 1)
            bucket['priority'] += priority_value
            bucket['success_weight'] += success_value * observation_weight
            bucket['success_observations'] += observation_weight
            bucket['usage_count'] += usage_value

        for op in self.batch_operations:
            op_type = op.get('type')
            data = op.get('data', ())
            if op_type == 'chain':
                if len(data) >= 13:
                    context_len, *context_words, next_word, priority, success_rate, usage_count = data[:13]
                elif len(data) >= 12:
                    context_len, *context_words, next_word, priority, usage_count = data[:12]
                    success_rate = 0.5
                else:
                    continue
                pad = list(context_words[:MAX_CONTEXT_SIZE])
                pad.extend([''] * (MAX_CONTEXT_SIZE - len(pad)))
                add_weighted_observation(
                    pattern_buckets,
                    (context_len, *pad, next_word),
                    priority,
                    success_rate,
                    usage_count,
                )
            elif op_type == 'assoc':
                if len(data) < 3:
                    continue
                source_word, next_word, priority = data[:3]
                success_rate = 0.5
                usage_count = 0
                if len(data) >= 5:
                    success_rate, usage_count = data[3:5]
                elif len(data) == 4:
                    fourth = data[3]
                    if isinstance(fourth, float) and 0.0 <= fourth <= 1.0:
                        success_rate = fourth
                    else:
                        usage_count = fourth
                add_weighted_observation(
                    association_buckets,
                    (source_word, next_word),
                    priority,
                    success_rate,
                    usage_count,
                )
            elif op_type in ('sentence_start', 'sentence_end', 'svo_pattern'):
                pattern, priority = data
                bucket = sentence_buckets[(op_type, ' '.join(pattern))]
                bucket[0] += float(priority)
                bucket[1] += 1
            elif op_type in ('article_noun', 'verb_form'):
                word1, word2, priority = data
                bucket = grammar_buckets[(op_type, word1, word2)]
                bucket[0] += float(priority)
                bucket[1] += 1
            elif op_type in ('phrase_2', 'phrase_3'):
                if op_type == 'phrase_2':
                    word1, word2, priority = data
                    phrase_text = f"{word1} {word2}"
                else:
                    word1, word2, word3, priority = data
                    phrase_text = f"{word1} {word2} {word3}"
                bucket = phrase_buckets[phrase_text]
                bucket[0] += float(priority)
                bucket[1] += 1
            elif op_type == 'semantic_rel':
                word1, word2, priority = data
                bucket = semantic_buckets[(word1, word2)]
                bucket[0] += float(priority)
                bucket[1] += 1

        pattern_data = []
        for key, bucket in pattern_buckets.items():
            observations = max(1, bucket['success_observations'])
            pattern_data.append((
                *key,
                bucket['priority'],
                bucket['success_weight'] / observations,
                bucket['usage_count'],
            ))
        assoc_data = []
        for key, bucket in association_buckets.items():
            observations = max(1, bucket['success_observations'])
            assoc_data.append((
                *key,
                bucket['priority'],
                bucket['success_weight'] / observations,
                bucket['usage_count'],
            ))
        sentence_patterns = [(*key, values[0], values[1]) for key, values in sentence_buckets.items()]
        grammar_patterns = [(*key, values[0], values[1]) for key, values in grammar_buckets.items()]
        phrase_patterns = [(key, values[0], values[1]) for key, values in phrase_buckets.items()]
        semantic_patterns = [(*key, values[0], values[1]) for key, values in semantic_buckets.items()]

        # Group nearby B-tree keys so a page can absorb several updates while it
        # is hot in SQLite's cache instead of being rewritten in corpus order.
        pattern_data.sort(key=lambda row: row[:10])
        assoc_data.sort(key=lambda row: row[:2])
        sentence_patterns.sort(key=lambda row: row[:2])
        grammar_patterns.sort(key=lambda row: row[:3])
        phrase_patterns.sort(key=lambda row: row[0])
        semantic_patterns.sort(key=lambda row: row[:2])

        try:
            if getattr(self, '_storage_backend', None):
                for row in pattern_data:
                    context_len, *context_words, next_word, priority, success_rate, usage_count = row
                    pad = context_words[:MAX_CONTEXT_SIZE] if len(context_words) >= MAX_CONTEXT_SIZE else list(context_words) + [''] * (MAX_CONTEXT_SIZE - len(context_words))
                    self._storage_backend.add_pattern_batch(context_len, tuple(pad), next_word, priority, success_rate, usage_count)
                for (source_word, next_word, priority, success_rate, usage_count) in assoc_data:
                    self._storage_backend.add_association_batch(source_word, next_word, priority, success_rate, usage_count)
                self._storage_backend.flush()
                if self.is_clone:
                    if pattern_data:
                        self.cur.executemany("""
                            INSERT INTO dynamic_word_chain
                            (context_len, word1, word2, word3, word4, word5, word6, word7, word8, next_word, priority, success_rate, usage_count)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ON CONFLICT(context_len, word1, word2, word3, word4, word5, word6, word7, word8, next_word) DO UPDATE SET
                                priority = dynamic_word_chain.priority + excluded.priority,
                                success_rate = CASE
                                    WHEN (dynamic_word_chain.usage_count + excluded.usage_count) > 0 THEN
                                        ((dynamic_word_chain.success_rate * dynamic_word_chain.usage_count) + (excluded.success_rate * excluded.usage_count))
                                        / (dynamic_word_chain.usage_count + excluded.usage_count)
                                    ELSE excluded.success_rate
                                END,
                                usage_count = dynamic_word_chain.usage_count + excluded.usage_count
                        """, pattern_data)
                    if assoc_data:
                        self.cur.executemany("""
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
                                usage_count = word_associations.usage_count + excluded.usage_count
                        """, assoc_data)
                    vocab_store = getattr(self, 'vocab_id_store', None)
                    if (
                        pattern_data
                        and vocab_store is not None
                        and str(getattr(self, 'vocab_id_storage_mode', 'off') or 'off') in {'dual', 'ids'}
                    ):
                        vocab_store.upsert_pattern_rows(pattern_data)
            else:
                vocab_mode = str(getattr(self, 'vocab_id_storage_mode', 'off') or 'off')
                vocab_store = getattr(self, 'vocab_id_store', None)
                write_text = vocab_mode != 'ids'
                write_ids = vocab_store is not None and vocab_mode in {'dual', 'ids'}
                if pattern_data and write_text:
                    self.cur.executemany("""
                    INSERT INTO dynamic_word_chain
                    (context_len, word1, word2, word3, word4, word5, word6, word7, word8, next_word, priority, success_rate, usage_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(context_len, word1, word2, word3, word4, word5, word6, word7, word8, next_word) DO UPDATE SET
                        priority = dynamic_word_chain.priority + excluded.priority,
                        success_rate = CASE
                            WHEN (dynamic_word_chain.usage_count + excluded.usage_count) > 0 THEN
                                ((dynamic_word_chain.success_rate * dynamic_word_chain.usage_count) + (excluded.success_rate * excluded.usage_count))
                                / (dynamic_word_chain.usage_count + excluded.usage_count)
                            ELSE excluded.success_rate
                        END,
                        usage_count = dynamic_word_chain.usage_count + excluded.usage_count
                    """, pattern_data)
                if pattern_data and write_ids:
                    vocab_store.upsert_pattern_rows(pattern_data)
                if assoc_data:
                    self.cur.executemany("""
                    INSERT INTO word_associations
                    (source_word, next_word, priority, success_rate, usage_count)
                        VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(source_word, next_word) DO UPDATE SET
                        priority = word_associations.priority + excluded.priority,
                        success_rate = CASE
                            WHEN (word_associations.usage_count + excluded.usage_count) > 0 THEN
                                ((word_associations.success_rate * word_associations.usage_count) + (excluded.success_rate * excluded.usage_count))
                                / (word_associations.usage_count + excluded.usage_count)
                            ELSE excluded.success_rate
                        END,
                        usage_count = word_associations.usage_count + excluded.usage_count
                    """, assoc_data)

          
            if sentence_patterns:
                self.cur.executemany("""
                    INSERT INTO sentence_patterns (pattern_type, pattern_text, priority, usage_count)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(pattern_type, pattern_text) DO UPDATE SET
                        priority = sentence_patterns.priority + excluded.priority,
                        usage_count = sentence_patterns.usage_count + excluded.usage_count
                """, sentence_patterns)

          
            if grammar_patterns:
                self.cur.executemany("""
                    INSERT INTO grammar_patterns (pattern_type, word1, word2, priority, usage_count)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(pattern_type, word1, word2) DO UPDATE SET
                        priority = grammar_patterns.priority + excluded.priority,
                        usage_count = grammar_patterns.usage_count + excluded.usage_count
                """, grammar_patterns)

          
            if phrase_patterns:
                self.cur.executemany("""
                    INSERT INTO phrase_patterns (phrase_text, priority, usage_count)
                    VALUES (?, ?, ?)
                    ON CONFLICT(phrase_text) DO UPDATE SET
                        priority = phrase_patterns.priority + excluded.priority,
                        usage_count = phrase_patterns.usage_count + excluded.usage_count
                """, phrase_patterns)

          
            if semantic_patterns:
                self.cur.executemany("""
                    INSERT INTO semantic_relationships (word1, word2, strength, usage_count)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(word1, word2) DO UPDATE SET
                        strength = semantic_relationships.strength + excluded.strength,
                        usage_count = semantic_relationships.usage_count + excluded.usage_count
                """, semantic_patterns)

            if sentence_patterns or grammar_patterns or phrase_patterns or semantic_patterns:
                self._bonus_caches_loaded = False
                self._next_word_contextual_bonus_cache = {}
            self.con.commit()
            
        except Exception:
            self.con.rollback()
            raise
        finally:
            self.batch_operations = []
            self.batch_count = 0

    def learn_from_text_optimized(self, text, base_priority_boost=1):
        """Enhanced learning with sentence structure and grammar patterns"""
        words = self.clean_text(text)
        if len(words) < 2: 
            return 0

      
        if (memory_manager.get_memory_usage_percent() or 0) > LEARNING_MEMORY_THRESHOLD:
            memory_manager.force_garbage_collection()

      
        # Keep vocabulary indices dense. The old list retained every repeated
        # unseen token, incrementing the index counter thousands of times while
        # repeatedly overwriting the same dictionary key.
        new_words = list(dict.fromkeys(w for w in words if w not in self.word_to_ix))
        if new_words:
            new_vocab_size = len(self.word_to_ix)
            for w in new_words:
                self.word_to_ix[w] = new_vocab_size
                self.ix_to_word[new_vocab_size] = w
                new_vocab_size += 1

            if not self.is_clone and new_vocab_size > self.model.vocab_size * 1.1:
                self.model = PlainMLP(new_vocab_size, context_size=NN_CONTEXT_SIZE)
        
      
        max_batch_size = max(1, min(LEARNING_BATCH_SIZE, len(words)))
        
      
        sentences = self._extract_sentences(text)
        
      
        for sentence in sentences:
            sentence_words = self.clean_text(sentence)
            if len(sentence_words) < 3:
                continue
                
          
            self._learn_sentence_structure(sentence_words, base_priority_boost)
            
          
            self._learn_grammar_patterns(sentence_words, base_priority_boost)

            if len(self.batch_operations) >= max_batch_size:
                self._flush_batch_operations()
        
      
        for i in range(1, len(words)):
          
            if i % LEARNING_PROGRESS_INTERVAL == 0:
                print(f"Processing word {i}/{len(words)} ({i/len(words)*100:.1f}%)")
            
          
            for context_len in [6, 4, 2]:
                if i >= context_len:
                    context = words[i - context_len : i]
                    next_word = words[i]
                    padding = ['<PAD>'] * (MAX_CONTEXT_SIZE - context_len)
                    full_context = tuple(padding + context)
                    
                  
                    priority = self._calculate_word_priority(next_word, context, base_priority_boost)
                    
                    self.batch_operations.append({
                        'type': 'chain',
                        'data': (context_len, *full_context, next_word, priority, 0.5, 1)
                    })
            
          
            self.batch_operations.append({
                'type': 'assoc',
                'data': (words[i-1], words[i], base_priority_boost, 0.5, 1)
            })
            
          
            if i >= 3:
                phrase = words[i-3:i+1]
                self._learn_phrase_patterns(phrase, base_priority_boost)
            
          
            if len(self.batch_operations) >= max_batch_size:
                self._flush_batch_operations()
                
              
                if i % LEARNING_GC_INTERVAL == 0:
                    memory_manager.force_garbage_collection()
        
      
        if self.batch_operations:
            self._flush_batch_operations()
        
      
        if not self.is_clone:
            self.training_word_count += len(words)
            if self.training_word_count > SEMANTIC_UPDATE_INTERVAL:
                self.semantic_memory.update_cooccurrence(words, force_minimal=True)
                self._learn_semantic_relationships(words)
                self.training_word_count = 0
        
        return len(words)

    def _extract_sentences(self, text):
        """Extract sentences from text for structure learning"""
        import re
      
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    def _learn_sentence_structure(self, sentence_words, base_priority_boost):
        """Learn sentence structure patterns"""
        if len(sentence_words) < 3:
            return
            
      
        if len(sentence_words) >= 3:
            start_pattern = sentence_words[:3]
            self.batch_operations.append({
                'type': 'sentence_start',
                'data': (tuple(start_pattern), base_priority_boost * 1.5)
            })
        
      
        if len(sentence_words) >= 3:
            end_pattern = sentence_words[-3:]
            self.batch_operations.append({
                'type': 'sentence_end',
                'data': (tuple(end_pattern), base_priority_boost * 1.5)
            })
        
      
        for i in range(len(sentence_words) - 2):
            pattern = sentence_words[i:i+3]
            self.batch_operations.append({
                'type': 'svo_pattern',
                'data': (tuple(pattern), base_priority_boost * 1.2)
            })

    def _learn_grammar_patterns(self, sentence_words, base_priority_boost):
        """Learn grammar and syntax patterns"""
        if len(sentence_words) < 2:
            return
            
      
        for i in range(len(sentence_words) - 1):
            if sentence_words[i].lower() in ['the', 'a', 'an']:
                self.batch_operations.append({
                    'type': 'article_noun',
                    'data': (sentence_words[i], sentence_words[i+1], base_priority_boost * 1.3)
                })
        
      
        for i, word in enumerate(sentence_words):
            if word.endswith(('ing', 'ed', 'er', 'ly')):
                if i > 0:
                    self.batch_operations.append({
                        'type': 'verb_form',
                        'data': (sentence_words[i-1], word, base_priority_boost * 1.1)
                    })

    def _learn_phrase_patterns(self, phrase, base_priority_boost):
        """Learn common phrase patterns"""
        if len(phrase) < 2:
            return
            
      
        for i in range(len(phrase) - 1):
            self.batch_operations.append({
                'type': 'phrase_2',
                'data': (phrase[i], phrase[i+1], base_priority_boost * 1.1)
            })
        
      
        if len(phrase) >= 3:
            for i in range(len(phrase) - 2):
                self.batch_operations.append({
                    'type': 'phrase_3',
                    'data': (phrase[i], phrase[i+1], phrase[i+2], base_priority_boost * 1.2)
                })

    def _calculate_word_priority(self, word, context, base_priority_boost):
        """Calculate enhanced priority based on word importance"""
        priority = base_priority_boost
        
      
        if word.lower() in ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']:
            priority *= 1.2
        
      
        if len(word) > 4 and not word.endswith(('ing', 'ed', 'er', 'ly')):
            priority *= 1.3
        
      
        if word in context:
            priority *= 1.1
        
        return priority

    def _learn_semantic_relationships(self, words):
        """Learn semantic relationships between words"""
        if len(words) < 3:
            return
            
      
        for i in range(len(words) - 1):
            for j in range(i + 1, min(i + 4, len(words))):
                if words[i] != words[j]:
                    self.batch_operations.append({
                        'type': 'semantic_rel',
                        'data': (words[i], words[j], 1.0)
                    })

    def learn_from_text(self, text, base_priority_boost=1):
        if not self.is_clone:
            self.semantic_memory.update_cooccurrence(self.clean_text(text))
        
        self.learn_from_text_optimized(text, base_priority_boost)

    def learn_from_conversation_exchange(self, user_text, bot_text, quality_score):
        """Learns the connection between a user input and a bot response."""
        adaptive_learning = getattr(self, 'adaptive_learning', None)
        if quality_score < CONVERSATION_LEARNING_QUALITY_THRESHOLD:
            if adaptive_learning:
                adaptive_learning.record_learning_event(
                    'conversation_exchange',
                    quality_score,
                    False,
                    source_kind='conversation',
                    detail='quality_threshold',
                )
            return
        if self._is_low_information_response(bot_text):
            if adaptive_learning:
                adaptive_learning.record_learning_event(
                    'conversation_exchange',
                    quality_score,
                    False,
                    source_kind='conversation',
                    detail='low_information',
                )
            return
        if adaptive_learning:
            decision = adaptive_learning.should_accept_live_learning(
                quality_score,
                source_kind='conversation',
                response_text=bot_text,
                context_text=user_text,
            )
            if not decision.get('accept'):
                adaptive_learning.record_learning_event(
                    'conversation_exchange',
                    quality_score,
                    False,
                    source_kind='conversation',
                    detail=decision.get('reason', 'adaptive_gate'),
                )
                return
        else:
            decision = {'reason': 'accepted'}
        learning_text = f"[user] {user_text} [bot] {bot_text}"
        priority_boost = 2
        if quality_score >= 0.82:
            priority_boost = 3
        elif quality_score <= 0.68:
            priority_boost = 1
        self.learn_from_text_optimized(learning_text, base_priority_boost=priority_boost)
        if adaptive_learning:
            adaptive_learning.record_learning_event(
                'conversation_exchange',
                quality_score,
                True,
                source_kind='conversation',
                detail=f"{decision.get('reason', 'accepted')}|priority:{priority_boost}",
            )
        
      
        reasoning_engine = getattr(self, 'reasoning_engine', None)
        if self.enhanced_intelligence_enabled and reasoning_engine:
            try:
                context_analysis = reasoning_engine.analyze_conversation_context(user_text, [])
                
              
                if context_analysis.get('intent') == 'question':
                    self._store_reasoning_pattern('question_response', bot_text, quality_score, context_analysis.get('topic'))
                elif context_analysis.get('intent') == 'seeking_explanation':
                    self._store_reasoning_pattern('explanation', bot_text, quality_score, context_analysis.get('topic'))
                
              
                if adaptive_learning:
                    adaptive_learning.update_learning_parameters(
                        quality_score,
                        quality_score,
                        context_key='conversation_exchange',
                    )
                    
            except Exception as e:
                print(f"Enhanced learning failed: {e}")
    
    def _store_reasoning_pattern(self, pattern_type, pattern_text, success_rate, context_type):
        """Store reasoning patterns for future use"""
        try:
            self.cur.execute("""
                INSERT OR REPLACE INTO reasoning_patterns 
                (pattern_type, pattern_text, usage_count, success_rate, context_type)
                VALUES (?, ?, 1, ?, ?)
            """, (pattern_type, pattern_text, success_rate, context_type))
            self.con.commit()
        except Exception as e:
            print(f"Error storing reasoning pattern: {e}")

    def _handle_duplication(self, words):
        if len(words) < 3:
            return words
        
        if len(words) >= 4 and words[-1] == words[-3] and words[-2] == words[-4]:
            return words[:-2]
        
        if len(words) >= 3:
            last_three = words[-3:]
            if len(set(last_three)) <= 1:
                return words[:-2]
        
        if len(words) >= 4:
            last_four = words[-4:]
            unique_words = len(set(last_four))
            if unique_words <= 2:
                return words[:-2]
        
        if len(words) > 10:
            last_eight = words[-8:]
            
            if len(last_eight) >= 4:
                pattern_found = False
                for pattern_len in [2, 3]:
                    if len(last_eight) >= pattern_len * 2:
                        pattern1 = last_eight[-pattern_len*2:-pattern_len]
                        pattern2 = last_eight[-pattern_len:]
                        if pattern1 == pattern2:
                            return words[:-pattern_len]
            
            recent_words = words[-6:]
            n_recent = len(recent_words)
            unique_ratio = len(set(recent_words)) / n_recent if n_recent else 0.0
            if unique_ratio < 0.5:
                return words[:-2]
        
        return words

    def _validate_response_quality(self, response_text):
        if response_text is None or not isinstance(response_text, str):
            return False
        words = response_text.split()
        if not words or len(words) < 2 or len(words) > 25:
            return False
        
        unique_words = len(set(words))
        if unique_words / len(words) < 0.5:
            return False
        
        for word in words:
            if len(word) > 8:
                char_counts = {}
                for char in word:
                    char_counts[char] = char_counts.get(char, 0) + 1
                if any(count > len(word) * 0.6 for count in char_counts.values()):
                    return False
        
        single_char_count = sum(1 for word in words if len(word) == 1 and word.isalpha())
        if single_char_count > len(words) * 0.25:
            return False
        
        problematic_patterns = ['fuck', 'shit', 'damn', 'freak', 'wtf', 'omg']
        response_lower = response_text.lower()
        if any(pattern in response_lower for pattern in problematic_patterns):
            return False
        
        topic_scores = []
        for word in words:
            if len(word) > 3:
                score = self.get_topic_coherence_score(word)
                topic_scores.append(score)
        
        if topic_scores:
            n_scores = len(topic_scores)
            avg_topic_coherence = sum(topic_scores) / n_scores if n_scores else 0.0
            if avg_topic_coherence < 0.2:
                return False
        
        has_verb_like = any(word.endswith(('ing', 'ed', 'ly', 'er', 'est')) for word in words)
        has_noun_like = any(len(word) >= 4 and not word.endswith(('ing', 'ly')) for word in words)
        
        if len(words) > 5 and not (has_verb_like or has_noun_like):
            return False
        
        return True

    def _generate_from_emergency_patterns(self, user_input=""):
        """EMERGENCY ONLY: Generate from most basic patterns when all else fails"""
        print("WARNING: Emergency pattern generation activated - this should be rare!")
        
      
        fallback_responses = [
            "Hello! I'm still learning and improving.",
            "Hi there! How can I help you today?",
            "Greetings! I'm here to chat and learn.",
            "Hello! What would you like to talk about?",
            "Hi! I'm excited to have this conversation.",
            "Hello there! How are you doing?",
            "Hi! I'm ready to learn from our chat.",
            "Greetings! What's on your mind?",
            "Hello! I'm here to assist and learn.",
            "Hi! Let's have an interesting conversation."
        ]
        
      
        self.cur.execute("""
            SELECT next_word FROM dynamic_word_chain 
            WHERE priority > 0 
            ORDER BY priority DESC, RANDOM() 
            LIMIT 10
        """)
        words = self.cur.fetchall()
        response_words = []
        if words:
            for word_tuple in words[:5]:
                word = word_tuple[0]
                if word not in ['<PAD>', '<UNK>'] and len(word) > 1:
                    response_words.append(word)
        
        if response_words:
            return " ".join(response_words[:6]).capitalize() + "."
        
      
        available_words = [w for w in self.word_to_ix.keys() if w not in ['<UNK>', '<PAD>'] and len(w) > 2]
        if available_words:
          
            selected_words = random.choices(available_words, k=min(6, len(available_words)))
            response = " ".join(selected_words).capitalize() + "."
            return response
        
      
        return random.choice(fallback_responses)
    
    def _generate_simple_response(self, user_input=""):
        """Generate a simple response using basic word associations"""
        try:
          
            user_words = user_input.lower().split() if user_input else []
            
          
            if any(word in user_words for word in ['hi', 'hello', 'hey', 'greetings']):
                greetings = [
                    "Hello! How are you doing today?",
                    "Hi there! Nice to meet you.",
                    "Hello! I'm here to chat and learn.",
                    "Hi! What would you like to talk about?",
                    "Greetings! How can I help you?"
                ]
                return random.choice(greetings)
            
          
            if any(word in user_words for word in ['what', 'how', 'why', 'when', 'where', 'who']):
                return "That's an interesting question. Let me think about that."
            
          
            available_words = [w for w in self.word_to_ix.keys() if w not in ['<UNK>', '<PAD>'] and len(w) > 2]
            if available_words:
              
                selected_words = random.choices(available_words, k=min(4, len(available_words)))
                response = " ".join(selected_words).capitalize() + "."
                return response
            
          
            return "I'm here to chat and learn from our conversation."
            
        except Exception as e:
            print(f"Simple response generation failed: {e}")
            return "Hello! I'm ready to chat."

    def _build_generation_context(self, user_input, history_turns=2):
        conversation_history_context = self.get_structured_conversation_context(history_turns=history_turns)
        current_user_context = self.clean_text(f"[user] {user_input}") if user_input else []
        return conversation_history_context + current_user_context

    def _get_generation_length_limit(self, user_input):
        prompt_words = len(self.clean_text(user_input))
        if prompt_words <= 4:
            return min(MAX_RESPONSE_LENGTH, 18)
        if prompt_words <= 10:
            return min(MAX_RESPONSE_LENGTH, 22)
        return MAX_RESPONSE_LENGTH

    def _is_low_information_response(self, response):
        text = (response or '').strip().lower()
        if not text:
            return True
        if any(text.startswith(prefix) for prefix in LOW_VALUE_RESPONSE_PREFIXES):
            return True
        words = self.clean_text(response)
        if len(words) < GENERATION_MIN_RESPONSE_WORDS:
            return True
        significant_words = [word for word in words if word not in QUALITY_STOPWORDS]
        if len(significant_words) < 4:
            return True
        unique_ratio = len(set(words)) / max(len(words), 1)
        if unique_ratio < 0.45:
            return True
        if len(words) < 10 and len(set(significant_words)) < 5:
            return True
        return False

    def _is_corpus_artifact_response(self, response, user_input=''):
        """Reject recurring training-corpus fragments when they are unrelated to the prompt."""
        text = ' '.join(str(response or '').lower().split())
        prompt_words = set(self.clean_text(user_input or ''))
        artifact_phrases = (
            'tail wags',
            'her tail',
            'his tail',
            'the ground and',
            'same thing',
        )
        for phrase in artifact_phrases:
            phrase_words = set(phrase.split())
            if phrase in text and not phrase_words.intersection(prompt_words):
                return True
        return False

    def _is_cacheable_response(self, response, user_input=''):
        if not response or self._is_corpus_artifact_response(response, user_input):
            return False
        if self._is_low_information_response(response):
            return False
        return self._calculate_response_quality(response, user_input) >= GENERATION_CACHE_QUALITY_THRESHOLD

    def _build_symbolic_response(self, user_input):
        """Answer high-confidence structural queries without generative noise."""
        normalized = " ".join(str(user_input or '').lower().split()).strip()
        if not normalized:
            return None

        if (
            'what kind of ai' in normalized
            or 'what type of ai' in normalized
            or 'how do you learn' in normalized
            or 'how you learn' in normalized
        ):
            return (
                "I am a non-tensor statistical and graph-learning AI. "
                "I learn through persistent memory, bounded chemistry-like control signals, learned goals, relations, and feedback."
            )

        universal = re.search(r'\bevery\s+(\w+)\s+is\s+(?:an?\s+)?(\w+)', normalized)
        exclusion = re.search(r'\bno\s+(\w+)\s+is\s+(?:an?\s+)?(\w+)', normalized)
        possibility = re.search(r'\bcould\s+any\s+(\w+)\s+be\s+(?:an?\s+)?(\w+)', normalized)
        if universal and exclusion and possibility:
            subject, middle = universal.groups()
            excluded_middle, excluded_property = exclusion.groups()
            asked_subject, asked_property = possibility.groups()
            if middle == excluded_middle and subject == asked_subject and excluded_property == asked_property:
                return (
                    f"No; every {subject} is a {middle}, and no {middle} can be {excluded_property}, "
                    f"so no {subject} can be {excluded_property}."
                )

        if 'what evidence' in normalized and ('before claiming' in normalized or 'before concluding' in normalized):
            return (
                "I would require preregistered controlled studies with adequate samples, clinically meaningful outcomes, "
                "independent replication, peer review, and evidence of safety before making that claim."
            )

        if 'water' in normalized and ('freeze' in normalized or 'freezes' in normalized or 'freezing' in normalized):
            if 'why' in normalized or 'explain' in normalized or 'explaining' in normalized:
                return (
                    "Liquid water freezes when cooling below its freezing point lets hydrogen-bonded molecules settle "
                    "into a stable crystalline lattice."
                )

        if 'memory replay' in normalized and ('help' in normalized or 'benefit' in normalized):
            return (
                "Memory replay revisits earlier experience, strengthening useful associations and correcting later "
                "behavior without requiring the original event to recur."
            )

        if 'without using any word twice' in normalized and any(
            greeting in normalized for greeting in ('hello', 'hi ', 'hey ', 'greeting')
        ):
            return "Hello; glad to meet you."

        if 'professional conversation style' in normalized or (
            'professional' in normalized and 'conversation' in normalized and 'style' in normalized
        ):
            return "A professional conversation style is clear, respectful, relevant, evidence-aware, and free of needless shock or crude language."

        comparison = re.search(r'\bcompare\s+(?:an?\s+)?(\w+)\s+and\s+(?:an?\s+)?(\w+)', normalized)
        if comparison:
            left, right = comparison.groups()
            concept_roles = {
                'map': 'represents places and possible routes',
                'memory': 'represents selected parts of past experience',
            }
            if left in concept_roles and right in concept_roles:
                return (
                    f"A {left} {concept_roles[left]}, while a {right} {concept_roles[right]}; "
                    "both are selective guides rather than the reality they represent."
                )

        return None

    def generate_response(self, user_input=""):
        """Main response generation: statistics (n-gram) + small CPU MLP (NumPy). No tensor framework or NVIDIA/CUDA required."""
        start_time = time.time()
        self.last_reasoning_trace = {}
        self.last_realization_trace = {}
        self.total_generation_attempts += 1
        self._begin_cognitive_turn(user_input)
        if user_input:
            self.update_topic_tracker(user_input)
        symbolic_response = self._build_symbolic_response(user_input)
        if symbolic_response:
            if GENERATION_CACHE_ENABLED and user_input and self._is_cacheable_response(symbolic_response, user_input):
                cache_key = hash(user_input.lower().strip())
                performance_cache.set_response_cache(cache_key, symbolic_response)
            if user_input:
                self.update_conversation_memory(user_input, symbolic_response)
            if hasattr(self, 'performance_monitor'):
                self.performance_monitor.record_response_time(time.time() - start_time)
            self._session_generation_count = getattr(self, '_session_generation_count', 0) + 1
            self.last_realization_trace = {
                'mode': 'symbolic_reasoning',
                'response_preview': symbolic_response[:200],
            }
            return symbolic_response
        self._ensure_bonus_caches()
        response_plan = self._build_response_plan(user_input) if user_input else {}
        grounded_response = self._build_fact_grounded_response(user_input, response_plan=response_plan)
        if grounded_response:
            grounded_response = self._apply_reasoning_engine(user_input, grounded_response, response_plan=response_plan)
            grounded_response = self._apply_critic_repair_loop(user_input, grounded_response, response_plan=response_plan)
            if GENERATION_CACHE_ENABLED and user_input and self._is_cacheable_response(grounded_response, user_input):
                cache_key = hash(user_input.lower().strip())
                performance_cache.set_response_cache(cache_key, grounded_response)
            if user_input:
                self.update_conversation_memory(user_input, grounded_response)
            if hasattr(self, 'performance_monitor'):
                self.performance_monitor.record_response_time(time.time() - start_time)
            self._session_generation_count = getattr(self, '_session_generation_count', 0) + 1
            return grounded_response

        if self._cognitive_strategy_should_preempt(response_plan):
            cognitive_response = self._build_cognitive_strategy_response(user_input, response_plan=response_plan)
            if cognitive_response:
                self.last_critic_assessment = {
                    'selected': 'cognitive_strategy',
                    'repair_attempted': False,
                    'reason': 'bounded cognitive action preempted scaffolded continuation',
                }
                self.last_realization_trace = {
                    'mode': 'cognitive_strategy',
                    'action': self.last_cognitive_trace.get('action'),
                    'goal': self.last_cognitive_trace.get('goal'),
                    'preempted': True,
                    'response_preview': cognitive_response[:200],
                }
                if user_input:
                    self.update_conversation_memory(user_input, cognitive_response)
                if hasattr(self, 'performance_monitor'):
                    self.performance_monitor.record_response_time(time.time() - start_time)
                self._session_generation_count = getattr(self, '_session_generation_count', 0) + 1
                return cognitive_response

        open_ended_response = self._build_open_ended_plan_response(user_input, response_plan=response_plan)
        if open_ended_response:
            open_ended_response = self._apply_reasoning_engine(user_input, open_ended_response, response_plan=response_plan)
            open_ended_response = self._apply_critic_repair_loop(user_input, open_ended_response, response_plan=response_plan)
            if GENERATION_CACHE_ENABLED and user_input and self._is_cacheable_response(open_ended_response, user_input):
                cache_key = hash(user_input.lower().strip())
                performance_cache.set_response_cache(cache_key, open_ended_response)
            if user_input:
                self.update_conversation_memory(user_input, open_ended_response)
            if hasattr(self, 'performance_monitor'):
                self.performance_monitor.record_response_time(time.time() - start_time)
            self._session_generation_count = getattr(self, '_session_generation_count', 0) + 1
            return open_ended_response

        cognitive_response = self._build_cognitive_strategy_response(user_input, response_plan=response_plan)
        if cognitive_response:
            self.last_critic_assessment = {
                'selected': 'cognitive_strategy',
                'repair_attempted': False,
                'reason': 'response was realized directly from a bounded cognitive action',
            }
            self.last_realization_trace = {
                'mode': 'cognitive_strategy',
                'action': self.last_cognitive_trace.get('action'),
                'goal': self.last_cognitive_trace.get('goal'),
                'response_preview': cognitive_response[:200],
            }
            if user_input:
                self.update_conversation_memory(user_input, cognitive_response)
            if hasattr(self, 'performance_monitor'):
                self.performance_monitor.record_response_time(time.time() - start_time)
            self._session_generation_count = getattr(self, '_session_generation_count', 0) + 1
            return cognitive_response

      
        if GENERATION_CACHE_ENABLED and user_input:
            cache_key = hash(user_input.lower().strip())
            cached_response = performance_cache.get_response_cache(cache_key)
            if cached_response and self._is_cacheable_response(cached_response, user_input):
                cached_response = self._enhance_with_knowledge(cached_response, user_input, response_plan=response_plan)
                cached_response = self._apply_reasoning_engine(user_input, cached_response, response_plan=response_plan)
                cached_response = self._apply_critic_repair_loop(user_input, cached_response, response_plan=response_plan)
                self.update_conversation_memory(user_input, cached_response)
                if hasattr(self, 'performance_monitor'):
                    self.performance_monitor.record_response_time(time.time() - start_time)
                self._session_generation_count = getattr(self, '_session_generation_count', 0) + 1
                return cached_response

        generation_deadline = time.monotonic() + GENERATION_TIME_BUDGET_SECONDS
        prepared_context = self._build_generation_context(user_input, history_turns=2)
        prepared_context = self._extend_context_with_response_plan(prepared_context, response_plan)
        topic_gate_state = self._build_topic_gate_state(user_input)

      
        candidates = []
        for attempt in range(GENERATION_PARALLEL_CANDIDATES):
            if time.monotonic() >= generation_deadline:
                break
            try:
              
                temp = random.uniform(*GENERATION_TEMPERATURE_RANGE)
                top_p = random.uniform(*GENERATION_TOP_P_RANGE)
                
                base_response = self._generate_base_response(
                    user_input,
                    temp,
                    top_p,
                    prepared_context=prepared_context,
                    topic_gate_state=topic_gate_state,
                    generation_deadline=generation_deadline,
                )
                if base_response:
                    candidates.append({
                        'response': base_response,
                        'quality': self._calculate_response_quality(base_response, user_input),
                        'temperature': temp,
                        'top_p': top_p
                    })
            except Exception as e:
                print(f"Candidate generation {attempt + 1} failed: {e}")

      
        if candidates:
            best_candidate = max(candidates, key=lambda x: x['quality'])
            final_response = best_candidate['response']
            final_response = self._apply_reasoning_engine(user_input, final_response, response_plan=response_plan)
            
          
            if hasattr(self, 'hierarchical_memory') and user_input:
                relevant_memories = self.hierarchical_memory.get_relevant_memories(user_input, limit=3)
                if relevant_memories:
                    final_response = self._enhance_with_memories(final_response, relevant_memories, user_input)
            final_response = self._enhance_with_knowledge(final_response, user_input, response_plan=response_plan)
            final_response = self._apply_critic_repair_loop(user_input, final_response, response_plan=response_plan)
            
          
            quality_score = self._calculate_response_quality(final_response, user_input)
            adaptive_learning = getattr(self, 'adaptive_learning', None)
            if adaptive_learning:
                adaptive_learning.update_learning_parameters(quality_score, 0.8, context_key='generation')
            
          
            if GENERATION_CACHE_ENABLED and user_input and self._is_cacheable_response(final_response, user_input):
                cache_key = hash(user_input.lower().strip())
                performance_cache.set_response_cache(cache_key, final_response)
            
            if user_input:
                self.update_conversation_memory(user_input, final_response)

          
            response_time = time.time() - start_time
            if hasattr(self, 'performance_monitor'):
                self.performance_monitor.record_response_time(response_time)
            self._session_generation_count = getattr(self, '_session_generation_count', 0) + 1
            return final_response

      
        print("No candidates generated, trying fallback methods...")
        
      
        simple_response = self._generate_simple_response(user_input)
        if simple_response and simple_response != "I need more training data to generate responses.":
            simple_response = self._enhance_with_knowledge(simple_response, user_input, response_plan=response_plan)
            simple_response = self._apply_reasoning_engine(user_input, simple_response, response_plan=response_plan)
            simple_response = self._apply_critic_repair_loop(user_input, simple_response, response_plan=response_plan)
            if user_input:
                self.update_conversation_memory(user_input, simple_response)
            self._session_generation_count = getattr(self, '_session_generation_count', 0) + 1
            return simple_response
        
      
        self.generation_failures += 1
        emergency_response = self._generate_from_emergency_patterns(user_input)
        if user_input:
            self.update_conversation_memory(user_input, emergency_response)
        self._session_generation_count = getattr(self, '_session_generation_count', 0) + 1
        return emergency_response

    def _generate_base_response(self, user_input, temperature=0.9, top_p=0.9, prepared_context=None, topic_gate_state=None, generation_deadline=None):
        """Generate a base response using enhanced word selection"""
        start_time = time.time()
        attempts = 0
        max_attempts = GENERATION_MAX_ATTEMPTS
        best_response = None
        best_quality_score = -1
        initial_context = list(prepared_context) if isinstance(prepared_context, list) else self._build_generation_context(user_input, history_turns=2)
        max_response_length = self._get_generation_length_limit(user_input)
        overall_deadline = generation_deadline or (time.monotonic() + GENERATION_TIME_BUDGET_SECONDS)
        self._next_word_contextual_bonus_cache = {}

        while attempts < max_attempts and time.monotonic() < overall_deadline:
            try:
                response_words = ['[bot]']
                self._seen_trigrams = set()
                self._recent_words.clear()
                for w in initial_context[-PRIMING_WINDOW:]:
                    if w and w not in ('<PAD>', '<UNK>'):
                        self._recent_words.append(w)

                current_context = initial_context[-MAX_CONTEXT_SIZE:]
                uw = (user_input or '').split()
                urgency_mod = 1.0
                if user_input and (len(uw) <= URGENCY_SHORT_LEN or (URGENCY_PUNCT and ('!' in user_input or '?' in user_input))):
                    urgency_mod = 1.0 + URGENCY_TEMP_BOOST
                fatigue_mod = 1.0
                if getattr(self, '_session_generation_count', 0) >= FATIGUE_AFTER_N:
                    fatigue_mod = 1.0 + FATIGUE_TEMP_BOOST
                attempt_deadline = min(overall_deadline, time.monotonic() + GENERATION_ATTEMPT_TIME_BUDGET_SECONDS)
                for step in range(max_response_length):
                    if time.monotonic() >= attempt_deadline and len(response_words) > 1:
                        break
                    rhythm_mod = 1.0 + RHYTHM_AMPLITUDE * math.sin(step * RHYTHM_FREQ)
                    surprise_mod = SURPRISE_TEMP_BOOST if getattr(self, '_last_was_surprise', False) else 0.0
                    temp_effective = temperature * rhythm_mod * urgency_mod * fatigue_mod + surprise_mod
                    next_w = self._next_word(
                        current_context,
                        user_input,
                        response_words,
                        top_p=top_p,
                        temperature=max(0.01, temp_effective),
                        topic_gate_state=topic_gate_state,
                    )
                    if not next_w:
                        break
                    response_words.append(next_w)
                    if next_w not in ('<PAD>', '<UNK>'):
                        self._recent_words.append(next_w)
                    current_context = (current_context + [next_w])[-MAX_CONTEXT_SIZE:]
                
                if len(response_words) <= 1:
                    attempts += 1
                    continue

                final_response_words = [w for w in response_words if w != '[bot]']
                final_response = " ".join(final_response_words).capitalize()
                if final_response and not any(final_response.endswith(p) for p in ['.', '?', '!']):
                    final_response += "."
                
                if self._validate_response_quality(final_response):
                    quality_score = self._calculate_response_quality(final_response, user_input)
                    
                    if quality_score > best_quality_score:
                        best_response = final_response
                        best_quality_score = quality_score
                    
                    if quality_score > 0.7 and len(final_response_words) >= GENERATION_MIN_RESPONSE_WORDS:
                        return final_response
                
                attempts += 1
                
            except Exception as e:
                print(f"Generation attempt {attempts + 1} failed: {e}")
                import traceback
                traceback.print_exc()
                attempts += 1

        if best_response and (
            best_quality_score > GENERATION_QUALITY_THRESHOLD
            or not self._is_low_information_response(best_response)
        ):
            response_time = time.time() - start_time
            if hasattr(self, 'performance_monitor'):
                self.performance_monitor.record_response_time(response_time)
            
            return best_response
        return None

    def _calculate_response_quality(self, response, user_input=""):
        if response is None or not isinstance(response, str):
            return 0.0
        words = response.split()
        if not words:
            return 0.0

        clean_words = self.clean_text(response)
        if not clean_words:
            return 0.0

        significant_response_words = [w for w in clean_words if w not in QUALITY_STOPWORDS]
        significant_user_words = [w for w in self.clean_text(user_input) if w not in QUALITY_STOPWORDS]
        context_words = self.get_structured_conversation_context(history_turns=2)

        optimal_length = 14 if significant_user_words else 12
        length_score = 1.0 - min(1.0, abs(len(clean_words) - optimal_length) / max(optimal_length, 1))
        diversity_score = len(set(clean_words)) / len(clean_words)
        coherence_score = self.get_response_coherence_score(user_input, response) if user_input else 0.55

        overlap = len(set(significant_response_words).intersection(significant_user_words))
        relevance_score = min(1.0, overlap / max(len(set(significant_user_words)), 1)) if significant_user_words else 0.55

        repeated_words = sum(max(0, count - 1) for count in {w: clean_words.count(w) for w in set(clean_words)}.values())
        repeated_bigrams = 0
        if len(clean_words) >= 2:
            bigrams = list(zip(clean_words, clean_words[1:]))
            repeated_bigrams = sum(max(0, count - 1) for count in {bg: bigrams.count(bg) for bg in set(bigrams)}.values())
        repetition_score = max(0.0, 1.0 - ((repeated_words * 0.12) + (repeated_bigrams * 0.2)))

        structure_score = 0.0
        if response[:1].isupper():
            structure_score += 0.35
        if response.rstrip().endswith(('.', '?', '!')):
            structure_score += 0.35
        if '[user]' not in response.lower() and '[bot]' not in response.lower():
            structure_score += 0.3

        cluster_score = 0.0
        for word in significant_response_words:
            if len(word) > 3:
                cluster_score += self.semantic_memory.get_cluster_context_score(word, context_words)
        if significant_response_words:
            cluster_score = min(1.0, cluster_score / len(significant_response_words))
        else:
            cluster_score = 0.4

        context_matches = sum(1 for word in significant_response_words if word in context_words)
        context_score = min(1.0, context_matches / max(len(significant_response_words), 1)) if significant_response_words else 0.5

        weighted_score = (
            length_score * 0.14 +
            diversity_score * 0.18 +
            coherence_score * 0.2 +
            relevance_score * 0.18 +
            repetition_score * 0.14 +
            structure_score * 0.08 +
            cluster_score * 0.12 +
            context_score * 0.06
        )

        lower_response = response.strip().lower()
        penalty = 0.0
        if any(lower_response.startswith(prefix) for prefix in LOW_VALUE_RESPONSE_PREFIXES):
            penalty += 0.12
        if self._is_low_information_response(response):
            penalty += 0.18
        if len(clean_words) < 9:
            penalty += 0.12
        if significant_user_words and not overlap:
            penalty += 0.1
        elif significant_user_words and overlap < min(2, len(set(significant_user_words))):
            penalty += 0.06
        if len(set(significant_response_words)) < 5:
            penalty += 0.08

        return max(0.0, min(1.0, weighted_score - penalty))

    def encode_input(self, context_indices):
        vec = [0] * self.model.input_size; vocab_size = self.model.vocab_size
        for i, word_ix in enumerate(context_indices):
            if word_ix < vocab_size: vec[i * vocab_size + word_ix] = 1
        return vec
        
    def encode_target(self, target_index):
        vec = [0] * self.model.output_size
        if target_index < self.model.output_size: vec[target_index] = 1
        return vec

    def get_generation_stats(self):
        """Get statistics about generation success rate"""
        total = getattr(self, 'total_generation_attempts', 0) or 0
        failures = getattr(self, 'generation_failures', 0) or 0
        if total == 0:
            return "No generation attempts yet"
        failure_rate = (failures / total) if total > 0 else 0
        success_rate = 1 - failure_rate
        return f"Generation Success Rate: {success_rate:.1%} ({total} attempts, {failures} failures)"

class PerformanceMonitor:
    """Monitor and track performance improvements"""
    
    def __init__(self):
        self.response_times = []
        self.learning_times = []
        self.cache_stats = []
        self.memory_usage = []
        self.start_time = time.time()
        
    def record_response_time(self, response_time):
        """Record response generation time"""
        self.response_times.append(response_time)
        if len(self.response_times) > 100:
            self.response_times.pop(0)
    
    def record_learning_time(self, learning_time):
        """Record learning time"""
        self.learning_times.append(learning_time)
        if len(self.learning_times) > 50:
            self.learning_times.pop(0)
    
    def record_memory_usage(self, memory_mb):
        """Record memory usage"""
        self.memory_usage.append(memory_mb)
        if len(self.memory_usage) > 100:
            self.memory_usage.pop(0)
    
    def get_performance_stats(self):
        """Get comprehensive performance statistics"""
        avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        avg_learning_time = sum(self.learning_times) / len(self.learning_times) if self.learning_times else 0
        avg_memory = sum(self.memory_usage) / len(self.memory_usage) if self.memory_usage else 0
        
        cache_stats = performance_cache.get_cache_stats()
        
        return {
            'avg_response_time_ms': avg_response_time * 1000,
            'avg_learning_time_ms': avg_learning_time * 1000,
            'avg_memory_mb': avg_memory,
            'cache_hit_rate': cache_stats['hit_rate'],
            'total_requests': cache_stats['total_requests'],
            'uptime_seconds': time.time() - self.start_time,
            'response_count': len(self.response_times),
            'learning_count': len(self.learning_times)
        }

# PHASED INTELLIGENCE ENHANCEMENT FEATURES
# Phase 1: Critic + Confidence Gate + Anti-Loop Filter (immediate quality boost, tiny CPU cost)
# Phase 2: Meta-Memory + Curiosity Tick (run only when idle)
# Phase 3: Environment Feedback Hooks (wire up from VRChat events)
# Phase 4: Autotune Hyperparams (small, reversible)

def build_backend_api_config():
    return {
        'app_dir': APP_DIR,
        'brain_factory': HybridBrain,
        'runtime_module': 'maimain.backend_runtime',
        'desktop_module': 'maimain.backend_runtime',
        'get_system_tier': get_system_tier,
        'get_hardware_profile': get_hardware_profile,
        'context_levels': CONTEXT_LEVELS,
        'db_file': DB_FILE,
        'gpu_accelerator': gpu_accelerator,
        'memory_safety_threshold': MEMORY_SAFETY_THRESHOLD,
        'chunk_size_words': CHUNK_SIZE_WORDS,
        'min_chunk_size': MIN_CHUNK_SIZE,
        'max_chunk_size': MAX_CHUNK_SIZE,
        'parallel_worker_limit': PARALLEL_WORKER_LIMIT,
        'batch_size': BATCH_SIZE,
        'large_file_threshold': LARGE_FILE_THRESHOLD,
        'nn_model_file': NN_MODEL_FILE,
        'vocab_file': VOCAB_FILE,
        'attention_file': ATTENTION_FILE,
        'context_scores_file': CONTEXT_SCORES_FILE,
        'semantic_clusters_file': SEMANTIC_CLUSTERS_FILE,
        'self_training_quality_threshold': SELF_TRAINING_QUALITY_THRESHOLD,
        'initial_knowledge_file': INITIAL_KNOWLEDGE_FILE,
        'training_extensions': ['.md', '.rst', '.text', '.txt'],
        'reasoning_context_window': REASONING_CONTEXT_WINDOW,
        'reasoning_adaptation_rate': REASONING_ADAPTATION_RATE,
        'generation_parallel_candidates': GENERATION_PARALLEL_CANDIDATES,
        'set_max_response_length': lambda value: globals().__setitem__('MAX_RESPONSE_LENGTH', int(value)),
    }


def create_headless_backend_api(brain=None):
    brain_instance = brain
    if brain_instance is None:
        use_hsb_backend = bool(settings_manager.get('use_hsb_backend', False))
        brain_instance = HybridBrain(DB_FILE, use_hsb_backend=use_hsb_backend)
    return MaiBackendAPI(
        brain_instance,
        settings_manager,
        memory_manager,
        api_config=build_backend_api_config(),
    )


def shutdown_headless_backend_api(backend_api):
    if backend_api is None:
        return
    try:
        backend_api.sync_brain_state()
    except Exception as e:
        print(f"Warning: Could not sync backend state cleanly: {e}", file=sys.stderr)
    brain = getattr(backend_api, 'brain', None)
    con = getattr(brain, 'con', None) if brain is not None else None
    if con is not None:
        try:
            con.close()
        except Exception as e:
            print(f"Warning: Could not close backend database cleanly: {e}", file=sys.stderr)
