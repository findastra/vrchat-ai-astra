import sys, sqlite3, re, random, os, shutil, json, math, copy, threading, time, gc, subprocess, ast
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import psutil
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                               QTextEdit, QLineEdit, QPushButton, QTabWidget, QLabel, 
                               QFileDialog, QProgressBar, QMessageBox, QGroupBox, QFrame, QTextBrowser,
                               QComboBox, QSlider, QSpinBox, QCheckBox)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QTextCursor

DB_FILE = 'mai_phoenix_brain.db'
NN_MODEL_FILE = 'mai_phoenix_model.json'
VOCAB_FILE = 'mai_phoenix_vocab.json'
ATTENTION_FILE = 'mai_attention_weights.json'
CONTEXT_SCORES_FILE = 'mai_context_scores.json'
SEMANTIC_CLUSTERS_FILE = 'mai_semantic_clusters.json'

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

LEARNING_BATCH_SIZE = 2000
LEARNING_PARALLEL_CHUNKS = 3
LEARNING_MEMORY_THRESHOLD = 0.75
LEARNING_GC_INTERVAL = 1500

GENERATION_MAX_ATTEMPTS = 8
GENERATION_QUALITY_THRESHOLD = 0.4
GENERATION_PARALLEL_CANDIDATES = 5
GENERATION_TEMPERATURE_RANGE = (0.7, 1.2)
GENERATION_TOP_P_RANGE = (0.8, 0.95)

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

# CONVERSATION INTELLIGENCE CONSTANTS
CONVERSATION_MEMORY_SIZE = 20
CONVERSATION_TOPIC_TRACKING = True
CONVERSATION_COHERENCE_THRESHOLD = 0.5
CONVERSATION_ADAPTATION_RATE = 0.15
GENERATION_CACHE_ENABLED = True
GENERATION_PARALLEL_CANDIDATES = 3

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
                return int(chunk_setting)
        
        available_mb = self.get_available_memory_mb()
        memory_threshold = settings_manager.get('memory_threshold', MEMORY_SAFETY_THRESHOLD) if 'settings_manager' in globals() else MEMORY_SAFETY_THRESHOLD
        
        estimated_mb_per_word = 0.001
        
        safe_memory_mb = available_mb * (1 - memory_threshold)
        safe_chunk_size = int(safe_memory_mb / estimated_mb_per_word)
        
        optimal_chunk = max(MIN_CHUNK_SIZE, min(MAX_CHUNK_SIZE, safe_chunk_size))
        
        if file_size_words > 1000000:
            optimal_chunk = min(optimal_chunk, CHUNK_SIZE_WORDS)
        
        return optimal_chunk
    
    def calculate_parallel_workers(self):
        """Calculate number of parallel workers based on memory and CPU"""
      
        if 'settings_manager' in globals():
            workers_setting = settings_manager.get('parallel_workers', 'auto')
            if workers_setting != 'auto':
                return int(workers_setting)
        
        available_mb = self.get_available_memory_mb()
        
        if available_mb < 2000:
            return 1
        elif available_mb < 4000:
            return min(2, PARALLEL_WORKER_LIMIT)
        else:
            return PARALLEL_WORKER_LIMIT
    
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
            import subprocess
            import re
            
          
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,compute_cap', '--format=csv,noheader,nounits'], 
                                     capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    for line in result.stdout.strip().split('\n'):
                        if line.strip():
                            parts = line.split(', ')
                            if len(parts) >= 3:
                                name = parts[0].strip()
                                memory = int(parts[1]) if parts[1].isdigit() else 0
                                compute_cap = parts[2].strip()
                                
                                gpu_info = {
                                    'name': name,
                                    'vendor': 'NVIDIA',
                                    'memory_gb': memory,
                                    'compute_capability': compute_cap,
                                    'supports_cuda': True,
                                    'supports_opencl': True,
                                    'ai_acceleration_score': self._calculate_nvidia_score(memory, compute_cap)
                                }
                                self.detected_gpus.append(gpu_info)
            except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
                pass
            
          
            try:
                import wmi
                c = wmi.WMI()
                for gpu in c.Win32_VideoController():
                    if gpu.Name and gpu.AdapterRAM:
                        memory_mb = int(gpu.AdapterRAM) // (1024 * 1024)
                        memory_gb = memory_mb // 1024
                        
                        gpu_name = gpu.Name.strip()
                        vendor = self._detect_vendor(gpu_name)
                        
                        if vendor and memory_gb >= 2:
                            gpu_info = {
                                'name': gpu_name,
                                'vendor': vendor,
                                'memory_gb': memory_gb,
                                'supports_cuda': vendor == 'NVIDIA',
                                'supports_opencl': vendor in ['NVIDIA', 'AMD', 'Intel'],
                                'ai_acceleration_score': self._calculate_general_score(vendor, memory_gb)
                            }
                            
                          
                            if not any(g['name'] == gpu_name for g in self.detected_gpus):
                                self.detected_gpus.append(gpu_info)
            except ImportError:
                pass
            
          
            if self.detected_gpus:
                self.recommended_gpu = max(self.detected_gpus, key=lambda x: x['ai_acceleration_score'])
                
            self.detection_results = {
                'total_gpus': len(self.detected_gpus),
                'ai_capable_gpus': len([g for g in self.detected_gpus if g['ai_acceleration_score'] > 50]),
                'best_gpu': self.recommended_gpu,
                'recommended_acceleration': self.recommended_gpu['ai_acceleration_score'] > 70 if self.recommended_gpu else False
            }
            
        except Exception as e:
            print(f"GPU detection failed: {e}")
            self.detection_results = {'error': str(e)}
    
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
    
    def _calculate_nvidia_score(self, memory_gb, compute_cap):
        """Calculate AI acceleration score for NVIDIA GPUs"""
        score = 0
        
      
        if memory_gb >= 8:
            score += 40
        elif memory_gb >= 6:
            score += 35
        elif memory_gb >= 4:
            score += 25
        elif memory_gb >= 2:
            score += 15
        
      
        try:
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
        except:
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
    
    def get_installation_recommendation(self):
        """Get recommendation for GPU acceleration installation"""
        if not self.recommended_gpu:
            return {
                'install': False,
                'reason': 'No suitable GPU detected',
                'packages': []
            }
        
        score = self.recommended_gpu['ai_acceleration_score']
        
        if score >= 80:
            return {
                'install': True,
                'reason': f'Excellent AI acceleration GPU detected: {self.recommended_gpu["name"]}',
                'packages': ['pycuda', 'pyopencl'],
                'priority': 'high'
            }
        elif score >= 60:
            return {
                'install': True,
                'reason': f'Good AI acceleration GPU detected: {self.recommended_gpu["name"]}',
                'packages': ['pyopencl'],
                'priority': 'medium'
            }
        elif score >= 40:
            return {
                'install': True,
                'reason': f'Moderate AI acceleration GPU detected: {self.recommended_gpu["name"]}',
                'packages': ['pyopencl'],
                'priority': 'low'
            }
        else:
            return {
                'install': False,
                'reason': f'GPU detected but not suitable for AI acceleration: {self.recommended_gpu["name"]}',
                'packages': [],
                'priority': 'none'
            }

class GPUAccelerator:
    def __init__(self, gpu_detector=None):
        self.gpu_detector = gpu_detector or GPUDetector()
        self.gpu_available = False
        self.gpu_type = "None"
        self.gpu_info = None
        self.initialize_gpu()
    
    def initialize_gpu(self):
        """Initialize GPU acceleration if available and suitable"""
        if not self.gpu_detector.recommended_gpu:
            print("GPU Acceleration: No suitable GPU detected for AI acceleration")
            return
        
        recommendation = self.gpu_detector.get_installation_recommendation()
        if not recommendation['install']:
            print(f"GPU Acceleration: {recommendation['reason']}")
            return
        
      
        if 'pyopencl' in recommendation['packages']:
            try:
                import pyopencl as cl
                platforms = cl.get_platforms()
                if platforms:
                    self.gpu_available = True
                    self.gpu_type = "OpenCL"
                    self.platform = platforms[0]
                    self.device = self.platform.get_devices(cl.device_type.GPU)[0]
                    self.context = cl.Context([self.device])
                    self.queue = cl.CommandQueue(self.context)
                    self.gpu_info = self.gpu_detector.recommended_gpu
                    print(f"GPU Acceleration: OpenCL initialized on {self.device.name}")
                    print(f"AI Acceleration Score: {self.gpu_info['ai_acceleration_score']}/100")
                    return
            except ImportError:
                print("GPU Acceleration: pyopencl not installed")
            except Exception as e:
                print(f"GPU Acceleration: OpenCL initialization failed: {e}")
        
      
        if 'pycuda' in recommendation['packages']:
            try:
                import pycuda.driver as cuda
                import pycuda.autoinit
                self.gpu_available = True
                self.gpu_type = "CUDA"
                self.device = cuda.Device(0)
                self.context = self.device.make_context()
                self.gpu_info = self.gpu_detector.recommended_gpu
                print(f"GPU Acceleration: CUDA initialized on {self.device.name()}")
                print(f"AI Acceleration Score: {self.gpu_info['ai_acceleration_score']}/100")
                return
            except ImportError:
                print("GPU Acceleration: pycuda not installed")
            except Exception as e:
                print(f"GPU Acceleration: CUDA initialization failed: {e}")
        
        print("GPU Acceleration: No compatible GPU acceleration available")
    
    def parallel_process_patterns(self, patterns, operation_type="similarity"):
        """Process patterns in parallel on GPU if available"""
        if not self.gpu_available:
            return self._cpu_fallback(patterns, operation_type)
        
        try:
            if self.gpu_type == "OpenCL":
                return self._opencl_process(patterns, operation_type)
            elif self.gpu_type == "CUDA":
                return self._cuda_process(patterns, operation_type)
        except Exception as e:
            print(f"GPU processing failed, falling back to CPU: {e}")
            return self._cpu_fallback(patterns, operation_type)
    
    def _opencl_process(self, patterns, operation_type):
        """Process using OpenCL"""
      
        kernel_code = """
        __kernel void process_patterns(__global float* input, __global float* output, int size) {
            int gid = get_global_id(0);
            if (gid < size) {
                output[gid] = input[gid] * 1.5f; // Example processing
            }
        }
        """
        program = cl.Program(self.context, kernel_code).build()
        
      
        pattern_data = [float(len(p)) for p in patterns]
        input_buffer = cl.Buffer(self.context, cl.mem_flags.READ_ONLY, size=len(pattern_data) * 4)
        output_buffer = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, size=len(pattern_data) * 4)
        
        cl.enqueue_copy(self.queue, input_buffer, np.array(pattern_data, dtype=np.float32))
        program.process_patterns(self.queue, (len(pattern_data),), None, input_buffer, output_buffer, np.int32(len(pattern_data)))
        
        result = np.empty(len(pattern_data), dtype=np.float32)
        cl.enqueue_copy(self.queue, result, output_buffer)
        return result.tolist()
    
    def _cuda_process(self, patterns, operation_type):
        """Process using CUDA"""
      
        pattern_data = np.array([float(len(p)) for p in patterns], dtype=np.float32)
        result = pattern_data * 1.5
        return result.tolist()
    
    def _cpu_fallback(self, patterns, operation_type):
        """CPU fallback for pattern processing"""
        return [float(len(p)) * 1.5 for p in patterns]

# Settings Manager
class SettingsManager:
    def __init__(self):
        self.settings_file = 'mai_settings.json'
        self.default_settings = {
            'gpu_acceleration': True,
            'gpu_acceleration_enabled': False,
            'gpu_acceleration_type': 'auto',  
            'gpu_memory_limit': 0.8,          
            'gpu_batch_size': 'auto',          
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
                with open(self.settings_file, 'r') as f:
                    return {**self.default_settings, **json.load(f)}
            else:
                return self.default_settings.copy()
        except Exception as e:
            print(f"Error loading settings: {e}")
            return self.default_settings.copy()
    
    def save_settings(self):
        """Save current settings to file"""
        try:
            with open(self.settings_file, 'w') as f:
                json.dump(self.settings, f, indent=2)
        except Exception as e:
            print(f"Error saving settings: {e}")
    
    def reset_to_defaults(self):
        """Reset all settings to default values"""
        self.settings = {
            'gpu_acceleration': True,
            'gpu_memory_limit': 2048,
            'gpu_batch_size': 1000,
            'performance_mode': 'balanced',
            'cache_enabled': True,
            'cache_size': 1000,
            'generation_attempts': 8,
            'generation_quality_threshold': 0.4,
            'enhanced_intelligence': True,
            'advanced_reasoning': True,
            'adaptive_learning': True,
            'creativity_factor': 0.5,
            'memory_replay_interval': MEMORY_REPLAY_INTERVAL,
            'memory_compression_threshold': MEMORY_COMPRESSION_THRESHOLD,
            'context_weight': 0.3,
            'semantic_weight': 0.4,
            'pattern_weight': 0.3
        }
        self.save_settings()
    
    def get(self, key, default=None):
        """Get setting value"""
        return self.settings.get(key, default)
    
    def set(self, key, value):
        """Set setting value"""
        self.settings[key] = value
        self.save_settings()

# Initialize Settings first (needed by MemoryManager)
settings_manager = SettingsManager()

# Initialize GPU Detection and Settings
gpu_detector = GPUDetector()
gpu_accelerator = GPUAccelerator(gpu_detector)

# Initialize Memory Manager after settings
memory_manager = MemoryManager()

# Initialize Performance Cache for speed optimization
performance_cache = PerformanceCache()

# Advanced Intelligence Enhancement Classes
class AdvancedReasoningEngine:
    """Enhanced reasoning capabilities for better response generation"""
    
    def __init__(self):
        self.reasoning_patterns = {
            'causal': ['because', 'therefore', 'as a result', 'consequently'],
            'comparative': ['however', 'although', 'while', 'whereas'],
            'sequential': ['first', 'then', 'next', 'finally'],
            'conditional': ['if', 'when', 'unless', 'provided that']
        }
        self.logical_connectors = self._build_logical_connectors()
    
    def _build_logical_connectors(self):
        """Build comprehensive logical connection patterns"""
        return {
            'cause_effect': ['because', 'since', 'as', 'due to', 'therefore', 'thus', 'consequently'],
            'contrast': ['but', 'however', 'although', 'though', 'while', 'whereas', 'nevertheless'],
            'addition': ['and', 'also', 'moreover', 'furthermore', 'in addition', 'besides'],
            'sequence': ['first', 'second', 'then', 'next', 'finally', 'lastly'],
            'example': ['for example', 'such as', 'like', 'including', 'specifically'],
            'conclusion': ['in conclusion', 'therefore', 'thus', 'hence', 'so']
        }
    
    def analyze_conversation_context(self, user_input, conversation_history):
        """Analyze conversation context for better understanding"""
        context_analysis = {
            'topic': self._extract_main_topic(user_input),
            'sentiment': self._analyze_sentiment(user_input),
            'intent': self._detect_user_intent(user_input),
            'complexity': self._assess_complexity(user_input),
            'context_clues': self._extract_context_clues(conversation_history)
        }
        return context_analysis
    
    def _extract_main_topic(self, text):
        """Extract the main topic from user input"""
        words = text.lower().split()
        topic_keywords = {
            'technology': ['computer', 'software', 'program', 'code', 'tech', 'digital'],
            'science': ['research', 'study', 'experiment', 'theory', 'scientific'],
            'philosophy': ['think', 'believe', 'exist', 'meaning', 'purpose', 'truth'],
            'emotion': ['feel', 'happy', 'sad', 'angry', 'love', 'hate', 'worry'],
            'practical': ['how', 'what', 'when', 'where', 'why', 'help', 'problem']
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in words for keyword in keywords):
                return topic
        return 'general'
    
    def _analyze_sentiment(self, text):
        """Analyze sentiment of user input"""
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'love', 'happy']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'angry', 'sad', 'worried']
        
        words = text.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'
    
    def _detect_user_intent(self, text):
        """Detect user intent from input"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['how', 'what', 'when', 'where', 'why', 'who']):
            return 'question'
        elif any(word in text_lower for word in ['help', 'assist', 'support']):
            return 'request_help'
        elif any(word in text_lower for word in ['think', 'believe', 'opinion']):
            return 'seeking_opinion'
        elif any(word in text_lower for word in ['explain', 'describe', 'tell']):
            return 'seeking_explanation'
        else:
            return 'statement'
    
    def _assess_complexity(self, text):
        """Assess complexity of user input"""
        words = text.split()
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        unique_words = len(set(words))
        complexity_score = (avg_word_length * 0.3) + (unique_words / len(words) * 0.7) if words else 0
        
        if complexity_score > 0.7:
            return 'high'
        elif complexity_score > 0.4:
            return 'medium'
        else:
            return 'low'
    
    def _extract_context_clues(self, conversation_history):
        """Extract context clues from conversation history"""
        if not conversation_history:
            return {}
        
        recent_exchanges = conversation_history[-3:]
        context_clues = {
            'recent_topics': [],
            'user_preferences': [],
            'conversation_style': 'formal'
        }
        
        for exchange in recent_exchanges:
            user_text = exchange.get('user', '').lower()
            if any(word in user_text for word in ['please', 'thank', 'sir', 'madam']):
                context_clues['conversation_style'] = 'formal'
            elif any(word in user_text for word in ['cool', 'awesome', 'hey', 'wow']):
                context_clues['conversation_style'] = 'casual'
        
        return context_clues

class EnhancedSemanticMemory:
    """Enhanced semantic memory with advanced clustering and relationship detection"""
    
    def __init__(self, base_semantic_memory):
        self.base_memory = base_semantic_memory
        self.concept_hierarchies = {}
        self.semantic_relationships = {}
        self.contextual_weights = {}
    
    def build_concept_hierarchy(self, words):
        """Build hierarchical concept relationships"""
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
        
      
        if 'user_input' in memory_data:
            user_input = memory_data['user_input']
          
            if any(word in user_input.lower() for word in ['what', 'how', 'why', 'explain', 'tell me']):
                importance += 0.2
            if len(user_input.split()) > 10:
                importance += 0.1
        
      
        if 'quality' in memory_data:
            importance += memory_data['quality'] * 0.3
        
      
        if 'sentiment' in memory_data:
            if memory_data['sentiment'] in ['positive', 'negative']:
                importance += 0.1
        
      
        if 'topics' in memory_data and memory_data['topics']:
            importance += 0.1
        
        return min(1.0, importance)
    
    def _update_importance_tracking(self, memory_entry):
        """Update importance tracking for memory"""
        memory_id = memory_entry['id']
        
      
        self.memory_frequency[memory_id] = self.memory_frequency.get(memory_id, 0) + 1
        
      
        self.memory_recency[memory_id] = time.time()
        
      
        frequency_score = min(1.0, self.memory_frequency[memory_id] / 10)
        recency_score = max(0.0, 1.0 - (time.time() - self.memory_recency[memory_id]) / 86400)
        
        overall_importance = (
            memory_entry['importance'] * 0.4 +
            frequency_score * MEMORY_FREQUENCY_WEIGHT +
            recency_score * MEMORY_RECENCY_WEIGHT
        )
        
        self.memory_importance[memory_id] = overall_importance
    
    def _create_contextual_links(self, memory_entry):
        """Create contextual links between episodic and semantic memories"""
        memory_id = memory_entry['id']
        memory_data = memory_entry['data']
        
      
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
        
      
        if 'user_input' in memory_data:
            user_input = memory_data['user_input'].lower()
          
            words = user_input.split()
            for word in words:
                if len(word) > 3:
                    concepts.add(word)
        
      
        if 'topics' in memory_data:
            concepts.update(memory_data['topics'])
        
      
        if 'sentiment' in memory_data:
            concepts.add(memory_data['sentiment'])
        
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
        if semantic_id in self.semantic_memory:
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
    
    def _update_all_importance_scores(self):
        """Update importance scores for all memories with decay"""
        current_time = time.time()
        
      
        for memory in self.short_term_memory + self.long_term_memory:
            memory_id = memory['id']
            if memory_id in self.memory_importance:
              
                self.memory_importance[memory_id] *= self.importance_decay_rate
                
              
                if self.memory_importance[memory_id] < self.min_importance_threshold:
                    self._forget_memory(memory)
        
      
        for semantic_id in list(self.semantic_memory.keys()):
            if semantic_id in self.semantic_memory:
                current_importance = self.semantic_memory[semantic_id].get('importance', 0.5)
                decayed_importance = current_importance * self.importance_decay_rate
                
                if decayed_importance < self.min_importance_threshold:
                  
                    self.forgotten_memories.append(self.semantic_memory[semantic_id])
                    del self.semantic_memory[semantic_id]
                else:
                    self.semantic_memory[semantic_id]['importance'] = decayed_importance
    
    def _forget_memory(self, memory):
        """Move memory to forgotten memories"""
        self.forgotten_memories.append(memory)
        
      
        if memory in self.short_term_memory:
            self.short_term_memory.remove(memory)
        if memory in self.long_term_memory:
            self.long_term_memory.remove(memory)
    
    def _promote_memory(self):
        """Promote important memories from short-term to long-term"""
        if not self.short_term_memory:
            return
        
      
        least_important = min(self.short_term_memory, key=lambda m: self.memory_importance.get(m['id'], 0))
        
      
        if self.memory_importance.get(least_important['id'], 0) >= MEMORY_PERSISTENCE_THRESHOLD:
          
            self.long_term_memory.append(least_important)
            if len(self.long_term_memory) > LONG_TERM_MEMORY_SIZE:
              
                self._compress_oldest_memory()
        else:
          
            pass
        
      
        self.short_term_memory.remove(least_important)
    
    def _compress_oldest_memory(self):
        """Compress the oldest memory into semantic form"""
        if not self.long_term_memory:
            return
        
      
        oldest_memory = min(self.long_term_memory, key=lambda m: m['timestamp'])
        
      
        semantic_key = self._create_semantic_key(oldest_memory)
        
        if semantic_key in self.semantic_memory:
          
            self.semantic_memory[semantic_key]['count'] += 1
            self.semantic_memory[semantic_key]['importance'] = max(
                self.semantic_memory[semantic_key]['importance'],
                oldest_memory['importance']
            )
        else:
          
            self.semantic_memory[semantic_key] = {
                'pattern': self._extract_semantic_pattern(oldest_memory),
                'count': 1,
                'importance': oldest_memory['importance'],
                'timestamp': oldest_memory['timestamp']
            }
        
      
        oldest_memory['compressed'] = True
        
      
        if len(self.semantic_memory) > SEMANTIC_MEMORY_SIZE:
            self._remove_least_important_semantic()
    
    def _create_semantic_key(self, memory):
        """Create a semantic key for memory compression"""
        if 'user_input' in memory['data']:
          
            words = memory['data']['user_input'].lower().split()
            key_words = [w for w in words if len(w) > 3][:3]
            return "_".join(key_words) if key_words else "general"
        elif 'topics' in memory['data'] and memory['data']['topics']:
            return "_".join(memory['data']['topics'][:2])
        else:
            return "general"
    
    def _extract_semantic_pattern(self, memory):
        """Extract semantic pattern from memory"""
        pattern = {
            'intent': memory['data'].get('intent', 'statement'),
            'sentiment': memory['data'].get('sentiment', 'neutral'),
            'complexity': memory['data'].get('complexity', 'medium'),
            'topics': memory['data'].get('topics', []),
            'quality': memory['data'].get('quality', 0.5)
        }
        return pattern
    
    def _remove_least_important_semantic(self):
        """Remove least important semantic memory"""
        if not self.semantic_memory:
            return
        
        least_important_key = min(
            self.semantic_memory.keys(),
            key=lambda k: self.semantic_memory[k]['importance']
        )
        del self.semantic_memory[least_important_key]
    
    def _perform_memory_replay(self):
        """Perform memory replay to reinforce connections"""
        print("Performing memory replay...")
        
      
        recent_memories = self.short_term_memory[-5:] + self.long_term_memory[-10:]
        
        for memory in recent_memories:
            if memory['type'] == 'conversation':
              
                self._relearn_from_memory(memory)
        
      
        self._recluster_semantic_memories()
        
      
        self._update_all_importance_scores()
        
        self.last_replay_time = time.time()
        print(f"Memory replay complete. {len(self.semantic_memory)} semantic memories active.")
    
    def _relearn_from_memory(self, memory):
        """Relearn from a memory to reinforce patterns"""
        if 'user_input' in memory['data'] and 'bot_response' in memory['data']:
          
            self.brain.learn_from_conversation_exchange(
                memory['data']['user_input'],
                memory['data']['bot_response'],
                memory['data'].get('quality', 0.5)
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
        
      
        if memory1['pattern']['intent'] == memory2['pattern']['intent']:
            similarity += 0.3
        
      
        if memory1['pattern']['sentiment'] == memory2['pattern']['sentiment']:
            similarity += 0.2
        
      
        topics1 = set(memory1['pattern']['topics'])
        topics2 = set(memory2['pattern']['topics'])
        if topics1 and topics2:
            topic_overlap = len(topics1.intersection(topics2)) / len(topics1.union(topics2))
            similarity += topic_overlap * 0.3
        
      
        quality_diff = abs(memory1['pattern']['quality'] - memory2['pattern']['quality'])
        similarity += (1.0 - quality_diff) * 0.2
        
        return similarity
    
    def _merge_semantic_memories(self, key1, key2):
        """Merge two semantic memories"""
        memory1 = self.semantic_memory[key1]
        memory2 = self.semantic_memory[key2]
        
      
        merged_memory = {
            'pattern': {
                'intent': memory1['pattern']['intent'],
                'sentiment': memory1['pattern']['sentiment'],
                'complexity': max(memory1['pattern']['complexity'], memory2['pattern']['complexity']),
                'topics': list(set(memory1['pattern']['topics'] + memory2['pattern']['topics'])),
                'quality': max(memory1['pattern']['quality'], memory2['pattern']['quality'])
            },
            'count': memory1['count'] + memory2['count'],
            'importance': max(memory1['importance'], memory2['importance']),
            'timestamp': min(memory1['timestamp'], memory2['timestamp'])
            }
        
      
        self.semantic_memory[key1] = merged_memory
        
      
        del self.semantic_memory[key2]
    
    def _update_all_importance_scores(self):
        """Update importance scores for all memories"""
        current_time = time.time()
        
        for memory in self.short_term_memory + self.long_term_memory:
            memory_id = memory['id']
            if memory_id in self.memory_recency:
                recency_score = max(0.0, 1.0 - (current_time - self.memory_recency[memory_id]) / 86400)
                frequency_score = min(1.0, self.memory_frequency.get(memory_id, 0) / 10)
                
                overall_importance = (
                    memory['importance'] * 0.4 +
                    frequency_score * MEMORY_FREQUENCY_WEIGHT +
                    recency_score * MEMORY_RECENCY_WEIGHT
                )
                
                self.memory_importance[memory_id] = overall_importance
    
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
        if 'user_input' in memory['data']:
            query_words = set(query.lower().split())
            memory_words = set(memory['data']['user_input'].lower().split())
            
            if memory_words:
                overlap = len(query_words.intersection(memory_words))
                return overlap / len(memory_words)
        
        return 0.0
    
    def _calculate_semantic_relevance(self, semantic_memory, query):
        """Calculate relevance of a semantic memory to a query"""
        query_words = set(query.lower().split())
        topic_words = set(semantic_memory['pattern']['topics'])
        
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

class AdaptiveLearningSystem:
    """Adaptive learning system that improves based on feedback and performance"""
    
    def __init__(self):
        self.learning_patterns = {}
        self.performance_metrics = {
            'response_quality': [],
            'user_satisfaction': [],
            'conversation_coherence': [],
            'learning_rate': 1.0
        }
        self.adaptive_parameters = {
            'context_weight': 0.3,
            'semantic_weight': 0.4,
            'pattern_weight': 0.3,
            'creativity_factor': 0.5
        }
    
    def update_learning_parameters(self, feedback_score, response_quality):
        """Update learning parameters based on feedback"""
        self.performance_metrics['response_quality'].append(response_quality)
        self.performance_metrics['user_satisfaction'].append(feedback_score)
        
      
        recent_quality = sum(self.performance_metrics['response_quality'][-10:]) / min(10, len(self.performance_metrics['response_quality']))
        if recent_quality > 0.7:
            self.performance_metrics['learning_rate'] = min(1.5, self.performance_metrics['learning_rate'] * 1.1)
        elif recent_quality < 0.3:
            self.performance_metrics['learning_rate'] = max(0.5, self.performance_metrics['learning_rate'] * 0.9)
    
    def get_adaptive_weights(self, context_analysis):
        """Get adaptive weights for response generation"""
        weights = self.adaptive_parameters.copy()
        
      
        if context_analysis.get('complexity') == 'high':
            weights['semantic_weight'] *= 1.2
            weights['context_weight'] *= 1.1
        elif context_analysis.get('complexity') == 'low':
            weights['pattern_weight'] *= 1.2
            weights['creativity_factor'] *= 1.1
        
      
        total = sum(weights.values())
        return {k: v/total for k, v in weights.items()}

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
        
        if analysis['reasoning_needed']:
            return self._apply_reasoning_patterns(base_response, analysis)
        else:
            return base_response
    
    def _extract_context_words(self, user_input, conversation_history):
        """Extract important context words"""
        all_text = user_input + " " + " ".join([str(h) for h in conversation_history[-3:]])
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
        input_lower = user_input.lower()
        
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
        words = user_input.split()
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
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
        reasoning_triggers = ['why', 'how', 'explain', 'because', 'reason', 'logic', 'think', 'opinion', 'believe']
        return any(trigger in user_input.lower() for trigger in reasoning_triggers)
    
    def _find_logical_connectors(self, user_input):
        """Find logical connectors in the input"""
        found_connectors = []
        for category, connectors in self.logical_connectors.items():
            for connector in connectors:
                if connector in user_input.lower():
                    found_connectors.append((category, connector))
        return found_connectors
    
    def _apply_reasoning_patterns(self, base_response, analysis):
        """Apply reasoning patterns to enhance the response"""
        enhanced_response = base_response
        
      
        if analysis['intent'] == 'question':
            if analysis['complexity'] == 'high':
                enhanced_response = f"Based on the context, {enhanced_response.lower()}"
            else:
                enhanced_response = f"I think {enhanced_response.lower()}"
        
        elif analysis['intent'] == 'seeking_explanation':
            enhanced_response = f"Let me explain: {enhanced_response.lower()}"
        
        elif analysis['intent'] == 'seeking_opinion':
            enhanced_response = f"In my view, {enhanced_response.lower()}"
        
      
        if analysis['complexity'] == 'high':
            enhanced_response = f"Given the complexity of this topic, {enhanced_response.lower()}"
        
      
        if analysis['sentiment'] == 'positive':
            enhanced_response = f"That's great! {enhanced_response}"
        elif analysis['sentiment'] == 'negative':
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
        quality_trend = [p['quality'] for p in recent_performance]
        
      
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
        
        recent_quality = [p['quality'] for p in self.performance_history[-5:]]
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
        
        recent_quality = [p['quality'] for p in self.performance_history[-10:]]
        avg_quality = sum(recent_quality) / len(recent_quality)
        trend = "improving" if recent_quality[-1] > recent_quality[0] else "declining" if recent_quality[-1] < recent_quality[0] else "stable"
        
        return f"Average Quality: {avg_quality:.3f}, Trend: {trend}, Learning Rates: {self.learning_rates}"

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
      
        response = self.brain.generate_response(user_input)
        
      
        if any(word in user_input.lower() for word in ['why', 'how', 'what']):
            reasoning_connector = random.choice(self.reasoning_engine.logical_connectors['cause_effect'])
            response = f"{response} {reasoning_connector} this approach considers multiple factors."
        
        return response
    
    def _generate_explanation_response(self, user_input, adaptive_weights):
        """Generate explanatory response"""
        response = self.brain.generate_response(user_input)
        
      
        if len(response.split()) > 10:
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
        return self.brain.generate_response(user_input)
    
    def _select_best_candidate(self, candidates, user_input, context_analysis):
        """Select the best response candidate"""
        if not candidates:
            return self.brain.generate_response(user_input)
        
      
        scored_candidates = []
        for candidate in candidates:
            score = self._score_candidate(candidate, user_input, context_analysis)
            scored_candidates.append((candidate, score))
        
      
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return scored_candidates[0][0]
    
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
        user_words = set(user_input.lower().split())
        response_words = set(response.lower().split())
        
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

# FIX: Move memory manager initialization earlier and add error handling
def process_text_chunk(chunk_data):
    try:
        chunk_text, chunk_id, total_chunks = chunk_data
        
        chunk_brain = HybridBrain(':memory:', is_clone=True)
        
      
        chunk_brain.clone_from_main()
        
      
        if not hasattr(chunk_brain, 'batch_operations'):
            chunk_brain.batch_operations = []
            chunk_brain.batch_count = 0
        
        words = chunk_brain.clean_text(chunk_text)
        
      
        processed_words = chunk_brain.learn_from_text_optimized(chunk_text, base_priority_boost=2)

        patterns = []
        chunk_brain.cur.execute("""
            SELECT context_len, word1, word2, word3, word4, word5, word6, word7, word8, next_word, priority
            FROM dynamic_word_chain
        """)
        patterns = chunk_brain.cur.fetchall()
        
        associations = []
        chunk_brain.cur.execute("""
            SELECT source_word, next_word, priority
            FROM word_associations
        """)
        associations = chunk_brain.cur.fetchall()
        
        sample_words = words[::5] if len(words) > 5000 else words
        
        chunk_brain.con.close()
        
        return {
            'chunk_id': chunk_id,
            'word_count': processed_words,
            'patterns': patterns,
            'associations': associations,
            'chunk_words': sample_words[:1000],
            'success': True
        }
        
    except Exception as e:
        return {
            'chunk_id': chunk_data[1] if len(chunk_data) > 1 else -1,
            'error': str(e),
            'success': False
        }

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
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
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
        with open(filename, 'w', encoding='utf-8') as f: 
            json.dump(data, f)
    
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
            except json.JSONDecodeError:
                print(f"Warning: Could not parse '{filename}'. Attention weights not loaded.")

class PlainMLP:
    def __init__(self, vocab_size, hidden_size=32, context_size=4):
        self.vocab_size, self.context_size, self.hidden_size, self.output_size = vocab_size, context_size, hidden_size, vocab_size
        self.input_size = self.vocab_size * self.context_size
        self.base_lr = 0.01
        self.adaptive_lr = {}
        self.prediction_accuracy = {}
        
        self.w1 = [[random.uniform(-0.5, 0.5) for _ in range(self.hidden_size)] for _ in range(self.input_size)]
        self.b1 = [0] * self.hidden_size
        self.w2 = [[random.uniform(-0.5, 0.5) for _ in range(self.output_size)] for _ in range(self.hidden_size)]
        self.b2 = [0] * self.output_size
        
    def sigmoid(self, x): 
        return 1 / (1 + math.exp(-max(-500, min(500, x))))
    
    def softmax(self, z):
        if not z: return []
        max_z = max(z)
        exp_z = [math.exp(i - max_z) for i in z]
        sum_exp_z = sum(exp_z)
        return [i / sum_exp_z for i in exp_z] if sum_exp_z > 0 else [1/len(z)] * len(z)
    
    def forward(self, x):
        self.z1 = [sum(x[i] * self.w1[i][j] for i in range(self.input_size)) + self.b1[j] for j in range(self.hidden_size)]
        self.a1 = [self.sigmoid(val) for val in self.z1]
        self.z2 = [sum(self.a1[i] * self.w2[i][j] for i in range(self.hidden_size)) + self.b2[j] for j in range(self.output_size)]
        return self.softmax(self.z2)
    
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
    
    def to_dict(self): 
        return {
            'w1': self.w1, 'b1': self.b1, 'w2': self.w2, 'b2': self.b2, 
            'vocab_size': self.vocab_size, 'context_size': self.context_size,
            'adaptive_lr': self.adaptive_lr, 'prediction_accuracy': self.prediction_accuracy
        }
    
    def from_dict(self, data):
        self.__init__(data['vocab_size'], context_size=data.get('context_size', NN_CONTEXT_SIZE))
        self.w1, self.b1, self.w2, self.b2 = data['w1'], data['b1'], data['w2'], data['b2']
        self.adaptive_lr = data.get('adaptive_lr', {})
        self.prediction_accuracy = data.get('prediction_accuracy', {})

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
        return sum(recent_performance) / len(recent_performance)
    
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
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f)
    
    def load_scores(self, filename):
        if os.path.exists(filename):
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.context_performance = data.get('context_performance', {})
                self.context_usage_count = data.get('context_usage_count', {})
            except Exception as e:
                print(f"Error loading context scores: {e}")

class FileTrainingWorker(QThread):
    progress_updated = Signal(int)
    status_updated = Signal(str)
    training_complete = Signal(str)
    memory_status_updated = Signal(str)
    
    def __init__(self, brain, file_paths):
        super().__init__()
        self.brain = brain
        self.file_paths = file_paths
        self.is_running = True
    
    def stream_and_chunk_file(self, file_path, chunk_size_words):
        """
        Reads a file from disk in chunks without loading the whole file into memory.
        This is a generator that yields text chunks.
        """
        buffer = []
        word_count = 0
        max_buffer_size = chunk_size_words * 2
        
      
        if not os.path.exists(file_path):
            self.status_updated.emit(f"ERROR: File {file_path} does not exist!")
            return
        
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            self.status_updated.emit(f"ERROR: File {file_path} is empty!")
            return
        
        self.status_updated.emit(f"Reading file: {file_path} ({file_size:,} bytes)")
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                line_count = 0
                for line in f:
                    if not self.is_running: 
                        self.status_updated.emit("Training stopped by user")
                        return
                    
                    line_count += 1
                    if line_count % 1000 == 0:
                        self.status_updated.emit(f"Reading line {line_count}...")
                    
                  
                    if memory_manager.get_memory_usage_percent() > MEMORY_SAFETY_THRESHOLD:
                        self.status_updated.emit("Memory pressure detected, forcing garbage collection...")
                        memory_manager.force_garbage_collection()
                    
                    words = line.split()
                    buffer.extend(words)
                    word_count += len(words)
                    
                  
                    while len(buffer) >= chunk_size_words:
                        chunk_words = buffer[:chunk_size_words]
                        buffer = buffer[chunk_size_words:]
                        chunk_text = ' '.join(chunk_words)
                        if chunk_text.strip():
                            yield chunk_text
                    
                  
                    if len(buffer) > max_buffer_size:
                        self.status_updated.emit("Buffer size limit reached, processing remaining words...")
                        remaining_chunk = ' '.join(buffer)
                        buffer = []
                        if remaining_chunk.strip():
                            yield remaining_chunk
                        
            if buffer:
                final_chunk = ' '.join(buffer)
                if final_chunk.strip():
                    yield final_chunk
            
            self.status_updated.emit(f"File reading complete: {line_count} lines, {word_count} words processed")
            
        except Exception as e:
            self.status_updated.emit(f"ERROR reading {file_path}: {e}")
            import traceback
            traceback.print_exc()

    def merge_chunk_results(self, chunk_results):
        self.status_updated.emit("Aggregating results from all chunks...")
        
        aggregated_patterns = defaultdict(int)
        aggregated_associations = defaultdict(int)
        for result in chunk_results:
            if not result.get('success'):
                print(f"Skipping failed chunk {result.get('chunk_id', 'unknown')}: {result.get('error', 'unknown error')}")
                continue
            for p in result.get('patterns', []):
                key, priority = tuple(p[:-1]), p[-1]
                aggregated_patterns[key] += priority
            for a in result.get('associations', []):
                key, priority = tuple(a[:-1]), a[-1]
                aggregated_associations[key] += priority

        self.status_updated.emit(f"Aggregated {len(aggregated_patterns):,} unique patterns. Writing to database...")
        merge_con = None
        try:
            merge_con = sqlite3.connect(self.brain.db_file, check_same_thread=False)
            merge_cur = merge_con.cursor()
            merge_cur.execute("PRAGMA journal_mode=WAL;")
            merge_cur.execute("PRAGMA synchronous=NORMAL;")
            merge_cur.execute("BEGIN TRANSACTION")
            pattern_data = [(*key, value, value) for key, value in aggregated_patterns.items()]
            assoc_data = [(*key, value, value) for key, value in aggregated_associations.items()]
            if pattern_data:
                merge_cur.executemany("""
                    INSERT INTO dynamic_word_chain (context_len, word1, word2, word3, word4, word5, word6, word7, word8, next_word, priority, usage_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(context_len, word1, word2, word3, word4, word5, word6, word7, word8, next_word) DO UPDATE SET
                        priority = priority + excluded.priority,
                        usage_count = usage_count + excluded.usage_count;
                """, pattern_data)
            if assoc_data:
                merge_cur.executemany("""
                    INSERT INTO word_associations (source_word, next_word, priority, usage_count)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(source_word, next_word) DO UPDATE SET
                        priority = priority + excluded.priority,
                        usage_count = usage_count + excluded.usage_count;
                """, assoc_data)
            merge_cur.execute("COMMIT")
        except Exception as e:
            if merge_con:
                try:
                    merge_cur.execute("ROLLBACK")
                except Exception as rollback_error:
                    print(f"Warning: Could not rollback transaction: {rollback_error}")
            raise e
        finally:
            if merge_con:
                try:
                    merge_con.close()
                except Exception as close_error:
                    print(f"Warning: Could not close merge connection: {close_error}")
        self.brain.con.commit()
        return len(aggregated_patterns), len(aggregated_associations)
    
    def process_large_file_memory_optimized(self, file_path):
        try:
            file_size_bytes = os.path.getsize(file_path)
          
            estimated_word_count = file_size_bytes // 6
            
            self.status_updated.emit(f"Processing file: {os.path.basename(file_path)} ({estimated_word_count:,} est. words)")
            
            chunk_size = memory_manager.calculate_optimal_chunk_size(estimated_word_count)
            self.status_updated.emit(f"Optimal chunk size: {chunk_size:,} words")
            
          
            chunks_generator = self.stream_and_chunk_file(file_path, chunk_size)
            
            parallel_workers = memory_manager.calculate_parallel_workers()
          
            chunks = list(chunks_generator)
            total_chunks = len(chunks)

            self.status_updated.emit(f"Generated {total_chunks} chunks from file")

            if not self.is_running: 
                self.status_updated.emit("Processing stopped by user")
                return 0
            
            if total_chunks == 0:
                self.status_updated.emit("ERROR: No chunks generated from file!")
                return 0

          
            if total_chunks <= 10:
                self.status_updated.emit(f"Using sequential processing for small file ({total_chunks} chunks).")
                return self.process_chunks_sequential(chunks)
            elif parallel_workers > 1 and total_chunks > 1:
                self.status_updated.emit(f"Using parallel processing with {parallel_workers} workers on {total_chunks} chunks.")
                return self.process_chunks_parallel(chunks, parallel_workers)
            else:
                self.status_updated.emit(f"Using sequential processing on {total_chunks} chunks.")
                return self.process_chunks_sequential(chunks)
                
        except Exception as e:
            self.status_updated.emit(f"ERROR in process_large_file_memory_optimized: {e}")
            import traceback
            traceback.print_exc()
            return 0
    
    def process_chunks_parallel(self, chunks, num_workers):
        chunk_data = [(chunk, i, len(chunks)) for i, chunk in enumerate(chunks)]
        processed_words = 0
        chunk_results = []
        all_chunk_words = []
        batch_size = min(num_workers * 2, len(chunks))

      
        use_gpu = settings_manager.get('gpu_acceleration', True) and gpu_accelerator.gpu_available
        if use_gpu:
            self.status_updated.emit(f"Using GPU acceleration ({gpu_accelerator.gpu_type}) with {num_workers} workers")
        else:
            self.status_updated.emit(f"Using CPU parallel processing with {num_workers} workers")

        for batch_start in range(0, len(chunk_data), batch_size):
            if not self.is_running: break
            batch_end = min(batch_start + batch_size, len(chunk_data))
            batch_data = chunk_data[batch_start:batch_end]
            memory_manager.force_garbage_collection()
            mem_usage = memory_manager.get_memory_usage_mb()
            self.memory_status_updated.emit(f"Memory: {mem_usage:.1f}MB")
            
            with Pool(processes=num_workers) as pool:
                batch_results = pool.map(process_text_chunk, batch_data)
                chunk_results.extend(batch_results)
                for result in batch_results:
                    if result.get('success'):
                        all_chunk_words.extend(result.get('chunk_words', []))
                        word_count = result.get('word_count', 0)
                        processed_words += word_count
                        self.status_updated.emit(f"Chunk {result.get('chunk_id', 'unknown')}: {word_count} words processed")
                    else:
                        self.status_updated.emit(f"Chunk {result.get('chunk_id', 'unknown')} failed: {result.get('error', 'unknown error')}")
                progress = int((batch_end / len(chunk_data)) * 100)
                self.progress_updated.emit(progress)
        
        if not self.is_running: return processed_words

        self.status_updated.emit("Merging parallel processing results...")
        self.merge_chunk_results(chunk_results)
        
        if all_chunk_words:
            self.status_updated.emit("Building unified semantic clusters with GPU acceleration...")
          
            if use_gpu:
                self.brain.semantic_memory.update_cooccurrence(all_chunk_words[:100000], force_minimal=True)
              
                cluster_patterns = list(self.brain.semantic_memory.clusters.keys())
                if cluster_patterns:
                    gpu_accelerator.parallel_process_patterns(cluster_patterns, "clustering")
            else:
                self.brain.semantic_memory.update_cooccurrence(all_chunk_words[:100000], force_minimal=True)
            
            self.brain.semantic_memory.rebuild_clusters()
            cluster_count = len(self.brain.semantic_memory.clusters)
            self.status_updated.emit(f"Unified semantic clustering complete: {cluster_count} clusters created.")
        
        self.status_updated.emit(f"Parallel processing complete.")
        return processed_words
    
    def process_chunks_sequential(self, chunks):
        processed_words = 0
        for i, chunk in enumerate(chunks):
            if not self.is_running: break
            mem_usage = memory_manager.get_memory_usage_mb()
            mem_percent = memory_manager.get_memory_usage_percent()
            self.memory_status_updated.emit(f"Memory: {mem_usage:.1f}MB ({mem_percent*100:.1f}%)")
            if mem_percent > MEMORY_SAFETY_THRESHOLD:
                self.status_updated.emit("Memory pressure detected, forcing garbage collection...")
                memory_manager.force_garbage_collection()
            
            self.status_updated.emit(f"Processing chunk {i+1}/{len(chunks)}")
            chunk_words_processed = self.brain.learn_from_text_optimized(chunk, base_priority_boost=2)
            processed_words += chunk_words_processed
            self.status_updated.emit(f"Chunk {i+1}: {chunk_words_processed} words processed")
            
            progress = int((i + 1) / len(chunks) * 100)
            self.progress_updated.emit(progress)
        
        self.brain._flush_batch_operations()
        return processed_words
    
    def run(self):
        try:
            total_files = len(self.file_paths)
            total_words_processed = 0
            for i, file_path in enumerate(self.file_paths):
                if not self.is_running: break
                try:
                    words_processed = self.process_large_file_memory_optimized(file_path)
                    total_words_processed += words_processed
                    memory_manager.force_garbage_collection()
                except Exception as e:
                    self.status_updated.emit(f"Error processing {file_path}: {e}")
                self.progress_updated.emit(int((i + 1) / total_files * 100))
            
            self.brain._flush_batch_operations()
            self.brain.save_state()
            
            result_msg = f"Training complete! Processed approx. {total_words_processed:,} words."
            self.training_complete.emit(result_msg)
            
        except Exception as e:
            self.training_complete.emit(f"Training failed: {str(e)}")
    
    def stop(self):
        self.is_running = False

class HybridBrain:
    def __init__(self, db_file, is_clone=False):
        self.db_file = db_file
        self.is_clone = is_clone
        self.con = sqlite3.connect(db_file if not is_clone else ':memory:', check_same_thread=False)
        self.cur = self.con.cursor()
        self.io_lock = threading.Lock() if not is_clone else None
        
      
        self.enhanced_intelligence_enabled = True
        self.creative_generator = None
        self.reasoning_engine = None
        self.adaptive_learning = None
        
        if not is_clone:
            self._initialize_enhanced_intelligence()

        self.cur.execute("PRAGMA journal_mode=WAL;")
        self.cur.execute("PRAGMA synchronous=NORMAL;")
        
      
        if not is_clone:
            self._setup_enhanced_intelligence_tables()
    
    def _initialize_enhanced_intelligence(self):
        """Initialize enhanced intelligence components"""
        try:
          
            self.semantic_memory = SemanticMemoryCluster()
            
            self.reasoning_engine = AdvancedReasoningEngine()
            self.adaptive_learning = AdaptiveLearningSystem()
            self.creative_generator = CreativeResponseGenerator(self)
            self.hierarchical_memory = HierarchicalMemorySystem(self)
            
          
            self.critic = Critic(self)
            self.confidence_gate = ConfidenceGate(self)
            self.anti_loop_filter = AntiLoopFilter(self)
            self.meta_memory = MetaMemory(self)
            self.curiosity = Curiosity(self)
            self.env_feedback = EnvironmentFeedback(self)
            self.autotune = Autotune(self)
            
          
            self.response_learning = ResponseLearningSystem(self)
            self.truth_fact_table = TruthFactTable(self)
            self.topic_detection = TopicDetectionSystem(self)
            
            print("Enhanced intelligence components initialized successfully")
        except Exception as e:
            print(f"Warning: Could not initialize enhanced intelligence: {e}")
            self.enhanced_intelligence_enabled = False
    
    def _setup_enhanced_intelligence_tables(self):
        """Setup database tables for enhanced intelligence features"""
        if self.is_clone:
            return
            
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
        
        self.generation_failures = 0
        self.total_generation_attempts = 0
        
        self.response_quality_history = []
        
        self.batch_operations = []
        self.batch_count = 0
        self.training_word_count = 0
        
        self.setup_database()
        
        self.word_to_ix, self.ix_to_word = {"<UNK>": 0, "<PAD>": 1}, {0: "<UNK>", 1: "<PAD>"}
        if not self.is_clone and os.path.exists(VOCAB_FILE):
            try:
                with open(VOCAB_FILE, 'r', encoding='utf-8') as f: 
                    self.word_to_ix = json.load(f)
                self.ix_to_word = {int(i): w for w, i in self.word_to_ix.items()}
            except (json.JSONDecodeError, TypeError) as e:
                 print(f"Warning: Could not load vocab file '{VOCAB_FILE}': {e}")

        self.model = PlainMLP(len(self.word_to_ix), context_size=NN_CONTEXT_SIZE)
        if not self.is_clone and os.path.exists(NN_MODEL_FILE):
            try:
                with open(NN_MODEL_FILE, 'r', encoding='utf-8') as f: 
                    self.model.from_dict(json.load(f))
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not load NN model file '{NN_MODEL_FILE}': {e}")
        
        self.attention = AnalogAttention()
        if not self.is_clone: self.attention.load_weights(ATTENTION_FILE)
        
        self.context_scorer = ContextScorer()
        if not self.is_clone: self.context_scorer.load_scores(CONTEXT_SCORES_FILE)
        
      
        self.reasoning_engine = AdvancedReasoningEngine()
        self.adaptive_learning = AdaptiveLearningSystem()
        self.creative_generator = CreativeResponseGenerator(self)
        self.hierarchical_memory = HierarchicalMemorySystem(self)
        
        if self.is_clone: self.clone_from_main()

    def _kn_unigram(self):
        rows = self.cur.execute("SELECT next_word, SUM(priority) FROM dynamic_word_chain GROUP BY next_word").fetchall()
        tot = sum(c for _, c in rows) or 1
        return {w: c / tot for w, c in rows}

    def _kn_next_probs(self, last_words, D=0.75):
        if not last_words: return self._kn_unigram()
        
      
        cache_key = tuple(last_words)
        cached_probs = performance_cache.get_word_prob_cache(cache_key)
        if cached_probs:
            return cached_probs
        
        last_words = [w for w in last_words if w]
        for L in range(min(MAX_CONTEXT_SIZE, len(last_words)), 0, -1):
            ctx = last_words[-L:]
            
            padding = ['<PAD>'] * (MAX_CONTEXT_SIZE - L)
            full_context_tuple = tuple(padding + ctx)

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
                p = {w: max(c - D, 0) / total for w, c in rows}
                lam = (D * len(rows)) / total
                lower = self._kn_next_probs(last_words[-(L-1):], D) if L > 1 else self._kn_unigram()
                for w, q in lower.items():
                    p[w] = p.get(w, 0.0) + lam * q
                
              
                performance_cache.set_word_prob_cache(cache_key, p)
                return p
        return self._kn_unigram()

    def _topic_gate(self, user_text, cand_word):
        stop = {"the","a","an","of","to","and","or","is","it","i","you","we","they","he","she","lol"}
        uw = [w for w in re.findall(r"[a-z]+", user_text.lower()) if w not in stop]
        if not uw: return 1.0
        return 1.0 if cand_word in uw else (0.55 if cand_word[:4] in {u[:4] for u in uw} else 0.35)

    def _idf_penalty(self, word):
        row = self.cur.execute("SELECT SUM(priority) FROM dynamic_word_chain WHERE next_word=?", (word,)).fetchone()
        f = (row[0] or 1)
        return 1.0 / (1.0 + math.log1p(f))

    def _repeat_block(self, generated_words, cand):
        if len(generated_words) >= 2:
            tri = (generated_words[-2], generated_words[-1], cand)
            if tri in self._seen_trigrams:
                return 0.2
        return 1.0

    def _get_sentence_structure_bonus(self, generated_words, candidate_word):
        """Get bonus for sentence structure patterns"""
        if not generated_words:
            return 1.0
            
        try:
          
            if len(generated_words) <= 3:
                pattern = ' '.join(generated_words + [candidate_word])
                row = self.cur.execute("""
                    SELECT priority FROM sentence_patterns 
                    WHERE pattern_type = 'sentence_start' AND pattern_text LIKE ?
                """, (f"{pattern}%",)).fetchone()
                if row:
                    return 1.0 + (row[0] * 0.3)
            
          
            if len(generated_words) >= 2:
                pattern = ' '.join(generated_words[-2:] + [candidate_word])
                row = self.cur.execute("""
                    SELECT priority FROM sentence_patterns 
                    WHERE pattern_type = 'sentence_end' AND pattern_text LIKE ?
                """, (f"%{pattern}",)).fetchone()
                if row:
                    return 1.0 + (row[0] * 0.4)
            
          
            if len(generated_words) >= 2:
                pattern = ' '.join(generated_words[-2:] + [candidate_word])
                row = self.cur.execute("""
                    SELECT priority FROM sentence_patterns 
                    WHERE pattern_type = 'svo_pattern' AND pattern_text LIKE ?
                """, (f"%{pattern}%",)).fetchone()
                if row:
                    return 1.0 + (row[0] * 0.2)
                    
        except Exception as e:
            pass
            
        return 1.0

    def _get_grammar_pattern_bonus(self, generated_words, candidate_word):
        """Get bonus for grammar patterns"""
        if not generated_words:
            return 1.0
            
        try:
            last_word = generated_words[-1]
            
          
            if last_word.lower() in ['the', 'a', 'an']:
                row = self.cur.execute("""
                    SELECT priority FROM grammar_patterns 
                    WHERE pattern_type = 'article_noun' AND word1 = ? AND word2 = ?
                """, (last_word.lower(), candidate_word)).fetchone()
                if row:
                    return 1.0 + (row[0] * 0.5)
            
          
            if candidate_word.endswith(('ing', 'ed', 'er', 'ly')):
                row = self.cur.execute("""
                    SELECT priority FROM grammar_patterns 
                    WHERE pattern_type = 'verb_form' AND word1 = ? AND word2 = ?
                """, (last_word, candidate_word)).fetchone()
                if row:
                    return 1.0 + (row[0] * 0.3)
                    
        except Exception as e:
            pass
            
        return 1.0

    def _get_phrase_pattern_bonus(self, generated_words, candidate_word):
        """Get bonus for phrase patterns"""
        if not generated_words:
            return 1.0
            
        try:
          
            if len(generated_words) >= 1:
                phrase = f"{generated_words[-1]} {candidate_word}"
                row = self.cur.execute("""
                    SELECT priority FROM phrase_patterns 
                    WHERE phrase_text = ?
                """, (phrase,)).fetchone()
                if row:
                    return 1.0 + (row[0] * 0.4)
            
          
            if len(generated_words) >= 2:
                phrase = f"{generated_words[-2]} {generated_words[-1]} {candidate_word}"
                row = self.cur.execute("""
                    SELECT priority FROM phrase_patterns 
                    WHERE phrase_text = ?
                """, (phrase,)).fetchone()
                if row:
                    return 1.0 + (row[0] * 0.5)
                    
        except Exception as e:
            pass
            
        return 1.0

    def _get_semantic_relationship_bonus(self, context_words, candidate_word):
        """Get bonus for semantic relationships"""
        if not context_words:
            return 1.0
            
        try:
          
            for context_word in context_words[-3:]:
                row = self.cur.execute("""
                    SELECT strength FROM semantic_relationships 
                    WHERE (word1 = ? AND word2 = ?) OR (word1 = ? AND word2 = ?)
                """, (context_word, candidate_word, candidate_word, context_word)).fetchone()
                if row:
                    return 1.0 + (row[0] * 0.3)
                    
        except Exception as e:
            pass
            
        return 1.0

    def _next_word(self, context_words, user_text, generated_words, top_p=0.9, temperature=0.9, rep=1.15):
        base = self._kn_next_probs(context_words)
        if not base:
            return None

        scored = []
        for w, p in base.items():
            if not w.strip() or w in ('<PAD>', '<UNK>'):
                continue
            
          
            gate = self._topic_gate(user_text, w)
            idf = self._idf_penalty(w)
            rep_pen = (1.0 / rep) if (generated_words and w == generated_words[-1]) else 1.0
            tri_pen = self._repeat_block(generated_words, w)
            
          
            sentence_bonus = self._get_sentence_structure_bonus(generated_words, w)
            
          
            grammar_bonus = self._get_grammar_pattern_bonus(generated_words, w)
            
          
            phrase_bonus = self._get_phrase_pattern_bonus(generated_words, w)
            
          
            semantic_bonus = self._get_semantic_relationship_bonus(context_words, w)
            
            score = (p ** (1.0 / max(temperature, 1e-6))) * gate * idf * rep_pen * tri_pen * sentence_bonus * grammar_bonus * phrase_bonus * semantic_bonus
            scored.append((w, score))
        
        if not scored: return None

        scored.sort(key=lambda x: x[1], reverse=True)
        
        total_score = sum(s for _, s in scored)
        if total_score < 1e-9: return scored[0][0]
            
        cumulative_p, out = 0.0, []
        for w, s in scored:
            prob = s / total_score
            out.append((w, prob))
            cumulative_p += prob
            if cumulative_p >= top_p:
                break
        
        if not out: return scored[0][0]
        
        final_dist_total = sum(p for _, p in out)
        if final_dist_total < 1e-9: return out[0][0]

        r = random.random()
        acc = 0.0
        for w, p in out:
            acc += p / final_dist_total
            if r <= acc:
                if len(generated_words) >= 2:
                    self._seen_trigrams.add((generated_words[-2], generated_words[-1], w))
                return w
        return out[0][0]

    def reinforce(self, rows, k=1.25):
        for L, ctx, nxt in rows:
            padding = ['<PAD>'] * (MAX_CONTEXT_SIZE - L)
            full_context = tuple(padding + list(ctx))
            conds = " AND ".join([f"word{i+1}=?" for i in range(MAX_CONTEXT_SIZE)])
            sql = f"UPDATE dynamic_word_chain SET priority = priority * ? WHERE context_len=? AND {conds} AND next_word=?"
            self.cur.execute(sql, (k, L, *full_context, nxt))
        self.con.commit()

    def discourage(self, rows, k=0.8):
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
        if len(words) < 2:
            return

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

    def clone_from_main(self):
        main_con = None
        try:
            main_con = sqlite3.connect(f'file:{DB_FILE}?mode=ro', uri=True, check_same_thread=False)
            main_con.backup(self.con)
        except sqlite3.OperationalError as e:
            print(f"Clone failed, could not open main DB (is it locked?): {e}")
        except Exception as e:
            print(f"Clone failed with unexpected error: {e}")
        finally:
            if main_con:
                try:
                    main_con.close()
                except Exception as e:
                    print(f"Warning: Could not close main DB connection: {e}")
        
        try:
            if os.path.exists(NN_MODEL_FILE):
                with open(NN_MODEL_FILE, 'r', encoding='utf-8') as f: 
                    self.model.from_dict(json.load(f))
        except Exception as e:
            print(f"Warning: Could not load NN model: {e}")
            
        try:
            if os.path.exists(VOCAB_FILE):
                with open(VOCAB_FILE, 'r', encoding='utf-8') as f: 
                    self.word_to_ix = json.load(f)
                self.ix_to_word = {int(i): w for w, i in self.word_to_ix.items()}
        except Exception as e:
            print(f"Warning: Could not load vocabulary: {e}")
            
        try:
            self.semantic_memory.load_clusters(SEMANTIC_CLUSTERS_FILE)
        except Exception as e:
            print(f"Warning: Could not load semantic clusters: {e}")

    def save_state(self):
        if self.is_clone or not self.io_lock: return
        threading.Thread(target=self._save_state_threaded).start()

    def _save_state_threaded(self):
        with self.io_lock:
            try:
                with open(VOCAB_FILE, 'w', encoding='utf-8') as f: json.dump(self.word_to_ix, f)
                with open(NN_MODEL_FILE, 'w', encoding='utf-8') as f: json.dump(self.model.to_dict(), f)
                self.attention.save_weights(ATTENTION_FILE)
                self.context_scorer.save_scores(CONTEXT_SCORES_FILE)
                self.semantic_memory.save_clusters(SEMANTIC_CLUSTERS_FILE)
            except Exception as e:
                print(f"Error during threaded save: {e}")

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
        self.response_quality_history.append(quality_score)
        if len(self.response_quality_history) > 50:
            self.response_quality_history.pop(0)
        
        self.learn_from_conversation_exchange(user_input, bot_response, quality_score)
        
      
        memory_entry = {
            "user": user_input, 
            "bot": bot_response, 
            "quality": quality_score,
            "topics": list(self.current_topics.keys())[:5], 
            "timestamp": time.time(),
            "sentiment": self._analyze_sentiment(user_input),
            "complexity": self._assess_complexity(user_input),
            "intent": self._identify_intent(user_input),
            "coherence_score": self._calculate_conversation_coherence(user_input, bot_response)
        }
        
        self.conversation_memory.append(memory_entry)
        if len(self.conversation_memory) > self.max_memory_length:
            self.conversation_memory.pop(0)
        
      
        if hasattr(self, 'hierarchical_memory'):
            self.hierarchical_memory.add_memory(memory_entry, "conversation")
        
      
        self._update_conversation_flow(user_input, bot_response, quality_score)
        self.update_topic_tracker(user_input, bot_response)

    def get_response_coherence_score(self, user_input, bot_response):
        bot_words = set(self.clean_text(bot_response))
        if not bot_words: return 0.5
        topic_relevance = sum(self.get_topic_coherence_score(word) for word in bot_words) / len(bot_words)
        return min(1.0, topic_relevance)

    def _analyze_sentiment(self, text):
        """Analyze sentiment of text"""
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
        else:
            return 'low'

    def _identify_intent(self, text):
        """Identify intent of text"""
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
        
      
        if hasattr(self, 'adaptive_learning'):
            adaptation_rate = self.adaptive_learning.get_learning_rate('conversation_flow')
            self.conversation_coherence_score *= (1 + adaptation_rate * (quality_score - 0.5))

    def _enhance_with_memories(self, response, relevant_memories, user_input):
        """Enhance response with relevant memories"""
        if not relevant_memories:
            return response
        
      
        best_memory = relevant_memories[0]
        
      
        if isinstance(best_memory, dict) and 'data' in best_memory:
          
            if 'bot_response' in best_memory['data']:
              
                memory_response = best_memory['data']['bot_response']
                if len(memory_response) > len(response) * 0.8:
                  
                    response = f"Based on our previous conversation, {response.lower()}"
        
        elif isinstance(best_memory, dict) and 'pattern' in best_memory:
          
            pattern = best_memory['pattern']
            if pattern['intent'] == 'question' and 'what' in user_input.lower():
                response = f"From what I remember, {response.lower()}"
            elif pattern['sentiment'] == 'positive':
                response = f"I'm glad you're interested! {response}"
            elif pattern['sentiment'] == 'negative':
                response = f"I understand this might be challenging. {response}"
        
        return response

    def get_structured_conversation_context(self, history_turns=3):
        """Builds a single list of words representing the last few conversational turns."""
        context_parts = []
        recent_exchanges = self.conversation_memory[-history_turns:]
        for exchange in recent_exchanges:
            context_parts.append(f"[user] {exchange['user']}")
            context_parts.append(f"[bot] {exchange['bot']}")
        full_context_string = " ".join(context_parts)
        return self.clean_text(full_context_string)

  

    def _flush_batch_operations(self):
        if not self.batch_operations: return

      
        pattern_data = []
        assoc_data = []
        sentence_patterns = []
        grammar_patterns = []
        phrase_patterns = []
        semantic_patterns = []

        for op in self.batch_operations:
            if op['type'] == 'chain':
                context_len, *context_words, next_word, priority, usage_count = op['data']
                pattern_data.append((context_len, *context_words, next_word, priority, usage_count))
            elif op['type'] == 'assoc':
                source_word, next_word, priority, usage_count = op['data'][:4]
                assoc_data.append((source_word, next_word, priority, usage_count))
            elif op['type'] in ['sentence_start', 'sentence_end', 'svo_pattern']:
                pattern, priority = op['data']
                sentence_patterns.append((op['type'], ' '.join(pattern), priority))
            elif op['type'] in ['article_noun', 'verb_form']:
                word1, word2, priority = op['data']
                grammar_patterns.append((op['type'], word1, word2, priority))
            elif op['type'] in ['phrase_2', 'phrase_3']:
                if op['type'] == 'phrase_2':
                    word1, word2, priority = op['data']
                    phrase_patterns.append((word1, word2, priority))
                else:
                    word1, word2, word3, priority = op['data']
                    phrase_patterns.append((f"{word1} {word2} {word3}", priority))
            elif op['type'] == 'semantic_rel':
                word1, word2, priority = op['data']
                semantic_patterns.append((word1, word2, priority))

        try:
          
            if pattern_data:
                    self.cur.executemany("""
                    INSERT OR REPLACE INTO dynamic_word_chain 
                    (context_len, word1, word2, word3, word4, word5, word6, word7, word8, next_word, priority, usage_count)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, pattern_data)

            if assoc_data:
                    self.cur.executemany("""
                    INSERT OR REPLACE INTO word_associations 
                    (source_word, next_word, priority, usage_count)
                        VALUES (?, ?, ?, ?)
                    """, assoc_data)

          
            if sentence_patterns:
                self.cur.executemany("""
                    INSERT OR REPLACE INTO sentence_patterns 
                    (pattern_type, pattern_text, priority)
                    VALUES (?, ?, ?)
                """, sentence_patterns)

          
            if grammar_patterns:
                self.cur.executemany("""
                    INSERT OR REPLACE INTO grammar_patterns 
                    (pattern_type, word1, word2, priority)
                    VALUES (?, ?, ?, ?)
                """, grammar_patterns)

          
            if phrase_patterns:
                self.cur.executemany("""
                    INSERT OR REPLACE INTO phrase_patterns 
                    (phrase_text, priority)
                    VALUES (?, ?)
                """, phrase_patterns)

          
            if semantic_patterns:
                self.cur.executemany("""
                    INSERT OR REPLACE INTO semantic_relationships 
                    (word1, word2, strength)
                    VALUES (?, ?, ?)
                """, semantic_patterns)

          
            self.con.commit()
            
        except Exception as e:
            print(f"Error during batch flush: {e}")
          
        finally:
            self.batch_operations = []
            self.batch_count = 0

    def learn_from_text_optimized(self, text, base_priority_boost=1):
        """Enhanced learning with sentence structure and grammar patterns"""
        words = self.clean_text(text)
        if len(words) < 2: 
            return 0

      
        if memory_manager.get_memory_usage_percent() > LEARNING_MEMORY_THRESHOLD:
            memory_manager.force_garbage_collection()

      
        new_words = [w for w in words if w not in self.word_to_ix]
        if new_words:
            new_vocab_size = len(self.word_to_ix)
            for w in new_words:
                self.word_to_ix[w] = new_vocab_size
                self.ix_to_word[new_vocab_size] = w
                new_vocab_size += 1

            if not self.is_clone and new_vocab_size > self.model.vocab_size * 1.1:
                self.model = PlainMLP(new_vocab_size, context_size=NN_CONTEXT_SIZE)
        
      
        max_batch_size = min(1000, len(words) // 5) if len(words) > 2000 else len(words)
        
      
        sentences = self._extract_sentences(text)
        
      
        for sentence in sentences:
            sentence_words = self.clean_text(sentence)
            if len(sentence_words) < 3:
                continue
                
          
            self._learn_sentence_structure(sentence_words, base_priority_boost)
            
          
            self._learn_grammar_patterns(sentence_words, base_priority_boost)
        
      
        for i in range(1, len(words)):
          
            if i % 1000 == 0:
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
                'data': (words[i-1], words[i], base_priority_boost, 0.5)
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
        if quality_score < 0.6:
            return
        learning_text = f"[user] {user_text} [bot] {bot_text}"
        self.learn_from_text_optimized(learning_text, base_priority_boost=2)
        
      
        if self.enhanced_intelligence_enabled and self.reasoning_engine:
            try:
                context_analysis = self.reasoning_engine.analyze_conversation_context(user_text, [])
                
              
                if context_analysis.get('intent') == 'question':
                    self._store_reasoning_pattern('question_response', bot_text, quality_score, context_analysis.get('topic'))
                elif context_analysis.get('intent') == 'seeking_explanation':
                    self._store_reasoning_pattern('explanation', bot_text, quality_score, context_analysis.get('topic'))
                
              
                if self.adaptive_learning:
                    self.adaptive_learning.update_learning_parameters(quality_score, quality_score)
                    
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
            unique_ratio = len(set(recent_words)) / len(recent_words)
            if unique_ratio < 0.5:
                return words[:-2]
        
        return words

    def _validate_response_quality(self, response_text):
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
            avg_topic_coherence = sum(topic_scores) / len(topic_scores)
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
        
        if words:
            response_words = []
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
        start_time = time.time()
        self.total_generation_attempts += 1
        if user_input:
            self.update_topic_tracker(user_input)

      
        if GENERATION_CACHE_ENABLED and user_input:
            cache_key = hash(user_input.lower().strip())
            cached_response = performance_cache.get_response_cache(cache_key)
            if cached_response:
                return cached_response

      
        candidates = []
        for attempt in range(GENERATION_PARALLEL_CANDIDATES):
            try:
              
                temp = random.uniform(*GENERATION_TEMPERATURE_RANGE)
                top_p = random.uniform(*GENERATION_TOP_P_RANGE)
                
                base_response = self._generate_base_response(user_input, temp, top_p)
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
            
          
            if hasattr(self, 'reasoning_engine'):
                conversation_history = self.conversation_memory[-5:] if self.conversation_memory else []
                final_response = self.reasoning_engine.generate_reasoned_response(user_input, conversation_history, final_response)
            
          
            if hasattr(self, 'hierarchical_memory') and user_input:
                relevant_memories = self.hierarchical_memory.get_relevant_memories(user_input, limit=3)
                if relevant_memories:
                    final_response = self._enhance_with_memories(final_response, relevant_memories, user_input)
            
          
            quality_score = best_candidate['quality']
            if hasattr(self, 'adaptive_learning'):
                self.adaptive_learning.update_learning_parameters(quality_score, 0.8)
            
          
            if GENERATION_CACHE_ENABLED and user_input:
                cache_key = hash(user_input.lower().strip())
                performance_cache.set_response_cache(cache_key, final_response)
            
          
            response_time = time.time() - start_time
            if hasattr(self, 'performance_monitor'):
                self.performance_monitor.record_response_time(response_time)
            
            return final_response

      
        print("No candidates generated, trying fallback methods...")
        
      
        simple_response = self._generate_simple_response(user_input)
        if simple_response and simple_response != "I need more training data to generate responses.":
            if user_input:
                self.update_conversation_memory(user_input, simple_response)
            return simple_response
        
      
        self.generation_failures += 1
        emergency_response = self._generate_from_emergency_patterns(user_input)
        if user_input:
            self.update_conversation_memory(user_input, emergency_response)
        return emergency_response

    def generate_response(self, user_input=""):
        """Main response generation method"""
        start_time = time.time()
        self.total_generation_attempts += 1
        if user_input:
            self.update_topic_tracker(user_input)

      
        if GENERATION_CACHE_ENABLED and user_input:
            cache_key = hash(user_input.lower().strip())
            cached_response = performance_cache.get_response_cache(cache_key)
            if cached_response:
                return cached_response

      
        candidates = []
        for attempt in range(GENERATION_PARALLEL_CANDIDATES):
            try:
              
                temp = random.uniform(*GENERATION_TEMPERATURE_RANGE)
                top_p = random.uniform(*GENERATION_TOP_P_RANGE)
                
                base_response = self._generate_base_response(user_input, temp, top_p)
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
            
          
            if hasattr(self, 'reasoning_engine'):
                conversation_history = self.conversation_memory[-5:] if self.conversation_memory else []
                final_response = self.reasoning_engine.generate_reasoned_response(user_input, conversation_history, final_response)
            
          
            if hasattr(self, 'hierarchical_memory') and user_input:
                relevant_memories = self.hierarchical_memory.get_relevant_memories(user_input, limit=3)
                if relevant_memories:
                    final_response = self._enhance_with_memories(final_response, relevant_memories, user_input)
            
          
            quality_score = best_candidate['quality']
            if hasattr(self, 'adaptive_learning'):
                self.adaptive_learning.update_learning_parameters(quality_score, 0.8)
            
          
            if GENERATION_CACHE_ENABLED and user_input:
                cache_key = hash(user_input.lower().strip())
                performance_cache.set_response_cache(cache_key, final_response)
            
          
            response_time = time.time() - start_time
            if hasattr(self, 'performance_monitor'):
                self.performance_monitor.record_response_time(response_time)
            
            return final_response

      
        print("No candidates generated, trying fallback methods...")
        
      
        simple_response = self._generate_simple_response(user_input)
        if simple_response and simple_response != "I need more training data to generate responses.":
            if user_input:
                self.update_conversation_memory(user_input, simple_response)
            return simple_response
        
      
        self.generation_failures += 1
        emergency_response = self._generate_from_emergency_patterns(user_input)
        if user_input:
            self.update_conversation_memory(user_input, emergency_response)
        return emergency_response

    def _generate_base_response(self, user_input, temperature=0.9, top_p=0.9):
        """Generate a base response using enhanced word selection"""
        attempts = 0
        max_attempts = GENERATION_MAX_ATTEMPTS
        best_response = None
        best_quality_score = -1

        while attempts < max_attempts:
            try:
                conversation_history_context = self.get_structured_conversation_context(history_turns=2)
                current_user_context = self.clean_text(f"[user] {user_input}") if user_input else []
                initial_context = conversation_history_context + current_user_context
                
                response_words = ['[bot]']
                self._seen_trigrams = set() 

                current_context = initial_context[-MAX_CONTEXT_SIZE:]

                for _ in range(MAX_RESPONSE_LENGTH):
                    next_w = self._next_word(current_context, user_input, response_words)
                    
                    if not next_w:
                        break
                    
                    response_words.append(next_w)
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
                    
                    if quality_score > 0.65:
                        if user_input:
                            self.update_conversation_memory(user_input, final_response)
                        return final_response
                
                attempts += 1
                
            except Exception as e:
                print(f"Generation attempt {attempts + 1} failed: {e}")
                import traceback
                traceback.print_exc()
                attempts += 1

        if best_response and best_quality_score > GENERATION_QUALITY_THRESHOLD:
            if user_input:
                self.update_conversation_memory(user_input, best_response)
            
          
            response_time = time.time() - start_time
            if hasattr(self, 'performance_monitor'):
                self.performance_monitor.record_response_time(response_time)
            
            return best_response
        else:
            self.generation_failures += 1
            emergency_response = self._generate_from_emergency_patterns(user_input)
            if user_input:
                self.update_conversation_memory(user_input, emergency_response)
            return emergency_response

    def _calculate_response_quality(self, response, user_input=""):
        words = response.split()
        if not words:
            return 0.0
        
        scores = []
        
        optimal_length = 13
        length_score = 1.0 - min(1.0, abs(len(words) - optimal_length) / optimal_length)
        scores.append(length_score)
        
        diversity_score = len(set(words)) / len(words)
        scores.append(diversity_score)
        
        if user_input:
            coherence_score = self.get_response_coherence_score(user_input, response)
            scores.append(coherence_score)
        
        has_articles = any(word in words for word in ['the', 'a', 'an'])
        has_verbs = any(word.endswith(('ing', 'ed', 'er', 'ly')) for word in words)
        structure_score = (0.5 if has_articles else 0) + (0.5 if has_verbs else 0)
        scores.append(structure_score)
        
        context_words = self.get_structured_conversation_context(history_turns=2)
        context_matches = sum(1 for word in words if word in context_words)
        context_score = min(1.0, context_matches / max(len(words) * 0.3, 1))
        scores.append(context_score)
        
        cluster_score = 0
        for word in words:
            if len(word) > 3:
                cluster_score += self.semantic_memory.get_cluster_context_score(word, context_words)
        if len(words) > 0:
            cluster_score = min(1.0, cluster_score / len(words))
        scores.append(cluster_score)
        
        return sum(scores) / len(scores)

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
        if self.total_generation_attempts == 0:
            return "No generation attempts yet"
        
        failure_rate = (self.generation_failures / self.total_generation_attempts) if self.total_generation_attempts > 0 else 0
        success_rate = 1 - failure_rate
        
        return f"Generation Success Rate: {success_rate:.1%} ({self.total_generation_attempts} attempts, {self.generation_failures} failures)"

class SelfTrainWorker(QThread):
    log_updated = Signal(str)
    def __init__(self, main_brain):
        super().__init__()
        self.main_brain = main_brain
        self.is_running = False
        self.training_quality_history = []
    
    def run(self):
        self.is_running = True
        student = None
        try:
            student = HybridBrain(DB_FILE, is_clone=True)
            teacher_last_said = self.main_brain.generate_response()
            training_round = 0
            
            while self.is_running:
                try:
                    time.sleep(SELF_TRAINING_INTERVAL)
                    training_round += 1
                    
                  
                    if training_round % SELF_TRAINING_MEMORY_CHECK_INTERVAL == 0:
                        if memory_manager.get_memory_usage_percent() > MEMORY_SAFETY_THRESHOLD:
                            self.log_updated.emit("Memory pressure detected, forcing garbage collection...")
                            memory_manager.force_garbage_collection()
                    
                    student_response = student.generate_response(teacher_last_said)
                    self.log_updated.emit(f"Round {training_round}:\nTeacher: {teacher_last_said}\nStudent: {student_response}\n")
                    
                    student_words = self.main_brain.clean_text(student_response)
                    base_quality = len(set(student_words)) / max(len(student_words), 1)
                    
                    length_bonus = 1.0 if 5 <= len(student_words) <= 20 else 0.5
                    coherence_bonus = 0.5 if any(word in teacher_last_said.lower() for word in student_words) else 0
                    
                    semantic_bonus = 0.0
                    if hasattr(student, 'semantic_memory'):
                        for word in student_words:
                            related_words = student.semantic_memory.get_related_words(word, 3)
                            if any(related in teacher_last_said.lower() for related in related_words):
                                semantic_bonus += 0.1
                    
                    quality_score = (base_quality + length_bonus + coherence_bonus + semantic_bonus) / 4
                    self.training_quality_history.append(quality_score)
                    
                    if quality_score > SELF_TRAINING_QUALITY_THRESHOLD:
                        self.main_brain.apply_feedback(student_response, is_positive=True)

                    student.learn_from_text(teacher_last_said)
            
                    if random.random() < 0.15 and quality_score > SELF_TRAINING_QUALITY_THRESHOLD:
                        for word, weight in student.attention.focus_weights.items():
                            if word in self.main_brain.attention.focus_weights: 
                                current_weight = self.main_brain.attention.focus_weights[word]
                                quality_factor = min(1.0, quality_score + 0.3)
                                new_weight = (current_weight + weight * quality_factor) / (1 + quality_factor)
                                self.main_brain.attention.focus_weights[word] = new_weight
                    
                    teacher_last_said = self.main_brain.generate_response(student_response)
                    
                    if training_round % 10 == 0:
                        recent_quality = sum(self.training_quality_history[-10:]) / min(10, len(self.training_quality_history))
                        cluster_count = len(self.main_brain.semantic_memory.clusters)
                        gen_stats = self.main_brain.get_generation_stats()
                        memory_usage = memory_manager.get_memory_usage_mb()
                        self.log_updated.emit(f"--- Generation Report (Round {training_round}) ---\n{gen_stats}\nQuality: {recent_quality:.2f}, Clusters: {cluster_count}\nMemory: {memory_usage:.1f}MB\n")
                        
                except Exception as e:
                    self.log_updated.emit(f"Error in training round {training_round}: {e}")
                  
                    continue
                    
        except Exception as e:
            self.log_updated.emit(f"Critical error in self-training: {e}")
        finally:
            if student:
                try:
                    student.con.close()
                except Exception as e:
                    self.log_updated.emit(f"Warning: Could not close student brain: {e}")
    
    def stop(self):
        self.is_running = False

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

class Critic:
    """Quality assessment system for response evaluation"""
    def __init__(self, brain):
        self.brain = brain
        self.quality_history = []
        self.confidence_threshold = 0.7
        
    def assess(self, user_input, response, context_hints=None):
        """Assess response quality and confidence"""
        try:
          
            length_score = min(len(response.split()) / 10, 1.0)
            coherence_score = self._assess_coherence(response)
            relevance_score = self._assess_relevance(user_input, response)
            creativity_score = self._assess_creativity(response)
            
          
            confidence = (coherence_score + relevance_score + creativity_score) / 3
            
          
            assessment = {
                'confidence': confidence,
                'length_score': length_score,
                'coherence_score': coherence_score,
                'relevance_score': relevance_score,
                'creativity_score': creativity_score,
                'timestamp': time.time()
            }
            
            self.quality_history.append(assessment)
            
          
            if len(self.quality_history) > 100:
                self.quality_history = self.quality_history[-100:]
                
            return assessment
            
        except Exception as e:
            print(f"Critic assessment error: {e}")
            return {'confidence': 0.5, 'error': str(e)}
    
    def _assess_coherence(self, response):
        """Assess response coherence using word associations"""
        try:
            words = response.lower().split()
            if len(words) < 2:
                return 0.3
                
            coherence_score = 0
            for i in range(len(words) - 1):
                word1, word2 = words[i], words[i + 1]
                if word1 in self.brain.word_associations and word2 in self.brain.word_associations[word1]:
                    coherence_score += self.brain.word_associations[word1][word2]
                    
            return min(coherence_score / (len(words) - 1), 1.0)
        except:
            return 0.5
    
    def _assess_relevance(self, user_input, response):
        """Assess response relevance to user input"""
        try:
            input_words = set(user_input.lower().split())
            response_words = set(response.lower().split())
            
            if not input_words or not response_words:
                return 0.3
                
            overlap = len(input_words.intersection(response_words))
            return min(overlap / len(input_words), 1.0)
        except:
            return 0.5
    
    def _assess_creativity(self, response):
        """Assess response creativity (avoid repetitive patterns)"""
        try:
            words = response.lower().split()
            if len(words) < 3:
                return 0.3
                
          
            unique_words = len(set(words))
            total_words = len(words)
            
            if total_words == 0:
                return 0.3
                
            diversity_ratio = unique_words / total_words
            return min(diversity_ratio * 1.5, 1.0)
        except:
            return 0.5

class ConfidenceGate:
    """Confidence-based response regeneration system"""
    def __init__(self, brain):
        self.brain = brain
        self.regeneration_threshold = 0.6
        self.max_regenerations = 2
        
    def should_regenerate(self, confidence, attempt_count=0):
        """Determine if response should be regenerated"""
        if attempt_count >= self.max_regenerations:
            return False
            
        return confidence < self.regeneration_threshold
    
    def regenerate_response(self, user_input, context, attempt_count=0):
        """Generate alternative response with different parameters"""
        try:
          
            creativity_boost = 0.1 + (attempt_count * 0.05)
            context_weight_reduction = 0.05 * attempt_count
            
          
            response = self.brain.generate_response(
                user_input, 
                context,
                creativity_factor=min(1.0, self.brain.settings_manager.get('creativity_factor', 0.7) + creativity_boost),
                context_weight=max(0.1, self.brain.settings_manager.get('context_weight', 0.3) - context_weight_reduction)
            )
            
            return response
        except Exception as e:
            print(f"Response regeneration error: {e}")
            return None

class AntiLoopFilter:
    """Prevent repetitive response patterns"""
    def __init__(self, brain):
        self.brain = brain
        self.recent_responses = []
        self.max_recent = 10
        self.similarity_threshold = 0.8
        
    def filter_response(self, response):
        """Filter out repetitive responses"""
        try:
            if not response or len(response.strip()) < 5:
                return response
                
          
            for recent in self.recent_responses:
                similarity = self._calculate_similarity(response, recent)
                if similarity > self.similarity_threshold:
                  
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
        """Calculate similarity between two responses"""
        try:
            words1 = set(response1.lower().split())
            words2 = set(response2.lower().split())
            
            if not words1 or not words2:
                return 0
                
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            return intersection / union if union > 0 else 0
        except:
            return 0
    
    def _modify_response(self, response):
        """Modify response to break repetition"""
        try:
            words = response.split()
            if len(words) < 3:
                return response
                
          
            variation_words = ["indeed", "certainly", "absolutely", "definitely", "surely"]
            variation_word = random.choice(variation_words)
            
          
            insert_pos = random.randint(1, len(words) - 1)
            words.insert(insert_pos, variation_word)
            
            return " ".join(words)
        except:
            return response

class MetaMemory:
    """Weakness tracking and improvement system"""
    def __init__(self, brain):
        self.brain = brain
        self.weakness_ledger = {}
        self.improvement_history = []
        
    def register_weakness(self, tag, context=None):
        """Register a weakness for tracking"""
        try:
            if tag not in self.weakness_ledger:
                self.weakness_ledger[tag] = {
                    'count': 0,
                    'first_seen': time.time(),
                    'last_seen': time.time(),
                    'contexts': []
                }
            
            self.weakness_ledger[tag]['count'] += 1
            self.weakness_ledger[tag]['last_seen'] = time.time()
            
            if context:
                self.weakness_ledger[tag]['contexts'].append(context)
              
                if len(self.weakness_ledger[tag]['contexts']) > 5:
                    self.weakness_ledger[tag]['contexts'] = self.weakness_ledger[tag]['contexts'][-5:]
                    
        except Exception as e:
            print(f"Meta-memory weakness registration error: {e}")
    
    def get_weakness_stats(self):
        """Get weakness statistics"""
        try:
            total_weaknesses = len(self.weakness_ledger)
            active_weaknesses = sum(1 for w in self.weakness_ledger.values() if w['count'] > 1)
            
            return {
                'total_weaknesses': total_weaknesses,
                'active_weaknesses': active_weaknesses,
                'most_common': max(self.weakness_ledger.items(), key=lambda x: x[1]['count']) if self.weakness_ledger else None
            }
        except:
            return {'total_weaknesses': 0, 'active_weaknesses': 0, 'most_common': None}

class Curiosity:
    """Idle-time curiosity and exploration system"""
    def __init__(self, brain):
        self.brain = brain
        self.last_curiosity_tick = 0
        self.curiosity_interval = 300
        self.exploration_topics = []
        
    def tick(self):
        """Run curiosity tick if enough time has passed"""
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
        """Explore new topics during idle time"""
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
        """Analyze conversation patterns for insights"""
        try:
          
            if hasattr(self.brain, 'conversation_memory') and self.brain.conversation_memory:
                recent_responses = list(self.brain.conversation_memory.values())[-10:]
                
              
                word_frequencies = {}
                for response in recent_responses:
                    if isinstance(response, dict) and 'response' in response:
                        words = response['response'].lower().split()
                        for word in words:
                            word_frequencies[word] = word_frequencies.get(word, 0) + 1
                
              
                if word_frequencies:
                    most_common = max(word_frequencies.items(), key=lambda x: x[1])
                    if most_common[1] > 2:
                        self.brain.semantic_memory.add_word_association(most_common[0], "common_pattern", 0.1)
                        
        except Exception as e:
            print(f"Pattern analysis error: {e}")
    
    def _update_learning_params(self):
        """Update learning parameters based on insights"""
        try:
          
            if hasattr(self.brain, 'adaptive_learning'):
                self.brain.adaptive_learning.adjust_learning_rate()
                
        except Exception as e:
            print(f"Learning parameter update error: {e}")

class EnvironmentFeedback:
    """Environment feedback integration system"""
    def __init__(self, brain):
        self.brain = brain
        self.feedback_handlers = {}
        self.feedback_history = []
        
    def register_handler(self, event_type, handler_func):
        """Register feedback handler for specific event type"""
        try:
            self.feedback_handlers[event_type] = handler_func
        except Exception as e:
            print(f"Feedback handler registration error: {e}")
    
    def process_feedback(self, event_type, data):
        """Process environment feedback"""
        try:
            if event_type in self.feedback_handlers:
                result = self.feedback_handlers[event_type](data)
                
              
                self.feedback_history.append({
                    'event_type': event_type,
                    'data': data,
                    'result': result,
                    'timestamp': time.time()
                })
                
              
                if len(self.feedback_history) > 50:
                    self.feedback_history = self.feedback_history[-50:]
                    
                return result
            else:
                print(f"No handler registered for event type: {event_type}")
                return None
                
        except Exception as e:
            print(f"Feedback processing error: {e}")
            return None
    
    def get_feedback_stats(self):
        """Get feedback statistics"""
        try:
            event_counts = {}
            for feedback in self.feedback_history:
                event_type = feedback['event_type']
                event_counts[event_type] = event_counts.get(event_type, 0) + 1
                
            return {
                'total_feedback': len(self.feedback_history),
                'event_counts': event_counts,
                'recent_feedback': self.feedback_history[-5:] if self.feedback_history else []
            }
        except:
            return {'total_feedback': 0, 'event_counts': {}, 'recent_feedback': []}

class Autotune:
    """Automatic hyperparameter tuning system"""
    def __init__(self, brain):
        self.brain = brain
        self.tuning_history = []
        self.parameter_ranges = {
            'creativity_factor': (0.1, 1.0),
            'context_weight': (0.1, 0.8),
            'semantic_weight': (0.1, 0.8),
            'pattern_weight': (0.1, 0.8)
        }
        
    def tune_from_quality(self, confidence_score):
        """Tune parameters based on quality feedback"""
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
                'timestamp': time.time()
            })
            
          
            if len(self.tuning_history) > 100:
                self.tuning_history = self.tuning_history[-100:]
                
        except Exception as e:
            print(f"Autotune error: {e}")
    
    def _adjust_parameters(self, adjustment_type):
        """Adjust parameters based on adjustment type"""
        try:
            current_settings = self.brain.settings_manager.settings
            
            if adjustment_type == 'increase':
              
                new_creativity = min(1.0, current_settings.get('creativity_factor', 0.7) + 0.05)
                new_context = max(0.1, current_settings.get('context_weight', 0.3) - 0.02)
                
                self.brain.settings_manager.set('creativity_factor', new_creativity)
                self.brain.settings_manager.set('context_weight', new_context)
                
            elif adjustment_type == 'slight':
              
                new_creativity = min(1.0, current_settings.get('creativity_factor', 0.7) + 0.02)
                new_semantic = min(0.8, current_settings.get('semantic_weight', 0.4) + 0.01)
                
                self.brain.settings_manager.set('creativity_factor', new_creativity)
                self.brain.settings_manager.set('semantic_weight', new_semantic)
                
          
            self.brain.settings_manager.save_settings()
            
        except Exception as e:
            print(f"Parameter adjustment error: {e}")
    
    def get_tuning_stats(self):
        """Get autotune statistics"""
        try:
            if not self.tuning_history:
                return {'total_adjustments': 0, 'recent_adjustments': []}
                
            recent_adjustments = self.tuning_history[-10:]
            adjustment_counts = {}
            
            for tuning in self.tuning_history:
                adj_type = tuning['adjustment']
                adjustment_counts[adj_type] = adjustment_counts.get(adj_type, 0) + 1
                
            return {
                'total_adjustments': len(self.tuning_history),
                'adjustment_counts': adjustment_counts,
                'recent_adjustments': recent_adjustments
            }
        except:
            return {'total_adjustments': 0, 'recent_adjustments': []}

# RESPONSE LEARNING & FACTUAL KNOWLEDGE SYSTEMS
# Clean integration without disrupting core SGM architecture

class ResponseLearningSystem:
    """Learns from human responses to improve conversation patterns"""
    def __init__(self, brain):
        self.brain = brain
        self.response_chains = {}
        self.response_patterns = {}
        self.sentiment_learning = {}
        self.conversation_flow = []
        
    def learn_from_response(self, user_input, mai_response, human_response):
        """Learn from human response to Mai's output"""
        try:
            if not user_input or not mai_response or not human_response:
                return
                
          
            chain_key = f"{user_input.lower().strip()} -> {mai_response.lower().strip()}"
            if chain_key not in self.response_chains:
                self.response_chains[chain_key] = []
            
            self.response_chains[chain_key].append({
                'human_response': human_response,
                'timestamp': time.time(),
                'context': self._extract_context(user_input, mai_response)
            })
            
          
            self._learn_response_patterns(mai_response, human_response)
            
          
            self._learn_sentiment_context(user_input, mai_response, human_response)
            
          
            self._update_conversation_flow(user_input, mai_response, human_response)
            
          
            if len(self.response_chains[chain_key]) > 10:
                self.response_chains[chain_key] = self.response_chains[chain_key][-10:]
                
        except Exception as e:
            print(f"Response learning error: {e}")
    
    def _extract_context(self, user_input, mai_response):
        """Extract contextual information"""
        try:
            return {
                'user_length': len(user_input.split()),
                'mai_length': len(mai_response.split()),
                'user_sentiment': self._analyze_sentiment(user_input),
                'mai_sentiment': self._analyze_sentiment(mai_response),
                'topics': self._extract_topics(user_input + " " + mai_response)
            }
        except:
            return {}
    
    def _learn_response_patterns(self, mai_response, human_response):
        """Learn patterns from human responses"""
        try:
            mai_words = mai_response.lower().split()
            human_words = human_response.lower().split()
            
          
            for mai_word in mai_words:
                if mai_word not in self.response_patterns:
                    self.response_patterns[mai_word] = {}
                
                for human_word in human_words:
                    if human_word not in self.response_patterns[mai_word]:
                        self.response_patterns[mai_word][human_word] = 0
                    self.response_patterns[mai_word][human_word] += 1
                    
        except Exception as e:
            print(f"Pattern learning error: {e}")
    
    def _learn_sentiment_context(self, user_input, mai_response, human_response):
        """Learn emotional context of responses"""
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
        """Update conversation flow patterns"""
        try:
            self.conversation_flow.append({
                'user': user_input,
                'mai': mai_response,
                'human': human_response,
                'timestamp': time.time()
            })
            
          
            if len(self.conversation_flow) > 50:
                self.conversation_flow = self.conversation_flow[-50:]
                
        except Exception as e:
            print(f"Conversation flow update error: {e}")
    
    def _analyze_sentiment(self, text):
        """Simple sentiment analysis"""
        try:
            positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'like', 'happy', 'pleased']
            negative_words = ['bad', 'terrible', 'awful', 'hate', 'dislike', 'angry', 'sad', 'disappointed', 'frustrated']
            
            words = text.lower().split()
            positive_count = sum(1 for word in words if word in positive_words)
            negative_count = sum(1 for word in words if word in negative_words)
            
            if positive_count > negative_count:
                return 'positive'
            elif negative_count > positive_count:
                return 'negative'
            else:
                return 'neutral'
        except:
            return 'neutral'
    
    def _extract_topics(self, text):
        """Extract topics from text"""
        try:
          
            words = text.lower().split()
            word_freq = {}
            for word in words:
                if len(word) > 3:
                    word_freq[word] = word_freq.get(word, 0) + 1
            
          
            return sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:3]
        except:
            return []
    
    def get_response_suggestions(self, user_input, mai_response):
        """Get response suggestions based on learned patterns"""
        try:
            suggestions = []
            
          
            for chain_key, responses in self.response_chains.items():
                if user_input.lower().strip() in chain_key:
                    for response_data in responses[-3:]:
                        suggestions.append({
                            'response': response_data['human_response'],
                            'confidence': 0.7,
                            'source': 'response_chain'
                        })
            
          
            mai_words = mai_response.lower().split()
            for mai_word in mai_words:
                if mai_word in self.response_patterns:
                    patterns = self.response_patterns[mai_word]
                    for human_word, count in sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:3]:
                        suggestions.append({
                            'response': f"Response involving '{human_word}'",
                            'confidence': min(count / 10, 1.0),
                            'source': 'pattern'
                        })
            
            return suggestions[:5]
            
        except Exception as e:
            print(f"Response suggestion error: {e}")
            return []
    
    def get_learning_stats(self):
        """Get learning statistics"""
        try:
            return {
                'response_chains': len(self.response_chains),
                'response_patterns': len(self.response_patterns),
                'sentiment_contexts': len(self.sentiment_learning),
                'conversation_flow_length': len(self.conversation_flow),
                'total_learned_responses': sum(len(responses) for responses in self.response_chains.values())
            }
        except:
            return {'response_chains': 0, 'response_patterns': 0, 'sentiment_contexts': 0, 'conversation_flow_length': 0, 'total_learned_responses': 0}

class TruthFactTable:
    """Stores and manages factual information about topics"""
    def __init__(self, brain):
        self.brain = brain
        self.fact_table = {}
        self.fact_sources = {}
        self.fact_confidence = {}
        self.topic_frequency = {}
        
    def add_fact(self, topic, fact, source="conversation", confidence=0.5):
        """Add a fact about a topic"""
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
        """Get facts about a topic"""
        try:
            topic = topic.lower().strip()
            if topic not in self.fact_table:
                return []
                
          
            facts = []
            for fact in self.fact_table[topic]:
                confidence = self.fact_confidence.get(fact, 0.5)
                facts.append({
                    'fact': fact,
                    'confidence': confidence,
                    'source': self.fact_sources.get(fact, 'unknown')
                })
            
          
            facts.sort(key=lambda x: x['confidence'], reverse=True)
            return facts[:limit]
            
        except Exception as e:
            print(f"Fact retrieval error: {e}")
            return []
    
    def update_fact_confidence(self, fact, feedback_score):
        """Update fact confidence based on feedback"""
        try:
            if fact in self.fact_confidence:
              
                current_confidence = self.fact_confidence[fact]
                new_confidence = (current_confidence + feedback_score) / 2
                self.fact_confidence[fact] = max(0.1, min(1.0, new_confidence))
                
        except Exception as e:
            print(f"Confidence update error: {e}")
    
    def get_topic_frequency(self, topic):
        """Get how often a topic appears in conversations"""
        try:
            return self.topic_frequency.get(topic.lower().strip(), 0)
        except:
            return 0
    
    def get_fact_stats(self):
        """Get fact table statistics"""
        try:
            total_facts = sum(len(facts) for facts in self.fact_table.values())
            total_topics = len(self.fact_table)
            avg_confidence = sum(self.fact_confidence.values()) / len(self.fact_confidence) if self.fact_confidence else 0
            
            return {
                'total_facts': total_facts,
                'total_topics': total_topics,
                'average_confidence': avg_confidence,
                'most_frequent_topic': max(self.topic_frequency.items(), key=lambda x: x[1]) if self.topic_frequency else None
            }
        except:
            return {'total_facts': 0, 'total_topics': 0, 'average_confidence': 0, 'most_frequent_topic': None}

class TopicDetectionSystem:
    """Detects when topics need factual context"""
    def __init__(self, brain):
        self.brain = brain
        self.topic_threshold = 0.1
        self.rare_topic_threshold = 0.05
        self.topic_contexts = {}
        
    def detect_topics_needing_facts(self, text):
        """Detect topics that might need factual context"""
        try:
            words = text.lower().split()
            word_freq = {}
            
          
            for word in words:
                if len(word) > 3:
                    word_freq[word] = word_freq.get(word, 0) + 1
            
          
            total_words = len(words)
            topics_needing_facts = []
            
            for word, count in word_freq.items():
                frequency = count / total_words
                
              
                if frequency < self.rare_topic_threshold:
                    topics_needing_facts.append({
                        'topic': word,
                        'frequency': frequency,
                        'reason': 'rare_topic',
                        'confidence': 0.8
                    })
                
              
                elif frequency < self.topic_threshold:
                    topics_needing_facts.append({
                        'topic': word,
                        'frequency': frequency,
                        'reason': 'medium_topic',
                        'confidence': 0.6
                    })
            
            return topics_needing_facts
            
        except Exception as e:
            print(f"Topic detection error: {e}")
            return []
    
    def should_provide_facts(self, topic, context):
        """Determine if facts should be provided for a topic"""
        try:
          
            if topic in self.topic_contexts:
                context_data = self.topic_contexts[topic]
                if context_data.get('frequency', 1) < self.rare_topic_threshold:
                    return True
            
          
            if 'user_input' in context and 'mai_response' in context:
                user_words = set(context['user_input'].lower().split())
                mai_words = set(context['mai_response'].lower().split())
                
                if topic in user_words and topic not in mai_words:
                    return True
            
            return False
            
        except Exception as e:
            print(f"Fact provision decision error: {e}")
            return False
    
    def update_topic_context(self, topic, context_data):
        """Update context information for a topic"""
        try:
            self.topic_contexts[topic] = {
                'frequency': context_data.get('frequency', 0),
                'last_seen': time.time(),
                'context_type': context_data.get('context_type', 'general')
            }
            
        except Exception as e:
            print(f"Topic context update error: {e}")
    
    def get_topic_stats(self):
        """Get topic detection statistics"""
        try:
            return {
                'total_topics_tracked': len(self.topic_contexts),
                'rare_topics': sum(1 for ctx in self.topic_contexts.values() if ctx.get('frequency', 0) < self.rare_topic_threshold),
                'medium_topics': sum(1 for ctx in self.topic_contexts.values() if self.rare_topic_threshold <= ctx.get('frequency', 0) < self.topic_threshold)
            }
        except:
            return {'total_topics_tracked': 0, 'rare_topics': 0, 'medium_topics': 0}

class MaiApp(QMainWindow):
    def __init__(self):
        super().__init__()
        print("Initializing Mai's Memory Optimized Edition...")
        self.brain = HybridBrain(DB_FILE)
        self.self_train_worker = SelfTrainWorker(self.brain)
        self.self_train_worker.log_updated.connect(self.update_self_train_log)
        
      
        self.performance_monitor = PerformanceMonitor()
        self.correction_context = ""
        self.correction_mai_response = ""
        self.file_training_worker = None
        
        print("Setting up UI...")
      
        self.setWindowTitle("Mai Generative v4.5 - Consolidated Build")
        self.setGeometry(100, 100, 950, 800)
        self.setStyleSheet("""
            QMainWindow { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #F0F0F0, stop:1 #E0E0E0); }
            QWidget { background-color: #F0F0F0; font-family: "Tahoma", Arial, sans-serif; color: #333333; }
            
            QTabWidget::pane { 
                border: 2px solid #999999; 
                background-color: white;
                border-radius: 8px;
            }
            QTabBar::tab { 
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #E0E0E0, stop:1 #C0C0C0);
                color: #333333; 
                padding: 12px 18px; 
                font-size: 13px; 
                font-weight: bold;
                border: 1px solid #999999;
                margin-right: 2px;
                border-radius: 8px 8px 0px 0px;
            }
            QTabBar::tab:hover { 
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #F0F0F0, stop:1 #D0D0D0);
            }
            QTabBar::tab:selected { 
                background: white;
                border-bottom: 2px solid white;
                font-weight: bold;
            }
            
            QTextBrowser { 
                background-color: white; 
                color: #000000; 
                border: 2px inset #C0C0C0;
                font-family: "Tahoma", Arial, sans-serif;
                font-size: 12px;
                selection-background-color: #4A90E2;
                line-height: 1.4;
            }
            
            QLineEdit { 
                background-color: white; 
                color: #000000; 
                border: 2px inset #C0C0C0;
                padding: 6px;
                font-family: "Tahoma", Arial, sans-serif;
                font-size: 12px;
                border-radius: 3px;
            }
            QLineEdit:focus {
                border: 2px solid #4A90E2;
                background-color: #FAFAFA;
            }
            
            QPushButton { 
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #FAFAFA, stop:1 #E0E0E0);
                color: #333333; 
                border: 2px outset #C0C0C0;
                padding: 8px 14px;
                font-family: "Tahoma", Arial, sans-serif;
                font-weight: bold;
                font-size: 12px;
                border-radius: 5px;
            }
            QPushButton:hover { 
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #FFFFFF, stop:1 #F0F0F0);
            }
            QPushButton:pressed {
                border: 2px inset #C0C0C0;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #E0E0E0, stop:1 #D0D0D0);
            }
            
            QLabel { color: #333333; font-weight: bold; }
            QProgressBar { 
                text-align: center; 
                border: 2px inset #C0C0C0;
                background: white;
                color: #333333;
                font-weight: bold;
                border-radius: 3px;
                height: 20px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #87CEEB, stop:1 #4682B4);
                border-radius: 2px;
            }
            
            QGroupBox { 
                font-weight: bold; 
                color: #333333;
                border: 2px solid #999999;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 12px;
                background-color: rgba(255, 255, 255, 180);
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 10px 0 10px;
                background-color: #F0F0F0;
                border-radius: 4px;
                font-size: 13px;
            }
            
            #BrainView, #SelfTrainLog { 
                background-color: #1E1E1E; 
                color: #00FF00; 
                font-family: "Courier New", Courier, monospace; 
                border: 2px inset #C0C0C0;
                font-size: 11px;
                line-height: 1.3;
            }
        """)
        
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        self.create_conversation_tab()
        self.create_teaching_tab()
        self.create_brain_tab()
        self.create_settings_tab()
        
        print("Mai Smart GPU Acceleration Build v6.1 is ready!")

    def create_conversation_tab(self):
        tab = QWidget()
        main_layout = QVBoxLayout(tab)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
      
        top_section = QWidget()
        top_section.setFixedHeight(45)
        top_section.setStyleSheet("""
            QWidget {
                background: #2C3E50;
                border-radius: 8px;
                padding: 3px;
            }
        """)
        top_layout = QHBoxLayout(top_section)
        top_layout.setContentsMargins(15, 5, 15, 5)
        top_layout.setSpacing(15)
        
        self.status_label = QLabel("Mai AI v6.3 - Enhanced Memory System")
        self.status_label.setStyleSheet("color: white; font-weight: bold; font-size: 13px;")
        
        self.quality_label = QLabel("Quality: Initializing...")
        self.quality_label.setStyleSheet("color: white; font-size: 11px; background: #34495E; padding: 3px 8px; border-radius: 4px;")
        
        self.cluster_label = QLabel("Clusters: Loading...")
        self.cluster_label.setStyleSheet("color: white; font-size: 11px; background: #34495E; padding: 3px 8px; border-radius: 4px;")
        
        self.gen_stats_label = QLabel("Stats: Loading...")
        self.gen_stats_label.setStyleSheet("color: white; font-size: 11px; background: #34495E; padding: 3px 8px; border-radius: 4px;")
        
        self.memory_label = QLabel(f"RAM: {memory_manager.get_memory_usage_mb():.1f}MB")
        self.memory_label.setStyleSheet("color: white; font-size: 11px; background: #34495E; padding: 3px 8px; border-radius: 4px;")
        
        top_layout.addWidget(self.status_label)
        top_layout.addStretch()
        top_layout.addWidget(self.quality_label)
        top_layout.addWidget(self.cluster_label)
        top_layout.addWidget(self.gen_stats_label)
        top_layout.addWidget(self.memory_label)
        
        main_layout.addWidget(top_section)
        
      
        self.chat_window = QTextBrowser()
        self.chat_window.setReadOnly(True)
        self.chat_window.anchorClicked.connect(self.handle_feedback)
        self.chat_window.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.chat_window.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.chat_window.setStyleSheet("""
            QTextBrowser {
                background-color: white;
                border: 1px solid #BDC3C7;
                border-radius: 8px;
                padding: 15px;
                font-size: 13px;
                line-height: 1.5;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                color: #2C3E50;
            }
            QTextBrowser:focus {
                border-color: #3498DB;
            }
            QScrollBar:vertical {
                background: #ECF0F1;
                width: 12px;
                border-radius: 6px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #BDC3C7;
                border-radius: 6px;
                min-height: 20px;
                margin: 1px;
            }
            QScrollBar::handle:vertical:hover {
                background: #95A5A6;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        main_layout.addWidget(self.chat_window, 1)
        
      
        input_container = QWidget()
        input_container.setMinimumHeight(80)
        input_container.setMaximumHeight(120)
        input_container.setStyleSheet("""
            QWidget {
                background: #ECF0F1;
                border: 1px solid #BDC3C7;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        input_layout = QVBoxLayout(input_container)
        input_layout.setContentsMargins(10, 10, 10, 10)
        input_layout.setSpacing(8)
        
      
        input_title = QLabel("Chat Input")
        input_title.setStyleSheet("""
            color: #2C3E50;
            font-size: 14px;
            font-weight: bold;
            text-align: center;
        """)
        input_layout.addWidget(input_title)
        
      
        self.message_input = QLineEdit()
        self.message_input.setPlaceholderText("Type your message here and press Enter...")
        self.message_input.setMinimumHeight(35)
        self.message_input.setMaximumHeight(40)
        self.message_input.setStyleSheet("""
            QLineEdit {
                font-size: 13px;
                padding: 10px 15px;
                border: 1px solid #BDC3C7;
                border-radius: 6px;
                background-color: white;
                color: #2C3E50;
            }
            QLineEdit:focus {
                border: 2px solid #3498DB;
                background-color: white;
            }
            QLineEdit::placeholder {
                color: #95A5A6;
                font-style: italic;
            }
        """)
        self.message_input.returnPressed.connect(self.send_message)
        
      
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)
        
        send_button = QPushButton("Send")
        send_button.setMinimumHeight(35)
        send_button.setMaximumHeight(40)
        send_button.setStyleSheet("""
            QPushButton {
                font-size: 12px;
                font-weight: bold;
                background: #3498DB;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 15px;
            }
            QPushButton:hover {
                background: #2980B9;
            }
            QPushButton:pressed {
                background: #21618C;
            }
        """)
        send_button.clicked.connect(self.send_message)
        
        clear_button = QPushButton("Clear")
        clear_button.setMinimumHeight(35)
        clear_button.setMaximumHeight(40)
        clear_button.setStyleSheet("""
            QPushButton {
                font-size: 12px;
                font-weight: bold;
                background: #95A5A6;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 15px;
            }
            QPushButton:hover {
                background: #7F8C8D;
            }
        """)
        clear_button.clicked.connect(self.clear_chat)
        
      
        input_row = QHBoxLayout()
        input_row.setSpacing(8)
        input_row.setContentsMargins(0, 0, 0, 0)
        
      
        self.message_input.setMinimumHeight(35)
        self.message_input.setStyleSheet("""
            QLineEdit {
                font-size: 13px;
                padding: 8px 12px;
                border: 2px solid #3498DB;
                border-radius: 6px;
                background-color: white;
                color: #2C3E50;
            }
            QLineEdit:focus {
                border: 2px solid #2980B9;
                background-color: white;
            }
            QLineEdit::placeholder {
                color: #95A5A6;
                font-style: italic;
            }
        """)
        
        input_row.addWidget(self.message_input, stretch=4)
        input_row.addWidget(send_button, stretch=1)
        input_row.addWidget(clear_button, stretch=1)
        
        input_layout.addLayout(input_row)
        main_layout.addWidget(input_container, 0)
        
      
      
        
      
        welcome_message = '''
        <div style="text-align: center; padding: 25px; background: #ECF0F1; border: 1px solid #BDC3C7; border-radius: 10px; margin: 15px; color: #2C3E50;">
            <h2 style="margin-bottom: 15px; font-size: 24px; color: #2C3E50;">Welcome to Mai AI v6.3 - SGM Architecture</h2>
            <p style="font-size: 14px; margin-bottom: 15px; line-height: 1.5;">I'm a <b>Statistical Generative Model (SGM)</b> - a tensor-free alternative to traditional AI architectures.</p>
            
            <div style="background: white; padding: 15px; border-radius: 8px; margin: 15px 0; border: 1px solid #BDC3C7;">
                <h3 style="margin-bottom: 10px; font-size: 16px; color: #2C3E50;">SGM Architecture Features:</h3>
                <ul style="text-align: left; font-size: 13px; line-height: 1.6; color: #34495E;">
                    <li><b>Tensor-Free Design:</b> No complex tensor operations, optimized for CPU processing</li>
                    <li><b>Statistical Learning:</b> Generates responses from learned statistical patterns</li>
                    <li><b>Memory Hierarchy:</b> Short-term, long-term, and semantic memory systems</li>
                    <li><b>Contextual Intelligence:</b> Cross-references conversations and semantic patterns</li>
                    <li><b>Adaptive Learning:</b> Self-improving through conversation feedback</li>
                </ul>
            </div>
            
            <div style="background: #F8F9FA; padding: 12px; border-radius: 6px; margin-top: 15px; border: 1px solid #E9ECEF;">
                <p style="font-size: 13px; margin: 0; color: #6C757D;"><b>Start:</b> Ask me questions, share stories, or have a casual conversation. I learn and adapt from every interaction!</p>
            </div>
        </div>
        '''
        self.chat_window.append(welcome_message)
        
        self.tabs.addTab(tab, "Chat")

    def create_teaching_tab(self):
        tab = QWidget()
        main_layout = QVBoxLayout(tab)
        
        file_group = QGroupBox("Manual Teaching (Memory Optimized for 2M+ Word Files)")
        file_layout = QVBoxLayout(file_group)
        file_layout.addWidget(QLabel("<p>Select .txt files to teach Mai statistical patterns.<br/><b>MEMORY OPTIMIZED:</b> Handles 2M+ word files with RAM monitoring and chunked processing!</p>"))
        select_layout = QHBoxLayout()
        self.file_input_label = QLabel("No files selected.")
        file_button = QPushButton("Select Files...")
        file_button.clicked.connect(self.select_files)
        select_layout.addWidget(file_button); select_layout.addWidget(self.file_input_label); select_layout.addStretch()
        file_layout.addLayout(select_layout)
        teach_button = QPushButton("Teach Patterns (Memory Optimized)")
        teach_button.clicked.connect(self.teach_from_files)
        file_layout.addWidget(teach_button)
        self.teach_progress = QProgressBar(); file_layout.addWidget(self.teach_progress)
        
        self.training_status = QLabel("Ready for memory optimized training")
        file_layout.addWidget(self.training_status)
        
        self.memory_status = QLabel(f"Memory: {memory_manager.get_memory_usage_mb():.1f}MB")
        file_layout.addWidget(self.memory_status)
        
        convo_group = QGroupBox("Conversational Correction (Generation Training)")
        convo_layout = QVBoxLayout(convo_group)
        self.correction_status = QLabel("Click 'Get New Prompt' to start generative training.")
        convo_layout.addWidget(self.correction_status)
        self.btn_get_prompt = QPushButton("Get New Prompt"); self.btn_get_prompt.clicked.connect(self.get_new_correction_prompt)
        convo_layout.addWidget(self.btn_get_prompt)
        self.correction_user_input = QLineEdit(); self.correction_user_input.setPlaceholderText("Your reply to Mai...")
        self.btn_get_mai_response = QPushButton("Generate Mai's Response"); self.btn_get_mai_response.clicked.connect(self.get_correction_response)
        user_reply_layout = QHBoxLayout(); user_reply_layout.addWidget(self.correction_user_input); user_reply_layout.addWidget(self.btn_get_mai_response)
        convo_layout.addLayout(user_reply_layout)
        self.correction_mai_output = QLabel("<i>Mai's generated response will appear here.</i>"); self.correction_mai_output.setWordWrap(True)
        convo_layout.addWidget(self.correction_mai_output)
        self.correction_feedback_input = QLineEdit(); self.correction_feedback_input.setPlaceholderText("Type a better response here...")
        feedback_buttons_layout = QHBoxLayout()
        self.btn_encourage = QPushButton("Encourage Pattern"); self.btn_encourage.clicked.connect(self.encourage_response)
        self.btn_discourage = QPushButton("Discourage Pattern"); self.btn_discourage.clicked.connect(self.discourage_response)
        self.btn_teach_correction = QPushButton("Teach Better Pattern"); self.btn_teach_correction.clicked.connect(self.teach_correction)
        feedback_buttons_layout.addWidget(self.btn_encourage); feedback_buttons_layout.addWidget(self.btn_discourage)
        convo_layout.addWidget(self.correction_feedback_input)
        convo_layout.addLayout(feedback_buttons_layout)
        convo_layout.addWidget(self.btn_teach_correction)
        self.reset_correction_ui()

        self_train_group = QGroupBox("Autonomous Self-Training (Memory Optimized)")
        self_train_layout = QVBoxLayout(self_train_group)
        self_train_layout.addWidget(QLabel("<p>Mai will talk to a clone using statistical generation with memory monitoring.</p>"))
        self.self_train_button = QPushButton("Begin Self-Training"); self.self_train_button.setCheckable(True)
        self.self_train_button.clicked.connect(self.toggle_self_train)
        self_train_layout.addWidget(self.self_train_button)
        self.self_train_log = QTextEdit(); self.self_train_log.setObjectName("SelfTrainLog"); self.self_train_log.setReadOnly(True)
        self.self_train_log.setVisible(False)
        self_train_layout.addWidget(QLabel("<h3>Memory Optimized Self-Training Log:</h3>")); self_train_layout.addWidget(self.self_train_log)
        
        main_layout.addWidget(file_group)
        main_layout.addWidget(convo_group)
        main_layout.addWidget(self_train_group)
        main_layout.addStretch()
        self.tabs.addTab(tab, "Teaching")

    def create_brain_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
      
        layout.addWidget(QLabel("<h2>Mai's Smart GPU Acceleration Brain v6.1 (Advanced Build)</h2>"))
        
        controls_layout = QHBoxLayout()
        btn_view_brain = QPushButton("View Statistical Patterns"); btn_view_brain.clicked.connect(self.view_brain_data)
        btn_view_attn = QPushButton("View Attention Weights"); btn_view_attn.clicked.connect(self.view_attention_weights)
        btn_view_context = QPushButton("View Context Performance"); btn_view_context.clicked.connect(self.view_context_performance)
        btn_view_clusters = QPushButton("View Semantic Clusters"); btn_view_clusters.clicked.connect(self.view_semantic_clusters)
        btn_view_gen_stats = QPushButton("View Generation Stats"); btn_view_gen_stats.clicked.connect(self.view_generation_stats)
        controls_layout.addWidget(btn_view_brain); controls_layout.addWidget(btn_view_attn); controls_layout.addWidget(btn_view_context); controls_layout.addWidget(btn_view_clusters); controls_layout.addWidget(btn_view_gen_stats)
        
        controls_layout2 = QHBoxLayout()
        btn_save_knowledge = QPushButton("Save Patterns"); btn_save_knowledge.clicked.connect(self.save_to_initial_knowledge)
        btn_export = QPushButton("Export Brain"); btn_export.clicked.connect(self.export_brain)
        btn_import = QPushButton("Import Brain"); btn_import.clicked.connect(self.import_brain)
        btn_memory_stats = QPushButton("Memory Statistics"); btn_memory_stats.clicked.connect(self.view_memory_statistics)
        btn_launch_viewer = QPushButton("Launch 3D Brain Viewer")
        btn_launch_viewer.clicked.connect(self.launch_brain_viewer)
        controls_layout2.addWidget(btn_save_knowledge); controls_layout2.addWidget(btn_export); controls_layout2.addWidget(btn_import); controls_layout2.addWidget(btn_memory_stats)
        controls_layout2.addWidget(btn_launch_viewer)
        
        self.brain_view = QTextEdit(); self.brain_view.setObjectName("BrainView"); self.brain_view.setReadOnly(True)
        self.brain_view.setText("Memory optimized brain data will appear here. No preset responses - all statistical patterns.")
        
        layout.addLayout(controls_layout)
        layout.addLayout(controls_layout2)
        layout.addWidget(self.brain_view)
        self.tabs.addTab(tab, "Brain")

    def create_settings_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        layout.addWidget(QLabel("<h2>Mai AI Settings & GPU Acceleration</h2>"))
        
      
        self.gpu_group = QGroupBox("GPU Acceleration")
        gpu_layout = QVBoxLayout(self.gpu_group)
        
      
        if gpu_detector.recommended_gpu:
            gpu_info = gpu_detector.recommended_gpu
            self.gpu_status_label = QLabel(f"GPU Detected: {gpu_info['name']} ({gpu_info['vendor']})")
            self.gpu_status_label.setStyleSheet("color: green; font-weight: bold;")
            gpu_layout.addWidget(self.gpu_status_label)
            
          
            score_label = QLabel(f"AI Acceleration Score: {gpu_info['ai_acceleration_score']}/100")
            if gpu_info['ai_acceleration_score'] >= 80:
                score_label.setStyleSheet("color: green; font-weight: bold;")
            elif gpu_info['ai_acceleration_score'] >= 60:
                score_label.setStyleSheet("color: orange; font-weight: bold;")
            else:
                score_label.setStyleSheet("color: red; font-weight: bold;")
            gpu_layout.addWidget(score_label)
            
          
            memory_label = QLabel(f"GPU Memory: {gpu_info['memory_gb']}GB")
            gpu_layout.addWidget(memory_label)
            
          
            if gpu_info['supports_cuda'] or gpu_info['supports_opencl']:
                self.gpu_enable_checkbox = QCheckBox("Enable GPU Acceleration")
                self.gpu_enable_checkbox.setChecked(settings_manager.get('gpu_acceleration_enabled', False))
                self.gpu_enable_checkbox.stateChanged.connect(self.toggle_gpu_acceleration)
                gpu_layout.addWidget(self.gpu_enable_checkbox)
                
              
                if gpu_info['supports_cuda'] and gpu_info['supports_opencl']:
                    accel_type_layout = QHBoxLayout()
                    accel_type_layout.addWidget(QLabel("Acceleration Type:"))
                    self.gpu_accel_type_combo = QComboBox()
                    self.gpu_accel_type_combo.addItems(['auto', 'cuda', 'opencl'])
                    current_type = settings_manager.get('gpu_acceleration_type', 'auto')
                    self.gpu_accel_type_combo.setCurrentText(current_type)
                    self.gpu_accel_type_combo.currentTextChanged.connect(self.update_gpu_acceleration_type)
                    accel_type_layout.addWidget(self.gpu_accel_type_combo)
                    gpu_layout.addLayout(accel_type_layout)
                
              
                gpu_mem_layout = QHBoxLayout()
                gpu_mem_layout.addWidget(QLabel("GPU Memory Limit:"))
                self.gpu_mem_slider = QSlider(Qt.Horizontal)
                self.gpu_mem_slider.setRange(50, 95)
                self.gpu_mem_slider.setValue(int(settings_manager.get('gpu_memory_limit', 0.8) * 100))
                self.gpu_mem_slider.valueChanged.connect(self.update_gpu_memory_limit)
                gpu_mem_layout.addWidget(self.gpu_mem_slider)
                self.gpu_mem_label = QLabel(f"{self.gpu_mem_slider.value()}%")
                gpu_mem_layout.addWidget(self.gpu_mem_label)
                gpu_layout.addLayout(gpu_mem_layout)
                
              
                gpu_batch_layout = QHBoxLayout()
                gpu_batch_layout.addWidget(QLabel("GPU Batch Size:"))
                self.gpu_batch_combo = QComboBox()
                self.gpu_batch_combo.addItems(['auto', 'small', 'medium', 'large'])
                current_batch = settings_manager.get('gpu_batch_size', 'auto')
                self.gpu_batch_combo.setCurrentText(current_batch)
                self.gpu_batch_combo.currentTextChanged.connect(self.update_gpu_batch_size)
                gpu_batch_layout.addWidget(self.gpu_batch_combo)
                gpu_layout.addLayout(gpu_batch_layout)
                
              
                current_status = "Available" if gpu_accelerator.gpu_available else "Not Available"
                current_type = gpu_accelerator.gpu_type if gpu_accelerator.gpu_available else "None"
                status_text = f"Current Status: {current_status}\n"
                status_text += f"Active Type: {current_type}\n"
                status_text += f"Acceleration: {'Enabled' if settings_manager.get('gpu_acceleration_enabled') else 'Disabled'}"
                
                gpu_info_text = QTextEdit()
                gpu_info_text.setMaximumHeight(80)
                gpu_info_text.setReadOnly(True)
                gpu_info_text.setText(status_text)
                gpu_layout.addWidget(gpu_info_text)
                
            else:
                no_support_label = QLabel("GPU detected but doesn't support AI acceleration")
                no_support_label.setStyleSheet("color: red; font-weight: bold;")
                gpu_layout.addWidget(no_support_label)
        else:
          
            no_gpu_label = QLabel("No suitable GPU detected for AI acceleration")
            no_gpu_label.setStyleSheet("color: red; font-weight: bold;")
            gpu_layout.addWidget(no_gpu_label)
            
          
            if gpu_detector.detected_gpus:
                detection_info = QTextEdit()
                detection_info.setMaximumHeight(100)
                detection_info.setReadOnly(True)
                info_text = "Detected GPUs:\n"
                for gpu in gpu_detector.detected_gpus:
                    info_text += f"- {gpu['name']} ({gpu['vendor']}, {gpu['memory_gb']}GB, Score: {gpu['ai_acceleration_score']}/100)\n"
                detection_info.setText(info_text)
                gpu_layout.addWidget(detection_info)
            
          
            self.gpu_group.setVisible(False)
        
      
        perf_group = QGroupBox("Performance Settings")
        perf_layout = QVBoxLayout(perf_group)
        
      
        workers_layout = QHBoxLayout()
        workers_layout.addWidget(QLabel("Parallel Workers:"))
        self.workers_combo = QComboBox()
        self.workers_combo.addItems(['auto', '1', '2', '4', '8', '16'])
        current_workers = settings_manager.get('parallel_workers', 'auto')
        self.workers_combo.setCurrentText(str(current_workers))
        self.workers_combo.currentTextChanged.connect(self.update_parallel_workers)
        workers_layout.addWidget(self.workers_combo)
        perf_layout.addLayout(workers_layout)
        
      
        chunk_layout = QHBoxLayout()
        chunk_layout.addWidget(QLabel("Chunk Size:"))
        self.chunk_combo = QComboBox()
        self.chunk_combo.addItems(['auto', '10000', '50000', '100000', '500000'])
        current_chunk = settings_manager.get('chunk_size', 'auto')
        self.chunk_combo.setCurrentText(str(current_chunk))
        self.chunk_combo.currentTextChanged.connect(self.update_chunk_size)
        chunk_layout.addWidget(self.chunk_combo)
        perf_layout.addLayout(chunk_layout)
        
      
        mem_layout = QHBoxLayout()
        mem_layout.addWidget(QLabel("Memory Threshold:"))
        self.mem_slider = QSlider(Qt.Horizontal)
        self.mem_slider.setRange(50, 95)
        self.mem_slider.setValue(int(settings_manager.get('memory_threshold', 0.85) * 100))
        self.mem_slider.valueChanged.connect(self.update_memory_threshold)
        mem_layout.addWidget(self.mem_slider)
        self.mem_label = QLabel(f"{self.mem_slider.value()}%")
        mem_layout.addWidget(self.mem_label)
        perf_layout.addLayout(mem_layout)
        
      
        gen_group = QGroupBox("Generation Settings")
        gen_layout = QVBoxLayout(gen_group)
        
      
        resp_layout = QHBoxLayout()
        resp_layout.addWidget(QLabel("Max Response Length:"))
        self.resp_length_spin = QSpinBox()
        self.resp_length_spin.setRange(5, 100)
        self.resp_length_spin.setValue(settings_manager.get('max_response_length', 25))
        self.resp_length_spin.valueChanged.connect(self.update_max_response_length)
        resp_layout.addWidget(self.resp_length_spin)
        gen_layout.addLayout(resp_layout)
        
      
        qual_layout = QHBoxLayout()
        qual_layout.addWidget(QLabel("Quality Threshold:"))
        self.qual_slider = QSlider(Qt.Horizontal)
        self.qual_slider.setRange(10, 90)
        self.qual_slider.setValue(int(settings_manager.get('quality_threshold', 0.5) * 100))
        self.qual_slider.valueChanged.connect(self.update_quality_threshold)
        qual_layout.addWidget(self.qual_slider)
        self.qual_label = QLabel(f"{self.qual_slider.value()}%")
        qual_layout.addWidget(self.qual_label)
        gen_layout.addLayout(qual_layout)
        
      
        sys_group = QGroupBox("System Settings")
        sys_layout = QVBoxLayout(sys_group)
        
        self.mem_monitor_checkbox = QCheckBox("Enable Memory Monitoring")
        self.mem_monitor_checkbox.setChecked(settings_manager.get('enable_memory_monitoring', True))
        self.mem_monitor_checkbox.stateChanged.connect(self.toggle_memory_monitoring)
        sys_layout.addWidget(self.mem_monitor_checkbox)
        
        self.gc_checkbox = QCheckBox("Enable Garbage Collection")
        self.gc_checkbox.setChecked(settings_manager.get('enable_garbage_collection', True))
        self.gc_checkbox.stateChanged.connect(self.toggle_garbage_collection)
        sys_layout.addWidget(self.gc_checkbox)
        
      
        intel_group = QGroupBox("Enhanced Intelligence")
        intel_layout = QVBoxLayout(intel_group)
        
        self.enhanced_intel_checkbox = QCheckBox("Enable Enhanced Intelligence")
        self.enhanced_intel_checkbox.setChecked(settings_manager.get('enhanced_intelligence', True))
        self.enhanced_intel_checkbox.stateChanged.connect(self.toggle_enhanced_intelligence)
        intel_layout.addWidget(self.enhanced_intel_checkbox)
        
        self.creativity_slider = QSlider(Qt.Horizontal)
        self.creativity_slider.setRange(0, 100)
        self.creativity_slider.setValue(int(settings_manager.get('creativity_factor', 0.5) * 100))
        self.creativity_slider.valueChanged.connect(self.update_creativity_factor)
        creativity_layout = QHBoxLayout()
        creativity_layout.addWidget(QLabel("Creativity Factor:"))
        creativity_layout.addWidget(self.creativity_slider)
        self.creativity_label = QLabel(f"{self.creativity_slider.value()}%")
        creativity_layout.addWidget(self.creativity_label)
        intel_layout.addLayout(creativity_layout)
        
        self.reasoning_checkbox = QCheckBox("Enable Advanced Reasoning")
        self.reasoning_checkbox.setChecked(settings_manager.get('advanced_reasoning', True))
        self.reasoning_checkbox.stateChanged.connect(self.toggle_advanced_reasoning)
        intel_layout.addWidget(self.reasoning_checkbox)
        
        self.adaptive_learning_checkbox = QCheckBox("Enable Adaptive Learning")
        self.adaptive_learning_checkbox.setChecked(settings_manager.get('adaptive_learning', True))
        self.adaptive_learning_checkbox.stateChanged.connect(self.toggle_adaptive_learning)
        intel_layout.addWidget(self.adaptive_learning_checkbox)
        
      
        memory_group = QGroupBox("Hierarchical Memory System")
        memory_layout = QVBoxLayout(memory_group)
        
      
        self.hierarchical_memory_label = QLabel("Hierarchical Memory: Loading...")
        memory_layout.addWidget(self.hierarchical_memory_label)
        
      
        self.memory_efficiency_label = QLabel("Memory Efficiency: Loading...")
        memory_layout.addWidget(self.memory_efficiency_label)
        
        self.contextual_links_label = QLabel("Contextual Links: Loading...")
        memory_layout.addWidget(self.contextual_links_label)
        
        self.forgotten_memories_label = QLabel("Forgotten Memories: Loading...")
        memory_layout.addWidget(self.forgotten_memories_label)
        
      
        replay_layout = QHBoxLayout()
        replay_layout.addWidget(QLabel("Memory Replay Interval:"))
        self.replay_interval_spin = QSpinBox()
        self.replay_interval_spin.setRange(10, 200)
        self.replay_interval_spin.setValue(settings_manager.get('memory_replay_interval', MEMORY_REPLAY_INTERVAL))
        self.replay_interval_spin.valueChanged.connect(self.update_memory_replay_interval)
        replay_layout.addWidget(self.replay_interval_spin)
        memory_layout.addLayout(replay_layout)
        
      
        compression_layout = QHBoxLayout()
        compression_layout.addWidget(QLabel("Compression Threshold:"))
        self.compression_slider = QSlider(Qt.Horizontal)
        self.compression_slider.setRange(30, 90)
        self.compression_slider.setValue(int(settings_manager.get('memory_compression_threshold', MEMORY_COMPRESSION_THRESHOLD) * 100))
        self.compression_slider.valueChanged.connect(self.update_memory_compression_threshold)
        compression_layout.addWidget(self.compression_slider)
        self.compression_label = QLabel(f"{self.compression_slider.value()}%")
        compression_layout.addWidget(self.compression_label)
        memory_layout.addLayout(compression_layout)
        
      
        self.force_replay_btn = QPushButton("Force Memory Replay")
        self.force_replay_btn.clicked.connect(self.force_memory_replay)
        memory_layout.addWidget(self.force_replay_btn)
        
      
        action_layout = QHBoxLayout()
        self.save_settings_btn = QPushButton("Save Settings")
        self.save_settings_btn.clicked.connect(self.save_all_settings)
        self.reset_settings_btn = QPushButton("Reset to Defaults")
        self.reset_settings_btn.clicked.connect(self.reset_settings)
        self.test_gpu_btn = QPushButton("Test GPU Acceleration")
        self.test_gpu_btn.clicked.connect(self.test_gpu_acceleration)
        action_layout.addWidget(self.save_settings_btn)
        action_layout.addWidget(self.reset_settings_btn)
        action_layout.addWidget(self.test_gpu_btn)
        
      
        layout.addWidget(self.gpu_group)
        layout.addWidget(perf_group)
        layout.addWidget(gen_group)
        layout.addWidget(sys_group)
        layout.addWidget(intel_group)
        layout.addWidget(memory_group)
        
      
        phased_group = QGroupBox("Phased Intelligence Enhancement")
        phased_layout = QVBoxLayout(phased_group)
        
      
        phase1_group = QGroupBox("Phase 1: Quality & Confidence (Immediate Boost)")
        phase1_layout = QVBoxLayout(phase1_group)
        
        self.critic_checkbox = QCheckBox("Critic: Quality Assessment System")
        self.critic_checkbox.setChecked(settings_manager.get('features', {}).get('critic', False))
        self.critic_checkbox.stateChanged.connect(self.toggle_critic)
        phase1_layout.addWidget(self.critic_checkbox)
        
        self.confidence_gate_checkbox = QCheckBox("Confidence Gate: Auto-Regeneration")
        self.confidence_gate_checkbox.setChecked(settings_manager.get('features', {}).get('confidence_gate', False))
        self.confidence_gate_checkbox.stateChanged.connect(self.toggle_confidence_gate)
        phase1_layout.addWidget(self.confidence_gate_checkbox)
        
        self.anti_loop_checkbox = QCheckBox("Anti-Loop Filter: Prevent Repetition")
        self.anti_loop_checkbox.setChecked(settings_manager.get('features', {}).get('anti_loop_filter', False))
        self.anti_loop_checkbox.stateChanged.connect(self.toggle_anti_loop_filter)
        phase1_layout.addWidget(self.anti_loop_checkbox)
        
        phased_layout.addWidget(phase1_group)
        
      
        phase2_group = QGroupBox("Phase 2: Meta-Learning & Exploration (Idle Time)")
        phase2_layout = QVBoxLayout(phase2_group)
        
        self.meta_memory_checkbox = QCheckBox("Meta-Memory: Weakness Tracking")
        self.meta_memory_checkbox.setChecked(settings_manager.get('features', {}).get('meta_memory', False))
        self.meta_memory_checkbox.stateChanged.connect(self.toggle_meta_memory)
        phase2_layout.addWidget(self.meta_memory_checkbox)
        
        self.curiosity_checkbox = QCheckBox("Curiosity: Idle-Time Exploration")
        self.curiosity_checkbox.setChecked(settings_manager.get('features', {}).get('curiosity', False))
        self.curiosity_checkbox.stateChanged.connect(self.toggle_curiosity)
        phase2_layout.addWidget(self.curiosity_checkbox)
        
        phased_layout.addWidget(phase2_group)
        
      
        phase3_group = QGroupBox("Phase 3: Environment Integration")
        phase3_layout = QVBoxLayout(phase3_group)
        
        self.env_feedback_checkbox = QCheckBox("Environment Feedback: External Events")
        self.env_feedback_checkbox.setChecked(settings_manager.get('features', {}).get('env_feedback', False))
        self.env_feedback_checkbox.stateChanged.connect(self.toggle_env_feedback)
        phase3_layout.addWidget(self.env_feedback_checkbox)
        
        phased_layout.addWidget(phase3_group)
        
      
        phase4_group = QGroupBox("Phase 4: Auto-Tuning")
        phase4_layout = QVBoxLayout(phase4_group)
        
        self.autotune_checkbox = QCheckBox("Autotune: Automatic Parameter Optimization")
        self.autotune_checkbox.setChecked(settings_manager.get('features', {}).get('autotune', False))
        self.autotune_checkbox.stateChanged.connect(self.toggle_autotune)
        phase4_layout.addWidget(self.autotune_checkbox)
        
        phased_layout.addWidget(phase4_group)
        
      
        learning_group = QGroupBox("Response Learning & Factual Knowledge")
        learning_layout = QVBoxLayout(learning_group)
        
        self.response_learning_checkbox = QCheckBox("Response Learning: Learn from Human Responses")
        self.response_learning_checkbox.setChecked(settings_manager.get('features', {}).get('response_learning', False))
        self.response_learning_checkbox.stateChanged.connect(self.toggle_response_learning)
        learning_layout.addWidget(self.response_learning_checkbox)
        
        self.truth_fact_checkbox = QCheckBox("Truth/Fact Table: Store Factual Information")
        self.truth_fact_checkbox.setChecked(settings_manager.get('features', {}).get('truth_fact_table', False))
        self.truth_fact_checkbox.stateChanged.connect(self.toggle_truth_fact_table)
        learning_layout.addWidget(self.truth_fact_checkbox)
        
        self.topic_detection_checkbox = QCheckBox("Topic Detection: Identify Topics Needing Facts")
        self.topic_detection_checkbox.setChecked(settings_manager.get('features', {}).get('topic_detection', False))
        self.topic_detection_checkbox.stateChanged.connect(self.toggle_topic_detection)
        learning_layout.addWidget(self.topic_detection_checkbox)
        
        phased_layout.addWidget(learning_group)
        
      
        status_group = QGroupBox("Feature Status")
        status_layout = QVBoxLayout(status_group)
        
        self.feature_status_label = QLabel("All features disabled by default for safety")
        self.feature_status_label.setStyleSheet("color: orange; font-weight: bold;")
        status_layout.addWidget(self.feature_status_label)
        
        phased_layout.addWidget(status_group)
        
        layout.addWidget(phased_group)
        layout.addLayout(action_layout)
        layout.addStretch()
        
        self.tabs.addTab(tab, "Settings")
    
    def launch_brain_viewer(self):
        viewer_script = "mai_brain_viewer.py"
        python_executable = sys.executable
        if not os.path.exists(viewer_script):
            QMessageBox.critical(self, "Error", f"Viewer script '{viewer_script}' not found. It should be in the same directory.")
            return
        
        try:
            subprocess.Popen([python_executable, viewer_script])
            QMessageBox.information(self, "Launching Viewer", "The 3D Brain Viewer is launching in a new window.")
        except Exception as e:
            QMessageBox.critical(self, "Launch Error", f"Failed to launch the 3D Brain Viewer:\n{e}")

    def update_status_labels(self):
        if hasattr(self.brain, 'response_quality_history') and self.brain.response_quality_history:
            recent_quality = sum(self.brain.response_quality_history[-5:]) / min(5, len(self.brain.response_quality_history))
            self.quality_label.setText(f"Quality: {recent_quality:.2f}")
        
        cluster_count = len(self.brain.semantic_memory.clusters)
        cooccurrence_count = len(self.brain.semantic_memory.word_cooccurrence)
        self.cluster_label.setText(f"Clusters: {cluster_count}, Patterns: {cooccurrence_count}")
        
        gen_stats = self.brain.get_generation_stats()
        self.gen_stats_label.setText(gen_stats)
        
        memory_usage = memory_manager.get_memory_usage_mb()
        available_memory = memory_manager.get_available_memory_mb()
        self.memory_label.setText(f"Memory: {memory_usage:.1f}MB (Available: {available_memory:.1f}MB)")
        
        memory_count = len(self.brain.conversation_memory)
        topic_count = len(self.brain.current_topics)
        
      
        if hasattr(self.brain, 'hierarchical_memory'):
            memory_stats = self.brain.hierarchical_memory.get_memory_stats()
            memory_info = f"ST: {memory_stats['short_term_count']}, LT: {memory_stats['long_term_count']}, SM: {memory_stats['semantic_count']}"
            
          
            if hasattr(self, 'hierarchical_memory_label'):
                self.hierarchical_memory_label.setText(
                    f"Short-term: {memory_stats['short_term_count']}, "
                    f"Long-term: {memory_stats['long_term_count']}, "
                    f"Semantic: {memory_stats['semantic_count']}, "
                    f"Interactions: {memory_stats['total_interactions']}"
                )
                
              
                if hasattr(self, 'memory_efficiency_label'):
                    efficiency = memory_stats.get('memory_efficiency', 0)
                    self.memory_efficiency_label.setText(f"Memory Efficiency: {efficiency:.2f}")
                
                if hasattr(self, 'contextual_links_label'):
                    contextual_links = memory_stats.get('contextual_links', 0)
                    self.contextual_links_label.setText(f"Contextual Links: {contextual_links}")
                
                if hasattr(self, 'forgotten_memories_label'):
                    forgotten_count = memory_stats.get('forgotten_count', 0)
                    self.forgotten_memories_label.setText(f"Forgotten Memories: {forgotten_count}")
        else:
            memory_info = f"Memory: {memory_count}"
        
      
        self.status_label.setText(f"Smart GPU Acceleration Build v6.1 - {memory_info}, Topics: {topic_count}")

    def send_message(self):
        user_text = self.message_input.text().strip()
        if not user_text: return
        self.add_message("You", user_text)
        self.message_input.clear()
        QApplication.processEvents()
        
        response = self.brain.generate_response(user_text)
        quality_score = self.brain._calculate_response_quality(response, user_text) if hasattr(self.brain, '_calculate_response_quality') else 0.5
        
        self.add_message("Mai", response, reinforce=True, quality=quality_score)
        self.update_status_labels()

    def add_message(self, sender, text, reinforce=False, quality=None):
        import datetime
        timestamp = datetime.datetime.now().strftime("%I:%M:%S %p")
        
        if sender == "You":
          
            message_html = f'''
            <div style="margin: 8px 0; text-align: right; padding: 0 10px;">
                <div style="display: inline-block; max-width: 75%; text-align: left;">
                    <div style="background: #3498DB; color: white; padding: 10px 14px; border-radius: 18px 18px 4px 18px; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; font-size: 13px; line-height: 1.4; word-wrap: break-word; box-shadow: 0 1px 2px rgba(0,0,0,0.1);">
                        {text}
                    </div>
                    <div style="color: #95A5A6; font-size: 11px; margin-top: 4px; text-align: right; padding-right: 4px;">{timestamp}</div>
                </div>
            </div>
            '''
        else:
          
            quality_indicator = ""
            if quality is not None:
                if quality > 0.7:
                    quality_indicator = " <span style='color: #27AE60; font-size: 11px;'>â—</span>"
                elif quality > 0.5:
                    quality_indicator = " <span style='color: #F39C12; font-size: 11px;'>â—</span>"
                else:
                    quality_indicator = " <span style='color: #E74C3C; font-size: 11px;'>â—</span>"
            
            message_html = f'''
            <div style="margin: 8px 0; text-align: left; padding: 0 10px;">
                <div style="display: inline-block; max-width: 75%;">
                    <div style="background: #F8F9FA; color: #2C3E50; padding: 12px 16px; border-radius: 18px 18px 18px 4px; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; font-size: 13px; line-height: 1.4; word-wrap: break-word; border: 1px solid #E9ECEF; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 4px;">
                        <div style="font-weight: 600; color: #34495E; margin-bottom: 6px; font-size: 12px; border-bottom: 1px solid #E9ECEF; padding-bottom: 4px;">
                            Mai{quality_indicator}
                        </div>
                        <div style="color: #2C3E50; font-size: 13px; line-height: 1.5;">
                            {text}
                        </div>
                    </div>
                    <div style="color: #95A5A6; font-size: 11px; margin-top: 2px; padding-left: 4px;">{timestamp}</div>
                </div>
            </div>
            '''
        
        self.chat_window.append(message_html)
        
        if reinforce:
            feedback_html = f'''
            <div style="margin: 8px 10px 15px 10px; text-align: left; font-size: 11px; padding: 8px; background: #F8F9FA; border-radius: 6px; border: 1px solid #E9ECEF;">
                <div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
                    <a href="feedback:+:{text}" style="color: #27AE60; text-decoration: none; padding: 6px 12px; border: 1px solid #27AE60; border-radius: 4px; font-weight: 500; background: rgba(39,174,96,0.1); display: inline-block; min-width: 60px; text-align: center;">Reinforce</a>
                    <a href="feedback:-:{text}" style="color: #E74C3C; text-decoration: none; padding: 6px 12px; border: 1px solid #E74C3C; border-radius: 4px; font-weight: 500; background: rgba(231,76,60,0.1); display: inline-block; min-width: 60px; text-align: center;">Discourage</a>
                    {f'<span style="color: #7F8C8D; font-weight: 500; padding: 6px 12px; background: rgba(127,140,141,0.1); border-radius: 4px; display: inline-block; min-width: 60px; text-align: center;">Quality: {quality:.2f}</span>' if quality is not None else ''}
                </div>
            </div>
            '''
            self.chat_window.append(feedback_html)
        
      
        self.chat_window.moveCursor(QTextCursor.End)
        scrollbar = self.chat_window.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def clear_chat(self):
        """Clear the chat window"""
        self.chat_window.clear()
        self.chat_window.append('<div style="color: #666666; font-style: italic;">Chat cleared. Start a new conversation!</div>')
    
    def handle_feedback(self, url):
        parts = url.toString().split(':', 2)
        if len(parts) == 3 and parts[0] == "feedback":
            score_char, sentence = parts[1], parts[2]
            self.brain.apply_feedback(sentence, is_positive=(score_char == '+'))
            feedback_type = "positive" if score_char == '+' else "negative"
            self.chat_window.append(f'<i style="color: #666666;">(Memory optimized {feedback_type} pattern feedback recorded for: "{sentence[:40]}...")</i>')
            self.update_status_labels()

  
    def toggle_gpu_acceleration(self, state):
        settings_manager.set('gpu_acceleration_enabled', state == Qt.Checked)
        if hasattr(self, 'gpu_status_label'):
            gpu_info = gpu_detector.recommended_gpu
            self.gpu_status_label.setText(f"GPU Detected: {gpu_info['name']} ({gpu_info['vendor']}) - {'Enabled' if state == Qt.Checked else 'Disabled'}")
    
    def update_gpu_acceleration_type(self, value):
        settings_manager.set('gpu_acceleration_type', value)
    
    def update_gpu_memory_limit(self, value):
        settings_manager.set('gpu_memory_limit', value / 100.0)
        self.gpu_mem_label.setText(f"{value}%")
    
    def update_gpu_batch_size(self, value):
        settings_manager.set('gpu_batch_size', value)
    
    def update_parallel_workers(self, value):
        settings_manager.set('parallel_workers', value)
    
    def update_chunk_size(self, value):
        settings_manager.set('chunk_size', value)
    
    def update_memory_threshold(self, value):
        settings_manager.set('memory_threshold', value / 100.0)
        self.mem_label.setText(f"{value}%")
    
    def update_max_response_length(self, value):
        settings_manager.set('max_response_length', value)
        global MAX_RESPONSE_LENGTH
        MAX_RESPONSE_LENGTH = value
    
    def update_quality_threshold(self, value):
        settings_manager.set('quality_threshold', value / 100.0)
        self.qual_label.setText(f"{value}%")
    
    def toggle_memory_monitoring(self, state):
        settings_manager.set('enable_memory_monitoring', state == Qt.Checked)
    
    def toggle_garbage_collection(self, state):
        settings_manager.set('enable_garbage_collection', state == Qt.Checked)
    
    def toggle_enhanced_intelligence(self, state):
        settings_manager.set('enhanced_intelligence', state == Qt.Checked)
        if hasattr(self, 'brain') and self.brain:
            self.brain.enhanced_intelligence_enabled = state == Qt.Checked
    
    def update_creativity_factor(self, value):
        settings_manager.set('creativity_factor', value / 100.0)
        self.creativity_label.setText(f"{value}%")
        if hasattr(self, 'brain') and self.brain and self.brain.adaptive_learning:
            self.brain.adaptive_learning.adaptive_parameters['creativity_factor'] = value / 100.0
    
    def toggle_advanced_reasoning(self, state):
        settings_manager.set('advanced_reasoning', state == Qt.Checked)
    
    def toggle_adaptive_learning(self, state):
        settings_manager.set('adaptive_learning', state == Qt.Checked)
    
    def update_memory_replay_interval(self, value):
        settings_manager.set('memory_replay_interval', value)
        if hasattr(self.brain, 'hierarchical_memory'):
            self.brain.hierarchical_memory.memory_replay_interval = value
        QMessageBox.information(self, "Settings", f"Memory replay interval updated to {value}")
    
    def update_memory_compression_threshold(self, value):
        threshold = value / 100.0
        settings_manager.set('memory_compression_threshold', threshold)
        self.compression_label.setText(f"{value}%")
        QMessageBox.information(self, "Settings", f"Compression threshold updated to {value}%")
    
    def force_memory_replay(self):
        if hasattr(self.brain, 'hierarchical_memory'):
            try:
                self.brain.hierarchical_memory._perform_memory_replay()
                QMessageBox.information(self, "Memory Replay", "Memory replay completed successfully!")
            except Exception as e:
                QMessageBox.warning(self, "Memory Replay", f"Memory replay failed: {e}")
        else:
            QMessageBox.warning(self, "Memory Replay", "Hierarchical memory system not available.")
    
    def toggle_enhanced_intelligence(self, state):
        try:
            settings_manager.set('enhanced_intelligence', state == Qt.Checked)
            if hasattr(self.brain, 'enhanced_intelligence_enabled'):
                self.brain.enhanced_intelligence_enabled = (state == Qt.Checked)
            QMessageBox.information(self, "Settings", "Enhanced intelligence setting updated!")
        except Exception as e:
            QMessageBox.warning(self, "Settings", f"Failed to update enhanced intelligence: {e}")
    
    def toggle_advanced_reasoning(self, state):
        try:
            settings_manager.set('advanced_reasoning', state == Qt.Checked)
            QMessageBox.information(self, "Settings", "Advanced reasoning setting updated!")
        except Exception as e:
            QMessageBox.warning(self, "Settings", f"Failed to update advanced reasoning: {e}")
    
    def reset_settings(self):
        try:
            settings_manager.reset_to_defaults()
            QMessageBox.information(self, "Settings", "Settings reset to defaults!")
        except Exception as e:
            QMessageBox.warning(self, "Settings", f"Failed to reset settings: {e}")
    
    def test_gpu_acceleration(self):
        try:
            if hasattr(self, 'gpu_accelerator') and self.gpu_accelerator.gpu_available:
                QMessageBox.information(self, "GPU Test", f"GPU acceleration is available: {self.gpu_accelerator.gpu_type}")
            else:
                QMessageBox.information(self, "GPU Test", "GPU acceleration is not available. Using CPU mode.")
        except Exception as e:
            QMessageBox.warning(self, "GPU Test", f"GPU test failed: {e}")
    
    def toggle_adaptive_learning(self, state):
        try:
            settings_manager.set('adaptive_learning', state == Qt.Checked)
            QMessageBox.information(self, "Settings", "Adaptive learning setting updated!")
        except Exception as e:
            QMessageBox.warning(self, "Settings", f"Failed to update adaptive learning: {e}")
    
    def update_creativity_factor(self, value):
        try:
            if hasattr(self.brain, 'adaptive_learning') and hasattr(self.brain.adaptive_learning, 'adaptive_parameters'):
                self.brain.adaptive_learning.adaptive_parameters['creativity_factor'] = value / 100.0
            settings_manager.set('creativity_factor', value / 100.0)
            self.creativity_label.setText(f"{value}%")
            QMessageBox.information(self, "Settings", f"Creativity factor updated to {value}%")
        except Exception as e:
            QMessageBox.warning(self, "Settings", f"Failed to update creativity factor: {e}")
    
    def update_context_weight(self, value):
        try:
            if hasattr(self.brain, 'adaptive_learning') and hasattr(self.brain.adaptive_learning, 'adaptive_parameters'):
                self.brain.adaptive_learning.adaptive_parameters['context_weight'] = value / 100.0
            settings_manager.set('context_weight', value / 100.0)
            self.context_label.setText(f"{value}%")
            QMessageBox.information(self, "Settings", f"Context weight updated to {value}%")
        except Exception as e:
            QMessageBox.warning(self, "Settings", f"Failed to update context weight: {e}")
    
    def update_semantic_weight(self, value):
        try:
            if hasattr(self.brain, 'adaptive_learning') and hasattr(self.brain.adaptive_learning, 'adaptive_parameters'):
                self.brain.adaptive_learning.adaptive_parameters['semantic_weight'] = value / 100.0
            settings_manager.set('semantic_weight', value / 100.0)
            self.semantic_label.setText(f"{value}%")
            QMessageBox.information(self, "Settings", f"Semantic weight updated to {value}%")
        except Exception as e:
            QMessageBox.warning(self, "Settings", f"Failed to update semantic weight: {e}")
    
    def update_pattern_weight(self, value):
        try:
            if hasattr(self.brain, 'adaptive_learning') and hasattr(self.brain.adaptive_learning, 'adaptive_parameters'):
                self.brain.adaptive_learning.adaptive_parameters['pattern_weight'] = value / 100.0
            settings_manager.set('pattern_weight', value / 100.0)
            self.pattern_label.setText(f"{value}%")
            QMessageBox.information(self, "Settings", f"Pattern weight updated to {value}%")
        except Exception as e:
            QMessageBox.warning(self, "Settings", f"Failed to update pattern weight: {e}")
    
    def save_all_settings(self):
        settings_manager.save_settings()
        QMessageBox.information(self, "Settings Saved", "All settings have been saved successfully!")
    
    def reset_settings(self):
        reply = QMessageBox.question(self, "Reset Settings", 
                                   "Are you sure you want to reset all settings to defaults?",
                                   QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            settings_manager.settings = settings_manager.default_settings.copy()
            settings_manager.save_settings()
            self.load_settings_ui()
            QMessageBox.information(self, "Settings Reset", "All settings have been reset to defaults!")
    
    def load_settings_ui(self):
        """Reload all settings UI elements with current values"""
        self.gpu_enable_checkbox.setChecked(settings_manager.get('gpu_acceleration', True))
        self.workers_combo.setCurrentText(str(settings_manager.get('parallel_workers', 'auto')))
        self.chunk_combo.setCurrentText(str(settings_manager.get('chunk_size', 'auto')))
        self.mem_slider.setValue(int(settings_manager.get('memory_threshold', 0.85) * 100))
        self.resp_length_spin.setValue(settings_manager.get('max_response_length', 25))
        self.qual_slider.setValue(int(settings_manager.get('quality_threshold', 0.5) * 100))
        self.mem_monitor_checkbox.setChecked(settings_manager.get('enable_memory_monitoring', True))
        self.gc_checkbox.setChecked(settings_manager.get('enable_garbage_collection', True))
    
    def test_gpu_acceleration(self):
        """Test GPU acceleration with sample data"""
        try:
            test_patterns = ["hello world", "test pattern", "gpu acceleration", "parallel processing"]
            start_time = time.time()
            result = gpu_accelerator.parallel_process_patterns(test_patterns, "test")
            end_time = time.time()
            
            processing_time = end_time - start_time
            gpu_type = gpu_accelerator.gpu_type
            available = gpu_accelerator.gpu_available
            
            message = f"GPU Test Results:\n"
            message += f"GPU Type: {gpu_type}\n"
            message += f"Available: {'Yes' if available else 'No'}\n"
            message += f"Processing Time: {processing_time:.4f} seconds\n"
            message += f"Test Patterns: {len(test_patterns)}\n"
            message += f"Result: {result[:2]}..."
            
            QMessageBox.information(self, "GPU Test Results", message)
            
        except Exception as e:
            QMessageBox.critical(self, "GPU Test Error", f"Error testing GPU acceleration:\n{e}")

  
    def toggle_critic(self, state):
        """Toggle Critic feature"""
        try:
            enabled = state == Qt.Checked
            settings_manager.set('features', {**settings_manager.get('features', {}), 'critic': enabled})
            self.update_feature_status()
            QMessageBox.information(self, "Critic Feature", f"Critic {'enabled' if enabled else 'disabled'}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to toggle Critic: {e}")
    
    def toggle_confidence_gate(self, state):
        """Toggle Confidence Gate feature"""
        try:
            enabled = state == Qt.Checked
            settings_manager.set('features', {**settings_manager.get('features', {}), 'confidence_gate': enabled})
            self.update_feature_status()
            QMessageBox.information(self, "Confidence Gate Feature", f"Confidence Gate {'enabled' if enabled else 'disabled'}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to toggle Confidence Gate: {e}")
    
    def toggle_anti_loop_filter(self, state):
        """Toggle Anti-Loop Filter feature"""
        try:
            enabled = state == Qt.Checked
            settings_manager.set('features', {**settings_manager.get('features', {}), 'anti_loop_filter': enabled})
            self.update_feature_status()
            QMessageBox.information(self, "Anti-Loop Filter Feature", f"Anti-Loop Filter {'enabled' if enabled else 'disabled'}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to toggle Anti-Loop Filter: {e}")
    
    def toggle_meta_memory(self, state):
        """Toggle Meta-Memory feature"""
        try:
            enabled = state == Qt.Checked
            settings_manager.set('features', {**settings_manager.get('features', {}), 'meta_memory': enabled})
            self.update_feature_status()
            QMessageBox.information(self, "Meta-Memory Feature", f"Meta-Memory {'enabled' if enabled else 'disabled'}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to toggle Meta-Memory: {e}")
    
    def toggle_curiosity(self, state):
        """Toggle Curiosity feature"""
        try:
            enabled = state == Qt.Checked
            settings_manager.set('features', {**settings_manager.get('features', {}), 'curiosity': enabled})
            self.update_feature_status()
            QMessageBox.information(self, "Curiosity Feature", f"Curiosity {'enabled' if enabled else 'disabled'}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to toggle Curiosity: {e}")
    
    def toggle_env_feedback(self, state):
        """Toggle Environment Feedback feature"""
        try:
            enabled = state == Qt.Checked
            settings_manager.set('features', {**settings_manager.get('features', {}), 'env_feedback': enabled})
            self.update_feature_status()
            QMessageBox.information(self, "Environment Feedback Feature", f"Environment Feedback {'enabled' if enabled else 'disabled'}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to toggle Environment Feedback: {e}")
    
    def toggle_autotune(self, state):
        """Toggle Autotune feature"""
        try:
            enabled = state == Qt.Checked
            settings_manager.set('features', {**settings_manager.get('features', {}), 'autotune': enabled})
            self.update_feature_status()
            QMessageBox.information(self, "Autotune Feature", f"Autotune {'enabled' if enabled else 'disabled'}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to toggle Autotune: {e}")
    
    def update_feature_status(self):
        """Update the feature status display"""
        try:
            features = settings_manager.get('features', {})
            enabled_count = sum(1 for enabled in features.values() if enabled)
            total_count = len(features)
            
            if enabled_count == 0:
                status_text = "All features disabled by default for safety"
                color = "orange"
            elif enabled_count == total_count:
                status_text = f"All {total_count} features enabled - Advanced mode active"
                color = "green"
            else:
                status_text = f"{enabled_count}/{total_count} features enabled"
                color = "blue"
            
            self.feature_status_label.setText(status_text)
            self.feature_status_label.setStyleSheet(f"color: {color}; font-weight: bold;")
        except Exception as e:
            print(f"Error updating feature status: {e}")
    
  
    def toggle_response_learning(self, state):
        """Toggle Response Learning feature"""
        try:
            enabled = state == Qt.Checked
            settings_manager.set('features', {**settings_manager.get('features', {}), 'response_learning': enabled})
            self.update_feature_status()
            QMessageBox.information(self, "Response Learning Feature", f"Response Learning {'enabled' if enabled else 'disabled'}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to toggle Response Learning: {e}")
    
    def toggle_truth_fact_table(self, state):
        """Toggle Truth/Fact Table feature"""
        try:
            enabled = state == Qt.Checked
            settings_manager.set('features', {**settings_manager.get('features', {}), 'truth_fact_table': enabled})
            self.update_feature_status()
            QMessageBox.information(self, "Truth/Fact Table Feature", f"Truth/Fact Table {'enabled' if enabled else 'disabled'}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to toggle Truth/Fact Table: {e}")
    
    def toggle_topic_detection(self, state):
        """Toggle Topic Detection feature"""
        try:
            enabled = state == Qt.Checked
            settings_manager.set('features', {**settings_manager.get('features', {}), 'topic_detection': enabled})
            self.update_feature_status()
            QMessageBox.information(self, "Topic Detection Feature", f"Topic Detection {'enabled' if enabled else 'disabled'}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to toggle Topic Detection: {e}")

    def select_files(self):
        self.selected_files, _ = QFileDialog.getOpenFileNames(self, "Select Text Files", "", "Text Files (*.txt)")
        self.file_input_label.setText(f"{len(self.selected_files)} files selected." if self.selected_files else "No files selected.")
    
    def teach_from_files(self):
        if not hasattr(self, 'selected_files') or not self.selected_files:
            QMessageBox.warning(self, "No Files", "Please select files to teach from first."); return
        
        if self.file_training_worker and self.file_training_worker.isRunning():
            self.file_training_worker.stop()
            self.file_training_worker.wait()
        
        self.file_training_worker = FileTrainingWorker(self.brain, self.selected_files)
        self.file_training_worker.progress_updated.connect(self.teach_progress.setValue)
        self.file_training_worker.status_updated.connect(self.training_status.setText)
        self.file_training_worker.training_complete.connect(self.on_training_complete)
        self.file_training_worker.memory_status_updated.connect(self.memory_status.setText)
        
        self.teach_progress.setValue(0)
        self.training_status.setText("Starting memory optimized training...")
        self.file_training_worker.start()
    
    def on_training_complete(self, message):
        QMessageBox.information(self, "Memory Optimized Training Complete", message)
        self.training_status.setText("Memory optimized training complete!")
        self.update_status_labels()

    def toggle_self_train(self, checked):
        if checked:
            self.self_train_button.setText("Stop Self-Training")
            self.self_train_log.setVisible(True)
            self.self_train_log.clear()
            self.self_train_log.append("=== Memory Optimized Self-Training Started ===\nNo preset responses - statistical generation with RAM monitoring\n")
            self.self_train_worker.start()
        else:
            self.self_train_button.setText("Begin Self-Training")
            self.self_train_worker.stop()
            if hasattr(self.self_train_worker, 'training_quality_history') and self.self_train_worker.training_quality_history:
                avg_quality = sum(self.self_train_worker.training_quality_history) / len(self.self_train_worker.training_quality_history)
                cluster_count = len(self.brain.semantic_memory.clusters)
                gen_stats = self.brain.get_generation_stats()
                memory_usage = memory_manager.get_memory_usage_mb()
                self.self_train_log.append(f"\n=== Training Complete ===\nAverage Quality: {avg_quality:.3f}\nFinal Clusters: {cluster_count}\nFinal Memory: {memory_usage:.1f}MB\n{gen_stats}")
    
    def update_self_train_log(self, text):
        self.self_train_log.append(text)
        self.self_train_log.moveCursor(QTextCursor.End)

    def reset_correction_ui(self, is_prompt_state=True):
        self.correction_user_input.setEnabled(not is_prompt_state)
        self.btn_get_mai_response.setEnabled(not is_prompt_state)
        self.btn_encourage.setEnabled(False); self.btn_discourage.setEnabled(False); self.btn_teach_correction.setEnabled(False)
        self.correction_feedback_input.setEnabled(False)
        if is_prompt_state:
            self.correction_user_input.clear(); self.correction_mai_output.setText("<i>Your reply will trigger Mai's generated response.</i>")
            self.correction_feedback_input.clear()

    def get_new_correction_prompt(self):
        prompt = self.brain.generate_response()
        self.correction_context = prompt
        self.correction_status.setText(f'<b>Mai says (Memory Optimized):</b> "{prompt}"')
        self.reset_correction_ui(is_prompt_state=False)

    def get_correction_response(self):
        user_text = self.correction_user_input.text().strip()
        if not user_text: return
        self.correction_context = user_text
        response = self.brain.generate_response(user_text)
        
        quality_score = self.brain._calculate_response_quality(response, user_text) if hasattr(self.brain, '_calculate_response_quality') else 0.5
        
        response_words = self.brain.clean_text(response)
        context_words = self.brain.get_structured_conversation_context(history_turns=2)
        semantic_matches = 0
        for word in response_words:
            if word in self.brain.semantic_memory.word_to_cluster:
                cluster_id = self.brain.semantic_memory.word_to_cluster[word]
                for context_word in context_words:
                    if context_word in self.brain.semantic_memory.word_to_cluster:
                        if self.brain.semantic_memory.word_to_cluster[context_word] == cluster_id:
                            semantic_matches += 1
                            break
        
        semantic_coherence = semantic_matches / max(len(response_words), 1)
        gen_stats = self.brain.get_generation_stats()
        memory_usage = memory_manager.get_memory_usage_mb()
        quality_text = f" (Quality: {quality_score:.2f}, Semantic: {semantic_coherence:.2f}, Memory: {memory_usage:.1f}MB)"
        
        self.correction_mai_response = response
        self.correction_mai_output.setText(f'<b>Mai\'s memory optimized response:</b> "{response}"{quality_text}<br/><small>{gen_stats}</small>')
        self.btn_encourage.setEnabled(True); self.btn_discourage.setEnabled(True)
        self.btn_teach_correction.setEnabled(True); self.correction_feedback_input.setEnabled(True)

    def encourage_response(self):
        full_sentence = self.correction_context + " " + self.correction_mai_response
        self.brain.apply_feedback(full_sentence, is_positive=True)
        QMessageBox.information(self, "Pattern Reinforcement", "Mai's generated response pattern has been strongly reinforced.")
        self.reset_correction_ui()

    def discourage_response(self):
        full_sentence = self.correction_context + " " + self.correction_mai_response
        self.brain.apply_feedback(full_sentence, is_positive=False)
        QMessageBox.information(self, "Pattern Discouragement", "Mai's generated response pattern has been discouraged.")
        self.reset_correction_ui()

    def teach_correction(self):
        correction = self.correction_feedback_input.text().strip()
        if not correction:
            QMessageBox.warning(self, "Empty Correction", "Please type a corrected response first."); return
        
        bad_sentence = self.correction_context + " " + self.correction_mai_response
        good_sentence = self.correction_context + " " + correction
        self.brain.apply_feedback(bad_sentence, is_positive=False)
        self.brain.apply_feedback(good_sentence, is_positive=True)

        QMessageBox.information(self, "Pattern Correction", "Mai has learned your corrected response as a statistical pattern.")
        self.reset_correction_ui()

    def view_brain_data(self):
        self.brain_view.setText("Loading memory optimized brain data..."); QApplication.processEvents()
        
        if self.file_training_worker and self.file_training_worker.isRunning():
            self.brain_view.setText("Training in progress. Please wait for training to complete before viewing brain data.")
            return
        
        try:
            read_con = sqlite3.connect(self.brain.db_file, check_same_thread=False)
            read_cur = read_con.cursor()
            
            read_cur.execute("""
                SELECT context_len, word1, word2, word3, word4, word5, word6, word7, word8, next_word, priority, success_rate, usage_count 
                FROM dynamic_word_chain 
                WHERE context_len = 8 
                ORDER BY priority DESC 
                LIMIT 100
            """)
            data = read_cur.fetchall()
            read_con.close()
            
        except Exception as e:
            self.brain_view.setText(f"Error loading brain data: {e}\nThis may happen if training is still in progress.")
            return
        
        if not data: 
            self.brain_view.setText("The memory optimized brain is empty. Teach Mai with text files using background training."); return
        
        header = "MEMORY OPTIMIZED BRAIN DATA - STRONGEST STATISTICAL PATTERNS\n"
        header += "NO PRESET RESPONSES - ALL GENERATED FROM LEARNED PATTERNS\n"
        header += "PRIORITY | SUCCESS | USAGE | STATISTICAL PATTERN -> NEXT WORD\n"
        header += "=" * 120 + "\n"
        
        content_lines = []
        for row in data:
            context_len, w1, w2, w3, w4, w5, w6, w7, w8, next_word, priority, success_rate, usage_count = row
            priority_str = str(priority).ljust(8)
            success_str = f"{success_rate:.2f}".ljust(7)
            usage_str = str(usage_count).ljust(5)
            pattern = f"'{w1}' '{w2}' '{w3}' '{w4}' '{w5}' '{w6}' '{w7}' '{w8}' -> '{next_word}'"
            
            content_lines.append(f"{priority_str} | {success_str} | {usage_str} | {pattern}")
        
        gen_stats = self.brain.get_generation_stats()
        memory_usage = memory_manager.get_memory_usage_mb()
        available_memory = memory_manager.get_available_memory_mb()
        footer = f"\n\nMemory Optimization: RAM monitoring, chunked processing for 2M+ word files"
        footer += f"\nOptimal chunk size calculation based on available memory"
        footer += f"\nCurrent Memory: {memory_usage:.1f}MB, Available: {available_memory:.1f}MB"
        footer += f"\n{gen_stats}"
        
        self.brain_view.setText(header + "\n".join(content_lines) + footer)

    def view_attention_weights(self):
        self.brain_view.setText("Loading memory optimized attention weights..."); QApplication.processEvents()
        weights = self.brain.attention.focus_weights
        adaptive_rates = getattr(self.brain.attention, 'adaptive_rates', {})
        success_history = getattr(self.brain.attention, 'success_history', {})
        
        if not weights: 
            self.brain_view.setText("No attention weights learned yet in memory optimized mode."); return
        
        header = "MEMORY OPTIMIZED ATTENTION WEIGHTS WITH SEMANTIC CLUSTERING\n"
        header += "NO PRESET RESPONSES - STATISTICAL FOCUS\n"
        header += "WEIGHT  | RATE | SUCCESS% | CLUSTER | WORD\n"
        header += "=" * 70 + "\n"
        
        sorted_weights = sorted(weights.items(), key=lambda item: item[1], reverse=True)
        content_lines = []
        
        for word, weight in sorted_weights[:50]:
            weight_str = f"{weight:.3f}".ljust(7)
            rate_str = f"{adaptive_rates.get(word, 0.1):.3f}".ljust(5)
            
            history = success_history.get(word, [])
            success_pct = (sum(history) / len(history) * 100) if history else 50.0
            success_str = f"{success_pct:.1f}%".ljust(8)
            
            cluster_id = self.brain.semantic_memory.word_to_cluster.get(word, "None")
            cluster_str = str(cluster_id).ljust(7)
            
            content_lines.append(f"{weight_str} | {rate_str} | {success_str} | {cluster_str} | '{word}'")
        
        gen_stats = self.brain.get_generation_stats()
        memory_usage = memory_manager.get_memory_usage_mb()
        footer = f"\n\nMemory Optimization: Background training with RAM monitoring"
        footer += f"\nCurrent Memory Usage: {memory_usage:.1f}MB"
        footer += f"\n{gen_stats}"
        
        self.brain_view.setText(header + "\n".join(content_lines) + footer)

    def view_context_performance(self):
        self.brain_view.setText("Loading memory optimized context performance..."); QApplication.processEvents()
        
        scorer = self.brain.context_scorer
        header = "MEMORY OPTIMIZED CONTEXT PERFORMANCE ANALYSIS\n"
        header += "NO PRESET RESPONSES - STATISTICAL PATTERN ANALYSIS\n"
        header += "CONTEXT | SUCCESS RATE | USAGE COUNT | PERFORMANCE\n"
        header += "=" * 60 + "\n"
        
        content_lines = []
        for context_len in CONTEXT_LEVELS:
            score = scorer.get_context_score(context_len)
            usage = scorer.context_usage_count.get(str(context_len), 0)
            
            context_str = f"{context_len}-word".ljust(7)
            score_str = f"{score:.3f}".ljust(11)
            usage_str = str(usage).ljust(11)
            
            if score > 0.7:
                performance = "Excellent"
            elif score > 0.5:
                performance = "Good"
            elif score > 0.3:
                performance = "Fair"
            else:
                performance = "Poor"
            
            content_lines.append(f"{context_str} | {score_str} | {usage_str} | {performance}")
        
        best_order = scorer.get_best_context_order()
        content_lines.append(f"\nOptimal Context Order: {' -> '.join(map(str, best_order))}")
        
        cluster_count = len(self.brain.semantic_memory.clusters)
        cooccurrence_count = len(self.brain.semantic_memory.word_cooccurrence)
        content_lines.append(f"\nSemantic Clusters: {cluster_count}")
        content_lines.append(f"Word Co-occurrence Patterns: {cooccurrence_count:,}")
        content_lines.append(f"Conversation coherence: {self.brain.conversation_coherence_score:.3f}")
        
        memory_usage = memory_manager.get_memory_usage_mb()
        available_memory = memory_manager.get_available_memory_mb()
        content_lines.append(f"\nMemory Optimization Settings:")
        content_lines.append(f"  Current Memory Usage: {memory_usage:.1f}MB")
        content_lines.append(f"  Available Memory: {available_memory:.1f}MB")
        content_lines.append(f"  Memory Safety Threshold: {MEMORY_SAFETY_THRESHOLD*100:.0f}%")
        content_lines.append(f"  Chunk Size Range: {MIN_CHUNK_SIZE:,} - {MAX_CHUNK_SIZE:,} words")
        content_lines.append(f"  Parallel Workers: {memory_manager.calculate_parallel_workers()}")
        
        gen_stats = self.brain.get_generation_stats()
        content_lines.append(f"\n{gen_stats}")
        
        self.brain_view.setText(header + "\n".join(content_lines))

    def view_semantic_clusters(self):
        self.brain_view.setText("Loading memory optimized semantic clusters..."); QApplication.processEvents()
        
        clusters = self.brain.semantic_memory.clusters
        cluster_strength = self.brain.semantic_memory.cluster_strength
        
        if not clusters:
            self.brain_view.setText("No semantic clusters formed yet. Teach Mai more content using memory optimized training."); return
        
        header = "MEMORY OPTIMIZED SEMANTIC MEMORY CLUSTERS\n"
        header += "NO PRESET RESPONSES - STATISTICAL CONCEPT RELATIONSHIPS\n"
        header += "CLUSTER | STRENGTH | SIZE | RELATED WORDS\n"
        header += "=" * 100 + "\n"
        
        content_lines = []
        
        sorted_clusters = sorted(clusters.items(), key=lambda x: cluster_strength.get(x[0], 0), reverse=True)
        
        for cluster_id, words in sorted_clusters[:20]:
            cluster_str = str(cluster_id).ljust(7)
            strength = cluster_strength.get(cluster_id, 0)
            strength_str = str(strength).ljust(8)
            size_str = str(len(words)).ljust(4)
            
            word_list = list(words)[:10]
            if len(words) > 10:
                words_str = ", ".join(f"'{w}'" for w in word_list) + f" ... (+{len(words)-10} more)"
            else:
                words_str = ", ".join(f"'{w}'" for w in word_list)
            
            content_lines.append(f"{cluster_str} | {strength_str} | {size_str} | {words_str}")
        
        content_lines.append(f"\n=== MEMORY OPTIMIZED CLUSTERING STATISTICS ===")
        content_lines.append(f"Total Clusters: {len(clusters)}")
        content_lines.append(f"Total Words in Clusters: {sum(len(words) for words in clusters.values())}")
        content_lines.append(f"Average Cluster Size: {sum(len(words) for words in clusters.values()) / len(clusters):.1f}")
        content_lines.append(f"Co-occurrence Patterns: {len(self.brain.semantic_memory.word_cooccurrence):,}")
        
        if len(self.brain.semantic_memory.word_to_cluster) > 0:
            content_lines.append(f"\n=== EXAMPLE STATISTICAL WORD RELATIONSHIPS ===")
            example_words = list(self.brain.semantic_memory.word_to_cluster.keys())[:5]
            for word in example_words:
                related = self.brain.semantic_memory.get_related_words(word, 5)
                if related:
                    related_str = ", ".join(f"'{w}'" for w in related)
                    content_lines.append(f"'{word}' -> {related_str}")
        
        memory_usage = memory_manager.get_memory_usage_mb()
        available_memory = memory_manager.get_available_memory_mb()
        content_lines.append(f"\n=== MEMORY OPTIMIZATION ===")
        content_lines.append(f"Memory usage monitoring enabled")
        content_lines.append(f"Current Memory: {memory_usage:.1f}MB, Available: {available_memory:.1f}MB")
        content_lines.append(f"Chunked processing for large files (2M+ words)")
        content_lines.append(f"Parallel/sequential processing based on RAM availability")
        
        gen_stats = self.brain.get_generation_stats()
        content_lines.append(f"\n{gen_stats}")
        
        self.brain_view.setText(header + "\n".join(content_lines))

    def view_generation_stats(self):
        self.brain_view.setText("Loading generation statistics..."); QApplication.processEvents()
        
        if self.file_training_worker and self.file_training_worker.isRunning():
            self.brain_view.setText("Training in progress. Please wait for training to complete before viewing statistics.")
            return
        
        gen_stats = self.brain.get_generation_stats()
        
        header = "MEMORY OPTIMIZED GENERATIVE MODEL STATISTICS\n"
        header += "NO PRESET RESPONSES - STATISTICAL GENERATION ANALYSIS\n"
        header += "=" * 80 + "\n"
        
        content_lines = []
        content_lines.append(f"Primary Statistics:")
        content_lines.append(f"  {gen_stats}")
        
        try:
            read_con = sqlite3.connect(self.brain.db_file, check_same_thread=False)
            read_cur = read_con.cursor()
            
            read_cur.execute("SELECT COUNT(*) FROM dynamic_word_chain")
            chain_count = read_cur.fetchone()[0]
            
            read_cur.execute("SELECT COUNT(*) FROM word_associations")
            assoc_count = read_cur.fetchone()[0]
            
            read_cur.execute("SELECT COUNT(DISTINCT next_word) FROM dynamic_word_chain")
            unique_words = read_cur.fetchone()[0]
            
            read_con.close()
            
            content_lines.append(f"\nDatabase Statistics:")
            content_lines.append(f"  Word Chain Patterns: {chain_count:,}")
            content_lines.append(f"  Word Associations: {assoc_count:,}")
            content_lines.append(f"  Unique Vocabulary: {unique_words:,}")
            
        except Exception as e:
            content_lines.append(f"\nDatabase Statistics: Error loading ({e})")
        
        cluster_count = len(self.brain.semantic_memory.clusters)
        cooccurrence_count = len(self.brain.semantic_memory.word_cooccurrence)
        
        content_lines.append(f"\nSemantic Clustering Statistics:")
        content_lines.append(f"  Semantic Clusters: {cluster_count}")
        content_lines.append(f"  Co-occurrence Patterns: {cooccurrence_count:,}")
        
        if self.brain.response_quality_history:
            avg_quality = sum(self.brain.response_quality_history) / len(self.brain.response_quality_history)
            recent_quality = sum(self.brain.response_quality_history[-10:]) / min(10, len(self.brain.response_quality_history))
            
            content_lines.append(f"\nResponse Quality Statistics:")
            content_lines.append(f"  Average Quality: {avg_quality:.3f}")
            content_lines.append(f"  Recent Quality (last 10): {recent_quality:.3f}")
            content_lines.append(f"  Quality History Length: {len(self.brain.response_quality_history)}")
        
        content_lines.append(f"\nContext Performance:")
        for context_len in CONTEXT_LEVELS:
            score = self.brain.context_scorer.get_context_score(context_len)
            usage = self.brain.context_scorer.context_usage_count.get(str(context_len), 0)
            content_lines.append(f"  {context_len}-word context: {score:.3f} success rate ({usage} uses)")
        
        content_lines.append(f"\nMemory Statistics:")
        content_lines.append(f"  Conversation Memory: {len(self.brain.conversation_memory)}")
        content_lines.append(f"  Topic Words: {len(self.brain.topic_words)}")
        content_lines.append(f"  Current Topics: {len(self.brain.current_topics)}")
        content_lines.append(f"  Conversation Coherence: {self.brain.conversation_coherence_score:.3f}")
        
        content_lines.append(f"\nNeural Network Statistics:")
        content_lines.append(f"  Vocabulary Size: {self.brain.model.vocab_size}")
        content_lines.append(f"  Context Size: {self.brain.model.context_size}")
        content_lines.append(f"  Hidden Size: {self.brain.model.hidden_size}")
        content_lines.append(f"  Input Size: {self.brain.model.input_size}")
        
        memory_usage = memory_manager.get_memory_usage_mb()
        available_memory = memory_manager.get_available_memory_mb()
        memory_percent = memory_manager.get_memory_usage_percent()
        
        content_lines.append(f"\nMemory Optimization Statistics:")
        content_lines.append(f"  Current Memory Usage: {memory_usage:.1f}MB ({memory_percent:.1f}%)")
        content_lines.append(f"  Available Memory: {available_memory:.1f}MB")
        content_lines.append(f"  Memory Safety Threshold: {MEMORY_SAFETY_THRESHOLD*100:.0f}%")
        content_lines.append(f"  Base Chunk Size: {CHUNK_SIZE_WORDS:,} words")
        content_lines.append(f"  Chunk Size Range: {MIN_CHUNK_SIZE:,} - {MAX_CHUNK_SIZE:,} words")
        content_lines.append(f"  Optimal Chunk Size (current): {memory_manager.calculate_optimal_chunk_size(1000000):,} words")
        content_lines.append(f"  Parallel Workers Available: {memory_manager.calculate_parallel_workers()}")
        batch_count = getattr(self.brain, 'batch_count', 0)
        content_lines.append(f"  Current Batch Operations: {batch_count}")
        
        content_lines.append(f"\n=== MEMORY OPTIMIZED GENERATIVE MODEL SUMMARY ===")
        content_lines.append(f"This is a Statistical Generative Model (SGM) with:")
        content_lines.append(f"- No preset responses or fallback text")
        content_lines.append(f"- All responses generated from learned statistical patterns")
        content_lines.append(f"- Semantic clustering for conceptual coherence")
        content_lines.append(f"- Dynamic context scoring and attention mechanisms")
        content_lines.append(f"- Continuous learning from user interactions")
        content_lines.append(f"- MEMORY OPTIMIZED: Handles 2M+ word files with RAM monitoring")
        content_lines.append(f"- MEMORY OPTIMIZED: Chunked processing with parallel/sequential selection")
        content_lines.append(f"- MEMORY OPTIMIZED: Dynamic batch sizing based on available RAM")
        content_lines.append(f"- MEMORY OPTIMIZED: Garbage collection and memory pressure detection")
        content_lines.append(f"- CURSOR PROTECTION: Database operations protected from conflicts")
        
        self.brain_view.setText(header + "\n".join(content_lines))

    def view_memory_statistics(self):
        self.brain_view.setText("Loading memory statistics..."); QApplication.processEvents()
        
        header = "DETAILED MEMORY OPTIMIZATION STATISTICS\n"
        header += "REAL-TIME RAM MONITORING AND OPTIMIZATION\n"
        header += "=" * 80 + "\n"
        
        content_lines = []
        
        memory_usage = memory_manager.get_memory_usage_mb()
        available_memory = memory_manager.get_available_memory_mb()
        memory_percent = memory_manager.get_memory_usage_percent()
        total_memory = psutil.virtual_memory().total / 1024 / 1024
        
        content_lines.append(f"=== CURRENT MEMORY STATE ===")
        content_lines.append(f"Process Memory Usage: {memory_usage:.1f}MB")
        content_lines.append(f"System Memory Usage: {memory_percent:.1f}%")
        content_lines.append(f"Available Memory: {available_memory:.1f}MB")
        content_lines.append(f"Total System Memory: {total_memory:.1f}MB")
        content_lines.append(f"Memory Safety Threshold: {MEMORY_SAFETY_THRESHOLD*100:.0f}%")
        
        content_lines.append(f"\n=== MEMORY OPTIMIZATION SETTINGS ===")
        content_lines.append(f"Base Batch Size: {BATCH_SIZE:,} operations")
        content_lines.append(f"Large File Threshold: {LARGE_FILE_THRESHOLD:,} words")
        content_lines.append(f"Base Chunk Size: {CHUNK_SIZE_WORDS:,} words")
        content_lines.append(f"Min Chunk Size: {MIN_CHUNK_SIZE:,} words")
        content_lines.append(f"Max Chunk Size: {MAX_CHUNK_SIZE:,} words")
        content_lines.append(f"Parallel Worker Limit: {PARALLEL_WORKER_LIMIT}")
        
        optimal_chunk_size = memory_manager.calculate_optimal_chunk_size(1000000)
        parallel_workers = memory_manager.calculate_parallel_workers()
        should_reduce_batch = memory_manager.should_reduce_batch_size()
        
        content_lines.append(f"\n=== DYNAMIC OPTIMIZATION CALCULATIONS ===")
        content_lines.append(f"Optimal Chunk Size (1M words): {optimal_chunk_size:,} words")
        content_lines.append(f"Recommended Parallel Workers: {parallel_workers}")
        content_lines.append(f"Should Reduce Batch Size: {'Yes' if should_reduce_batch else 'No'}")
        
        cpu_count_logical = cpu_count()
        cpu_count_physical = psutil.cpu_count(logical=False)
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        content_lines.append(f"\n=== SYSTEM RESOURCES ===")
        content_lines.append(f"CPU Cores (Logical): {cpu_count_logical}")
        content_lines.append(f"CPU Cores (Physical): {cpu_count_physical}")
        content_lines.append(f"Current CPU Usage: {cpu_percent:.1f}%")
        
        content_lines.append(f"\n=== FILE SIZE PROCESSING RECOMMENDATIONS ===")
        
        file_sizes = [10000, 50000, 100000, 500000, 1000000, 2000000, 5000000]
        for file_size in file_sizes:
            chunk_size = memory_manager.calculate_optimal_chunk_size(file_size)
            chunks_needed = (file_size + chunk_size - 1) // chunk_size
            processing_mode = "Parallel" if parallel_workers > 1 and chunks_needed > 1 else "Sequential"
            
            content_lines.append(f"  {file_size:,} words: {chunks_needed} chunks of {chunk_size:,} words ({processing_mode})")

        content_lines.append(f"\n=== MEMORY PRESSURE SCENARIOS ===")
        scenarios = [
            ("Low Memory (1GB available)", 1024),
            ("Medium Memory (4GB available)", 4096),
            ("High Memory (8GB+ available)", 8192)
        ]
        
        for scenario_name, available_mb in scenarios:
            safe_memory_mb = available_mb * (1 - MEMORY_SAFETY_THRESHOLD)
            estimated_mb_per_word = 0.001
            safe_chunk_size = int(safe_memory_mb / estimated_mb_per_word)
            safe_chunk_size = max(MIN_CHUNK_SIZE, min(MAX_CHUNK_SIZE, safe_chunk_size))
            
            workers = 1 if available_mb < 2000 else (min(2, PARALLEL_WORKER_LIMIT) if available_mb < 4000 else PARALLEL_WORKER_LIMIT)
            
            content_lines.append(f"  {scenario_name}:")
            content_lines.append(f"    Chunk Size: {safe_chunk_size:,} words")
            content_lines.append(f"    Parallel Workers: {workers}")
            content_lines.append(f"    Processing Mode: {'Parallel' if workers > 1 else 'Sequential'}")
        
        content_lines.append(f"\n=== GARBAGE COLLECTION STATUS ===")
        content_lines.append(f"Garbage Collection: Available and Active")
        content_lines.append(f"Auto-GC on Memory Pressure: Enabled")
        content_lines.append(f"Force GC After Large Operations: Enabled")
        
        content_lines.append(f"\n=== PERFORMANCE OPTIMIZATION TIPS ===")
        content_lines.append(f"- Close other applications before processing very large files")
        content_lines.append(f"- Files over 1M words will be automatically chunked")
        content_lines.append(f"- Parallel processing is automatically selected when beneficial")
        content_lines.append(f"- Memory usage is monitored and optimized in real-time")
        content_lines.append(f"- Training will adapt batch sizes if memory pressure is detected")
        
        self.brain_view.setText(header + "\n".join(content_lines))

    def save_to_initial_knowledge(self):
        reply = QMessageBox.question(self, "Save Patterns", 
                                   "This will add recent high-quality statistical patterns to the initial knowledge file. Continue?")
        if reply == QMessageBox.StandardButton.Yes:
            try:
                if self.file_training_worker and self.file_training_worker.isRunning():
                    QMessageBox.warning(self, "Training in Progress", "Please wait for training to complete before saving patterns.")
                    return
                
                read_con = sqlite3.connect(self.brain.db_file, check_same_thread=False)
                read_cur = read_con.cursor()
                
                read_cur.execute("""
                    SELECT word1, word2, word3, word4, word5, word6, word7, word8, next_word, priority, success_rate, usage_count
                    FROM dynamic_word_chain 
                    WHERE priority > 15 AND context_len = 8 AND usage_count > 2
                    ORDER BY (priority * LOG(usage_count + 1)) DESC LIMIT 30
                """)
                chains = read_cur.fetchall()
                read_con.close()
                
                learned_sentences = []
                for chain in chains:
                    words = [w for w in chain[:-3] if w != '<PAD>']
                    words.append(chain[-3])
                    if len(words) > 4:
                        sentence = " ".join(words).strip()
                        if sentence not in learned_sentences:
                            learned_sentences.append(sentence)
                
                knowledge_file = 'training/initial_knowledge.txt'
                current_content = ""
                if os.path.exists(knowledge_file):
                    with open(knowledge_file, 'r', encoding='utf-8') as f:
                        current_content = f.read().strip()
                
                if learned_sentences:
                    cluster_count = len(self.brain.semantic_memory.clusters)
                    gen_stats = self.brain.get_generation_stats()
                    memory_usage = memory_manager.get_memory_usage_mb()
                    new_content = current_content + f"\n\n# Memory optimized patterns (v4.5 - {time.strftime('%Y-%m-%d')}):\n"
                    new_content += f"# {gen_stats}\n"
                    new_content += f"# Semantic clusters: {cluster_count}\n"
                    new_content += f"# Memory optimization: 2M+ word file support, RAM monitoring\n"
                    new_content += f"# Memory usage: {memory_usage:.1f}MB\n"
                    new_content += "\n".join(learned_sentences[:15])
                    
                    with open(knowledge_file, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    
                    QMessageBox.information(self, "Patterns Saved", 
                                          f"Added {len(learned_sentences[:15])} high-quality statistical patterns to initial knowledge file.")
                else:
                    QMessageBox.information(self, "No New Patterns", "No high-quality patterns found to save.")
                    
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Could not save patterns: {str(e)}")

    def export_brain(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Export Memory Optimized Brain", DB_FILE, "Database Files (*.db)")
        if file_path:
            try:
                shutil.copy(DB_FILE, file_path)
                
                memory_usage = memory_manager.get_memory_usage_mb()
                available_memory = memory_manager.get_available_memory_mb()
                
                metadata = {
                    'version': '4.5_consolidated_build',
                    'export_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'generation_stats': self.brain.get_generation_stats(),
                    'context_performance': self.brain.context_scorer.context_performance,
                    'semantic_clusters': len(self.brain.semantic_memory.clusters),
                    'word_cooccurrence_patterns': len(self.brain.semantic_memory.word_cooccurrence),
                    'no_preset_responses': True,
                    'statistical_generation': True,
                    'memory_optimized': True,
                    'memory_optimization_settings': {
                        'memory_safety_threshold': MEMORY_SAFETY_THRESHOLD,
                        'chunk_size_words': CHUNK_SIZE_WORDS,
                        'min_chunk_size': MIN_CHUNK_SIZE,
                        'max_chunk_size': MAX_CHUNK_SIZE,
                        'parallel_worker_limit': PARALLEL_WORKER_LIMIT,
                        'large_file_threshold': LARGE_FILE_THRESHOLD
                    },
                    'memory_stats_at_export': {
                        'memory_usage_mb': memory_usage,
                        'available_memory_mb': available_memory,
                        'optimal_chunk_size': memory_manager.calculate_optimal_chunk_size(1000000),
                        'parallel_workers': memory_manager.calculate_parallel_workers()
                    },
                    'conversation_stats': {
                        'memory_length': len(self.brain.conversation_memory),
                        'topics_tracked': len(self.brain.current_topics),
                        'quality_history_length': len(getattr(self.brain, 'response_quality_history', []))
                    }
                }
                
                metadata_file = file_path.replace('.db', '_metadata.json')
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2)
                
                QMessageBox.information(self, "Memory Optimized Export Complete", 
                                      f"Memory optimized brain exported to {file_path}\nMetadata saved to {metadata_file}")
            except Exception as e:
                QMessageBox.warning(self, "Export Error", f"Could not export brain: {str(e)}")

    def import_brain(self):
        reply = QMessageBox.question(self, "Import Memory Optimized Brain", 
                                   "This will overwrite Mai's current brain and require a restart. Are you sure?")
        if reply == QMessageBox.StandardButton.Yes:
            file_path, _ = QFileDialog.getOpenFileName(self, "Import Memory Optimized Brain", "", "Database Files (*.db)")
            if file_path:
                try:
                    self.self_train_worker.stop()
                    if self.file_training_worker and self.file_training_worker.isRunning():
                        self.file_training_worker.stop()
                        self.file_training_worker.wait()
                    
                    self.brain.con.close()
                    shutil.copy(file_path, DB_FILE)
                    
                    metadata_file = file_path.replace('.db', '_metadata.json')
                    if os.path.exists(metadata_file):
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                        cluster_info = f"Clusters: {metadata.get('semantic_clusters', 'unknown')}, " if 'semantic_clusters' in metadata else ""
                        gen_info = "Statistical Generation: YES" if metadata.get('statistical_generation', False) else "Statistical Generation: UNKNOWN"
                        mem_info = " (Memory Optimized)" if metadata.get('memory_optimized', False) else ""
                        QMessageBox.information(self, "Memory Optimized Import Complete", 
                                              f"Memory optimized brain imported successfully.\nVersion: {metadata.get('version', 'unknown')}\n{cluster_info}{gen_info}{mem_info}\nPlease restart the application.")
                    else:
                        QMessageBox.information(self, "Import Complete", 
                                              "Brain imported successfully. Please restart the application.")
                    self.close()
                except Exception as e:
                    QMessageBox.warning(self, "Import Error", f"Could not import brain: {str(e)}")

    def closeEvent(self, event):
        print("Shutting down Memory Optimized Edition...")
        self.self_train_worker.stop()
        if self.self_train_worker.isRunning():
            self.self_train_worker.wait()
        
        if self.file_training_worker and self.file_training_worker.isRunning():
            self.file_training_worker.stop()
            self.file_training_worker.wait()
        
        if hasattr(self.brain, '_flush_batch_operations'):
            self.brain._flush_batch_operations()
        
        self.brain.save_state()
      
        try:
            if self.brain.io_lock:
                with self.brain.io_lock:
                    self.brain.con.close()
            else:
                self.brain.con.close()
        except Exception as e:
            print(f"Warning: Could not close database cleanly: {e}")
        print("Memory Optimized Edition shutdown complete.")
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = MaiApp()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    if not os.path.exists(DB_FILE):
        print("No brain found. Performing memory optimized initial learning setup...")
        try:
            temp_brain = HybridBrain(DB_FILE)
            with open('training/initial_knowledge.txt', 'r', encoding='utf-8') as f: 
                content = f.read()
                temp_brain.learn_from_text(content, base_priority_boost=3)
            
            if hasattr(temp_brain, '_flush_batch_operations'):
                temp_brain._flush_batch_operations()
            
            cluster_count = len(temp_brain.semantic_memory.clusters)
            gen_stats = temp_brain.get_generation_stats()
            memory_usage = memory_manager.get_memory_usage_mb()
            print(f"Smart GPU acceleration initial learning complete. Mai v6.1 is ready.")
            print(f"Clusters: {cluster_count}")
            print(f"{gen_stats}")
            print(f"Memory usage: {memory_usage:.1f}MB")
            print("NO PRESET RESPONSES - All responses generated from learned patterns!")
            print(f"SMART GPU ACCELERATION: Intelligent hardware detection, conditional acceleration, memory management, batch optimization")
            
            temp_brain.save_state()
            temp_brain.con.close()
        except Exception as e:
            print(f"Error during initial learning: {e}")
    
  
    mp.freeze_support()
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    main()
