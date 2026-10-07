"""
Proprietary High-Speed AI Brain File Format (.hsb)
Single-file format with custom extension and dedicated reader
"""

import os
import struct
import zlib
import json
import pickle
import time
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

# Module-level format constants for consumers that don't need the class
HSB_MAGIC = b'HSB\x00'
HSB_VERSION = 1
HSB_HEADER_SIZE = 256

class HSBFileFormat:
    """Proprietary High-Speed Brain (.hsb) file format"""
    
    # File format constants (also exposed at module level as HSB_MAGIC, HSB_VERSION, HSB_HEADER_SIZE)
    HSB_MAGIC = b'HSB\x00'  # High-Speed Brain magic bytes
    HSB_VERSION = 1
    HEADER_SIZE = 256  # Increased header size to accommodate larger headers
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.file_handle = None
        self.header = {}
        
    def create_file(self):
        """Create new HSB file with header"""
        if not self.file_path or not isinstance(self.file_path, str):
            raise ValueError("HSB file path must be a non-empty string")
        if self.file_handle is not None:
            try:
                self.file_handle.close()
            except Exception:
                pass
            self.file_handle = None
        parent = os.path.dirname(self.file_path)
        if parent:
            try:
                os.makedirs(parent, exist_ok=True)
            except OSError as e:
                raise OSError(f"Cannot create parent directory for HSB file: {e}") from e
        self.file_handle = open(self.file_path, 'wb')
        try:
            # Write magic bytes and version
            self.file_handle.write(self.HSB_MAGIC)
            self.file_handle.write(struct.pack('<I', self.HSB_VERSION))
            
            # Initialize header
            self.header = {
                'created': time.time(),
                'modified': time.time(),
                'data_offset': self.HEADER_SIZE,
                'data_size': 0,
                'clusters_offset': 0,
                'clusters_size': 0,
                'enhanced_offset': 0,
                'enhanced_size': 0,
                'checksum': 0
            }
            
            # Write initial header (protocol 4 for compatibility)
            header_data = zlib.compress(pickle.dumps(self.header, protocol=4))
            
            # Ensure header data fits in reserved space
            if len(header_data) > self.HEADER_SIZE - 8:
                raise ValueError(f"Header data too large: {len(header_data)} > {self.HEADER_SIZE - 8}")
            
            # Pad header data to fill reserved space and flush for durability
            padded_header = header_data + b'\x00' * (self.HEADER_SIZE - 8 - len(header_data))
            self.file_handle.write(padded_header)
            self.file_handle.flush()
        except Exception:
            self.file_handle.close()
            self.file_handle = None
            raise
        
    def open_file(self, mode='rb'):
        """Open existing HSB file"""
        if not self.file_path or not isinstance(self.file_path, str):
            raise ValueError("HSB file path must be a non-empty string")
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"HSB file not found: {self.file_path}")
        if not os.path.isfile(self.file_path):
            raise ValueError(f"HSB path is not a file: {self.file_path}")
        if self.file_handle is not None:
            try:
                self.file_handle.close()
            except Exception:
                pass
            self.file_handle = None
        self.file_handle = open(self.file_path, mode)
        try:
            # Read and validate header
            magic = self.file_handle.read(4)
            if len(magic) < 4:
                raise ValueError("HSB file truncated (short magic)")
            if magic != self.HSB_MAGIC:
                raise ValueError("Invalid HSB file format")
            version_bytes = self.file_handle.read(4)
            if len(version_bytes) < 4:
                raise ValueError("HSB file truncated")
            version = struct.unpack('<I', version_bytes)[0]
            if version != self.HSB_VERSION:
                raise ValueError(f"Unsupported HSB version: {version}")
        except Exception:
            self.file_handle.close()
            self.file_handle = None
            raise
            
        # Read header data
        header_data = self.file_handle.read(self.HEADER_SIZE - 8)
        if not header_data or len(header_data) < 4:
            self.file_handle.close()
            self.file_handle = None
            raise ValueError("HSB file truncated (header data too short)")
        # Try to decompress header data with multiple fallback strategies
        try:
            # First try direct decompression
            decompressed_data = zlib.decompress(header_data)
            self.header = pickle.loads(decompressed_data)
            if not isinstance(self.header, dict):
                self.header = {'created': time.time(), 'modified': time.time(), 'data_offset': self.HEADER_SIZE,
                               'data_size': 0, 'clusters_offset': 0, 'clusters_size': 0, 'enhanced_offset': 0, 'enhanced_size': 0, 'checksum': 0}
        except zlib.error:
            try:
                # Try removing trailing null bytes
                trimmed_data = header_data.rstrip(b'\x00')
                decompressed_data = zlib.decompress(trimmed_data)
                self.header = pickle.loads(decompressed_data)
                if not isinstance(self.header, dict):
                    self.header = {'created': time.time(), 'modified': time.time(), 'data_offset': self.HEADER_SIZE,
                                   'data_size': 0, 'clusters_offset': 0, 'clusters_size': 0, 'enhanced_offset': 0, 'enhanced_size': 0, 'checksum': 0}
            except zlib.error:
                try:
                    # Try removing leading null bytes too
                    trimmed_data = header_data.strip(b'\x00')
                    decompressed_data = zlib.decompress(trimmed_data)
                    self.header = pickle.loads(decompressed_data)
                    if not isinstance(self.header, dict):
                        self.header = {'created': time.time(), 'modified': time.time(), 'data_offset': self.HEADER_SIZE,
                                       'data_size': 0, 'clusters_offset': 0, 'clusters_size': 0, 'enhanced_offset': 0, 'enhanced_size': 0, 'checksum': 0}
                except zlib.error as e:
                    # If all decompression attempts fail, create a default header
                    print(f"Warning: Could not decompress header data: {e}")
                    print("Creating default header and continuing...")
                    self.header = {
                        'created': time.time(),
                        'modified': time.time(),
                        'data_offset': self.HEADER_SIZE,
                        'data_size': 0,
                        'clusters_offset': 0,
                        'clusters_size': 0,
                        'enhanced_offset': 0,
                        'enhanced_size': 0,
                        'checksum': 0
                    }
        
    def write_data_section(self, data: Dict[str, Any]):
        """Write main data section"""
        if self.file_handle is None:
            raise RuntimeError("HSB file not open for writing; call create_file() first.")
        if not isinstance(data, dict):
            data = {}
        compressed_data = zlib.compress(pickle.dumps(data, protocol=4))
        
        # Update header
        self.header['data_offset'] = self.file_handle.tell()
        self.header['data_size'] = len(compressed_data)
        self.header['modified'] = time.time()
        
        # Write data and flush for durability
        self.file_handle.write(compressed_data)
        self.file_handle.flush()
        
    def write_clusters_section(self, clusters_data: Dict[str, Any]):
        """Write semantic clusters section"""
        if self.file_handle is None:
            raise RuntimeError("HSB file not open for writing; call create_file() first.")
        if not isinstance(clusters_data, dict):
            clusters_data = {}
        compressed_data = zlib.compress(pickle.dumps(clusters_data, protocol=4))
        
        # Update header
        self.header['clusters_offset'] = self.file_handle.tell()
        self.header['clusters_size'] = len(compressed_data)
        self.header['modified'] = time.time()
        
        # Write clusters and flush
        self.file_handle.write(compressed_data)
        self.file_handle.flush()
        
    def write_enhanced_section(self, enhanced_data: Dict[str, Any]):
        """Write enhanced intelligence section"""
        if self.file_handle is None:
            raise RuntimeError("HSB file not open for writing; call create_file() first.")
        if not isinstance(enhanced_data, dict):
            enhanced_data = {}
        compressed_data = zlib.compress(pickle.dumps(enhanced_data, protocol=4))
        
        # Update header
        self.header['enhanced_offset'] = self.file_handle.tell()
        self.header['enhanced_size'] = len(compressed_data)
        self.header['modified'] = time.time()
        
        # Write enhanced data and flush
        self.file_handle.write(compressed_data)
        self.file_handle.flush()
        
    def read_data_section(self) -> Dict[str, Any]:
        """Read main data section"""
        if self.file_handle is None:
            return {}
        if not isinstance(self.header, dict):
            return {}
        data_size = self.header.get('data_size', 0)
        if data_size <= 0:
            return {}
        max_sane = 500 * 1024 * 1024  # 500 MB
        if not isinstance(data_size, (int, float)) or data_size > max_sane:
            return {}
        data_offset = self.header.get('data_offset', self.HEADER_SIZE)
        if not isinstance(data_offset, (int, float)) or data_offset < 8:
            return {}
        try:
            data_size = int(data_size)
            self.file_handle.seek(int(data_offset))
            compressed_data = self.file_handle.read(data_size)
            if not compressed_data or len(compressed_data) < data_size:
                return {}
            data = pickle.loads(zlib.decompress(compressed_data))
            return data if isinstance(data, dict) else {}
        except (zlib.error, pickle.UnpicklingError, EOFError, Exception):
            return {}
        
    def read_clusters_section(self) -> Dict[str, Any]:
        """Read semantic clusters section"""
        if self.file_handle is None:
            return {}
        if not isinstance(self.header, dict):
            return {}
        size = self.header.get('clusters_size', 0)
        if size <= 0:
            return {}
        max_sane = 100 * 1024 * 1024  # 100 MB
        if not isinstance(size, (int, float)) or size > max_sane:
            return {}
        clusters_offset = self.header.get('clusters_offset', self.HEADER_SIZE)
        if not isinstance(clusters_offset, (int, float)) or clusters_offset < 8:
            return {}
        try:
            size = int(size)
            self.file_handle.seek(int(clusters_offset))
            compressed_data = self.file_handle.read(size)
            if not compressed_data or len(compressed_data) < size:
                return {}
            data = pickle.loads(zlib.decompress(compressed_data))
            return data if isinstance(data, dict) else {}
        except (zlib.error, pickle.UnpicklingError, EOFError, Exception):
            return {}
        
    def read_enhanced_section(self) -> Dict[str, Any]:
        """Read enhanced intelligence section"""
        if self.file_handle is None:
            return {}
        if not isinstance(self.header, dict):
            return {}
        size = self.header.get('enhanced_size', 0)
        if size <= 0:
            return {}
        max_sane = 100 * 1024 * 1024  # 100 MB
        if not isinstance(size, (int, float)) or size > max_sane:
            return {}
        enhanced_offset = self.header.get('enhanced_offset', self.HEADER_SIZE)
        if not isinstance(enhanced_offset, (int, float)) or enhanced_offset < 8:
            return {}
        try:
            size = int(size)
            self.file_handle.seek(int(enhanced_offset))
            compressed_data = self.file_handle.read(size)
            if not compressed_data or len(compressed_data) < size:
                return {}
            data = pickle.loads(zlib.decompress(compressed_data))
            return data if isinstance(data, dict) else {}
        except (zlib.error, pickle.UnpicklingError, EOFError, Exception):
            return {}
        
    def update_header(self):
        """Update file header"""
        if self.file_handle is None:
            return
        if not isinstance(self.header, dict):
            return
        try:
            current_pos = self.file_handle.tell()
            self.file_handle.seek(self.HEADER_SIZE)
            data = self.file_handle.read()
            self.header['checksum'] = zlib.crc32(data) & 0xffffffff if data else 0
            self.file_handle.seek(current_pos)
            self.file_handle.seek(8)
            header_data = zlib.compress(pickle.dumps(self.header, protocol=4))
            if len(header_data) > self.HEADER_SIZE - 8:
                raise ValueError(f"Header data too large: {len(header_data)} > {self.HEADER_SIZE - 8}")
            padded_header = header_data + b'\x00' * (self.HEADER_SIZE - 8 - len(header_data))
            self.file_handle.write(padded_header)
            self.file_handle.flush()
        except (OSError, IOError, ValueError):
            pass

    def close(self):
        """Close file and update header. File handle is always cleared in finally so state is consistent."""
        if not self.file_handle:
            return
        try:
            if hasattr(self.file_handle, 'mode') and 'w' in self.file_handle.mode:
                self.update_header()
        except Exception:
            pass
        try:
            self.file_handle.close()
        except Exception:
            pass
        finally:
            self.file_handle = None

    def get_format_version(self) -> int:
        """Return the HSB format version number (e.g. 1)."""
        return self.HSB_VERSION
            
    def get_file_info(self) -> Dict[str, Any]:
        """Get file information"""
        default = {'file_path': '', 'file_size_mb': 0, 'created': 0, 'modified': 0,
                   'data_size_mb': 0, 'clusters_size_mb': 0, 'enhanced_size_mb': 0, 'checksum': 0}
        if not self.file_path:
            return default.copy()
        if not isinstance(self.header, dict):
            return default.copy()
        try:
            file_size = os.path.getsize(self.file_path) if os.path.exists(self.file_path) else 0
        except OSError:
            file_size = 0
        try:
            created = float(self.header.get('created', 0) or 0)
            modified = float(self.header.get('modified', 0) or 0)
            data_size = float(self.header.get('data_size', 0) or 0)
            clusters_size = float(self.header.get('clusters_size', 0) or 0)
            enhanced_size = float(self.header.get('enhanced_size', 0) or 0)
        except (TypeError, ValueError):
            created = modified = data_size = clusters_size = enhanced_size = 0
        denom = 1024 * 1024
        return {
            'file_path': self.file_path or '',
            'file_size_mb': file_size / denom if file_size else 0,
            'created': created,
            'modified': modified,
            'data_size_mb': data_size / denom if data_size else 0,
            'clusters_size_mb': clusters_size / denom if clusters_size else 0,
            'enhanced_size_mb': enhanced_size / denom if enhanced_size else 0,
            'checksum': int(self.header.get('checksum', 0) or 0)
        }

class HSBBrainReader:
    """Dedicated reader for HSB brain files"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.hsb_file = None
        self.data = {}
        self.clusters = {}
        self.enhanced = {}
        
    def __enter__(self) -> "HSBBrainReader":
        """Context manager entry: load_brain must be called before use (e.g. after read_hsb_brain)."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit: close the reader."""
        self.close()
        return None

    def load_brain(self, verbose: bool = True):
        """Load complete brain from HSB file. Set verbose=False to suppress load/section messages."""
        if not self.file_path or not isinstance(self.file_path, str):
            raise ValueError("HSB file path must be a non-empty string")
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"HSB file not found: {self.file_path}")
        if not os.path.isfile(self.file_path):
            raise ValueError(f"HSB path is not a file: {self.file_path}")

        self.hsb_file = HSBFileFormat(self.file_path)
        self.hsb_file.open_file('rb')

        # Load all sections safely
        try:
            self.data = self.hsb_file.read_data_section()
        except Exception as e:
            if verbose:
                print(f"Warning: Could not read data section: {e}")
            self.data = {}

        try:
            self.clusters = self.hsb_file.read_clusters_section()
        except Exception as e:
            if verbose:
                print(f"Warning: Could not read clusters section: {e}")
            self.clusters = {}

        try:
            self.enhanced = self.hsb_file.read_enhanced_section()
        except Exception as e:
            if verbose:
                print(f"Warning: Could not read enhanced section: {e}")
            self.enhanced = {}

        if verbose:
            try:
                size_mb = os.path.getsize(self.file_path) / 1024 / 1024
                print(f"Loaded HSB brain: {size_mb:.2f} MB")
            except OSError:
                print("Loaded HSB brain (size unknown)")
        
    def get_pattern_count(self) -> int:
        """Return number of patterns. Uses stored pattern_count when present to avoid building the full list."""
        if not isinstance(self.data, dict):
            return 0
        val = self.data.get('pattern_count')
        if isinstance(val, (int, float)) and val >= 0:
            return int(val)
        return len(self.get_patterns())

    def get_association_count(self) -> int:
        """Return number of associations. Uses stored association_count when present to avoid building the full list."""
        if not isinstance(self.data, dict):
            return 0
        val = self.data.get('association_count')
        if isinstance(val, (int, float)) and val >= 0:
            return int(val)
        return len(self.get_associations())

    def get_patterns(self) -> List[Tuple]:
        """Get all patterns. Normalizes each item to a tuple and skips non-sequences."""
        if not isinstance(self.data, dict):
            return []
        out = self.data.get('patterns')
        if out is None:
            return []
        if not isinstance(out, (list, tuple)):
            return []
        result = []
        for item in out:
            if not isinstance(item, (list, tuple)):
                continue
            try:
                result.append(tuple(item))
            except (TypeError, ValueError):
                continue
        return result
        
    def get_associations(self) -> List[Tuple]:
        """Get all associations. Normalizes each item to a tuple and skips non-sequences."""
        if not isinstance(self.data, dict):
            return []
        out = self.data.get('associations')
        if out is None:
            return []
        if not isinstance(out, (list, tuple)):
            return []
        result = []
        for item in out:
            if not isinstance(item, (list, tuple)):
                continue
            try:
                result.append(tuple(item))
            except (TypeError, ValueError):
                continue
        return result
        
    def get_semantic_clusters(self) -> Dict[str, Any]:
        """Get semantic clusters"""
        return self.clusters if isinstance(self.clusters, dict) else {}
        
    def get_enhanced_intelligence(self) -> Dict[str, Any]:
        """Get enhanced intelligence data"""
        return self.enhanced if isinstance(self.enhanced, dict) else {}
        
    def get_section_offsets(self) -> Dict[str, Any]:
        """Return section offsets and sizes from the header for debugging/diagnostics. Empty dict if not loaded."""
        if not self.hsb_file or not isinstance(getattr(self.hsb_file, 'header', None), dict):
            return {}
        h = self.hsb_file.header
        return {
            'data_offset': h.get('data_offset'),
            'data_size': h.get('data_size'),
            'clusters_offset': h.get('clusters_offset'),
            'clusters_size': h.get('clusters_size'),
            'enhanced_offset': h.get('enhanced_offset'),
            'enhanced_size': h.get('enhanced_size'),
        }

    def get_brain_stats(self) -> Dict[str, Any]:
        """Get comprehensive brain statistics"""
        default_stats = {
            'file_info': {'file_size_mb': 0, 'file_path': getattr(self, 'file_path', None) or '', 'created': 0, 'modified': 0},
            'patterns_count': 0, 'associations_count': 0, 'clusters_count': 0, 'words_in_clusters': 0, 'enhanced_tables': 0
        }
        if not self.hsb_file or not getattr(self, 'file_path', None):
            return default_stats
        try:
            file_size = os.path.getsize(self.file_path) if os.path.exists(self.file_path) else 0
        except OSError:
            file_size = 0
        header = self.hsb_file.header if isinstance(getattr(self.hsb_file, 'header', None), dict) else {}
        stats = {
            'file_info': {
                'file_size_mb': file_size / (1024 * 1024) if file_size else 0,
                'file_path': self.file_path or '',
                'created': header.get('created', 0),
                'modified': header.get('modified', 0)
            },
            'patterns_count': self.get_pattern_count(),
            'associations_count': self.get_association_count(),
            'clusters_count': len((self.clusters or {}).get('clusters', {})) if isinstance(self.clusters, dict) else 0,
            'words_in_clusters': len((self.clusters or {}).get('word_to_cluster', {})) if isinstance(self.clusters, dict) else 0,
            'enhanced_tables': len(self.enhanced) if isinstance(self.enhanced, dict) else 0
        }
        return stats

    def verify_integrity(self) -> Dict[str, Any]:
        """Verify brain data integrity. Returns {'ok': bool, 'errors': list of str}."""
        errors = []
        if not isinstance(self.data, dict):
            errors.append("data section is not a dict")
        else:
            for key in ('patterns', 'associations'):
                val = self.data.get(key)
                if val is None:
                    continue
                if not isinstance(val, (list, tuple)):
                    errors.append(f"data['{key}'] is not a list or tuple")
                    continue
                for i, item in enumerate(val):
                    if not isinstance(item, (list, tuple)):
                        errors.append(f"data['{key}'][{i}] is not a sequence")
                    elif len(item) < (3 if key == 'patterns' else 2):
                        errors.append(f"data['{key}'][{i}] has too few elements (got {len(item)})")
                # Check stored count consistency when present
                count_key = 'pattern_count' if key == 'patterns' else 'association_count'
                stored = self.data.get(count_key)
                if isinstance(stored, (int, float)) and stored >= 0 and int(stored) != len(val):
                    errors.append(f"data['{count_key}'] ({int(stored)}) does not match len(data['{key}']) ({len(val)})")
        if not isinstance(self.clusters, dict) and self.clusters is not None:
            errors.append("clusters is not a dict or None")
        if not isinstance(self.enhanced, dict) and self.enhanced is not None:
            errors.append("enhanced is not a dict or None")
        return {'ok': len(errors) == 0, 'errors': errors}

    def close(self):
        """Close brain reader and clear cached sections so post-close access returns empty and refs are released."""
        if self.hsb_file:
            try:
                self.hsb_file.close()
            except Exception:
                pass
            self.hsb_file = None
        self.data = {}
        self.clusters = {}
        self.enhanced = {}

    @property
    def is_loaded(self) -> bool:
        """True if load_brain() has been called and the brain file is open."""
        return self.hsb_file is not None

class HSBBrainWriter:
    """Dedicated writer for HSB brain files"""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.hsb_file = None

    def __enter__(self) -> "HSBBrainWriter":
        """Context manager entry. Call create_brain() after entering."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit: finalize (and close) the brain file."""
        self.finalize(verbose=False)
        return None

    def create_brain(self):
        """Create new HSB brain file"""
        if not self.file_path or not isinstance(self.file_path, str):
            raise ValueError("HSBBrainWriter: file_path must be a non-empty string")
        self.hsb_file = HSBFileFormat(self.file_path)
        self.hsb_file.create_file()
        
    def write_patterns(self, patterns: List[Tuple]):
        """Write patterns to brain. Validates and normalizes each pattern (min 3 elements); skips invalid rows."""
        if not self.hsb_file:
            raise RuntimeError("Brain file not created")
        raw = list(patterns) if patterns is not None else []
        normalized = []
        skipped = 0
        for item in raw:
            if not isinstance(item, (list, tuple)):
                skipped += 1
                continue
            if len(item) < 3:
                skipped += 1
                continue
            try:
                normalized.append(tuple(item))
            except (TypeError, ValueError):
                skipped += 1
        if skipped and __name__ != "__main__":
            print(f"HSB format: write_patterns skipped {skipped} invalid pattern(s)")
        data = {
            'patterns': normalized,
            'pattern_count': len(normalized),
            'timestamp': time.time()
        }
        self.hsb_file.write_data_section(data)
        self.hsb_file._data_section_written = True
        
    def write_associations(self, associations: List[Tuple]):
        """Write associations to brain. Validates and normalizes each association (min 2 elements); skips invalid rows."""
        if not self.hsb_file:
            raise RuntimeError("Brain file not created")
        raw = list(associations) if associations is not None else []
        normalized = []
        skipped = 0
        for item in raw:
            if not isinstance(item, (list, tuple)):
                skipped += 1
                continue
            if len(item) < 2:
                skipped += 1
                continue
            try:
                normalized.append(tuple(item))
            except (TypeError, ValueError):
                skipped += 1
        if skipped and __name__ != "__main__":
            print(f"HSB format: write_associations skipped {skipped} invalid association(s)")
        associations = normalized
        # Add associations to existing data or create new section
        if getattr(self.hsb_file, '_data_section_written', False):
            try:
                existing_data = self.hsb_file.read_data_section()
            except Exception:
                existing_data = {}
            if not isinstance(existing_data, dict):
                existing_data = {}
            existing_data['associations'] = associations
            existing_data['association_count'] = len(associations)
            existing_data['timestamp'] = time.time()
            self.hsb_file.write_data_section(existing_data)
        else:
            # Write new data section
            data = {
                'associations': associations,
                'association_count': len(associations),
                'timestamp': time.time()
            }
            self.hsb_file.write_data_section(data)
            
    def write_semantic_clusters(self, clusters_data: Dict[str, Any]):
        """Write semantic clusters to brain"""
        if not self.hsb_file:
            raise RuntimeError("Brain file not created")
        clusters_data = clusters_data if isinstance(clusters_data, dict) else {}
        self.hsb_file.write_clusters_section(clusters_data)
        
    def write_enhanced_intelligence(self, enhanced_data: Dict[str, Any]):
        """Write enhanced intelligence data to brain"""
        if not self.hsb_file:
            raise RuntimeError("Brain file not created")
        enhanced_data = enhanced_data if isinstance(enhanced_data, dict) else {}
        self.hsb_file.write_enhanced_section(enhanced_data)
        
    def finalize(self, verbose: bool = True):
        """Finalize and close brain file. Set verbose=False to suppress the save message. Safe to call multiple times."""
        if self.hsb_file:
            try:
                self.hsb_file.close()
            except Exception:
                pass
            if verbose:
                print(f"HSB brain saved: {self.file_path}")
            self.hsb_file = None

def create_hsb_brain_from_data(output_path: str, patterns: List[Tuple],
                              associations: List[Tuple], clusters: Dict[str, Any] = None,
                              enhanced: Dict[str, Any] = None, verbose: bool = True):
    """Create HSB brain file from data. Set verbose=False to suppress progress prints."""
    if not output_path or not isinstance(output_path, str):
        raise ValueError("create_hsb_brain_from_data: output_path must be a non-empty string")
    patterns = list(patterns) if patterns is not None else []
    associations = list(associations) if associations is not None else []
    if verbose:
        print(f"HSB format: Creating brain file: {output_path}")
        print(f"HSB format: Patterns: {len(patterns)}, Associations: {len(associations)}")

    writer = HSBBrainWriter(output_path)
    if verbose:
        print("HSB format: Initialized HSB brain writer")

    writer.create_brain()
    if verbose:
        print("HSB format: Created brain file")

    if verbose:
        print("HSB format: Writing patterns...")
    writer.write_patterns(patterns)
    if verbose:
        print("HSB format: Patterns written")

    if verbose:
        print("HSB format: Writing associations...")
    writer.write_associations(associations)
    if verbose:
        print("HSB format: Associations written")

    if clusters:
        if verbose:
            print("HSB format: Writing semantic clusters...")
        writer.write_semantic_clusters(clusters)
        if verbose:
            print("HSB format: Semantic clusters written")

    if enhanced:
        if verbose:
            print("HSB format: Writing enhanced intelligence...")
        writer.write_enhanced_intelligence(enhanced)
        if verbose:
            print("HSB format: Enhanced intelligence written")

    if verbose:
        print("HSB format: Finalizing file...")
    writer.finalize(verbose=verbose)
    if verbose:
        print(f"HSB format: Brain file created successfully: {output_path}")

    return output_path

def read_hsb_brain(file_path: str, verbose: bool = True, validate: bool = False) -> HSBBrainReader:
    """Read HSB brain file. verbose=False suppresses load/section messages. validate=True checks is_hsb_file() first."""
    if validate and not is_hsb_file(file_path):
        raise ValueError(f"Path is not a valid HSB file or does not exist: {file_path!r}")
    reader = HSBBrainReader(file_path)
    reader.load_brain(verbose=verbose)
    return reader


def is_hsb_file(path: str) -> bool:
    """Return True if path exists, is a file, and has valid HSB magic bytes. Safe for missing/invalid paths."""
    if not path or not isinstance(path, str):
        return False
    try:
        if not os.path.exists(path) or not os.path.isfile(path):
            return False
        with open(path, 'rb') as f:
            magic = f.read(4)
        return len(magic) == 4 and magic == HSB_MAGIC
    except (OSError, IOError):
        return False

# Example usage
if __name__ == "__main__":
    # Create test data
    patterns = [
        (3, ("the", "machine", "learning"), "algorithm", 85, 0.9, 150),
        (2, ("neural", "network"), "training", 80, 0.85, 200)
    ]
    
    associations = [
        ("neural", "network", 80, 0.85, 200),
        ("machine", "learning", 85, 0.9, 250)
    ]
    
    clusters = {
        'clusters': {0: {"machine", "learning", "algorithm"}, 1: {"neural", "network"}},
        'word_to_cluster': {"machine": 0, "learning": 0, "algorithm": 0, "neural": 1, "network": 1},
        'cluster_strength': {0: 100, 1: 80},
        'cluster_coherence': {0: 0.9, 1: 0.85}
    }
    
    # Create HSB brain
    hsb_file = create_hsb_brain_from_data("test_brain.hsb", patterns, associations, clusters)
    reader = None
    try:
        reader = read_hsb_brain(hsb_file)
        stats = reader.get_brain_stats()
        print(f"Created HSB brain with {stats.get('patterns_count', 0)} patterns and {stats.get('associations_count', 0)} associations")
        fi = stats.get('file_info') or {}
        print(f"File size: {fi.get('file_size_mb', 0):.2f} MB")
    finally:
        if reader is not None:
            reader.close()
