#ollash/embedding_utils.py
import subprocess
import json
import numpy as np
from typing import Optional, List, Dict,  Any
import hashlib
import re
from pathlib import Path
import mmap
import pickle
import time
from concurrent.futures import ThreadPoolExecutor

class EmbeddingManager:
    """Optimized embedding management with local caching and batch processing"""
    
    def __init__(self, model: str = "llama3"):
        self.model = model
        self.cache_dir = Path.home() / ".ollash" / "embeddings_cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        # Memory-mapped cache for faster access
        self.cache_mmap_path = self.cache_dir / "embeddings.mmap"
        self._init_mmap_cache()
        
        # Thread pool for parallel operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Precompile regex patterns
        self._clean_pattern = re.compile(r'[^\w\s\-\.\/_\|\&\$\(\)\[\]\{\}<>]')
        self._num_pattern = re.compile(r'-?\d+\.?\d*')

    def _init_mmap_cache(self):
        """Initialize memory-mapped cache file"""
        if not self.cache_mmap_path.exists():
            with open(self.cache_mmap_path, 'wb') as f:
                pickle.dump({}, f)
        
        self.mmap_file = open(self.cache_mmap_path, 'r+b')
        self.mmap = mmap.mmap(self.mmap_file.fileno(), 0)
        self.cache = pickle.loads(self.mmap)

    def _update_mmap_cache(self):
        """Update memory-mapped cache"""
        self.mmap.seek(0)
        pickle.dump(self.cache, self.mmap)
        self.mmap.flush()

    def get_embedding(self, text: str, use_cache: bool = True) -> Optional[np.ndarray]:
        """Get embedding with optimized caching and fallbacks"""
        if not text:
            return None
            
        normalized_text = self._normalize_text(text)
        cache_key = self._get_cache_key(normalized_text)
        
        # Check memory-mapped cache first
        if use_cache and cache_key in self.cache:
            return self.cache[cache_key]
        
        # Try to get embedding (with timeout)
        try:
            embedding = self._get_embedding_with_timeout(normalized_text, timeout=5)
        except Exception as e:
            print(f"Embedding generation failed: {e}")
            embedding = None
            
        # Fallback to hash-based embedding if needed
        if embedding is None:
            embedding = self._get_hash_embedding(normalized_text)
        
        # Update cache if successful
        if embedding is not None and use_cache:
            self.cache[cache_key] = embedding
            self._update_mmap_cache()
            
        return embedding

    def _get_embedding_with_timeout(self, text: str, timeout: int = 5) -> Optional[np.ndarray]:
        """Get embedding with timeout protection"""
        try:
            # Try Ollama embeddings API first
            result = subprocess.run(
                ["ollama", "embeddings", self.model, text],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if result.returncode == 0:
                try:
                    response = json.loads(result.stdout)
                    if 'embedding' in response:
                        embedding = np.array(response['embedding'], dtype=np.float32)
                        return embedding / np.linalg.norm(embedding)  # Normalize
                except json.JSONDecodeError:
                    pass
                    
            # Fallback to model inference if embeddings API not available
            prompt = f"Convert this to space-separated numbers: {text}"
            result = subprocess.run(
                ["ollama", "run", self.model, prompt],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if result.returncode == 0:
                numbers = self._num_pattern.findall(result.stdout)
                if len(numbers) >= 8:
                    embedding = np.array([float(n) for n in numbers[:384]], dtype=np.float32)
                    norm = np.linalg.norm(embedding)
                    return embedding / norm if norm > 0 else None
                    
        except Exception:
            return None
            
        return None

    def get_embeddings_batch(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        """Get embeddings for multiple texts in parallel"""
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(self.get_embedding, text) for text in texts]
            return [future.result() for future in futures]

    def _normalize_text(self, text: str) -> str:
        """Optimized text normalization"""
        text = text.lower().strip()
        text = ' '.join(text.split())
        return self._clean_pattern.sub(' ', text)

    def _get_cache_key(self, text: str) -> str:
        """Fast cache key generation"""
        return hashlib.blake2b(f"{self.model}:{text}".encode(), digest_size=16).hexdigest()

    def _get_hash_embedding(self, text: str, dims: int = 384) -> np.ndarray:
        """Optimized hash-based embedding generator"""
        # Use multiple hash algorithms for better distribution
        hash_funcs = [hashlib.blake2b, hashlib.sha3_256, hashlib.sha256]
        hashes = []
        
        for i, hash_func in enumerate(hash_funcs):
            h = hash_func(f"{text}:{i}".encode())
            hashes.extend([x/255.0 for x in h.digest()])
        
        # Ensure correct dimensions
        embedding = np.array(hashes[:dims], dtype=np.float32)
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else np.zeros(dims, dtype=np.float32)

    def compute_similarity_batch(self, query_embedding: np.ndarray, 
                               target_embeddings: List[np.ndarray]) -> np.ndarray:
        """Vectorized similarity computation"""
        if not target_embeddings:
            return np.array([])
            
        targets = np.stack(target_embeddings)
        dots = np.dot(targets, query_embedding)
        norms = np.linalg.norm(targets, axis=1)
        valid = (norms > 0) & (np.linalg.norm(query_embedding) > 0)
        similarities = np.zeros(len(target_embeddings))
        similarities[valid] = dots[valid] / (norms[valid] * np.linalg.norm(query_embedding))
        return similarities

    def clear_cache(self):
        """Clear all cached embeddings"""
        self.cache.clear()
        self._update_mmap_cache()

    def get_cache_info(self) -> Dict[str, float]:
        """Get cache statistics"""
        return {
            "size": len(self.cache),
            "memory_usage": self.mmap.size() / (1024 * 1024)  # MB
        }

    def __del__(self):
        """Cleanup resources"""
        self.mmap.close()
        self.mmap_file.close()


def extract_command_features(command: str) -> Dict[str, Any]:
    """Optimized command feature extraction"""
    words = command.split()
    return {
        'command': words[0] if words else '',
        'has_pipe': '|' in command,
        'has_redirect': any(c in command for c in ['>', '<']),
        'has_sudo': command.startswith('sudo '),
        'flag_count': sum(1 for w in words if w.startswith('-')),
        'word_count': len(words),
        'path_depth': max(len(p.split('/')) for p in command.split()) if '/' in command else 1,
        'has_glob': any(c in command for c in ['*', '?', '[']),
        'has_subshell': any(c in command for c in ['$(', '`'])
    }


def calculate_command_similarity(cmd1: str, cmd2: str) -> float:
    """Optimized command similarity calculation"""
    if not cmd1 or not cmd2:
        return 0.0
        
    f1 = extract_command_features(cmd1)
    f2 = extract_command_features(cmd2)
    
    # Command name match (40% weight)
    name_score = 0.4 if f1['command'] == f2['command'] else 0
    
    # Structural features (30% weight)
    struct_features = ['has_pipe', 'has_redirect', 'has_sudo', 'has_glob', 'has_subshell']
    struct_score = 0.3 * sum(f1[f] == f2[f] for f in struct_features) / len(struct_features)
    
    # Length similarity (20% weight)
    len_score = 0.2 * (1 - abs(f1['word_count'] - f2['word_count']) / max(f1['word_count'], f2['word_count'], 1))
    
    # Flag count similarity (10% weight)
    flag_score = 0.1 * (1 - abs(f1['flag_count'] - f2['flag_count']) / max(f1['flag_count'], f2['flag_count'], 1))
    
    return min(name_score + struct_score + len_score + flag_score, 1.0)
