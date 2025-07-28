# ollash/embedding_utils.py
import subprocess
import json
import numpy as np
from typing import Optional, List
import hashlib
import re
from pathlib import Path


class EmbeddingManager:
    """Advanced embedding management for better semantic search"""
    
    def __init__(self, model: str = "llama3"):
        self.model = model if model else "llama3"  # Ensure model is not None
        self.cache_dir = Path.home() / ".ollash" / "embeddings_cache"
        self.cache_dir.mkdir(exist_ok=True)
    
    def get_embedding(self, text: str, use_cache: bool = True) -> Optional[np.ndarray]:
        """Get embedding for text with caching support"""
        # Clean and normalize text
        normalized_text = self._normalize_text(text)
        
        # Check cache first
        if use_cache:
            cached_embedding = self._get_cached_embedding(normalized_text)
            if cached_embedding is not None:
                return cached_embedding
        
        # Try different methods to get embeddings
        embedding = self._get_ollama_embedding(normalized_text)
        
        if embedding is None:
            # Fallback to simpler hash-based embedding
            embedding = self._get_hash_embedding(normalized_text)
        
        # Cache the result
        if embedding is not None and use_cache:
            self._cache_embedding(normalized_text, embedding)
        
        return embedding
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for consistent embeddings"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove special characters but keep important shell symbols
        text = re.sub(r'[^\w\s\-\.\/_\|\&\$\(\)\[\]\{\}<>]', ' ', text)
        
        return text.strip()
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        return hashlib.md5(f"{self.model}:{text}".encode()).hexdigest()
    
    def _get_cached_embedding(self, text: str) -> Optional[np.ndarray]:
        """Retrieve cached embedding if available"""
        cache_key = self._get_cache_key(text)
        cache_file = self.cache_dir / f"{cache_key}.npy"
        
        if cache_file.exists():
            try:
                return np.load(cache_file)
            except Exception:
                # Remove corrupted cache file
                cache_file.unlink(missing_ok=True)
        
        return None
    
    def _cache_embedding(self, text: str, embedding: np.ndarray):
        """Cache embedding for future use"""
        cache_key = self._get_cache_key(text)
        cache_file = self.cache_dir / f"{cache_key}.npy"
        
        try:
            np.save(cache_file, embedding)
        except Exception:
            pass  # Silently fail caching

    def _get_ollama_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding using Ollama (if available)"""
        if not text or not isinstance(text, str):
            return None
            
        try:
            # Try to use ollama embeddings API if available
            result = subprocess.run([
                "ollama", "embeddings", self.model, text
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                # Parse JSON response if ollama supports embeddings
                try:
                    response = json.loads(result.stdout)
                    if 'embedding' in response:
                        return np.array(response['embedding'], dtype=np.float32)
                except json.JSONDecodeError:
                    pass
        except (subprocess.TimeoutExpired, FileNotFoundError, TypeError):
            pass
        
        # Fallback: use the model to generate a pseudo-embedding
        try:
            if not self.model:  # Add this check
                return None
                
            prompt = f"Convert this text to a numerical representation (space-separated numbers): {text}"
            result = subprocess.run([
                "ollama", "run", self.model, prompt
            ], capture_output=True, text=True, timeout=15)
            
            if result.returncode == 0:
                # Try to extract numbers from response
                numbers = re.findall(r'-?\d+\.?\d*', result.stdout)
                if len(numbers) >= 8:  # Minimum viable embedding size
                    embedding = np.array([float(n) for n in numbers[:128]], dtype=np.float32)  # Cap at 128 dims
                    return embedding / np.linalg.norm(embedding)  # Normalize
        except (subprocess.TimeoutExpired, FileNotFoundError, TypeError, ValueError):
            pass
        
        return None
    
    def _get_hash_embedding(self, text: str, dims: int = 128) -> np.ndarray:
        """Generate a hash-based embedding as fallback"""
        # Create multiple hash variations for better distribution
        hashes = []
        
        # Use more hash rounds to fill the dimensions properly
        rounds_needed = (dims + 15) // 16  # Each hash gives us 16 values
        
        for i in range(rounds_needed):
            hash_input = f"{text}:{i}"
            hash_obj = hashlib.md5(hash_input.encode())
            hash_bytes = hash_obj.digest()
            
            # Convert bytes to floats
            hash_floats = [b / 255.0 for b in hash_bytes]
            hashes.extend(hash_floats)
        
        # Ensure we have exactly the right number of dimensions
        embedding = np.array(hashes[:dims], dtype=np.float32)
        
        # Normalize the embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        else:
            # If norm is 0, create a small random embedding
            embedding = np.random.normal(0, 0.1, dims).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
    
    def compute_similarity_batch(self, query_embedding: np.ndarray, 
                                target_embeddings: List[np.ndarray]) -> List[float]:
        """Compute cosine similarity for multiple targets efficiently"""
        if not target_embeddings:
            return []
        
        # Stack all target embeddings
        targets = np.stack(target_embeddings)
        
        # Compute dot products
        dot_products = np.dot(targets, query_embedding)
        
        # Compute norms
        query_norm = np.linalg.norm(query_embedding)
        target_norms = np.linalg.norm(targets, axis=1)
        
        # Avoid division by zero
        valid_norms = (query_norm > 0) & (target_norms > 0)
        similarities = np.zeros(len(target_embeddings))
        
        if np.any(valid_norms):
            similarities[valid_norms] = dot_products[valid_norms] / (query_norm * target_norms[valid_norms])
        
        return similarities.tolist()
    
    def clear_cache(self):
        """Clear embedding cache"""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(exist_ok=True)
    
    def get_cache_info(self) -> dict:
        """Get information about the embedding cache"""
        if not self.cache_dir.exists():
            return {"files": 0, "size_mb": 0}
        
        files = list(self.cache_dir.glob("*.npy"))
        total_size = sum(f.stat().st_size for f in files)
        
        return {
            "files": len(files),
            "size_mb": round(total_size / (1024 * 1024), 2)
        }


def extract_command_features(command: str) -> dict:
    """Extract features from a command for better similarity matching"""
    features = {
        'has_pipe': '|' in command,
        'has_redirect': any(op in command for op in ['>', '>>', '<']),
        'has_sudo': command.strip().startswith('sudo'),
        'has_flags': ' -' in command,
        'word_count': len(command.split()),
        'char_count': len(command),
        'has_path': '/' in command or '\\' in command,
        'has_glob': any(char in command for char in ['*', '?', '[']),
        'command_name': command.split()[0] if command.strip() else ''
    }
    
    return features


def calculate_command_similarity(cmd1: str, cmd2: str) -> float:
    """Calculate similarity between two commands based on structure and features"""
    features1 = extract_command_features(cmd1)
    features2 = extract_command_features(cmd2)
    
    similarity = 0.0
    
    # Same command name gets high score
    if features1['command_name'] == features2['command_name']:
        similarity += 0.4
    
    # Similar structural features
    structural_features = ['has_pipe', 'has_redirect', 'has_sudo', 'has_flags', 'has_path', 'has_glob']
    matching_features = sum(1 for feat in structural_features if features1[feat] == features2[feat])
    similarity += (matching_features / len(structural_features)) * 0.3
    
    # Similar length
    len_diff = abs(features1['word_count'] - features2['word_count'])
    length_similarity = max(0, 1 - len_diff / 10)  # Penalty for length difference
    similarity += length_similarity * 0.2
    
    # Character overlap
    chars1 = set(cmd1.lower())
    chars2 = set(cmd2.lower())
    char_overlap = len(chars1.intersection(chars2)) / len(chars1.union(chars2)) if chars1.union(chars2) else 0
    similarity += char_overlap * 0.1
    
    return min(similarity, 1.0)
