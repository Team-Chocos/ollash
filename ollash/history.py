import sqlite3
from datetime import datetime
from pathlib import Path
import json
import os
import numpy as np
from typing import List, Tuple, Optional
import subprocess
import threading
import queue
from ollash.embedding_utils import EmbeddingManager, calculate_command_similarity


class HistoryLogger:
    """Manage per-user command history with advanced semantic search in ~/.ollash/history.db"""
    def __init__(self, model: str = "llama3"):
        home = Path.home()
        db_dir = home / ".ollash"
        db_dir.mkdir(exist_ok=True)
        self.db_path = db_dir / "history.db"
        self.embedding_manager = EmbeddingManager(model)
        self._init_db()
        
        self._shutdown = False
        # Background embedding processing
        self.embedding_queue = queue.Queue()
        self.embedding_thread = threading.Thread(target=self._process_embeddings, daemon=True)
        self.embedding_thread.start()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    input TEXT,
                    generated_command TEXT,
                    execution_result TEXT,
                    cwd TEXT,
                    tags TEXT,
                    embedding BLOB
                )
            """)
            
            # Add new columns if they don't exist (for backward compatibility)
            try:
                conn.execute("ALTER TABLE history ADD COLUMN tags TEXT")
            except sqlite3.OperationalError:
                pass
            
            try:
                conn.execute("ALTER TABLE history ADD COLUMN embedding BLOB")
            except sqlite3.OperationalError:
                pass

    def _get_embedding(self, text: str, model: str = "llama3") -> Optional[np.ndarray]:
        """Get embedding for text using the embedding manager"""
        if self.embedding_manager is None:
            return None
        return self.embedding_manager.get_embedding(text)

    def _serialize_embedding(self, embedding: np.ndarray) -> bytes:
        """Serialize numpy array to bytes for storage"""
        return embedding.tobytes()

    def _deserialize_embedding(self, blob: bytes) -> np.ndarray:
        """Deserialize bytes back to numpy array"""
        return np.frombuffer(blob, dtype=np.float32)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)

    def log(self, input_text: str, generated_command: str, execution_result: str, 
            cwd: str = None, tags: str = None, model: str = "llama3", 
            generate_embedding: bool = True):
        """Log command with optional asynchronous embedding generation"""
        timestamp = datetime.utcnow().isoformat()
        cwd = cwd or os.getcwd()
        
        # Insert the record immediately without embedding
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO history (timestamp, input, generated_command, execution_result, cwd, tags, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (timestamp, input_text, generated_command, execution_result, cwd, tags, None))
            
            record_id = cursor.lastrowid
        
        # Queue embedding generation for background processing if requested
        if generate_embedding:
            combined_text = f"{input_text} {generated_command}"
            self.embedding_queue.put((record_id, combined_text, model))
    
    def _process_embeddings(self):
        """Background thread to process embedding generation"""
        while not self._shutdown:
            try:
                # Wait for embedding tasks with timeout
                record_id, combined_text, model = self.embedding_queue.get(timeout=1.0)
                
                # Generate embedding with proper error handling
                try:
                    embedding = self._get_embedding(combined_text, model)

                    if embedding is not None:
                        embedding_blob = self._serialize_embedding(embedding)

                        # Update the record with the embedding
                        with sqlite3.connect(self.db_path) as conn:
                            conn.execute("""
                                UPDATE history SET embedding = ? WHERE id = ?
                            """, (embedding_blob, record_id))
                    else:
                        # Log when embedding generation returns None
                        print(f"Warning: No embedding generated for text: '{combined_text[:50]}...'")
                    
                except Exception as e:
                    print(f"Warning: Failed to generate embedding for record {record_id}: {e}")
                
                # Always mark task as done after attempting to process
                self.embedding_queue.task_done()
                
            except queue.Empty:
                continue  # Timeout, check shutdown flag and continue
            except Exception as e:
                # Log error but don't crash the thread
                print(f"Warning: Failed to process embedding task: {e}")
                try:
                    self.embedding_queue.task_done()
                except:
                    pass

    def search_similar(self, query: str, potential_command: str = "", 
                      limit: int = 5, model: str = "llama3") -> List[Tuple[dict, float]]:
        """Search for similar past entries using advanced semantic similarity"""
        # Create query embedding
        search_text = f"{query} {potential_command}".strip()
        query_embedding = self._get_embedding(search_text, model)
        
        if query_embedding is None:
            # Fallback to text-based search
            return self._text_search(query, limit)
        
        results = []
        valid_embeddings = []
        entries_data = []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT id, timestamp, input, generated_command, execution_result, cwd, tags, embedding
                FROM history 
                WHERE embedding IS NOT NULL
                ORDER BY timestamp DESC
                LIMIT 100
            """)
            
            for row in cursor.fetchall():
                if row[7]:  # embedding exists
                    try:
                        stored_embedding = self._deserialize_embedding(row[7])
                        
                        # Check if embeddings have compatible dimensions
                        if stored_embedding.shape != query_embedding.shape:
                            continue  # Skip incompatible embeddings
                            
                        valid_embeddings.append(stored_embedding)
                        
                        entry = {
                            'id': row[0],
                            'timestamp': row[1],
                            'input': row[2],
                            'generated_command': row[3],
                            'execution_result': row[4],
                            'cwd': row[5],
                            'tags': row[6]
                        }
                        entries_data.append(entry)
                    except Exception as e:
                        continue  # Skip corrupted embeddings
        
        if not valid_embeddings:
            return self._text_search(query, limit)
        
        # Compute similarities efficiently
        try:
            similarities = self.embedding_manager.compute_similarity_batch(
                query_embedding, valid_embeddings
            )
        except Exception as e:
            # Fallback to individual similarity computation
            similarities = []
            for embedding in valid_embeddings:
                try:
                    sim = self._cosine_similarity(query_embedding, embedding)
                    similarities.append(sim)
                except:
                    similarities.append(0.0)
        
        # Combine results and add command structure similarity
        for entry, similarity in zip(entries_data, similarities):
            # Add command structure similarity bonus
            cmd_similarity = calculate_command_similarity(
                potential_command, entry['generated_command']
            ) if potential_command else 0
            
            # Weighted combination of semantic and structural similarity
            combined_similarity = 0.7 * similarity + 0.3 * cmd_similarity
            
            # Boost recent successful commands
            if entry['execution_result'] == 'success':
                combined_similarity *= 1.1
            
            results.append((entry, combined_similarity))
        
        # Sort by combined similarity and return top results
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    def _text_search(self, query: str, limit: int = 5) -> List[Tuple[dict, float]]:
        """Fallback text-based search when embeddings are not available"""
        query_lower = query.lower()
        results = []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT id, timestamp, input, generated_command, execution_result, cwd, tags
                FROM history 
                ORDER BY timestamp DESC
                LIMIT 50
            """)
            
            for row in cursor.fetchall():
                input_text = row[2].lower()
                command_text = row[3].lower()
                
                # Simple text matching score
                score = 0.0
                if query_lower in input_text:
                    score += 0.8
                if query_lower in command_text:
                    score += 0.6
                
                # Word overlap scoring
                query_words = set(query_lower.split())
                input_words = set(input_text.split())
                command_words = set(command_text.split())
                
                input_overlap = len(query_words.intersection(input_words)) / max(len(query_words), 1)
                command_overlap = len(query_words.intersection(command_words)) / max(len(query_words), 1)
                
                score += input_overlap * 0.4 + command_overlap * 0.3
                
                if score > 0.1:  # Only include if there's some relevance
                    entry = {
                        'id': row[0],
                        'timestamp': row[1],
                        'input': row[2],
                        'generated_command': row[3],
                        'execution_result': row[4],
                        'cwd': row[5],
                        'tags': row[6]
                    }
                    results.append((entry, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    def get_context_summary(self, similar_entries: List[Tuple[dict, float]]) -> str:
        """Generate a context summary from similar entries"""
        if not similar_entries:
            return ""
        
        context_parts = []
        context_parts.append("# Similar past commands:")
        
        for i, (entry, similarity) in enumerate(similar_entries, 1):
            context_parts.append(f"{i}. Input: '{entry['input']}'")
            context_parts.append(f"   Command: {entry['generated_command']}")
            context_parts.append(f"   Result: {entry['execution_result']}")
            context_parts.append(f"   Directory: {entry['cwd']}")
            if entry.get('tags'):
                context_parts.append(f"   Tags: {entry['tags']}")
            context_parts.append("")
        
        return "\n".join(context_parts)

    def add_tags(self, entry_id: int, tags: str):
        """Add tags to an existing history entry"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE history SET tags = ? WHERE id = ?", (tags, entry_id))

    def get_cache_stats(self) -> dict:
        """Get embedding cache statistics"""
        return self.embedding_manager.get_cache_info()
    
    def clear_embedding_cache(self):
        """Clear the embedding cache"""
        self.embedding_manager.clear_cache()
    
    
    
    def shutdown(self):
        """Shutdown the background embedding thread"""
        self._shutdown = True
        if self.embedding_thread.is_alive():
            self.embedding_thread.join(timeout=2.0)
    
    def get_recent_entries(self, limit: int = 10) -> List[dict]:
        """Get recent history entries"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT id, timestamp, input, generated_command, execution_result, cwd, tags
                FROM history 
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
            
            entries = []
            for row in cursor.fetchall():
                entries.append({
                    'id': row[0],
                    'timestamp': row[1],
                    'input': row[2],
                    'generated_command': row[3],
                    'execution_result': row[4],
                    'cwd': row[5],
                    'tags': row[6]
                })
            
            return entries
