#ollash/history.py
import sqlite3
from datetime import datetime
from pathlib import Path
import os
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import threading
import queue
import faiss
import pickle
import time

class OptimizedHistoryLogger:
    """Ultra-fast offline command history with llama3 embeddings and FAISS search"""
    
    def __init__(self):
        # Initialize storage paths
        home = Path.home()
        self.db_dir = home / ".ollash"
        self.db_dir.mkdir(exist_ok=True)
        
        # Database for metadata
        self.db_path = self.db_dir / "history.db"
        self._init_db()
        
        # Embeddings storage
        self.embeddings_path = self.db_dir / "embeddings.faiss"
        self.mapping_path = self.db_dir / "id_mapping.pkl"
        
        # FAISS index and ID mapping
        self.faiss_index = None
        self.id_to_idx = {}
        self.idx_to_id = []
        self._load_faiss_index()
        
        # Background processing
        self._shutdown = False
        self.embedding_queue = queue.Queue(maxsize=1000)
        self.embedding_thread = threading.Thread(
            target=self._process_embeddings, 
            daemon=True,
            name="EmbeddingProcessor"
        )
        self.embedding_thread.start()
        
        # Track pending embeddings
        self.pending_embeddings = set()

    def _init_db(self):
        """Initialize SQLite database with optimized schema"""
        with sqlite3.connect(self.db_path) as conn:
            # Main history table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    input TEXT,
                    generated_command TEXT,
                    execution_result TEXT,
                    cwd TEXT,
                    tags TEXT
                )
            """)
            
            # Try to add has_embedding column if it doesn't exist
            try:
                conn.execute("ALTER TABLE history ADD COLUMN has_embedding INTEGER DEFAULT 0")
            except sqlite3.OperationalError:
                pass  # Column already exists
            
            # Indexes for fast queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON history(timestamp DESC)
            """)
            
            try:
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_embedding_flag 
                    ON history(has_embedding) 
                    WHERE has_embedding = 1
                """)
            except sqlite3.OperationalError:
                pass  # Older SQLite might not support WHERE in index
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_execution_result 
                ON history(execution_result)
            """)

    def _load_faiss_index(self):
        """Load existing FAISS index or create new"""
        try:
            if self.embeddings_path.exists():
                self.faiss_index = faiss.read_index(str(self.embeddings_path))
                
            if self.mapping_path.exists():
                with open(self.mapping_path, 'rb') as f:
                    self.id_to_idx, self.idx_to_id = pickle.load(f)
            else:
                self.id_to_idx = {}
                self.idx_to_id = []
                
            if self.faiss_index is None:
                # Initialize empty index (dimension will be set at first addition)
                self.faiss_index = None
                
        except Exception as e:
            print(f"Warning: Failed to load FAISS index: {e}")
            self.faiss_index = None
            self.id_to_idx = {}
            self.idx_to_id = []

    def _save_faiss_index(self):
        """Persist FAISS index to disk"""
        try:
            if self.faiss_index is not None:
                faiss.write_index(self.faiss_index, str(self.embeddings_path))
            with open(self.mapping_path, 'wb') as f:
                pickle.dump((self.id_to_idx, self.idx_to_id), f)
        except Exception as e:
            print(f"Warning: Failed to save FAISS index: {e}")

    def log(self, 
            input_text: str, 
            generated_command: str, 
            execution_result: str,
            cwd: str = None, 
            tags: str = None,
            generate_embedding: bool = True) -> int:
        """Log a command with optional async embedding generation"""
        timestamp = datetime.utcnow().isoformat()
        cwd = cwd or os.getcwd()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO history (
                    timestamp, input, generated_command, 
                    execution_result, cwd, tags, has_embedding
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (timestamp, input_text, generated_command, 
                 execution_result, cwd, tags, 0))
            
            record_id = cursor.lastrowid
        
        if generate_embedding:
            combined_text = f"{input_text} {generated_command}"
            self.embedding_queue.put((record_id, combined_text))
            self.pending_embeddings.add(record_id)
            
        return record_id

    def _process_embeddings(self):
        """Background thread for generating and storing embeddings"""
        try:
            from ollash.embedding_utils import EmbeddingManager
        except ImportError:
            print("Warning: embedding_utils not available, skipping embedding processing")
            return
        
        embedding_manager = EmbeddingManager("llama3")
        
        while not self._shutdown:
            try:
                record_id, combined_text = self.embedding_queue.get(timeout=1.0)
                
                try:
                    # Generate embedding (this is the slow part)
                    start_time = time.time()
                    embedding = embedding_manager.get_embedding(combined_text)
                    
                    if embedding is not None:
                        # Convert to numpy array if needed
                        embedding = np.array(embedding, dtype='float32').flatten()
                        
                        # Initialize FAISS index if this is our first embedding
                        if self.faiss_index is None:
                            dim = embedding.shape[0]
                            self.faiss_index = faiss.IndexFlatIP(dim)
                        
                        # Add to FAISS index
                        embedding = embedding.reshape(1, -1)
                        faiss.normalize_L2(embedding)
                        self.faiss_index.add(embedding)
                        
                        # Update ID mapping
                        new_idx = len(self.idx_to_id)
                        self.id_to_idx[record_id] = new_idx
                        self.idx_to_id.append(record_id)
                        
                        # Mark as embedded in database
                        with sqlite3.connect(self.db_path) as conn:
                            conn.execute("""
                                UPDATE history 
                                SET has_embedding = 1 
                                WHERE id = ?
                            """, (record_id,))
                            
                        # Periodically save index
                        if len(self.idx_to_id) % 100 == 0:
                            self._save_faiss_index()
                            
                    self.pending_embeddings.discard(record_id)
                    
                except Exception as e:
                    print(f"Error generating embedding: {e}")
                    self.pending_embeddings.discard(record_id)
                
                finally:
                    self.embedding_queue.task_done()
                    
            except queue.Empty:
                continue

    def search_similar(self, 
                      query: str, 
                      potential_command: str = "", 
                      limit: int = 5) -> List[Tuple[Dict, float]]:
        """Fast semantic search with hybrid scoring"""
        # Fallback if no embeddings exist
        if self.faiss_index is None or not self.idx_to_id:
            return self._text_search(query, limit)
        
        # Generate query embedding
        try:
            from ollash.embedding_utils import EmbeddingManager, calculate_command_similarity
            embedding_manager = EmbeddingManager("llama3")
        except ImportError:
            print("Warning: embedding_utils not available, falling back to text search")
            return self._text_search(query, limit)
        
        search_text = f"{query} {potential_command}".strip()
        query_embedding = embedding_manager.get_embedding(search_text)
        
        if query_embedding is None:
            return self._text_search(query, limit)
            
        query_embedding = np.array(query_embedding, dtype='float32').reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        # Search FAISS index
        distances, indices = self.faiss_index.search(query_embedding, limit)
        
        # Retrieve metadata for results
        results = []
        with sqlite3.connect(self.db_path) as conn:
            for idx, distance in zip(indices[0], distances[0]):
                if idx == -1:
                    continue
                    
                record_id = self.idx_to_id[idx]
                cursor = conn.execute("""
                    SELECT id, timestamp, input, generated_command, 
                           execution_result, cwd, tags
                    FROM history 
                    WHERE id = ?
                """, (record_id,))
                
                row = cursor.fetchone()
                if row:
                    entry = {
                        'id': row[0],
                        'timestamp': row[1],
                        'input': row[2],
                        'generated_command': row[3],
                        'execution_result': row[4],
                        'cwd': row[5],
                        'tags': row[6]
                    }
                    
                    # Calculate command structure similarity if available
                    cmd_sim = 0
                    if potential_command:
                        try:
                            cmd_sim = calculate_command_similarity(
                                potential_command, 
                                entry['generated_command']
                            )
                        except:
                            cmd_sim = 0
                    
                    # Combined score (70% semantic, 30% structure)
                    semantic_score = (1 - distance)  # Convert distance to similarity
                    combined_score = 0.7 * semantic_score + 0.3 * cmd_sim
                    
                    # Boost successful commands
                    if entry['execution_result'] == 'success':
                        combined_score *= 1.1
                        
                    results.append((entry, combined_score))
        
        return sorted(results, key=lambda x: x[1], reverse=True)[:limit]

    def _text_search(self, query: str, limit: int = 5) -> List[Tuple[Dict, float]]:
        """Fallback text-based search"""
        query_lower = query.lower()
        results = []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT id, timestamp, input, generated_command, 
                       execution_result, cwd, tags
                FROM history 
                ORDER BY timestamp DESC
                LIMIT 50
            """)
            
            for row in cursor.fetchall():
                input_text = row[2].lower()
                command_text = row[3].lower()
                
                # Simple text matching
                score = 0.0
                if query_lower in input_text:
                    score += 0.8
                if query_lower in command_text:
                    score += 0.6
                
                # Word overlap
                query_words = set(query_lower.split())
                input_words = set(input_text.split())
                cmd_words = set(command_text.split())
                
                input_overlap = len(query_words & input_words) / max(len(query_words), 1)
                cmd_overlap = len(query_words & cmd_words) / max(len(query_words), 1)
                
                score += 0.4 * input_overlap + 0.3 * cmd_overlap
                
                if score > 0.1:
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
        
        return sorted(results, key=lambda x: x[1], reverse=True)[:limit]

    def get_recent_entries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most recent history entries"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT id, timestamp, input, generated_command, 
                       execution_result, cwd, tags, has_embedding
                FROM history 
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
            
            return [{
                'id': row['id'],
                'timestamp': row['timestamp'],
                'input': row['input'],
                'generated_command': row['generated_command'],
                'execution_result': row['execution_result'],
                'cwd': row['cwd'],
                'tags': row['tags'],
                'has_embedding': bool(row['has_embedding'])
            } for row in cursor.fetchall()]

    def shutdown(self):
        """Complete shutdown procedure with proper resource cleanup"""
        if not hasattr(self, '_shutdown'):
            return
            
        self._shutdown = True
        
        # Wait for embedding thread to finish if it exists
        if hasattr(self, 'embedding_thread') and self.embedding_thread.is_alive():
            # First try to process remaining items
            while not self.embedding_queue.empty() and self.embedding_thread.is_alive():
                time.sleep(0.1)
            
            # Then wait for thread to finish
            self.embedding_thread.join(timeout=2.0)
        
        # Ensure FAISS index is saved
        try:
            self._save_faiss_index()
        except Exception as e:
            print(f"Warning: Failed to save FAISS index during shutdown: {e}")
        
        # Clean up any other resources if needed
        if hasattr(self, 'pending_embeddings'):
            self.pending_embeddings.clear()

    def __del__(self):
        """Destructor to ensure proper cleanup"""
        try:
            self.shutdown()
        except Exception:
            pass  # Prevent exceptions during garbage collection