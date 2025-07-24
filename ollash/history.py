import sqlite3
from datetime import datetime
from pathlib import Path
import json
import os


class HistoryLogger:
    """Manage per-user command history in ~/.ollash/history.db"""
    def __init__(self):
        home = Path.home()
        db_dir = home / ".ollash"
        db_dir.mkdir(exist_ok=True)
        self.db_path = db_dir / "history.db"
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    input TEXT,
                    generated_command TEXT,
                    execution_result TEXT,
                    cwd TEXT
                )
            """)

    def log(self, input_text, generated_command, execution_result, cwd=None):
        timestamp = datetime.utcnow().isoformat()
        cwd = cwd or os.getcwd()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO history (timestamp, input, generated_command, execution_result, cwd)
                VALUES (?, ?, ?, ?, ?)
            """, (timestamp, input_text, generated_command, execution_result, cwd))
