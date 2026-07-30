
import sqlite3
from sqlite3 import Connection
import os
from time import strftime
from typing import Optional
import logging
from string import Template
import json

from bioguider.utils.constants import DEFAULT_TOKEN_USAGE

logging = logging.getLogger(__name__)

SUMMARIZED_FILES_TABLE_NAME = "SummarizedFiles"

summarized_files_create_table_query = f"""
CREATE TABLE IF NOT EXISTS {SUMMARIZED_FILES_TABLE_NAME} (
    file_path VARCHAR(512),
    instruction TEXT,
    summarize_prompt TEXT,
    summarize_level INTEGER,
    summarized_text TEXT,
    content_hash TEXT,
    token_usage  VARCHAR(512),
    datetime TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%f', 'now')),
    UNIQUE (file_path, instruction, summarize_level, summarize_prompt)
);
"""
summarized_files_upsert_query = f"""
INSERT INTO {SUMMARIZED_FILES_TABLE_NAME}(file_path, instruction, summarize_level, summarize_prompt, summarized_text, content_hash, token_usage, datetime)
VALUES (?, ?, ?, ?, ?, ?, ?, strftime('%Y-%m-%d %H:%M:%f', 'now'))
ON CONFLICT(file_path, instruction, summarize_level, summarize_prompt) DO UPDATE SET summarized_text=excluded.summarized_text,
content_hash=excluded.content_hash,
token_usage=excluded.token_usage,
datetime=strftime('%Y-%m-%d %H:%M:%f', 'now');
"""
# Base lookup (content-agnostic — preserves the pre-content_hash behaviour when
# no hash is supplied). The `_with_hash` variant additionally requires the stored
# summary to have been produced from the exact same file content, so a changed
# file misses the cache and is re-summarized instead of returning a stale entry.
summarized_files_select_query = f"""
SELECT summarized_text, datetime FROM {SUMMARIZED_FILES_TABLE_NAME}
where file_path = ? and instruction = ? and summarize_level = ? and summarize_prompt=?;
"""
summarized_files_select_with_hash_query = f"""
SELECT summarized_text, datetime FROM {SUMMARIZED_FILES_TABLE_NAME}
where file_path = ? and instruction = ? and summarize_level = ? and summarize_prompt=? and content_hash = ?;
"""

class SummarizedFilesDb:
    def __init__(self, author: str, repo_name: str, data_folder: str = None):
        self.author = author
        self.repo_name = repo_name
        self.connection: Connection | None = None
        self.data_folder = data_folder

    def _ensure_tables(self) -> bool:
        if self.connection is None:
            return False
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                summarized_files_create_table_query
            )
            # Migrate pre-existing databases that were created before the
            # content_hash column existed. Rows added before the migration keep
            # content_hash = NULL, so a hashed lookup misses them and they are
            # transparently re-summarized (self-healing stale caches).
            cursor.execute(f"PRAGMA table_info({SUMMARIZED_FILES_TABLE_NAME})")
            columns = [row[1] for row in cursor.fetchall()]
            if "content_hash" not in columns:
                cursor.execute(
                    f"ALTER TABLE {SUMMARIZED_FILES_TABLE_NAME} ADD COLUMN content_hash TEXT"
                )
            self.connection.commit()
            return True
        except Exception as e:
            logging.error(e)
            return False
        
    def _connect_to_db(self) -> bool:
        if self.connection is not None:
            return True
        db_path = self.data_folder
        if db_path is None:
            db_path = os.environ.get("DATA_FOLDER", "./data")
        db_path = os.path.join(db_path, "databases")
        # Ensure the local path exists
        try:
            os.makedirs(db_path, exist_ok=True)
        except Exception as e:
            logging.error(e)
            return False
        db_path = os.path.join(db_path, f"{self.author}_{self.repo_name}_summarized_file.db")
        if not os.path.exists(db_path):
            try:
                with open(db_path, "w"):
                    pass
            except Exception as e:
                logging.error(e)
                return False
        self.connection = sqlite3.connect(db_path)
        return True
    
    def upsert_summarized_file(
        self,
        file_path: str,
        instruction: str,
        summarize_level: int,
        summarize_prompt: str,
        summarized_text: str,
        token_usage: dict | None = None,
        content_hash: str | None = None,
    ):
        token_usage = token_usage if token_usage is not None else {**DEFAULT_TOKEN_USAGE}
        token_usage = json.dumps(token_usage)
        res = self._connect_to_db()
        assert res
        res = self._ensure_tables()
        assert res
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                summarized_files_upsert_query,
                (file_path, instruction, summarize_level, summarize_prompt, summarized_text, content_hash, token_usage, )
            )
            self.connection.commit()
            return True
        except Exception as e:
            logging.error(e)
            return False
        finally:
            self.connection.close()
            self.connection = None

    def select_summarized_text(
        self,
        file_path: str,
        instruction: str,
        summarize_level: int,
        summarize_prompt: str = "N/A",
        content_hash: str | None = None,
    ) -> str | None:
        """Return a cached summary, or None if there is no matching entry.

        When ``content_hash`` is provided, the cached summary must have been
        produced from file content with the same hash; otherwise the lookup
        misses and the caller re-summarizes. Passing ``None`` preserves the
        legacy content-agnostic behaviour (used by callers that do not track
        content, and by pre-migration rows).
        """
        self._connect_to_db()
        self._ensure_tables()
        try:
            cursor = self.connection.cursor()
            if content_hash is None:
                cursor.execute(
                    summarized_files_select_query,
                    (file_path, instruction, summarize_level, summarize_prompt,)
                )
            else:
                cursor.execute(
                    summarized_files_select_with_hash_query,
                    (file_path, instruction, summarize_level, summarize_prompt, content_hash,)
                )
            row = cursor.fetchone()
            if row is None:
                return None
            return row[0]
        except Exception as e:
            logging.error(e)
            return None
        finally:
            self.connection.close()
            self.connection = None
        
    def get_db_file(self):
        """Get the database file path (matches the file opened by _connect_to_db).

        Must mirror the path built in _connect_to_db(), otherwise callers
        (e.g. test teardown) operate on a file that does not exist.
        """
        db_path = self.data_folder
        if db_path is None:
            db_path = os.environ.get("DATA_FOLDER", "./data")
        db_path = os.path.join(db_path, "databases")
        return os.path.join(db_path, f"{self.author}_{self.repo_name}_summarized_file.db")


