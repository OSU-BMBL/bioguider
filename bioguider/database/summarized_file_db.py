
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
    token_usage  VARCHAR(512),
    datetime TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%f', 'now')),
    UNIQUE (file_path, instruction, summarize_level, summarize_prompt)
);
"""
summarized_files_upsert_query = f"""
INSERT INTO {SUMMARIZED_FILES_TABLE_NAME}(file_path, instruction, summarize_level, summarize_prompt, summarized_text, token_usage, datetime)
VALUES (?, ?, ?, ?, ?, ?, strftime('%Y-%m-%d %H:%M:%f', 'now'))
ON CONFLICT(file_path, instruction, summarize_level, summarize_prompt) DO UPDATE SET summarized_text=excluded.summarized_text,
datetime=strftime('%Y-%m-%d %H:%M:%f', 'now');
"""
summarized_files_select_query = f"""
SELECT summarized_text, datetime FROM {SUMMARIZED_FILES_TABLE_NAME} 
where file_path = ? and instruction = ? and summarize_level = ? and summarize_prompt=?;
"""

class SummarizedFilesDb:
    def __init__(self, author: str, repo_name: str):
        self.author = author
        self.repo_name = repo_name
        self.connection: Connection | None = None

    def _ensure_tables(self) -> bool:
        if self.connection is None:
            return False
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                summarized_files_create_table_query
            )
            self.connection.commit()
            return True
        except Exception as e:
            logging.error(e)
            return False
        
    def _connect_to_db(self) -> bool:
        if self.connection is not None:
            return True
        db_path = os.environ.get("DATA_FOLDER", "./data")
        db_path = os.path.join(db_path, "databases")
        # Ensure the local path exists
        try:
            os.makedirs(db_path, exist_ok=True)
        except Exception as e:
            logging.error(e)
            return False
        db_path = os.path.join(db_path, f"{self.author}_{self.repo_name}.db")
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
        token_usage: dict | None = None
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
                (file_path, instruction, summarize_level, summarize_prompt, summarized_text, token_usage, )
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
        summarize_prompt: str,
    ) -> str | None:
        self._connect_to_db()
        self._ensure_tables()
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                summarized_files_select_query, 
                (file_path, instruction, summarize_level, summarize_prompt,)
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
        db_path = os.environ.get("DATA_FOLDER", "./data")
        db_path = os.path.join(db_path, f"{self.author}_{self.repo_name}.db")
        return db_path


