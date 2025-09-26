import sqlite3
from sqlite3 import Connection
import os
from time import strftime
from typing import Optional, List, Dict, Any
import logging
import json

logging = logging.getLogger(__name__)

CODE_STRUCTURE_TABLE_NAME = "SourceCodeStructure"

code_structure_create_table_query = f"""
CREATE TABLE IF NOT EXISTS {CODE_STRUCTURE_TABLE_NAME} (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(256) NOT NULL,
    path VARCHAR(512) NOT NULL,
    start_lineno INTEGER NOT NULL,
    end_lineno INTEGER NOT NULL,
    parent VARCHAR(256),
    doc_string TEXT,
    params TEXT,
    reference_to TEXT,
    reference_by TEXT,
    datetime TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%f', 'now')),
    UNIQUE (name, path, start_lineno, end_lineno, parent)
);
"""

code_structure_insert_query = f"""
INSERT INTO {CODE_STRUCTURE_TABLE_NAME}(name, path, start_lineno, end_lineno, parent, doc_string, params, reference_to, reference_by, datetime)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, strftime('%Y-%m-%d %H:%M:%f', 'now'))
ON CONFLICT(name, path, start_lineno, end_lineno, parent) DO UPDATE SET doc_string=excluded.doc_string, params=excluded.params, 
reference_to=excluded.reference_to, reference_by=excluded.reference_by, datetime=strftime('%Y-%m-%d %H:%M:%f', 'now');
"""

code_structure_select_by_path_query = f"""
SELECT id, name, path, start_lineno, end_lineno, parent, doc_string, params, reference_to, reference_by, datetime 
FROM {CODE_STRUCTURE_TABLE_NAME} 
WHERE path = ?;
"""

code_structure_select_by_name_query = f"""
SELECT id, name, path, start_lineno, end_lineno, parent, doc_string, params, reference_to, reference_by, datetime 
FROM {CODE_STRUCTURE_TABLE_NAME} 
WHERE name = ?;
"""

code_structure_select_by_name_and_path_query = f"""
SELECT id, name, path, start_lineno, end_lineno, parent, doc_string, params, reference_to, reference_by, datetime 
FROM {CODE_STRUCTURE_TABLE_NAME} 
WHERE name = ? AND path = ?;
"""

code_structure_select_by_id_query = f"""
SELECT id, name, path, start_lineno, end_lineno, parent, doc_string, params, reference_to, reference_by, datetime 
FROM {CODE_STRUCTURE_TABLE_NAME} 
WHERE id = ?;
"""

code_structure_select_by_parent_and_parentpath_query = f"""
SELECT id, name, path, start_lineno, end_lineno, parent, doc_string, params, reference_to, reference_by, datetime 
FROM {CODE_STRUCTURE_TABLE_NAME} 
WHERE parent = ? AND path = ?;
"""

code_structure_select_by_parent_query = f"""
SELECT id, name, path, start_lineno, end_lineno, parent, doc_string, params, reference_to, reference_by, datetime 
FROM {CODE_STRUCTURE_TABLE_NAME} 
WHERE parent = ?;
"""

code_structure_update_query = f"""
UPDATE {CODE_STRUCTURE_TABLE_NAME} 
SET name = ?, path = ?, start_lineno = ?, end_lineno = ?, parent = ?, doc_string = ?, params = ?, reference_to = ?, reference_by = ?, datetime = strftime('%Y-%m-%d %H:%M:%f', 'now')
WHERE id = ?;
"""

code_structure_delete_query = f"""
DELETE FROM {CODE_STRUCTURE_TABLE_NAME} WHERE id = ?;
"""

code_structure_select_by_name_and_parent_and_path_query = f"""
SELECT id, name, path, start_lineno, end_lineno, parent, doc_string, params, reference_to, reference_by, datetime 
FROM {CODE_STRUCTURE_TABLE_NAME} 
WHERE name = ? AND parent = ? AND path = ?;
"""

code_structure_select_by_name_and_parent_query = f"""
SELECT id, name, path, start_lineno, end_lineno, parent, doc_string, params, reference_to, reference_by, datetime 
FROM {CODE_STRUCTURE_TABLE_NAME} 
WHERE name = ? AND parent = ?;
"""

class CodeStructureDb:
    def __init__(self, author: str, repo_name: str, data_folder: str = None):
        self.author = author
        self.repo_name = repo_name
        self.data_folder = data_folder
        self.connection: Connection | None = None

    def _ensure_tables(self) -> bool:
        if self.connection is None:
            return False
        try:
            cursor = self.connection.cursor()
            cursor.execute(code_structure_create_table_query)
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
        if not os.path.exists(db_path):
            try:
                os.makedirs(db_path, exist_ok=True)
            except Exception as e:
                logging.error(e)
                return False        
        db_path = os.path.join(db_path, f"{self.author}_{self.repo_name}_code_structure.db")
        if not os.path.exists(db_path):
            try:
                with open(db_path, "w"):
                    pass
            except Exception as e:
                logging.error(e)
                return False
        self.connection = sqlite3.connect(db_path)
        return True
    
    def is_database_built(self) -> bool:
        res = self._connect_to_db()
        if not res:
            return False
        res = self._ensure_tables()
        if not res:
            return False
        try:
            cursor = self.connection.cursor()
            cursor.execute(f"SELECT * FROM {CODE_STRUCTURE_TABLE_NAME}")
            return cursor.fetchone() is not None
        except Exception as e:
            logging.error(e)
            return False
        finally:
            self.connection.close()
            self.connection = None

    def insert_code_structure(
        self,
        name: str,
        path: str,
        start_lineno: int,
        end_lineno: int,
        parent: str = None,
        doc_string: str = None,
        params: str = None,
        reference_to: str = None,
        reference_by: str = None
    ) -> bool:
        """Insert a new code structure entry into the database."""
        if parent is None:
            parent = ""
        if path is None:
            path = ""
        res = self._connect_to_db()
        if not res:
            return False
        res = self._ensure_tables()
        if not res:
            return False
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                code_structure_insert_query, 
                (name, path, start_lineno, end_lineno, parent, doc_string, json.dumps(params) if params is not None else None, reference_to, reference_by)
            )
            self.connection.commit()
            return True
        except Exception as e:
            logging.error(e)
            return False
        finally:
            self.connection.close()
            self.connection = None

    def select_by_path(self, path: str) -> List[Dict[str, Any]]:
        """Select all code structures by file path."""
        res = self._connect_to_db()
        if not res:
            return []
        res = self._ensure_tables()
        if not res:
            return []
        try:
            cursor = self.connection.cursor()
            cursor.execute(code_structure_select_by_path_query, (path,))
            rows = cursor.fetchall()
            return [
                {
                    "id": row[0],
                    "name": row[1],
                    "path": row[2],
                    "start_lineno": row[3],
                    "end_lineno": row[4],
                    "parent": row[5],
                    "doc_string": row[6],
                    "params": row[7],
                    "reference_to": row[8],
                    "reference_by": row[9],
                    "datetime": row[10]
                }
                for row in rows
            ]
        except Exception as e:
            logging.error(e)
            return []
        finally:
            self.connection.close()
            self.connection = None

    def select_by_name(self, name: str) -> List[Dict[str, Any]]:
        """Select all code structures by name."""
        res = self._connect_to_db()
        if not res:
            return []
        res = self._ensure_tables()
        if not res:
            return []
        try:
            cursor = self.connection.cursor()
            cursor.execute(code_structure_select_by_name_query, (name,))
            rows = cursor.fetchall()
            return [
                {
                    "id": row[0],
                    "name": row[1],
                    "path": row[2],
                    "start_lineno": row[3],
                    "end_lineno": row[4],
                    "parent": row[5],
                    "doc_string": row[6],
                    "params": row[7],
                    "reference_to": row[8],
                    "reference_by": row[9],
                    "datetime": row[10]
                }
                for row in rows
            ]
        except Exception as e:
            logging.error(e)
            return []
        finally:
            self.connection.close()
            self.connection = None

    def select_by_name_and_path(self, name: str, path: str) -> Optional[Dict[str, Any]]:
        """Select a code structure by name and path."""
        res = self._connect_to_db()
        if not res:
            return None
        res = self._ensure_tables()
        if not res:
            return None
        try:
            cursor = self.connection.cursor()
            cursor.execute(code_structure_select_by_name_and_path_query, (name, path))
            row = cursor.fetchone()
            if row is None:
                return None
            return {
                "id": row[0],
                "name": row[1],
                "path": row[2],
                "start_lineno": row[3],
                "end_lineno": row[4],
                "parent": row[5],
                "doc_string": row[6],
                "params": row[7],
                "reference_to": row[8],
                "reference_by": row[9],
                "datetime": row[10]
            }
        except Exception as e:
            logging.error(e)
            return None
        finally:
            self.connection.close()
            self.connection = None

    def select_by_name_and_parent(self, name: str, parent: str) -> List[Dict[str, Any]]:
        """Select all code structures by name and parent."""
        res = self._connect_to_db()
        if not res:
            return []
        res = self._ensure_tables()
        if not res:
            return []
        try:
            cursor = self.connection.cursor()
            cursor.execute(code_structure_select_by_name_and_parent_query, (name, parent))
            rows = cursor.fetchall()
            return [
                {
                    "id": row[0],
                    "name": row[1],
                    "path": row[2],
                    "start_lineno": row[3],
                    "end_lineno": row[4],
                    "parent": row[5],
                    "doc_string": row[6],
                    "params": row[7],
                    "reference_to": row[8],
                    "reference_by": row[9],
                    "datetime": row[10]
                }
                for row in rows
            ]
        except Exception as e:
            logging.error(e)
            return []
        finally:
            self.connection.close()
            self.connection = None


    def select_by_name_and_parent_and_path(self, name: str, parent: str, path: str) -> Optional[Dict[str, Any]]:
        """Select a code structure by name and parent."""
        res = self._connect_to_db()
        if not res:
            return None
        res = self._ensure_tables()
        if not res:
            return None
        try:
            cursor = self.connection.cursor()
            cursor.execute(code_structure_select_by_name_and_parent_and_path_query, (name, parent, path))
            row = cursor.fetchone()
            if row is None:
                return None
            return {
                "id": row[0],
                "name": row[1],
                "path": row[2],
                "start_lineno": row[3],
                "end_lineno": row[4],
                "parent": row[5],
                "doc_string": row[6],
                "params": row[7],
                "reference_to": row[8],
                "reference_by": row[9],
                "datetime": row[10]
            }
        except Exception as e:
            logging.error(e)
            return None
        finally:
            self.connection.close()
            self.connection = None

    def select_by_id(self, id: int) -> Optional[Dict[str, Any]]:
        """Select a code structure by ID."""
        res = self._connect_to_db()
        if not res:
            return None
        res = self._ensure_tables()
        if not res:
            return None
        try:
            cursor = self.connection.cursor()
            cursor.execute(code_structure_select_by_id_query, (id,))
            row = cursor.fetchone()
            if row is None:
                return None
            return {
                "id": row[0],
                "name": row[1],
                "path": row[2],
                "start_lineno": row[3],
                "end_lineno": row[4],
                "parent": row[5],
                "doc_string": row[6],
                "params": row[7],
                "reference_to": row[8],
                "reference_by": row[9],
                "datetime": row[10]
            }
        except Exception as e:
            logging.error(e)
            return None
        finally:
            self.connection.close()
            self.connection = None

    def update_code_structure(
        self,
        id: int,
        name: str,
        path: str,
        start_lineno: int,
        end_lineno: int,
        parent: str = None,
        doc_string: str = None,
        params: str = None,
        reference_to: str = None,
        reference_by: str = None
    ) -> bool:
        """Update an existing code structure entry."""
        res = self._connect_to_db()
        if not res:
            return False
        res = self._ensure_tables()
        if not res:
            return False
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                code_structure_update_query, 
                (name, path, start_lineno, end_lineno, parent, doc_string, params, reference_to, reference_by, id)
            )
            self.connection.commit()
            return cursor.rowcount > 0
        except Exception as e:
            logging.error(e)
            return False
        finally:
            self.connection.close()
            self.connection = None

    def select_by_parent(self, parent: str, path: str | None = None) -> List[Dict[str, Any]]:
        """Select all code structures by parent."""
        res = self._connect_to_db()
        if not res:
            return []
        res = self._ensure_tables()
        if not res:
            return []
        try:
            cursor = self.connection.cursor()
            if path is not None:
                cursor.execute(code_structure_select_by_parent_and_parentpath_query, (parent, path))
            else:
                cursor.execute(code_structure_select_by_parent_query, (parent,))
            rows = cursor.fetchall()
            return [
                {
                    "id": row[0],
                    "name": row[1],
                    "path": row[2],
                    "start_lineno": row[3],
                    "end_lineno": row[4],
                    "parent": row[5],
                    "doc_string": row[6],
                    "params": row[7],
                    "reference_to": row[8],
                    "reference_by": row[9],
                    "datetime": row[10]
                }
                for row in rows
            ]
        except Exception as e:
            logging.error(e)
            return []
        finally:
            self.connection.close()
            self.connection = None

    def delete_code_structure(self, id: int) -> bool:
        """Delete a code structure entry by ID."""
        res = self._connect_to_db()
        if not res:
            return False
        res = self._ensure_tables()
        if not res:
            return False
        try:
            cursor = self.connection.cursor()
            cursor.execute(code_structure_delete_query, (id,))
            self.connection.commit()
            return cursor.rowcount > 0
        except Exception as e:
            logging.error(e)
            return False
        finally:
            self.connection.close()
            self.connection = None

    def get_db_file(self) -> str:
        """Get the database file path."""
        db_path = os.environ.get("DATA_FOLDER", "./data")
        db_path = os.path.join(db_path, "databases")
        db_path = os.path.join(db_path, f"{self.author}_{self.repo_name}.db")
        return db_path
