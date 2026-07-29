
import pytest
import unittest
import os
import sqlite3
import tempfile

from bioguider.database.summarized_file_db import (
    SummarizedFilesDb,
    SUMMARIZED_FILES_TABLE_NAME,
)

class SummarizedFilesDbTestCase(unittest.TestCase):
    def setUp(self):
        # Use a private temp folder so the on-disk sqlite file can't be
        # contaminated by (or leak into) other tests that write under ./data.
        self.tmp = tempfile.TemporaryDirectory()
        self.db = SummarizedFilesDb(
            author="foo",
            repo_name="bar",
            data_folder=self.tmp.name,
        )
        res = self.db.upsert_summarized_file(
            "111/222/333",
            "",
            3,
            "N/A",                                # summarize_prompt
            "balahbalah balahbalah balahbalah",   # summarized_text
        )
    def tearDown(self):
        self.tmp.cleanup()

    def test_upsert(self):
        res = self.db.upsert_summarized_file(
            "aaa/bbb/ccc",
            "",
            3,
            "N/A",                                # summarize_prompt
            "balahbalah balahbalah balahbalah",   # summarized_text
        )
        self.assertTrue(res)

    def test_select(self):
        text = self.db.select_summarized_text(
            "111/222/333",
            "",
            3, 
        )
        self.assertEqual(text, "balahbalah balahbalah balahbalah")

    def test_insert_with_token_usage(self):
        token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        res = self.db.upsert_summarized_file(
            "123/456/789",
            "",
            3,
            "N/A",          # summarize_prompt
            "balahbalah",   # summarized_text
            token_usage,
        )
        self.assertTrue(res)
        res = self.db.select_summarized_text(
            "123/456/789",
            "",
            3
        )
        self.assertEqual(res, "balahbalah")

    # --- content-hash cache invalidation (#1) --------------------------------

    def test_select_with_matching_hash_hits(self):
        self.db.upsert_summarized_file(
            "hash/path", "", 6, "N/A", "summary-v1",
            token_usage=None, content_hash="hash-v1",
        )
        got = self.db.select_summarized_text(
            "hash/path", "", 6, summarize_prompt="N/A", content_hash="hash-v1",
        )
        self.assertEqual(got, "summary-v1")

    def test_select_with_changed_hash_misses(self):
        """A file whose content changed (new hash) must not return the stale summary."""
        self.db.upsert_summarized_file(
            "hash/path", "", 6, "N/A", "summary-v1",
            token_usage=None, content_hash="hash-v1",
        )
        got = self.db.select_summarized_text(
            "hash/path", "", 6, summarize_prompt="N/A", content_hash="hash-v2",
        )
        self.assertIsNone(got)

    def test_reupsert_refreshes_hash_and_text(self):
        """Re-summarizing the same key overwrites in place (no stale row lingers)."""
        self.db.upsert_summarized_file(
            "hash/path", "", 6, "N/A", "summary-v1",
            token_usage=None, content_hash="hash-v1",
        )
        # content changed -> caller re-summarizes and upserts with the new hash
        self.db.upsert_summarized_file(
            "hash/path", "", 6, "N/A", "summary-v2",
            token_usage=None, content_hash="hash-v2",
        )
        # old hash no longer matches
        self.assertIsNone(
            self.db.select_summarized_text(
                "hash/path", "", 6, summarize_prompt="N/A", content_hash="hash-v1",
            )
        )
        # new hash returns the refreshed summary
        self.assertEqual(
            self.db.select_summarized_text(
                "hash/path", "", 6, summarize_prompt="N/A", content_hash="hash-v2",
            ),
            "summary-v2",
        )

    def test_hashless_lookup_is_backward_compatible(self):
        """Callers that don't track content still match on the legacy key."""
        self.db.upsert_summarized_file(
            "hash/path", "", 6, "N/A", "summary-v1",
            token_usage=None, content_hash="hash-v1",
        )
        got = self.db.select_summarized_text("hash/path", "", 6)
        self.assertEqual(got, "summary-v1")


class SummarizedFilesDbMigrationTestCase(unittest.TestCase):
    """A database created before the content_hash column existed must be
    migrated in place, and its pre-migration rows must self-heal (a hashed
    lookup misses them, forcing a re-summarize)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = SummarizedFilesDb(
            author="old", repo_name="schema", data_folder=self.tmp.name,
        )
        db_path = self.db.get_db_file()
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        # Create the OLD schema (no content_hash column) and seed a row.
        conn = sqlite3.connect(db_path)
        conn.execute(
            f"""
            CREATE TABLE {SUMMARIZED_FILES_TABLE_NAME} (
                file_path VARCHAR(512),
                instruction TEXT,
                summarize_prompt TEXT,
                summarize_level INTEGER,
                summarized_text TEXT,
                token_usage VARCHAR(512),
                datetime TEXT,
                UNIQUE (file_path, instruction, summarize_level, summarize_prompt)
            );
            """
        )
        conn.execute(
            f"INSERT INTO {SUMMARIZED_FILES_TABLE_NAME}"
            f"(file_path, instruction, summarize_prompt, summarize_level, summarized_text) "
            f"VALUES (?, ?, ?, ?, ?)",
            ("legacy/readme", "", "N/A", 6, "stale-summary"),
        )
        conn.commit()
        conn.close()

    def tearDown(self):
        self.tmp.cleanup()

    def test_migration_adds_column_and_heals_stale_rows(self):
        # Content-agnostic lookup still finds the legacy row (backward compatible)...
        self.assertEqual(
            self.db.select_summarized_text("legacy/readme", "", 6),
            "stale-summary",
        )
        # ...but a content-aware lookup misses it, so a changed file re-summarizes
        # instead of serving the stale entry.
        self.assertIsNone(
            self.db.select_summarized_text(
                "legacy/readme", "", 6, summarize_prompt="N/A", content_hash="new-hash",
            )
        )
        # And the column now exists on the migrated database.
        conn = sqlite3.connect(self.db.get_db_file())
        cols = [r[1] for r in conn.execute(
            f"PRAGMA table_info({SUMMARIZED_FILES_TABLE_NAME})"
        ).fetchall()]
        conn.close()
        self.assertIn("content_hash", cols)





