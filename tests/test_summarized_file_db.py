
import pytest
import unittest
import os

from bioguider.database.summarized_file_db import SummarizedFilesDb

class SummarizedFilesDbTestCase(unittest.TestCase):
    def setUp(self):
        self.db = SummarizedFilesDb(
            author="foo",
            repo_name="bar",
        )
        res = self.db.upsert_summarized_file(
            "111/222/333",
            "",
            3,
            "balahbalah balahbalah balahbalah",
            "N/A",
        )
    def tearDown(self):
        if self.db is None:
            return
        db_path = self.db.get_db_file()
        os.unlink(db_path)

    def test_upsert(self):
        res = self.db.upsert_summarized_file(
            "aaa/bbb/ccc",
            "",
            3,
            "balahbalah balahbalah balahbalah",
            "N/A",
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
            "balahbalah",
            "N/A",
            token_usage,
        )
        self.assertTrue(res)
        res = self.db.select_summarized_text(
            "123/456/789",
            "",
            3
        )
        self.assertEqual(res, "balahbalah")





