import textwrap

from bioguider.utils.code_structure_builder import CodeStructureBuilder
from bioguider.database.code_structure_db import CodeStructureDb


def _make_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".gitignore").write_text("", encoding="utf-8")
    (repo / "cli.py").write_text(textwrap.dedent("""
        import argparse

        def main():
            parser = argparse.ArgumentParser(prog="mytool", description="A tool.")
            parser.add_argument("-f", "--file", dest="input_file", default="aaa.dat")
            parser.add_argument("-b", action="store_true")
            parser.add_argument("mode", choices=["a", "b"])
            return parser.parse_args()

        class Helper:
            def run(self):
                return 1
    """), encoding="utf-8")
    (repo / "lib.py").write_text(textwrap.dedent("""
        def add(a, b):
            return a + b
    """), encoding="utf-8")
    return repo


def test_builder_persists_cli_arguments(tmp_path):
    repo = _make_repo(tmp_path)
    db = CodeStructureDb(author="acme", repo_name="repo", data_folder=str(tmp_path))
    builder = CodeStructureBuilder(str(repo), str(repo / ".gitignore"), db)
    builder.build_code_structure()

    # functions/classes still captured
    names = set(db.select_all_names())
    assert {"main", "Helper", "run", "add"} <= names

    # CLI arguments captured from cli.py only
    cli_rows = db.select_cli_arguments_by_path("cli.py")
    by_dest = {r["dest"]: r for r in cli_rows}
    assert set(by_dest) == {"input_file", "b", "mode"}
    assert by_dest["input_file"]["prog"] == "mytool"
    assert by_dest["input_file"]["description"] == "A tool."
    assert by_dest["input_file"]["option_strings"] == ["-f", "--file"]
    assert by_dest["input_file"]["default_value"] == "aaa.dat"
    assert by_dest["b"]["action"] == "store_true"
    assert by_dest["mode"]["option_strings"] == []
    assert by_dest["mode"]["choices"] == ["a", "b"]

    assert db.select_cli_arguments_by_path("lib.py") == []
    assert len(db.select_all_cli_arguments()) == 3


def test_builder_no_cli_in_repo(tmp_path):
    repo = tmp_path / "repo2"
    repo.mkdir()
    (repo / ".gitignore").write_text("", encoding="utf-8")
    (repo / "lib.py").write_text("def f():\n    return 0\n", encoding="utf-8")

    db = CodeStructureDb(author="acme", repo_name="repo2", data_folder=str(tmp_path))
    builder = CodeStructureBuilder(str(repo), str(repo / ".gitignore"), db)
    builder.build_code_structure()

    assert db.select_all_cli_arguments() == []
    assert "f" in set(db.select_all_names())


def test_builder_argparse_without_switches(tmp_path):
    # argparse is used, parse_args() is called, but no add_argument() -> no rows
    repo = tmp_path / "repo3"
    repo.mkdir()
    (repo / ".gitignore").write_text("", encoding="utf-8")
    (repo / "cli.py").write_text(textwrap.dedent("""
        import argparse

        def main():
            parser = argparse.ArgumentParser(prog="bare", description="No switches.")
            return parser.parse_args()
    """), encoding="utf-8")

    db = CodeStructureDb(author="acme", repo_name="repo3", data_folder=str(tmp_path))
    builder = CodeStructureBuilder(str(repo), str(repo / ".gitignore"), db)
    builder.build_code_structure()

    assert db.select_all_cli_arguments() == []
    assert db.select_cli_arguments_by_path("cli.py") == []
    assert "main" in set(db.select_all_names())


def test_builder_only_class_definition(tmp_path):
    # a .py file that only defines a class -> class captured, no CLI rows
    repo = tmp_path / "repo_cls"
    repo.mkdir()
    (repo / ".gitignore").write_text("", encoding="utf-8")
    (repo / "model.py").write_text(textwrap.dedent('''
        class Model:
            """A model."""
            def fit(self, x):
                return x

            def predict(self, x):
                return x
    '''), encoding="utf-8")

    db = CodeStructureDb(author="acme", repo_name="repo_cls", data_folder=str(tmp_path))
    builder = CodeStructureBuilder(str(repo), str(repo / ".gitignore"), db)
    builder.build_code_structure()

    assert db.select_all_cli_arguments() == []
    assert db.select_cli_arguments_by_path("model.py") == []
    assert {"Model", "fit", "predict"} <= set(db.select_all_names())


def test_builder_mixed_files_only_argparse_file_has_rows(tmp_path):
    # several .py files, only one uses argparse
    repo = tmp_path / "repo4"
    repo.mkdir()
    (repo / ".gitignore").write_text("", encoding="utf-8")
    (repo / "plain.py").write_text("def g(x):\n    return x\n", encoding="utf-8")
    (repo / "also_plain.py").write_text(
        "class C:\n    def m(self):\n        return 2\n", encoding="utf-8"
    )
    (repo / "tool.py").write_text(textwrap.dedent("""
        import argparse
        p = argparse.ArgumentParser(prog="tool")
        p.add_argument("--name", default="x")
    """), encoding="utf-8")

    db = CodeStructureDb(author="acme", repo_name="repo4", data_folder=str(tmp_path))
    builder = CodeStructureBuilder(str(repo), str(repo / ".gitignore"), db)
    builder.build_code_structure()

    assert db.select_cli_arguments_by_path("plain.py") == []
    assert db.select_cli_arguments_by_path("also_plain.py") == []
    rows = db.select_cli_arguments_by_path("tool.py")
    assert {r["dest"] for r in rows} == {"name"}
    assert len(db.select_all_cli_arguments()) == 1
    assert {"g", "C", "m"} <= set(db.select_all_names())
