import textwrap

from bioguider.utils.python_file_handler import PythonFileHandler


def _write(tmp_path, name, src):
    p = tmp_path / name
    p.write_text(textwrap.dedent(src), encoding="utf-8")
    return str(p)


def test_no_argparse_returns_none(tmp_path):
    path = _write(tmp_path, "plain.py", """
        def add(a, b):
            return a + b
    """)
    handler = PythonFileHandler(path)
    assert handler.get_cli_interface() is None
    assert handler.get_cli_arguments() == []


def test_basic_argparse(tmp_path):
    path = _write(tmp_path, "cli.py", """
        import argparse

        def main():
            parser = argparse.ArgumentParser(prog="mytool", description="Do a thing.")
            parser.add_argument("-f", "--file", dest="input_file", type=str,
                                default="aaa.dat", help="input data file")
            parser.add_argument("-b", "--brief", action="store_true",
                                help="brief output")
            parser.add_argument("-o", "--out", required=True, help="output path")
            parser.add_argument("mode", choices=["fast", "slow"])
            parser.add_argument("--level", type=int, default=3)
            return parser.parse_args()
    """)
    iface = PythonFileHandler(path).get_cli_interface()
    assert iface is not None
    assert iface.prog == "mytool"
    assert iface.description == "Do a thing."
    assert iface.subcommands == []

    by_dest = {a.dest: a for a in iface.arguments}
    assert set(by_dest) == {"input_file", "brief", "out", "mode", "level"}

    f = by_dest["input_file"]
    assert f.option_strings == ["-f", "--file"]
    assert f.arg_type == "str"
    assert f.default == "aaa.dat"
    assert f.action is None
    assert f.help == "input data file"
    assert f.subcommand is None

    b = by_dest["brief"]
    assert b.option_strings == ["-b", "--brief"]
    assert b.action == "store_true"

    o = by_dest["out"]
    assert o.required is True

    mode = by_dest["mode"]
    assert mode.option_strings == []          # positional
    assert mode.choices == ["fast", "slow"]

    level = by_dest["level"]
    assert level.arg_type == "int"
    assert level.default == "3"


def test_subparsers(tmp_path):
    path = _write(tmp_path, "subcli.py", """
        from argparse import ArgumentParser

        parser = ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")

        train_p = subparsers.add_parser("train")
        train_p.add_argument("--epochs", type=int, default=10)

        predict_p = subparsers.add_parser("predict")
        predict_p.add_argument("--model", required=True)

        parser.add_argument("--verbose", action="store_true")
    """)
    iface = PythonFileHandler(path).get_cli_interface()
    assert iface is not None
    assert sorted(iface.subcommands) == ["predict", "train"]

    by_dest = {(a.subcommand, a.dest): a for a in iface.arguments}
    assert by_dest[("train", "epochs")].arg_type == "int"
    assert by_dest[("train", "epochs")].default == "10"
    assert by_dest[("predict", "model")].required is True
    # top-level argument has no subcommand
    assert by_dest[(None, "verbose")].action == "store_true"


def test_dynamic_default_is_captured_as_source(tmp_path):
    path = _write(tmp_path, "dyn.py", """
        import argparse, os

        parser = argparse.ArgumentParser()
        parser.add_argument("--workers", type=int, default=os.cpu_count())
    """)
    iface = PythonFileHandler(path).get_cli_interface()
    assert iface is not None
    workers = iface.arguments[0]
    assert workers.dest == "workers"
    # not a literal -> stored as the source expression
    assert workers.default == "os.cpu_count()"


def test_get_cli_arguments_tuple_shape(tmp_path):
    path = _write(tmp_path, "cli2.py", """
        import argparse
        parser = argparse.ArgumentParser(prog="t")
        parser.add_argument("-x", default="1")
    """)
    rows = PythonFileHandler(path).get_cli_arguments()
    assert len(rows) == 1
    row = rows[0]
    assert len(row) == 15
    # (path, prog, description, subcommand, dest, option_strings, arg_type,
    #  default, action, nargs, choices, required, metavar, help, lineno)
    assert row[0] == path
    assert row[1] == "t"
    assert row[4] == "x"
    assert row[5] == ["-x"]
    assert row[7] == "1"


def test_parser_variable_with_arbitrary_name(tmp_path):
    # receiver isn't called "parser" — extraction should still work
    path = _write(tmp_path, "weird.py", """
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("--name")
        ap.add_argument("count", type=int)
    """)
    iface = PythonFileHandler(path).get_cli_interface()
    assert iface is not None
    by_dest = {a.dest: a for a in iface.arguments}
    assert by_dest["name"].option_strings == ["--name"]
    assert by_dest["count"].option_strings == []
    assert by_dest["count"].arg_type == "int"


def test_nargs_and_metavar_and_append(tmp_path):
    path = _write(tmp_path, "na.py", """
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("--include", action="append", metavar="PATTERN")
        p.add_argument("files", nargs="+")
        p.add_argument("--opt", nargs="?", default="x")
        p.add_argument("--count", action="count", default=0)
    """)
    iface = PythonFileHandler(path).get_cli_interface()
    assert iface is not None
    by_dest = {a.dest: a for a in iface.arguments}
    assert by_dest["include"].action == "append"
    assert by_dest["include"].metavar == "PATTERN"
    assert by_dest["files"].nargs == "+"
    assert by_dest["opt"].nargs == "?"
    assert by_dest["count"].action == "count"
    assert by_dest["count"].default == "0"


def test_parser_with_no_arguments(tmp_path):
    # ArgumentParser present but no add_argument calls -> still a (trivial) CLI
    path = _write(tmp_path, "empty.py", """
        import argparse
        parser = argparse.ArgumentParser(prog="x", description="d")
        parser.parse_args()
    """)
    iface = PythonFileHandler(path).get_cli_interface()
    assert iface is not None
    assert iface.prog == "x"
    assert iface.description == "d"
    assert iface.arguments == []


def test_from_import_argument_parser(tmp_path):
    path = _write(tmp_path, "fi.py", """
        from argparse import ArgumentParser
        parser = ArgumentParser(description="hi")
        parser.add_argument("-v", "--verbose", action="store_true")
    """)
    iface = PythonFileHandler(path).get_cli_interface()
    assert iface is not None
    assert iface.description == "hi"
    assert iface.arguments[0].dest == "verbose"
    assert iface.arguments[0].action == "store_true"


def test_short_option_only_dest(tmp_path):
    path = _write(tmp_path, "so.py", """
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("-n", type=int)
    """)
    iface = PythonFileHandler(path).get_cli_interface()
    assert iface is not None
    assert iface.arguments[0].dest == "n"
    assert iface.arguments[0].option_strings == ["-n"]


def test_explicit_dest_overrides(tmp_path):
    path = _write(tmp_path, "ed.py", """
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("--input-file", dest="src", default="a.dat")
    """)
    iface = PythonFileHandler(path).get_cli_interface()
    assert iface is not None
    a = iface.arguments[0]
    assert a.dest == "src"
    assert a.option_strings == ["--input-file"]
    assert a.default == "a.dat"


def test_syntax_error_file_returns_none(tmp_path):
    path = _write(tmp_path, "bad.py", "def f(:\n  pass\n")
    assert PythonFileHandler(path).get_cli_interface() is None


def test_functions_and_classes_still_work(tmp_path):
    path = _write(tmp_path, "mod.py", '''
        class Foo:
            """A foo."""
            def bar(self, x):
                """Bar method."""
                return x
        def baz(a, b):
            """Baz fn."""
            return a + b
    ''')
    fns = PythonFileHandler(path).get_functions_and_classes()
    names = {f[0] for f in fns}
    assert names == {"Foo", "bar", "baz"}
