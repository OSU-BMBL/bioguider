from bioguider.database.code_structure_db import CodeStructureDb


def _db(tmp_path):
    return CodeStructureDb(author="acme", repo_name="tool", data_folder=str(tmp_path))


def test_insert_and_select_cli_argument(tmp_path):
    db = _db(tmp_path)
    assert db.insert_cli_argument(
        path="cli.py",
        dest="input_file",
        option_strings=["-f", "--file"],
        prog="mytool",
        description="Do a thing.",
        subcommand=None,
        arg_type="str",
        default_value="aaa.dat",
        action=None,
        required=False,
        help="input data file",
        lineno=12,
    )
    assert db.insert_cli_argument(
        path="cli.py",
        dest="epochs",
        option_strings=["--epochs"],
        subcommand="train",
        arg_type="int",
        default_value="10",
        choices=None,
        required=None,
    )

    rows = db.select_cli_arguments_by_path("cli.py")
    assert len(rows) == 2
    by_dest = {r["dest"]: r for r in rows}

    f = by_dest["input_file"]
    assert f["option_strings"] == ["-f", "--file"]
    assert f["prog"] == "mytool"
    assert f["arg_type"] == "str"
    assert f["default_value"] == "aaa.dat"
    assert f["required"] == 0
    assert f["subcommand"] == ""

    e = by_dest["epochs"]
    assert e["subcommand"] == "train"
    assert e["arg_type"] == "int"
    assert e["default_value"] == "10"
    assert e["required"] is None

    assert len(db.select_all_cli_arguments()) == 2
    assert db.select_cli_arguments_by_path("other.py") == []


def test_upsert_cli_argument(tmp_path):
    db = _db(tmp_path)
    db.insert_cli_argument(path="cli.py", dest="x", option_strings=["-x"], default_value="1")
    db.insert_cli_argument(path="cli.py", dest="x", option_strings=["-x", "--xx"], default_value="2")
    rows = db.select_cli_arguments_by_path("cli.py")
    assert len(rows) == 1
    assert rows[0]["default_value"] == "2"
    assert rows[0]["option_strings"] == ["-x", "--xx"]
