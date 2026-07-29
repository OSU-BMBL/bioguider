"""Unit tests for ConsistencyCollectionResult — the schema produced by
ConsistencyCollectionStep.  No LLM is involved here; these only check that the
result model (and its JSON schema) round-trips the new ``cli_invocations`` field
the way the step relies on.
"""

from bioguider.agents.consistency_collection_step import (
    ConsistencyCollectionResult,
    ConsistencyCollectionResultJsonSchema,
)


def test_result_defaults_to_empty_lists():
    res = ConsistencyCollectionResult()
    assert res.functions_and_classes == []
    assert res.cli_invocations == []


def test_result_round_trips_functions_and_cli_invocations():
    payload = {
        "functions_and_classes": [
            {
                "name": "chat",
                "file_path": "chatllm.py",
                "parameters": "message",
                "parent": "ChatLLM",
            }
        ],
        "cli_invocations": [
            {
                "program": "scripts/run.py",
                "language": "python",
                "subcommand": "N/A",
                "options": [
                    {"name": "-f", "value": "aaa.dat", "kind": "option"},
                    {"name": "-b", "value": "N/A", "kind": "flag"},
                    {"name": "-o", "value": "out/", "kind": "option"},
                ],
                "source": "python scripts/run.py -f aaa.dat -b -o out/",
            },
            {
                "program": "bin/analyze.R",
                "language": "r",
                "subcommand": "N/A",
                "options": [
                    {"name": "--input", "value": "data.csv", "kind": "option"},
                    {"name": "--threshold", "value": "0.05", "kind": "option"},
                ],
                "source": "Rscript bin/analyze.R --input data.csv --threshold 0.05",
            },
        ],
    }
    res = ConsistencyCollectionResult.model_validate(payload)
    assert len(res.functions_and_classes) == 1
    assert res.functions_and_classes[0]["parent"] == "ChatLLM"
    assert [c["language"] for c in res.cli_invocations] == ["python", "r"]
    assert res.cli_invocations[0]["options"][0] == {
        "name": "-f", "value": "aaa.dat", "kind": "option",
    }


def test_result_validates_when_cli_invocations_missing():
    # the LLM may omit the key entirely; the model must still validate
    res = ConsistencyCollectionResult.model_validate(
        {"functions_and_classes": [{"name": "f", "file_path": "N/A",
                                    "parameters": "N/A", "parent": "N/A"}]}
    )
    assert res.cli_invocations == []


def test_json_schema_advertises_cli_invocations():
    schema = ConsistencyCollectionResultJsonSchema
    assert "cli_invocations" in schema["properties"]
    assert schema["properties"]["cli_invocations"]["type"] == "array"
    assert "cli_invocations" in schema["required"]
    assert "functions_and_classes" in schema["required"]
