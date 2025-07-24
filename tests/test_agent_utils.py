
import pytest

from bioguider.utils.utils import escape_braces

def test_escape_braces():
    # Test case with single braces
    input_text = "This is a {test} string with {single} braces."
    expected_output = "This is a {{test}} string with {{single}} braces."
    assert escape_braces(input_text) == expected_output

    # Test case with double braces
    input_text = "This is a {{test}} string with {{double}} braces."
    expected_output = "This is a {{test}} string with {{double}} braces."
    assert escape_braces(input_text) == expected_output

    # Test case with mixed single and double braces
    input_text = "This is a {test} string with {{double}} and {single} braces."
    expected_output = "This is a {{test}} string with {{double}} and {{single}} braces."
    assert escape_braces(input_text) == expected_output

