import pytest

from bioguider.agents.agent_utils import parse_final_answer

@pytest.mark.parametrize("final_answer, expected_result", [
    ('**FinalAnswer:**: { "final_answer": "test" }', { 'final_answer': 'test' }),
    ('**FinalAnswer**: { "final_answer": "test" }', { 'final_answer': 'test' }),
    ('FinalAnswer: { "final_answer": "test" }', { 'final_answer': 'test' }),
    ('**Final Answer:**: { "final_answer": "test" }', { 'final_answer': 'test' }),
    ('Final Answer: { "final_answer": "test" }', { 'final_answer': 'test' }),
    ('**final answer:**: { "final_answer": "test" }', { 'final_answer': 'test' }),
])
def test_parse_final_answer(final_answer, expected_result):
    result = parse_final_answer(final_answer)
    assert result == expected_result