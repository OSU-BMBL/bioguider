
import pytest

from bioguider.agents.common_agent import CommonConversation

def test_CommonConversation(llm):
    conversation = CommonConversation(llm=llm)
    res, token_usage = conversation.generate(
        system_prompt="Please act as an math teacher in a middle school.",
        instruction_prompt="Please draft me some geometry problems."
    )
    assert res is not None
    assert token_usage is not None


