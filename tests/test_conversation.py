import pytest

from bioguider.conversation import DeepSeekConversation

def test_set_api_key():
    conv = DeepSeekConversation()
    res = conv.set_api_key("abc")
    assert res == False
