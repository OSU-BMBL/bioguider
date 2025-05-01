
from abc import ABC, abstractmethod
from langchain_core.messages import BaseMessage
from langchain_deepseek import ChatDeepSeek
from openai import AuthenticationError
from pydantic import PositiveFloat, PositiveInt

class Conversation(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def chat(
        question: str,
        messages: list[BaseMessage] = None
    ):
        """ chat with LLM """

class DeepSeekConversation(Conversation):
    chatter: ChatDeepSeek | None = None
    model: str = "deepseek-chat"
    temperature: PositiveFloat = 0.1
    request_timeout: PositiveInt = 60
    base_url: str = "https://api.deepseek.com/v1"
    max_retries: PositiveInt = 3
    api_key: str | None = None
    def __init__(
        self,        
    ):
        super().__init__()

    def set_api_key(self, key: str):
        
        try:
            self.chatter = ChatDeepSeek(
                model=self.model,
                api_key=key,
                temperature=self.temperature,
                max_retries=self.max_retries,
                timeout=self.request_timeout,
                base_url=self.base_url,
            )
            # verify chat
            ai_msg = self.chatter.invoke(
                 [("system", "Hi")]
            )
            return True
        except AuthenticationError as e:
            self.chatter = None
            return False

    def chat(
        self,
        question: str,
        messages: list[BaseMessage] = None
    ):
        msgs = messages + [("user", question)] if messages is not None else \
               [("user", question)]
        
        try:
            res_msg = self.chatter.invoke(msgs)
            return res_msg
        except Exception as e:
            return str(e)



