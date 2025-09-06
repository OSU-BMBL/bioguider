from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_community.callbacks.openai_info import OpenAICallbackHandler
from bioguider.utils.utils import escape_braces

class CommonConversation:
    def __init__(self, llm: BaseChatOpenAI):
        self.llm = llm

    def generate(self, system_prompt: str, instruction_prompt: str):
        msgs = [
            SystemMessage(system_prompt),
            HumanMessage(instruction_prompt),
        ]
        callback_handler = OpenAICallbackHandler()
        result = self.llm.generate(
            messages=[msgs],
            callbacks=[callback_handler]
        )
        response = result.generations[0][0].text
        # Try to normalize token usage across providers
        token_usage = {}
        try:
            if hasattr(result, "llm_output") and result.llm_output is not None:
                raw = result.llm_output.get("token_usage") or result.llm_output.get("usage")
                if isinstance(raw, dict):
                    token_usage = {
                        "total_tokens": raw.get("total_tokens") or raw.get("total"),
                        "prompt_tokens": raw.get("prompt_tokens") or raw.get("prompt"),
                        "completion_tokens": raw.get("completion_tokens") or raw.get("completion"),
                    }
        except Exception:
            pass
        if not token_usage:
            token_usage = {
                "total_tokens": getattr(callback_handler, "total_tokens", 0),
                "prompt_tokens": getattr(callback_handler, "prompt_tokens", 0),
                "completion_tokens": getattr(callback_handler, "completion_tokens", 0),
            }
        return response, token_usage
    
    def generate_with_schema(self, system_prompt: str, instruction_prompt: str, schema: any):
        system_prompt = escape_braces(system_prompt)
        instruction_prompt = escape_braces(instruction_prompt)
        msgs = [
            SystemMessage(system_prompt),
            HumanMessage(instruction_prompt),
        ]
        msgs_template = ChatPromptTemplate.from_messages(messages=msgs)
        callback_handler = OpenAICallbackHandler()
        agent = msgs_template | self.llm.with_structured_output(schema)
        result = agent.invoke(
            input={},
            config={
                "callbacks": [callback_handler],
            },
        )
        token_usage = vars(callback_handler)
        return result, token_usage

