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
        token_usage = result.llm_output.get("token_usage")
        return response, vars(token_usage)
    
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

