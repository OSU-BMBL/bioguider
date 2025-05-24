from typing import Any, Callable, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_community.callbacks.openai_info import OpenAICallbackHandler
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_incrementing
import logging

from bioguider.agents.common_agent import (
    CommonAgent,
    RetryException,
)
from bioguider.agents.prompt_utils import COT_USER_INSTRUCTION

logger = logging.getLogger()


class CommonAgentTwoSteps(CommonAgent):
    def __init__(self, llm: BaseChatOpenAI):
        super().__init__(llm)

    def _initialize(self):
        self.exceptions = None
        self.token_usage = None

    def _get_retryexception_message(self) -> list[tuple[str, str]]:
        if self.exceptions is None:
            return None
        return [("human", str(excp)) for excp in self.exceptions]

    def _build_prompt_for_cot_step(
        self,
        system_prompt: str,
        instruction_prompt: str,
    ):
        msgs = [("system", system_prompt)]
        msgs + msgs + [("human", instruction_prompt)]
        exception_msgs = self._get_retryexception_message()
        if exception_msgs is not None:
            msgs = msgs + exception_msgs
        msgs = msgs + [("human", COT_USER_INSTRUCTION)]
        return ChatPromptTemplate.from_messages(msgs)
    
    def _build_prompt_for_final_step(
        self,
        system_prompt: str,
        cot_msg: str,
    ):
        msgs = [("system", system_prompt)]
        cot_msg = cot_msg.replace("{", "{{").replace("}", "}}")
        msgs = msgs + [(
            "human",
            f"Please review the following step-by-step reasoning and provide the answer based on it: ```{cot_msg}```"
        )]
        return ChatPromptTemplate.from_messages(msgs)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_incrementing(start=1.0, increment=3, max=10),
    )
    def _invoke_agent(
        self,
        system_prompt: str,
        instruction_prompt: str,
        schema: any,
        post_process: Optional[Callable] = None,
        **kwargs: Optional[Any],
    ):
        # Initialize the callback handler
        callback_handler = OpenAICallbackHandler()
        cot_prompt = self._build_prompt_for_cot_step(
            system_prompt=system_prompt, 
            instruction_prompt=instruction_prompt
        )

        try:
            # First, use llm to do CoT
            msgs = cot_prompt.invoke(input={}).to_messages()
            
            cot_res = self.llm.generate(messages=[msgs])
            reasoning_process = cot_res.generations[0][0].text
            token_usage = cot_res.llm_output.get("token_usage")
            cot_tokens = {
                "total_tokens": token_usage.get("total_tokens", 0),
                "prompt_tokens": token_usage.get("prompt_tokens", 0),
                "completion_tokens": token_usage.get("completion_tokens", 0),
            }
            self._incre_token_usage(cot_tokens)
        except Exception as e:
            logger.error(str(e))
            raise e
        
        # Then use the reasoning process to do the structured output
        updated_prompt = self._build_prompt_for_final_step(
            system_prompt=system_prompt,
            cot_msg=reasoning_process,
        )
        agent = updated_prompt | self.llm.with_structured_output(schema)
        try:
            res = agent.invoke(
                input={},
                config={
                    "callbacks": [callback_handler],
                },
            )
            self._incre_token_usage(callback_handler)
        except Exception as e:
            logger.error(str(e))
            raise e
        processed_res = None
        if post_process is not None:
            try:
                processed_res = post_process(res, **kwargs)
            except RetryException as e:
                logger.error(str(e))
                self.exceptions = [e] if self.exceptions is None else self.exceptions + [e]
                raise e
            except Exception as e:
                logger.error(str(e))
                raise e
        return res, processed_res, self.token_usage, reasoning_process