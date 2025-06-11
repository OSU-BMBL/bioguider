

from typing import Callable, TypedDict, Optional
from langchain_openai.chat_models.base import BaseChatOpenAI

class IdentificationWorkflowState(TypedDict):
    llm: BaseChatOpenAI
    step_output_callback: Optional[Callable]
    goal: str

    plan_actions: Optional[str]
    plan_reasoning: Optional[str]
    intermediate_steps: Optional[list[str]]
    final_answer: Optional[str]
    final_answer_example: Optional[str]
    step_output: Optional[str]
    step_analysis: Optional[str]
    step_thoughts: Optional[str]
