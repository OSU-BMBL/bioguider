

from typing import Optional
from langchain_openai.chat_models.base import BaseChatOpenAI
from pydantic import BaseModel, Field
from bioguider.agents.common_step import CommonState, CommonStep

class PEOWorkflowState(CommonState):
    intermediate_steps: Optional[str]
    step_output: Optional[str]
    step_analysis: Optional[str]
    step_thoughts: Optional[str]
    plan_actions: Optional[list[dict]]

class PEOCommonStep(CommonStep):
    """
    This class is a placeholder for common step functionality in the PEO agent.
    It is currently empty and can be extended in the future.
    """
    def __init__(self, llm: BaseChatOpenAI):
        super().__init__()
        self.llm = llm

    def _build_intermediate_steps(self, state: PEOWorkflowState):
        """
        Build intermediate steps for the PEO workflow.
        """
        intermediate_steps = ""
        # previous steps
        if "intermediate_steps" in state:
            for i in range(len(state['intermediate_steps'])):
                step = state['intermediate_steps'][i].replace("{", "(").replace("}", ")")
                intermediate_steps += step + "\n"
        # current step
        if "step_output" in state and state["step_output"] is not None:
            step_content = state["step_output"]
            step_content = step_content.replace("{", "(").replace("}", ")")
            intermediate_steps += step_content
        return intermediate_steps
    
    def _build_intermediate_analysis_and_thoughts(self, state: PEOWorkflowState):
        intermediate_analysis = "N/A" if "step_analysis" not in state or \
            state["step_analysis"] is None \
            else state["step_analysis"]
        intermediate_analysis = intermediate_analysis.replace("{", "(").replace("}", ")")
        intermediate_thoughts = "N/A" if "step_thoughts" not in state or \
            state["step_thoughts"] is None \
            else state["step_thoughts"]
        intermediate_thoughts = intermediate_thoughts.replace("{", "(").replace("}", ")")
        return intermediate_analysis, intermediate_thoughts

    @staticmethod
    def _reset_step_state(state):
        # move step_output to intermediate steps
        if "intermediate_steps" not in state or state["intermediate_steps"] is None:
            state["intermediate_steps"] = []
        intermediate_steps = state["intermediate_steps"]
        if "step_output" in state and state["step_output"] is not None:
            intermediate_steps.append(state["step_output"])
        state["intermediate_steps"] = intermediate_steps

        state["step_analysis"] = None
        state["step_thoughts"] = None
        state["step_output"] = None
