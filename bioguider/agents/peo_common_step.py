

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

class PlanAgentResult(BaseModel):
    """ Identification Plan Result """
    actions: list[dict] = Field(description="a list of action dictionary, e.g. [{'name': 'read_file', 'input': 'README.md'}, ...]")

PlanAgentResultJsonSchema = {
    "title": "identification_plan_result",
    "description": "plan result",
    "type": "object",
    "properties": {
        "actions": {
            "type": "array",
            "description": """a list of action dictionary, e.g. [{'name': 'read_file', 'input': 'README.md'}, ...]""",
            "title": "Actions",
            "items": {"type": "object"}
        },
    },
    "required": ["actions"],
}

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
                step = state['intermediate_steps'][i].replace("{", "_").replace("}", "_")
                intermediate_steps += step + "\n"
        # current step
        if "step_output" in state and state["step_output"] is not None:
            step_content = state["step_output"]
            step_content = step_content.replace("{", "_").replace("}", "_")
            intermediate_steps += step_content
        return intermediate_steps
    
    def _build_intermediate_analysis_and_thoughts(self, state: PEOWorkflowState):
        intermediate_analysis = "N/A" if "step_analysis" not in state or \
            state["step_analysis"] is None \
            else state["step_analysis"]
        intermediate_thoughts = "N/A" if "step_thoughts" not in state or \
            state["step_thoughts"] is None \
            else state["step_thoughts"]
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
