
from langchain_core.prompts import ChatPromptTemplate

from bioguider.agents.evaluation_identification_agent import (
    get_system_prompt,
    postprocess_validate_result,
    EvaluationIdentificationAgentResult,
)
from bioguider.agents.common_step import CommonAgentStep

class EvaluationIdentificationStep(CommonAgentStep):
    def __init__(self):
        super().__init__()
        self.start_title = "Identification Step"
        self.end_title = "Completed to Identify Project"

    def get_system_prompt(self, state):
        project_structure = state["project_structure"]
        return get_system_prompt(
            project_structure=project_structure,
        )
    
    def get_schema(self):
        return EvaluationIdentificationAgentResult
    
    def get_post_processor_and_kwargs(self, state):
        return postprocess_validate_result, None
    
    

