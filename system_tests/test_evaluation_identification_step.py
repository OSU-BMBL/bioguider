
from bioguider.agents.workflow_utils import BGWorkflowState
from bioguider.agents.evaluation_identification_step import EvaluationIdentificationStep

def test_EvaluationIdentificationStep(
    llm, 
    project_structure,
    step_callback,
):
    step = EvaluationIdentificationStep()
    state = BGWorkflowState()
    state['llm'] = llm
    state["step_callback"] = step_callback
    state["project_structure"] = project_structure

    state = step.execute(state)
    assert state is not None

