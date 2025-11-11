


from typing import Callable
from langchain_openai.chat_models.base import BaseChatOpenAI
from pydantic import BaseModel

from bioguider.agents.consistency_evaluation_task_utils import ConsistencyEvaluationState
from bioguider.database.code_structure_db import CodeStructureDb
from .consistency_collection_step import ConsistencyCollectionStep
from .consistency_query_step import ConsistencyQueryStep
from .consistency_observe_step import ConsistencyObserveStep

class ConsistencyEvaluationResult(BaseModel):
    score: int
    assessment: str
    development: list[str]
    strengths: list[str]

class ConsistencyEvaluationTask:
    def __init__(
        self, 
        llm: BaseChatOpenAI, 
        code_structure_db: CodeStructureDb, 
        step_callback: Callable | None = None
    ):
        self.llm = llm
        self.code_structure_db = code_structure_db
        self.step_callback = step_callback

    def evaluate(self, domain: str, documentation: str) -> ConsistencyEvaluationResult:
        collection_step = ConsistencyCollectionStep(llm=self.llm)
        query_step = ConsistencyQueryStep(code_structure_db=self.code_structure_db)
        observe_step = ConsistencyObserveStep(llm=self.llm)

        state = ConsistencyEvaluationState(
            domain=domain,
            documentation=documentation,
            step_output_callback=self.step_callback,
        )

        state = collection_step.execute(state)
        state = query_step.execute(state)
        state = observe_step.execute(state)

        score = state["consistency_score"]
        assessment = state["consistency_assessment"]
        development = state["consistency_development"]
        strengths = state["consistency_strengths"]

        return ConsistencyEvaluationResult(
            score=score,
            assessment=assessment,
            development=development,
            strengths=strengths,
        )

