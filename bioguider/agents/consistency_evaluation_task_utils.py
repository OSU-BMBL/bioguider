
from typing import Callable, Optional, TypedDict


class ConsistencyEvaluationState(TypedDict):
    domain: str
    documentation: str
    step_output_callback: Optional[Callable]
    functions_and_classes: Optional[list[dict]]
    all_query_rows: Optional[list[any]]
    consistency_score: Optional[int]
    consistency_assessment: Optional[str]
    consistency_development: Optional[list[str]]
    consistency_strengths: Optional[list[str]]
