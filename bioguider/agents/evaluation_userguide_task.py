
from pathlib import Path
import logging
from typing import Optional
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from bioguider.agents.collection_task import CollectionTask
from bioguider.agents.consistency_evaluation_task import ConsistencyEvaluationResult
from bioguider.agents.prompt_utils import CollectionGoalItemEnum
from bioguider.utils.constants import (
    DEFAULT_TOKEN_USAGE, 
)
from bioguider.utils.file_utils import flatten_files
from .evaluation_utils import (
    compute_readability_metrics,
    default_consistency_result,
    evaluate_consistency_on_content,
    normalize_evaluation_content,
    run_llm_evaluation,
    sanitize_files,
)

from .evaluation_task import EvaluationTask
from bioguider.utils.utils import get_overall_score, increase_token_usage
from .evaluation_userguide_prompts import INDIVIDUAL_USERGUIDE_EVALUATION_SYSTEM_PROMPT


class UserGuideEvaluationResult(BaseModel):
    overall_score: int=Field(description="A number between 0 and 100 representing the overall quality rating.")
    overall_key_strengths: str=Field(description="A string value, the key strengths of the user guide")
    
    readability_score: int=Field(description="A number between 0 and 100 representing the readability quality rating.")
    readability_error_count: Optional[int]=Field(default=0, description="Total number of ERROR INSTANCES found (count every occurrence, not types)")
    readability_errors_found: list[str]=Field(default_factory=list, description="List of ALL individual error instances with format: 'ERROR_TYPE: original → corrected - location'")
    readability_suggestions: list[str]=Field(description="A list of string values, suggestions to improve readability if necessary")
    context_and_purpose_score: int=Field(description="A number between 0 and 100 representing the context and purpose quality rating.")
    context_and_purpose_suggestions: list[str]=Field(description="A list of string values, suggestions to improve context and purpose if necessary")
    error_handling_score: int=Field(description="A number between 0 and 100 representing the error handling quality rating.")
    error_handling_suggestions: list[str]=Field(description="A list of string values, suggestions to improve error handling if necessary")

class IndividualUserGuideEvaluationResult(BaseModel):
    user_guide_evaluation: UserGuideEvaluationResult | None=Field(description="The evaluation result of the user guide")
    consistency_evaluation: ConsistencyEvaluationResult | None=Field(description="The evaluation result of the consistency of the user guide")

logger = logging.getLogger(__name__)

MAX_FILE_SIZE = 1024 * 100 # 100K


class EvaluationUserGuideTask(EvaluationTask):
    def __init__(
        self, 
        llm, 
        repo_path, 
        gitignore_path, 
        meta_data = None, 
        step_callback = None,
        summarized_files_db = None,
        code_structure_db = None,
        collected_files: list[str] | None = None,
    ):
        super().__init__(llm, repo_path, gitignore_path, meta_data, step_callback, summarized_files_db)
        self.evaluation_name = "User Guide Evaluation"
        self.code_structure_db = code_structure_db
        self.collected_files = collected_files

    def _collect_files(self):
        if self.collected_files is not None:
            return self.collected_files
        
        task = CollectionTask(
            llm=self.llm,
            step_callback=self.step_callback,
            summarized_files_db=self.summarized_files_db,
        )
        task.compile(
            repo_path=self.repo_path,
            gitignore_path=Path(self.repo_path, ".gitignore"),
            goal_item=CollectionGoalItemEnum.UserGuide.name,
        )
        files = task.collect()
        files = flatten_files(self.repo_path, files)
        files = sanitize_files(self.repo_path, files, max_size_bytes=MAX_FILE_SIZE)
        return files

    def _evaluate_individual_userguide(self, file: str) -> tuple[IndividualUserGuideEvaluationResult | None, dict]:
        content, readability_content = normalize_evaluation_content(
            self.repo_path, file
        )
        if content is None or readability_content is None:
            logger.error(f"Error in reading file {file}")
            return None, {**DEFAULT_TOKEN_USAGE}

        # evaluate general criteria
        flesch_reading_ease, flesch_kincaid_grade, gunning_fog_index, smog_index = \
            compute_readability_metrics(readability_content)
        system_prompt = ChatPromptTemplate.from_template(
            INDIVIDUAL_USERGUIDE_EVALUATION_SYSTEM_PROMPT
        ).format(
            flesch_reading_ease=flesch_reading_ease,
            flesch_kincaid_grade=flesch_kincaid_grade,
            gunning_fog_index=gunning_fog_index,
            smog_index=smog_index,
            userguide_content=readability_content,
        )
        res, token_usage, reasoning_process = run_llm_evaluation(
            llm=self.llm,
            system_prompt=system_prompt,
            instruction_prompt="Now, let's begin the user guide/API documentation evaluation.",
            schema=UserGuideEvaluationResult,
        )
        res: UserGuideEvaluationResult = res

        # evaluate consistency
        consistency_evaluation_result, _temp_token_usage = evaluate_consistency_on_content(
            llm=self.llm,
            code_structure_db=self.code_structure_db,
            step_callback=self.step_callback,
            domain="user guide/API",
            content=content,
        )
        if consistency_evaluation_result is None:
            # No sufficient information to evaluate the consistency of the user guide/API documentation
            consistency_evaluation_result = default_consistency_result("user guide/API")

        # calculate overall score
        res.overall_score = get_overall_score(
            [
                consistency_evaluation_result.score,
                res.readability_score, 
                res.context_and_purpose_score, 
                res.error_handling_score, 
            ],
            [2, 1, 1, 1],
        )
        
        return IndividualUserGuideEvaluationResult(
            user_guide_evaluation=res,
            consistency_evaluation=consistency_evaluation_result,
        ), token_usage

    def _evaluate(self, files: list[str] | None = None) -> tuple[dict[str, IndividualUserGuideEvaluationResult] | None, dict, list[str]]:
        total_token_usage = {**DEFAULT_TOKEN_USAGE}
        user_guide_evaluation_results = {}
        files = flatten_files(self.repo_path, files)
        for file in files:
            if file.endswith(".py") or file.endswith(".R"):
                continue
            user_guide_evaluation_result, token_usage = self._evaluate_individual_userguide(file)
            total_token_usage = increase_token_usage(total_token_usage, token_usage)
            user_guide_evaluation_results[file] = user_guide_evaluation_result

        return user_guide_evaluation_results, total_token_usage, files
