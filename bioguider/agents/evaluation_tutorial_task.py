

from pathlib import Path
from typing import Callable, Optional
from langchain.prompts import ChatPromptTemplate
from langchain_openai.chat_models.base import BaseChatOpenAI
from pydantic import BaseModel, Field
import logging

from bioguider.agents.consistency_evaluation_task import ConsistencyEvaluationResult
from bioguider.agents.evaluation_task import EvaluationTask
from bioguider.agents.collection_task import CollectionTask
from bioguider.agents.evaluation_tutorial_task_prompts import INDIVIDUAL_TUTORIAL_EVALUATION_SYSTEM_PROMPT
from bioguider.agents.prompt_utils import CollectionGoalItemEnum
from bioguider.utils.constants import DEFAULT_TOKEN_USAGE, ProjectMetadata
from bioguider.utils.file_utils import flatten_files
from bioguider.utils.utils import increase_token_usage, get_overall_score
from .evaluation_utils import (
    compute_readability_metrics,
    default_consistency_result,
    evaluate_consistency_on_content,
    normalize_evaluation_content,
    run_llm_evaluation,
    sanitize_files,
)

logger = logging.getLogger(__name__)

MAX_FILE_SIZE = 1024 * 100 # 100K

class TutorialEvaluationResult(BaseModel):
    overall_score: int=Field(description="A number between 0 and 100 representing the overall quality rating.")
    overall_key_strengths: str=Field(description="A string value, the key strengths of the tutorial")
    # overall_improvement_suggestions: str=Field(description="Suggestions to improve the overall score if necessary")
    readability_score: int=Field(description="A number between 0 and 100 representing the readability quality rating.")
    readability_error_count: Optional[int]=Field(default=0, description="Total number of ERROR INSTANCES found (count every occurrence, not types)")
    readability_errors_found: list[str]=Field(default_factory=list, description="List of ALL individual error instances with format: 'ERROR_TYPE: original → corrected - location'")
    readability_suggestions: list[str]=Field(default_factory=list, description="General readability improvement suggestions (non-error related)")
    setup_and_dependencies_score: int=Field(description="A number between 0 and 100 representing the setup and dependencies quality rating.")
    setup_and_dependencies_suggestions: list[str]=Field(description="A list of string values, suggestions to improve setup and dependencies if necessary")
    reproducibility_score: int=Field(description="A number between 0 and 100 representing the reproducibility quality rating.")
    reproducibility_suggestions: list[str]=Field(description="A list of string values, suggestions to improve reproducibility if necessary")
    structure_and_navigation_score: int=Field(description="A number between 0 and 100 representing the structure and navigation quality rating.")
    structure_and_navigation_suggestions: list[str]=Field(description="A list of string values, suggestions to improve structure and navigation if necessary")
    executable_code_quality_score: int=Field(description="A number between 0 and 100 representing the executable code quality rating.")
    executable_code_quality_suggestions: list[str]=Field(description="A list of string values, suggestions to improve executable code quality if necessary")
    result_verification_score: int=Field(description="A number between 0 and 100 representing the result verification quality rating.")
    result_verification_suggestions: list[str]=Field(description="A list of string values, suggestions to improve result verification if necessary")
    performance_and_resource_notes_score: int=Field(description="A number between 0 and 100 representing the performance and resource notes quality rating.")
    performance_and_resource_notes_suggestions: list[str]=Field(description="A list of string values, suggestions to improve performance and resource notes if necessary")
    
class IndividualTutorialEvaluationResult(BaseModel):
    tutorial_evaluation: TutorialEvaluationResult | None=Field(description="The evaluation result of the tutorial")
    consistency_evaluation: ConsistencyEvaluationResult | None=Field(description="The evaluation result of the consistency of the tutorial")

class EvaluationTutorialTask(EvaluationTask):
    def __init__(
        self, 
        llm: BaseChatOpenAI, 
        repo_path: str, 
        gitignore_path: str,
        meta_data: ProjectMetadata | None = None,
        step_callback: Callable | None = None,
        summarized_files_db = None,
        code_structure_db = None,
        collected_files: list[str] | None = None,
    ):
        super().__init__(llm, repo_path, gitignore_path, meta_data, step_callback, summarized_files_db)
        self.evaluation_name = "Tutorial Evaluation"
        self.code_structure_db = code_structure_db
        self.collected_files = collected_files

    def _sanitize_files(self, files: list[str]) -> list[str]:
        return sanitize_files(
            self.repo_path,
            files,
            max_size_bytes=MAX_FILE_SIZE,
            disallowed_exts={".svg"},
            check_ipynb_size=False,
        )

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
            goal_item=CollectionGoalItemEnum.Tutorial.name,
        )
        files = task.collect()
        files = flatten_files(self.repo_path, files)
        files = self._sanitize_files(files)
        return files

    def _evaluate_individual_tutorial(self, file: str) -> tuple[IndividualTutorialEvaluationResult | None, dict]:
        content, readability_content = normalize_evaluation_content(
            self.repo_path, file
        )
        if content is None or readability_content is None:
            logger.error(f"Error in sanitizing file {file} - {Path(self.repo_path, file).resolve()}")
            return None, {**DEFAULT_TOKEN_USAGE}
            
        # evaluate general criteria
        flesch_reading_ease, flesch_kincaid_grade, gunning_fog_index, smog_index = \
            compute_readability_metrics(readability_content)
        system_prompt = ChatPromptTemplate.from_template(
            INDIVIDUAL_TUTORIAL_EVALUATION_SYSTEM_PROMPT
        ).format(
            flesch_reading_ease=flesch_reading_ease,
            flesch_kincaid_grade=flesch_kincaid_grade,
            gunning_fog_index=gunning_fog_index,
            smog_index=smog_index,
            tutorial_file_content=readability_content,
        )
                
        res, token_usage, reasoning_process = run_llm_evaluation(
            llm=self.llm,
            system_prompt=system_prompt,
            instruction_prompt="Now, let's begin the tutorial evaluation.",
            schema=TutorialEvaluationResult,
        )
        res: TutorialEvaluationResult = res

        # evaluate consistency
        consistency_evaluation_result, _temp_token_usage = evaluate_consistency_on_content(
            llm=self.llm,
            code_structure_db=self.code_structure_db,
            step_callback=self.step_callback,
            domain="tutorial/vignette",
            content=content,
        )
        if consistency_evaluation_result is None:
            # No sufficient information to evaluate the consistency of the tutorial
            consistency_evaluation_result = default_consistency_result("tutorial/vignette")

        # calculate overall score
        res.overall_score = get_overall_score(
            [
                consistency_evaluation_result.score,
                res.readability_score, 
                res.setup_and_dependencies_score, 
                res.reproducibility_score, 
                res.structure_and_navigation_score, 
                res.executable_code_quality_score, 
                res.result_verification_score, 
                res.performance_and_resource_notes_score,
            ],
            [3, 3, 3, 1, 1, 2, 1, 1],
        )
        
        return IndividualTutorialEvaluationResult(
            tutorial_evaluation=res,
            consistency_evaluation=consistency_evaluation_result,
        ), token_usage

    def _evaluate(self, files: list[str] | None = None) -> tuple[dict[str, IndividualTutorialEvaluationResult] | None, dict, list[str]]:
        total_token_usage = {**DEFAULT_TOKEN_USAGE}
        tutorial_evaluation_results = {}
        for file in files:
            tutorial_evaluation_result, token_usage = self._evaluate_individual_tutorial(file)
            total_token_usage = increase_token_usage(total_token_usage, token_usage)
            tutorial_evaluation_results[file] = tutorial_evaluation_result
        return tutorial_evaluation_results, total_token_usage, files
