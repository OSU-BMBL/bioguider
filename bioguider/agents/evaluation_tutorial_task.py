

import json
from pathlib import Path
from typing import Callable, Tuple
from langchain.prompts import ChatPromptTemplate
from langchain_openai.chat_models.base import BaseChatOpenAI
from pydantic import BaseModel, Field
import logging

from bioguider.agents.agent_utils import read_file
from bioguider.agents.common_agent_2step import CommonAgentTwoSteps
from bioguider.agents.consistency_evaluation_task import ConsistencyEvaluationResult, ConsistencyEvaluationTask
from bioguider.agents.evaluation_task import EvaluationTask
from bioguider.agents.collection_task import CollectionTask
from bioguider.agents.evaluation_tutorial_task_prompts import INDIVIDUAL_TUTORIAL_EVALUATION_SYSTEM_PROMPT
from bioguider.agents.prompt_utils import CollectionGoalItemEnum
from bioguider.utils.constants import DEFAULT_TOKEN_USAGE, ProjectMetadata
from bioguider.utils.file_utils import detect_file_type, flatten_files
from bioguider.utils.notebook_utils import extract_markdown_from_notebook, strip_notebook_to_code_and_markdown
from bioguider.utils.pyphen_utils import PyphenReadability
from bioguider.utils.utils import convert_html_to_text, increase_token_usage, get_overall_score

logger = logging.getLogger(__name__)

MAX_FILE_SIZE = 1024 * 100 # 100K

class TutorialEvaluationResult(BaseModel):
    overall_score: int=Field(description="A number between 0 and 100 representing the overall quality rating.")
    overall_key_strengths: str=Field(description="A string value, the key strengths of the tutorial")
    # overall_improvement_suggestions: str=Field(description="Suggestions to improve the overall score if necessary")
    readability_score: int=Field(description="A number between 0 and 100 representing the readability quality rating.")
    readability_suggestions: list[str]=Field(description="A list of string values, suggestions to improve readability if necessary")
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
        sanitized_files = []
        for file in files:
            file_path = Path(self.repo_path, file)
            if not file_path.exists() or not file_path.is_file():
                continue
            if detect_file_type(file_path) == "binary":
                continue
            if file.endswith(".svg"):
                continue
            if not file.endswith(".ipynb") and file_path.stat().st_size > MAX_FILE_SIZE:
                continue
            sanitized_files.append(file)
        return sanitized_files

    def _sanitize_file_content(self, file: str) -> Tuple[str | None, str | None]:
        content = read_file(Path(self.repo_path, file))
        if content is None:
            logger.error(f"Error in reading file {file} - {Path(self.repo_path, file).resolve()}")
            return None, None

        if file.endswith(".ipynb") or file.endswith(".html") or file.endswith(".htm"):
            if file.endswith(".ipynb"):
                readability_content = extract_markdown_from_notebook(Path(self.repo_path, file))
                content = json.dumps(strip_notebook_to_code_and_markdown(Path(self.repo_path, file)))
            else:
                readability_content = convert_html_to_text(Path(self.repo_path, file))
                content = readability_content

            content = content.replace("{", "<<").replace("}", ">>")
        else:
            readability_content = content
        return content, readability_content

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

    def _evaluate_consistency(self, file: str) -> ConsistencyEvaluationResult:
        consistency_evaluation_task = ConsistencyEvaluationTask(
            llm=self.llm,
            code_structure_db=self.code_structure_db,
            step_callback=self.step_callback,
        )
        file = file.strip()
        with open(Path(self.repo_path, file), "r") as f:
            tutorial_content = f.read()
        return consistency_evaluation_task.evaluate(
            domain="tutorial/vignette",
            documentation=tutorial_content,
        )

    def _evaluate_consistency_on_content(self, content: str) -> ConsistencyEvaluationResult:
        consistency_evaluation_task = ConsistencyEvaluationTask(
            llm=self.llm,
            code_structure_db=self.code_structure_db,
            step_callback=self.step_callback,
        )
        return consistency_evaluation_task.evaluate(
            domain="tutorial/vignette",
            documentation=content,
        ), {**DEFAULT_TOKEN_USAGE}

    def _evaluate_individual_tutorial(self, file: str) -> tuple[IndividualTutorialEvaluationResult | None, dict]:
        content, readability_content = self._sanitize_file_content(file)
        if content is None or readability_content is None:
            logger.error(f"Error in sanitizing file {file} - {Path(self.repo_path, file).resolve()}")
            return None, {**DEFAULT_TOKEN_USAGE}
            
        # evaluate general criteria
        readability = PyphenReadability()
        flesch_reading_ease, flesch_kincaid_grade, gunning_fog_index, smog_index, \
                _, _, _, _, _ = readability.readability_metrics(readability_content)
        system_prompt = ChatPromptTemplate.from_template(
            INDIVIDUAL_TUTORIAL_EVALUATION_SYSTEM_PROMPT
        ).format(
            flesch_reading_ease=flesch_reading_ease,
            flesch_kincaid_grade=flesch_kincaid_grade,
            gunning_fog_index=gunning_fog_index,
            smog_index=smog_index,
            tutorial_file_content=readability_content,
        )
        agent = CommonAgentTwoSteps(llm=self.llm)
        res, _, token_usage, reasoning_process = agent.go(
            system_prompt=system_prompt,
            instruction_prompt="Now, let's begin the tutorial evaluation.",
            schema=TutorialEvaluationResult,
        )
        res: TutorialEvaluationResult = res

        # evaluate consistency
        consistency_evaluation_result, _temp_token_usage = self._evaluate_consistency_on_content(content)
        if consistency_evaluation_result is None:
            # No sufficient information to evaluate the consistency of the tutorial
            consistency_evaluation_result = ConsistencyEvaluationResult(
                consistency_score=0,
                consistency_assessment="No sufficient information to evaluate the consistency of the tutorial",
                consistency_development=[],
                consistency_strengths=[],
            )

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


