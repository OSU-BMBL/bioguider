
import os
from pathlib import Path
import logging
from langchain.prompts import ChatPromptTemplate
from markdownify import markdownify as md
from pydantic import BaseModel, Field

from bioguider.agents.agent_utils import read_file
from bioguider.agents.collection_task import CollectionTask
from bioguider.agents.consistency_evaluation_task import ConsistencyEvaluationTask
from bioguider.agents.prompt_utils import EVALUATION_INSTRUCTION, CollectionGoalItemEnum
from bioguider.utils.constants import (
    DEFAULT_TOKEN_USAGE, 
    ProjectMetadata,
    StructuredEvaluationInstallationResult,
    FreeEvaluationInstallationResult,
    EvaluationInstallationResult,
)
from bioguider.rag.data_pipeline import count_tokens
from .common_agent_2step import CommonAgentTwoSteps, CommonAgentTwoChainSteps
from ..utils.pyphen_utils import PyphenReadability

from .evaluation_task import EvaluationTask
from .agent_utils import read_file
from bioguider.utils.utils import increase_token_usage
from .evaluation_userguide_prompts import CONSISTENCY_EVAL_SYSTEM_PROMPT, INDIVIDUAL_USERGUIDE_EVALUATION_SYSTEM_PROMPT
from .consistency_collection_task import ConsistencyCollectionTask

class ConsistencyEvaluationResult(BaseModel):
    consistency_score: str=Field(description="A string value, could be `Poor`, `Fair`, `Good`, or `Excellent`")
    consistency_assessment: str=Field(description="Your evaluation of whether the user guide/API documentation is consistent with the code definitions")
    consistency_development: list[str]=Field(description="A list of inconsistent function/class/method name and inconsistent docstring")
    consistency_strengths: list[str]=Field(description="A list of strengths of the user guide/API documentation on consistency")

class UserGuideEvaluationResult(BaseModel):
    overall_score: str=Field(description="A string value, could be `Poor`, `Fair`, `Good`, or `Excellent`")
    overall_key_strengths: str=Field(description="A string value, the key strengths of the user guide")
    overall_improvement_suggestions: str=Field(description="Suggestions to improve the overall score if necessary")
    readability_score: str=Field(description="A string value, could be `Poor`, `Fair`, `Good`, or `Excellent`")
    readability_suggestions: str=Field(description="Suggestions to improve readability if necessary")
    context_and_purpose_score: str=Field(description="A string value, could be `Poor`, `Fair`, `Good`, or `Excellent`")
    context_and_purpose_suggestions: str=Field(description="Suggestions to improve context and purpose if necessary")
    error_handling_score: str=Field(description="A string value, could be `Poor`, `Fair`, `Good`, or `Excellent`")
    error_handling_suggestions: str=Field(description="Suggestions to improve error handling if necessary")

class IndividualUserGuideEvaluationResult(BaseModel):
    user_guide_evaluation: UserGuideEvaluationResult | None=Field(description="The evaluation result of the user guide")
    consistency_evaluation: ConsistencyEvaluationResult | None=Field(description="The evaluation result of the consistency of the user guide")

logger = logging.getLogger(__name__)

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
    ):
        super().__init__(llm, repo_path, gitignore_path, meta_data, step_callback, summarized_files_db)
        self.evaluation_name = "User Guide Evaluation"
        self.code_structure_db = code_structure_db

    def _collect_files(self):
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
        return files

    def _evaluate_consistency(self, file: str) -> tuple[EvaluationInstallationResult | None, dict, list[str]]:
        consistency_evaluation_task = ConsistencyEvaluationTask(
            llm=self.llm,
            code_structure_db=self.code_structure_db,
            step_callback=self.step_callback,
        )
        with open(Path(self.repo_path, file), "r") as f:
            user_guide_api_documentation = f.read()
        res = consistency_evaluation_task.evaluate(user_guide_api_documentation)

        consistency_evaluation_task.evaluate(file)

        consistency_collect_task = ConsistencyCollectionTask(
            llm=self.llm,
            code_structure_db=self.code_structure_db,
            step_callback=self.step_callback,
        )
        consistency_collect_task.compile(repo_path=self.repo_path, gitignore_path=Path(self.repo_path, ".gitignore"))
        with open(Path(self.repo_path, file), "r") as f:
            user_guide_api_documentation = f.read()
        res, code_definitions = consistency_collect_task.collect(user_guide_api_documentation)

        if not res:
            # No sufficient information to evaluate the consistency of the user guide/API documentation
            return None, {**DEFAULT_TOKEN_USAGE}

        system_prompt = ChatPromptTemplate.from_template(
            CONSISTENCY_EVAL_SYSTEM_PROMPT
        ).format(
            user_guide_api_documentation=user_guide_api_documentation,
            code_definitions=code_definitions,
        )
        agent = CommonAgentTwoSteps(llm=self.llm)
        res, _, token_usage, reasoning_process = agent.go(
            system_prompt=system_prompt,
            instruction_prompt="Now, let's begin the consistency evaluation step.",
            schema=ConsistencyEvaluationResult,
        )
        res: ConsistencyEvaluationResult = res
        self.print_step(step_output=f"Consistency Evaluation Result: {res}")
        self.print_step(step_output=f"Consistency Evaluation Reasoning Process: {reasoning_process}")
        self.print_step(token_usage=token_usage)

        return res, token_usage

    def _evaluate_individual_userguide(self, file: str) -> tuple[IndividualUserGuideEvaluationResult | None, dict]:
        content = read_file(Path(self.repo_path, file))
        
        if content is None:
            logger.error(f"Error in reading file {file}")
            return None, {**DEFAULT_TOKEN_USAGE}

        readability = PyphenReadability()
        flesch_reading_ease, flesch_kincaid_grade, gunning_fog_index, smog_index, \
                _, _, _, _, _ = readability.readability_metrics(content)
        system_prompt = ChatPromptTemplate.from_template(
            INDIVIDUAL_USERGUIDE_EVALUATION_SYSTEM_PROMPT
        ).format(
            flesch_reading_ease=flesch_reading_ease,
            flesch_kincaid_grade=flesch_kincaid_grade,
            gunning_fog_index=gunning_fog_index,
            smog_index=smog_index,
            userguide_content=content,
        )
        agent = CommonAgentTwoSteps(llm=self.llm)
        res, _, token_usage, reasoning_process = agent.go(
            system_prompt=system_prompt,
            instruction_prompt="Now, let's begin the user guide/API documentation evaluation.",
            schema=UserGuideEvaluationResult,
        )
        res: UserGuideEvaluationResult = res

        consistency_evaluation_result, _temp_token_usage = self._evaluate_consistency(file)
        if consistency_evaluation_result is None:
            # No sufficient information to evaluate the consistency of the user guide/API documentation
            consistency_evaluation_result = ConsistencyEvaluationResult(
                consistency_score="N/A",
                consistency_assessment="No sufficient information to evaluate the consistency of the user guide/API documentation",
                consistency_development=[],
                consistency_strengths=[],
            )
        return IndividualUserGuideEvaluationResult(
            user_guide_evaluation=res,
            consistency_evaluation=consistency_evaluation_result,
        ), token_usage

    def _evaluate(self, files: list[str] | None = None) -> tuple[dict[str, IndividualUserGuideEvaluationResult] | None, dict, list[str]]:
        total_token_usage = {**DEFAULT_TOKEN_USAGE}
        user_guide_evaluation_results = {}
        for file in files:
            user_guide_evaluation_result, token_usage = self._evaluate_individual_userguide(file)
            total_token_usage = increase_token_usage(total_token_usage, token_usage)
            user_guide_evaluation_results[file] = user_guide_evaluation_result

        return user_guide_evaluation_results, total_token_usage, files
