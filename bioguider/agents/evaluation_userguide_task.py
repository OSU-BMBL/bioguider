
import os
from pathlib import Path
import logging
from langchain.prompts import ChatPromptTemplate
from markdownify import markdownify as md
from pydantic import BaseModel, Field

from bioguider.agents.agent_utils import read_file
from bioguider.agents.collection_task import CollectionTask
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
from .evaluation_userguide_prompts import INDIVIDUAL_USERGUIDE_EVALUATION_SYSTEM_PROMPT

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
    ):
        super().__init__(llm, repo_path, gitignore_path, meta_data, step_callback, summarized_files_db)
        self.evaluation_name = "User Guide Evaluation"

    def _collect_files(self):
        task = CollectionTask(
            llm=self.llm,
            step_callback=self.step_callback,
        )
        task.compile(
            repo_path=self.repo_path,
            gitignore_path=Path(self.repo_path, ".gitignore"),
            db=self.summarized_files_db,
            goal_item=CollectionGoalItemEnum.UserGuide.name,
        )
        files = task.collect()
        return files

    def _evaluate_individual_userguide(self, file: str) -> tuple[EvaluationInstallationResult | None, dict, list[str]]:
        content = read_file(file)
        
        if content is None:
            logger.error(f"Error in reading file {file}")
            return None, DEFAULT_TOKEN_USAGE, []
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

    def _evaluate(self, files: list[str] | None = None) -> tuple[EvaluationInstallationResult | None, dict, list[str]]:
        pass
