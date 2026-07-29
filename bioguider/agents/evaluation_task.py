
import os
from pathlib import Path
import logging
from typing import Callable
from abc import ABC, abstractmethod
from langchain.prompts import ChatPromptTemplate
from langchain_openai.chat_models.base import BaseChatOpenAI

from bioguider.agents.agent_utils import read_file
from bioguider.agents.prompt_utils import EVALUATION_INSTRUCTION
from bioguider.database.summarized_file_db import SummarizedFilesDb
from bioguider.generation.prompts import load_prompt
from bioguider.utils.constants import DEFAULT_TOKEN_USAGE, ProjectMetadata
from .common_conversation import CommonConversation
from ..utils.pyphen_utils import PyphenReadability

logger = logging.getLogger(__name__)

# Extracted to bioguider/generation/prompts/evaluation_readme.txt so the same
# prompt text can be re-used verbatim in the Claude-Code comparison run.
EVALUATION_README_SYSTEM_PROMPT = load_prompt("evaluation_readme")

class EvaluationTask(ABC):
    def __init__(
        self, 
        llm: BaseChatOpenAI, 
        repo_path: str, 
        gitignore_path: str,
        meta_data: ProjectMetadata | None = None,
        step_callback: Callable | None = None,
        summarized_files_db: SummarizedFilesDb | None=None,
    ):
        self.evaluation_name = ""
        self.llm = llm
        self.repo_path = repo_path
        self.gitignore_path = gitignore_path
        self.step_callback = step_callback
        self.metadata = meta_data
        self.summarized_files_db = summarized_files_db

    def print_step(
        self,
        step_name: str | None = None,
        step_output: str | None = None,
        token_usage: dict | None = None,
    ):
        if self.step_callback is None:
            return
        self.step_callback(
            step_name=step_name,
            step_output=step_output,
            token_usage=token_usage,
        )

    def evaluate(self) -> tuple[dict, list[str]]:
        self._enter_evaluation()
        files = self._collect_files()
        evaluations, token_usage, files = self._evaluate(files)
        self._leave_evaluation(token_usage)
        return evaluations, files
    
    def _enter_evaluation(self):
        self.print_step(step_name=self.evaluation_name)

    def _leave_evaluation(self, token_usage):
        self.print_step(token_usage=token_usage)

    @abstractmethod
    def _evaluate(self, files: list[str]) -> tuple[dict, dict, list[str]]:
        pass

    @abstractmethod
    def _collect_files(self) -> list[str]:
        pass
