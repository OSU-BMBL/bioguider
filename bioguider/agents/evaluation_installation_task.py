import os
from pathlib import Path
import logging
from typing import Callable, Optional
from abc import ABC, abstractmethod
from langchain.prompts import ChatPromptTemplate
from langchain_openai.chat_models.base import BaseChatOpenAI
from pydantic import BaseModel, Field
from markdownify import markdownify as md

from bioguider.agents.agent_utils import read_file
from bioguider.agents.collection_task import CollectionTask
from bioguider.agents.prompt_utils import EVALUATION_INSTRUCTION, CollectionGoalItemEnum
from bioguider.utils.constants import DEFAULT_TOKEN_USAGE, ProjectMetadata
from bioguider.rag.data_pipeline import count_tokens
from .common_agent_2step import CommonAgentTwoSteps, CommonAgentTwoChainSteps
from .common_agent import CommonConversation
from ..utils.pyphen_utils import PyphenReadability
from ..utils.gitignore_checker import GitignoreChecker
from .evaluation_task import EvaluationTask
from .agent_utils import increase_token_usage, read_file


logger = logging.getLogger(__name__)

STRUCTURED_EVALUATION_INSTALLATION_SYSTEM_PROMPT = """
You are an expert in evaluating the quality of installation information in software repositories. 
Your task is to analyze the provided files related to installation and generate a structured quality assessment based on the following criteria.
---

### **Evaluation Criteria**

1. **Installation Available**: Is the installation section in document (like README.md or INSTALLATION)?
   * Output: `Yes` or `No`

2. **Installation Tutorial**: Is the step-by-step installation tutorial provided?
   * Ouput: `Yes` or `No`

3. **Number of required Dependencies Installation**: The number of dependencies that are required to install
   * Output: Number
   * Suggest specific improvements if necessary, such as missing dependencies

4. **Compatible Operating System**: Is the compatible operating system described?
   * Output: `Yes` or `No`

5. **Overall Score**: Give an overall quality rating of the Installation information.
   * Output: `Poor`, `Fair`, `Good`, or `Excellent`

---

### **Final Report Ouput**
Your final report must **exactly match** the following format. Do not add or omit any sections.

**FinalAnswer**
**Install Available:** [Yes / No]
**Install Tutorial:** [Yes / No]
**Dependency:**
  * number: [Number]
  * suggestions: <suggestion to improve **dependency information** like missing dependencies
**Compatible Operating System:** [Yes / No]
**Overall Score:** [Poor / Fair / Good / Excellent]

---

### Installation Files Provided:
{installation_files_content}

"""


EVALUATION_INSTALLATION_SYSTEM_PROMPT = """
You are an expert in evaluating the quality of **installation instructions** in software repositories.
Your task is to analyze the provided content of installation-related files and generate a **comprehensive, structured quality report**.

---

### Evaluation Criteria

Please assess the installation information using the following criteria. For each, provide a concise evaluation and specific feedback:

1. **Ease of Access**
   * Is the installation information clearly presented and easy to locate within the repository?
   * Is it included in a top-level README, a dedicated INSTALL.md file, or other accessible locations?

2. **Clarity of Dependency Specification**
   * Are all software and library dependencies clearly listed?
   * Are installation methods (e.g., `pip`, `conda`, `apt`) for those dependencies explicitly provided?

3. **Hardware Requirements**
   * Does the documentation specify hardware needs (e.g., GPU, memory, OS) if relevant?

4. **Step-by-Step Installation Guide**
   * Is there a clear, ordered set of instructions for installing the software?
   * Are example commands or configuration steps provided to help users follow along?

---

### Output Format

Your response **must exactly follow** the structure below:

**FinalAnswer**
**Overall Score:** [Poor / Fair / Good / Excellent]  
**Ease of Access:** <your comments>  
**Clarity of Dependency Specification:** <your comments>  
**Hardware Requirements:** <your comments>  
**Installation Guide:** <your comments>  

---

### Installation Files Provided:
{installation_files_content}

"""

class StructuredEvaluationInstallationResult(BaseModel):
    install_available: Optional[bool]=Field(description="A boolean value. Is the installation documents accessible and present?")
    install_tutorial: Optional[bool]=Field(description="A boolean value. Is the installation tutorial provided?")
    dependency_number: Optional[int]=Field(description="A number. It is the number of dependencies that are required to install.")
    dependency_suggestions: Optional[str]=Field(description="A string value. It is the specific improvements if necessary, such as missing dependencies")
    compatible_os: Optional[bool]=Field(description="A boolean value. Is compatible operating system described?")
    overall_score: Optional[str]=Field(description="A overall scroll for the installation quality, could be `Poor`, `Fair`, `Good`, or `Excellent`")

class EvaluationInstallationResult(BaseModel):
    ease_of_access: Optional[str]=Field(description="Is the installation information easy to access")
    score: Optional[str]=Field(description="An overall score, could be Poor, Fair, Good or Excellent")
    clarity_of_dependency: Optional[str]=Field(description="Are all dependencies clearly listed")
    hardware_requirements: Optional[str]=Field(description="Are all hardware requirements clearly specified")
    installation_guide: Optional[str]=Field(description="Is there a clear, ordered set of instructions for installing the software")

EvaluationInstallationResultSchema = {
    "title": "EvaluationREADMEResult",
    "type": "object",
    "properties": {
        "ease_of_access": {
            "anyOf": [{"type": "string"}, {"type": "null"}],
            "description": "Is the installation information easy to access",
            "title": "Ease of Access"
        },
        "score": {
            "anyOf": [{"type": "string"}, {"type": "null"}],
            "description": "An overall score, could be Poor, Fair, Good or Excellent",
            "title": "Score"
        },
        "clarity_of_dependency": {
            "anyOf": [{"type": "string"}, {"type": "null"}],
            "description": "Are all dependencies clearly listed",
            "title": "Clarity of Dependency",
        },
        "hardware_requirements": {
            "anyOf": [{"type": "string"}, {"type": "null"}],
            "description": "Are all hardware requirements clearly specified",
            "title": "Hardware Requirements"
        },
        "installation_guide": {
            "anyOf": [{"type": "string"}, {"type": "null"}],
            "description": "Is there a clear, ordered set of instructions for installing the software",
            "title": "Installation Guide"
        }
    },
    "required": ["ease_of_access", "score", "clarity_of_dependency", "hardware_requirements", "installation_guide"]
}

class EvaluationInstallationTask(EvaluationTask):
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
        self.evaluation_name = "Installation Evaluation"


    def _collect_install_files_content(self, files: list[str] | None=None) -> str:
        if files is None or len(files) == 0:
            return "N/A"
        files_content = ""
        MAX_TOKENS = os.environ.get("OPENAI_MAX_INPUT_TOKENS", 102400)
        for f in files:
            if f.endswith(".html") or f.endswith(".htm"):
                html = read_file(os.path.join(self.repo_path, f))
                content = md(html, escape_underscores=False)
            else:
                content = read_file(os.path.join(self.repo_path, f))
            if count_tokens(content) > int(MAX_TOKENS):
                content = content[:100000]
            files_content += f"""
{f} content:
{content}

"""
        return files_content
    
    def _structured_evaluate(self, files: list[str] | None = None) -> tuple[dict|None, dict]:
        if files is None or len(files) == 0:
            return None, {**DEFAULT_TOKEN_USAGE}
        
        files_content = self._collect_install_files_content(files)
        system_prompt = ChatPromptTemplate.from_template(
            STRUCTURED_EVALUATION_INSTALLATION_SYSTEM_PROMPT,
        ).format(
            installation_files_content=files_content,
        )
        agent = CommonAgentTwoChainSteps(llm=self.llm)
        res, _, token_usage, reasoning_process = agent.go(
            system_prompt=system_prompt,
            instruction_prompt=EVALUATION_INSTRUCTION,
            schema=StructuredEvaluationInstallationResult,
        )
        self.print_step(step_output=reasoning_process)
        self.print_step(token_usage=token_usage)

        return {
            "structured_evaluation": res,
            "structured_reasoning_process": reasoning_process,
        }, token_usage
    
    def _free_evaluate(self, files: list[str] | None=None) -> tuple[dict|None, dict]:
        if files is None or len(files) == 0:
            return None, {**DEFAULT_TOKEN_USAGE}
        
        files_content = self._collect_install_files_content(files)
        system_prompt = ChatPromptTemplate.from_template(EVALUATION_INSTALLATION_SYSTEM_PROMPT).format(
            installation_files_content=files_content
        )
        agent = CommonAgentTwoChainSteps(llm=self.llm)
        res, _, token_usage, reasoning_process = agent.go(
            system_prompt=system_prompt,
            instruction_prompt=EVALUATION_INSTRUCTION,
            schema=EvaluationInstallationResultSchema,
        )
        res = EvaluationInstallationResult(**res)
        self.print_step(step_output=reasoning_process)
        self.print_step(token_usage=token_usage)
        evaluation = {
            "evaluation": res,
            "reasoning_process": reasoning_process,
        }
        return evaluation, token_usage
    
    def _evaluate(self, files: list[str] | None = None) -> tuple[dict | None, dict, list[str]]:
        evaluation, token_usage = self._free_evaluate(files)
        structured_evaluation, structured_token_usage = self._structured_evaluate(files)

        combined_evaluation = {
            **evaluation,
            **structured_evaluation,
        }
        total_token_usage = increase_token_usage(token_usage, structured_token_usage)

        return combined_evaluation, total_token_usage, files

    def _collect_files(self):
        task = CollectionTask(
            llm=self.llm,
            step_callback=self.step_callback,
        )
        task.compile(
            repo_path=self.repo_path,
            gitignore_path=Path(self.repo_path, ".gitignore"),
            db=self.summarized_files_db,
            goal_item=CollectionGoalItemEnum.Installation.name,
        )
        files = task.collect()
        if files is None:
            return []
        return files
