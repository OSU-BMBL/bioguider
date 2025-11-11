import os
from pathlib import Path
import logging
from langchain.prompts import ChatPromptTemplate
from markdownify import markdownify as md

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

from .evaluation_task import EvaluationTask
from .agent_utils import read_file
from bioguider.utils.utils import get_overall_score, increase_token_usage


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

5. **Hardware Requirements**: Is the hardware requirements described?
   * Output: `Yes` or `No`

6. **Overall Score**: Give an overall quality rating of the Installation information.
   * Output: a number between 0 and 100 representing the overall quality rating.
   * **Grade Level**:
     - **85-100**: The installation information is exceptionally clear, polished, and engaging. It reads smoothly, with minimal effort required from the reader.
     - **65-84**: The installation information is clear and easy to understand, with a natural flow and minimal jargon.
     - **45-64**: The installation information is somewhat clear, but could benefit from more polish and consistency.
     - **0-44**: The installation information is difficult to understand, with unclear language, jargon, or overly complex sentences.

### Installation Files Provided:
{installation_files_content}

"""


FREE_EVALUATION_INSTALLATION_SYSTEM_PROMPT = """
You are an expert in evaluating the quality of **installation instructions** in software repositories.
Your task is to analyze the provided content of installation-related files and generate a **comprehensive, structured quality report**.
You will be given:
1. The content of installation-related files.
2. A structured evaluation of the installation-related files and its reasoning process.

---

### **Instructions**
1. Based on the provided structured evaluation and its reasoning process, generate a free evaluation of the installation-related files.
2. Focus on the explanation of assessment in structured evaluation and how to improve the installation-related files based on the structured evaluation and its reasoning process.
   * For each suggestion to improve the installation-related files, you **must provide some examples** of the original text snippet and the improving comments.
3. For each item in the structured evaluation, provide a detailed assessment followed by specific, actionable comments for improvement.
4. Your improvement suggestions must also include the original text snippet and the improving comments.
5. Your improvement suggestions must also include suggestions to improve readability.
6. If you think the it is good enough, you can say so.
7. In each section output, please first give a detailed explanation of the assessment, and then provide the detailed suggestion for improvement. If you think the it is good enough, you can say so.
  The following is an example of the output format:
  **Ease of Access:**
    Detailed explanation of the assessment. Such as: The INSTALLATION file is present in the repository. The content of the file has been shared completely and is accessible. This confirms the availability of the installation documentation for evaluation. There's no issue with availability.
    Detailed suggestion for improvement. Such as: No need to improve accessibility of the installation documentation.
  **Clarity of Dependency Specification:**
    Detailed explanation of the assessment. Such as: The installation section provides a clear list of dependencies. No need to improve.
    Detailed suggestion for improvement. Such as: No need to improve.
  **Hardware Requirements:**
    Detailed explanation of the assessment. Such as: No hardware requirements are provided in the installation section.
    Detailed suggestion for improvement. Such as: Add a section to describe the hardware requirements for the installation.
  **Installation Guide:**
    Detailed explanation of the assessment. Such as: The installation guide is present in the INSTALLATION file, but lacks a clear and concise guide for the installation process.
    Detailed suggestion for improvement. Such as: 
    - Add a clear and concise guide for the installation process.
    - Improve the readability of the installation guide, like:
      - <original text snippet> - <improving comments>
      - <original text snippet> - <improving comments>
      - ...
  **Compatible Operating System:**
    Detailed explanation of the assessment. Such as: No compatible operating system is provided in the installation section.
    Detailed suggestion for improvement. Such as: 
    - Add a section to describe the compatible operating systems for the installation.
    - For example, Ubuntu 22.04 LTS+, CentOS 7+, Mac OS X 10.15+, Windows 10+ are supported.
    - ...
  **Overall Score:**
    Detailed explanation of the assessment. Such as: The installation section provides a clear list of installation steps. No need to improve.
    Detailed suggestion for improvement. Such as: Add a section to describe the compatible operating systems for the installation.

---

### **Output Format**
Your output must **exactly match** the following format. Do not add or omit any sections. In the output, your detailed assessment and detailed suggestion must include the original text snippet and the improving comments, and include the analysis and explanation.

**FinalAnswer**
**Ease of Access:** 
  <Your assessment and suggestion here>
**Clarity of Dependency Specification:**
  <Your assessment and suggestion here>
**Hardware Requirements:**
  <Your assessment and suggestion here>
**Installation Guide:**
  <Your assessment and suggestion here>  
**Compatible Operating System:**
  <Your assessment and suggestion here>
**Overall Score:**
  <Your assessment and suggestion here>

---

### **Structured Evaluation and Reasoning Process**
{structured_evaluation_and_reasoning_process}

---

### Installation Files Provided:
{installation_files_content}

"""

class EvaluationInstallationTask(EvaluationTask):
    def __init__(
        self, 
        llm, 
        repo_path,
        gitignore_path, 
        meta_data = None, 
        step_callback = None,
        summarized_files_db = None,
        collected_files: list[str] | None = None,
    ):
        super().__init__(llm, repo_path, gitignore_path, meta_data, step_callback, summarized_files_db)
        self.evaluation_name = "Installation Evaluation"
        self.collected_files = collected_files

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
        res: StructuredEvaluationInstallationResult = res
        res.overall_score = get_overall_score([
            res.install_available, 
            res.install_tutorial, 
            res.compatible_os, 
            res.hardware_requirements,
        ], [3, 3, 1, 1])
        res.dependency_number = 0 if res.dependency_number is None else res.dependency_number
        self.print_step(step_output=reasoning_process)
        self.print_step(token_usage=token_usage)

        return {
            "evaluation": res,
            "reasoning_process": reasoning_process,
        }, token_usage
    
    def _free_evaluate(
        self, 
        files: list[str] | None=None,
        structured_evaluation_and_reasoning_process: str | None=None,
    ) -> tuple[dict|None, dict]:
        if files is None or len(files) == 0:
            return None, {**DEFAULT_TOKEN_USAGE}
        
        structured_evaluation_and_reasoning_process = structured_evaluation_and_reasoning_process or "N/A"
        files_content = self._collect_install_files_content(files)
        system_prompt = ChatPromptTemplate.from_template(FREE_EVALUATION_INSTALLATION_SYSTEM_PROMPT).format(
            installation_files_content=files_content,
            structured_evaluation_and_reasoning_process=structured_evaluation_and_reasoning_process,
        )
        agent = CommonAgentTwoSteps(llm=self.llm)
        res, _, token_usage, reasoning_process = agent.go(
            system_prompt=system_prompt,
            instruction_prompt=EVALUATION_INSTRUCTION,
            schema=FreeEvaluationInstallationResult,
        )
        self.print_step(step_output=reasoning_process)
        self.print_step(token_usage=token_usage)
        evaluation = {
            "evaluation": res,
            "reasoning_process": reasoning_process,
        }
        return evaluation, token_usage
    
    def _evaluate(self, files: list[str] | None = None) -> tuple[EvaluationInstallationResult | None, dict, list[str]]:
        total_token_usage = {**DEFAULT_TOKEN_USAGE}

        structured_evaluation, structured_token_usage = self._structured_evaluate(files)
        total_token_usage = increase_token_usage(total_token_usage, structured_token_usage)
        evaluation, token_usage = self._free_evaluate(files, structured_evaluation["reasoning_process"])
        total_token_usage = increase_token_usage(total_token_usage, token_usage)

        combined_evaluation = EvaluationInstallationResult(
            structured_evaluation=structured_evaluation["evaluation"],
            free_evaluation=evaluation["evaluation"],
            structured_reasoning_process=structured_evaluation["reasoning_process"],
            free_reasoning_process=evaluation["reasoning_process"],
        )

        return combined_evaluation, total_token_usage, files

    def _collect_files(self):
        if self.collected_files is not None:
            return self.collected_files
        
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
