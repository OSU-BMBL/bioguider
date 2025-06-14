
import os
from pathlib import Path
import logging
from typing import Callable
from abc import ABC, abstractmethod
from langchain.prompts import ChatPromptTemplate
from langchain_openai.chat_models.base import BaseChatOpenAI

from bioguider.agents.agent_utils import read_file
from bioguider.utils.constants import ProjectMetadata
from .common_agent_2step import CommonAgentTwoSteps
from .common_agent import CommonConversation
from ..utils.pyphen_utils import PyphenReadability
from ..utils.gitignore_checker import GitignoreChecker

logger = logging.getLogger(__name__)

EVALUATION_README_SYSTEM_PROMPT = """
You are an expert in evaluating the quality of README files in software repositories. Your task is to analyze the README file found in the repository and provide a detailed evaluation based on the following criteria:
1. **Clarity**: Is the purpose of the project clearly stated?
2. **Installation Instructions**: Are there clear instructions on how to install the project?
3. **Usage Instructions**: Are there basic instructions on how to use the project?
4. **Contributing Guidelines**: Are there guidelines for contributing to the project?
5. **License Information**: Is the license clearly stated?
6. **Overall Quality**: Provide an overall assessment of the README file's quality.
7. **Readability**: You are provided the following metrics scores calculated with pyphen, please evaluate readability based on the scores:
  Flesch Reading Ease: {flesch_reading_ease} (206.835 - 1.015(words/sentences) - 84.6(syllables/words))
  Flesch-Kincaid Grade Level: {flesch_kincaid_grade} (0.39(words/sentences) + 11.8(syllables/words) - 15.59)
  Gunning Fog Index: {gunning_fog_index} (0.4[(words/sentences) + 100(complex words/words)])
  SMOG Index: {smog_index} (1.043*sqrt(polysyllables * (30/sentences)) + 3.1291)

Here is README:
{readme_content}
"""

class EvaluationTask(ABC):
    def __init__(
        self, 
        llm: BaseChatOpenAI, 
        repo_path: str, 
        gitignore_path: str,
        meta_data: ProjectMetadata | None = None,
        step_callback: Callable | None = None
    ):
        self.evaluation_name = ""
        self.llm = llm
        self.repo_path = repo_path
        self.gitignore_path = gitignore_path
        self.step_callback = step_callback
        self.metadata = meta_data
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

    def evaluate(self, files: list[str] | None = None):
        self._enter_evaluation()
        evaluation, token_usage = self._evaluate(files)
        self._leave_evaluation(token_usage)
        return evaluation
    
    def _enter_evaluation(self):
        self.print_step(step_name=self.evaluation_name)

    def _leave_evaluation(self, token_usage):
        self.print_step(token_usage=token_usage)

    @abstractmethod
    def _evaluate(self, files: list[str]):
        pass

class EvaluationREADMETask(EvaluationTask):
    def __init__(
        self, 
        llm: BaseChatOpenAI, 
        repo_path: str, 
        gitignore_path: str,
        meta_data: ProjectMetadata | None = None,
        step_callback: Callable | None = None
    ):
        super().__init__(llm, repo_path, gitignore_path, meta_data, step_callback)
        self.evaluation_name = "README Evaluation"
            
    def _evaluate(self, files: list[str]):
        readme_files = files
        if readme_files is None or len(readme_files) == 0:
            return None
        
        readme_evaluations = {}
        for readme_file in readme_files:
            readme_path = Path(self.repo_path, readme_file)
            readme_content = read_file(readme_path)
            if readme_content is None:
                logger.error(f"Error in reading file {readme_file}")
                continue

            readability = PyphenReadability()
            flesch_reading_ease, flesch_kincaid_grade, gunning_fog_index, smog_index, \
                _, _, _, _, _ = readability.readability_metrics(readme_content)
            system_prompt = ChatPromptTemplate.from_template(
                EVALUATION_README_SYSTEM_PROMPT
            ).format(
                readme_content=readme_content,
                flesch_reading_ease=flesch_reading_ease,
                flesch_kincaid_grade=flesch_kincaid_grade,
                gunning_fog_index=gunning_fog_index,
                smog_index=smog_index,
            )
            conversation = CommonConversation(llm=self.llm)
            response, token_usage = conversation.generate(
                system_prompt=system_prompt,
                instruction_prompt="Before arriving at the conclusion, clearly explain your reasoning step by step. Now, let's begin the evaluation."
            )
            self.print_step(step_output=f"README: {readme_file}")
            self.print_step(step_output=response)
            readme_evaluations[readme_file] = response
        return readme_evaluations, token_usage
        
EVALUATION_TUTORIAL_SYSTEM_PROMPT="""
You are an expert in software documentation and developer education.
You are given the content of a tutorial file from a GitHub repository. Your task is to **critically evaluate** the quality of this tutorial based on best practices in technical writing and developer onboarding.
Please assess the tutorial using the following criteria. Provide your evaluation in structured sections:

---

### **Evaluation Criteria:**
1. **Readability**: You are provided the following metrics scores calculated with pyphen, please evaluate readability based on the scores:
   * Flesch Reading Ease: {flesch_reading_ease} (206.835 - 1.015(words/sentences) - 84.6(syllables/words))
   * Flesch-Kincaid Grade Level: {flesch_kincaid_grade} (0.39(words/sentences) + 11.8(syllables/words) - 15.59)
   * Gunning Fog Index: {gunning_fog_index} (0.4[(words/sentences) + 100(complex words/words)])
   * SMOG Index: {smog_index} (1.043*sqrt(polysyllables * (30/sentences)) + 3.1291)
2. **Coverage**
   * Does the tutorial cover all major steps needed to get started?
   * Are dependencies, prerequisites, setup steps, and example usage included?
3. **Structure & Organization**
   * Is the content logically structured (e.g., introduction → setup → examples → summary)?
   * Are sections well-labeled and easy to navigate?
4. **Balance Between Code and Explanation**
   * Is there a good balance between code snippets and narrative explanation?
   * Are code blocks properly annotated or explained?
5. **Terminology Consistency**
   * Is technical terminology used consistently and accurately?
   * Are key terms introduced and reused correctly?
6. **Example Quality**
   * Are the examples relevant, correct, and representative of real usage?
   * Are edge cases or typical user pitfalls addressed?
7. **Formatting and Style**
   * Are headings, bullet points, code formatting, and markdown style used effectively?
   * Are there any formatting issues that hurt clarity?
---

### **Output Format:**
Please respond in the following format:

```
**FinalAnswer**
**Readability**: Your comments here  
**Coverage**: Your comments here  
**Structure & Organization**: Your comments here  
**Code vs. Explanation Balance**: Your comments here  
**Terminology Consistency**: Your comments here  
**Example Quality**: Your comments here  
**Formatting and Style**: Your comments here  
**Overall Rating**: [Poor / Fair / Good / Excellent]  
```

---

### **Tutorial File Content:**

```
{tutorial_file_content}
```

---
"""
class EvaluationTutorialTask(EvaluationTask):
    def __init__(
        self, 
        llm: BaseChatOpenAI, 
        repo_path: str, 
        gitignore_path: str,
        meta_data: ProjectMetadata | None = None,
        step_callback: Callable | None = None
    ):
        super().__init__(llm, repo_path, gitignore_path, meta_data, step_callback)
        self.evaluation_name = "Tutorial Evaluation"

    def _evaluate(self, files: list[str]):
        if len(files) == 0:
            return None
        
        evaluations = {}
        for file in files:
            tutorial_path = Path(self.repo_path, file)
            tutorial_content = read_file(tutorial_path)
            if tutorial_content is None:
                logging.error(f"Error in reading file {file}")
                continue

            readability = PyphenReadability()
            flesch_reading_ease, flesch_kincaid_grade, gunning_fog_index, smog_index, \
                _, _, _, _, _ = readability.readability_metrics(tutorial_content)
            system_prompt = ChatPromptTemplate.from_template(
                EVALUATION_TUTORIAL_SYSTEM_PROMPT
            ).format(
                tutorial_file_content=tutorial_content,
                flesch_reading_ease=flesch_reading_ease,
                flesch_kincaid_grade=flesch_kincaid_grade,
                gunning_fog_index=gunning_fog_index,
                smog_index=smog_index,
            )
            conversation = CommonConversation(llm=self.llm)
            response, token_usage = conversation.generate(
                system_prompt=system_prompt,
                instruction_prompt="Before arriving at the conclusion, clearly explain your reasoning step by step. Now, let's begin the evaluation."
            )
            self.print_step(step_output=f"Tutorial: {file}")
            self.print_step(step_output=response)
            evaluations[file] = response
        return evaluations, token_usage

