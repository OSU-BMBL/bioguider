
import os
from pathlib import Path
import logging
from typing import Callable, Optional
from abc import ABC, abstractmethod
from langchain.prompts import ChatPromptTemplate
from langchain_openai.chat_models.base import BaseChatOpenAI
from pydantic import BaseModel, Field

from bioguider.agents.agent_utils import read_file
from bioguider.utils.constants import DEFAULT_TOKEN_USAGE, ProjectMetadata
from .common_agent_2step import CommonAgentTwoSteps, CommonAgentTwoChainSteps
from .common_agent import CommonConversation
from ..utils.pyphen_utils import PyphenReadability
from ..utils.gitignore_checker import GitignoreChecker

logger = logging.getLogger(__name__)

EVALUATION_README_SYSTEM_PROMPT = """
You are an expert in evaluating the quality of README files in software repositories. Your task is to analyze the provided README file and generate a comprehensive quality report.

---

### **Step 1:  Identify README type

First, determine whether the provided README is a **project-level README** (typically at the root of a repository) or a **folder-level README** (typically inside subdirectories).

---

### **Project-level README Evaluation**

If the README is a **project-level** file, evaluate it using the following criteria.
For each criterion below, provide a brief assessment followed by specific, actionable comments for improvement.

**1. Project Clarity & Purpose**
 * **Assessment**: [Your evaluation of whether the project's purpose is clear.]
 * **Improvement Suggestions**:
    * **Original text:** [Quote a specific line/section from the README.]
    * **Improving comments:** [Provide your suggestions to improve clarity.]

**2. Installation Instructions**
 * **Assessment**: [Your evaluation of the installation instructions.]
 * **Improvement Suggestions**:
    * **Original text:** [Quote text related to installation.]
    * **Improving comments:** [Provide your suggestions.]

**3. Usage Instructions**
 * **Assessment**: [Your evaluation of the usage instructions.]
 * **Improvement Suggestions**:
    * **Original text:** [Quote text related to usage.]
    * **Improving comments:** [Provide your suggestions.]

**4. Contributing Guidelines**
 * **Assessment**: [Your evaluation of the contributing guidelines.]
 * **Improvement Suggestions**:
    * **Original text:** [Quote text related to contributions.]
    * **Improving comments:** [Provide your suggestions.]

**5. License Information**
 * **Assessment**: [Your evaluation of the license information.]
 * **Improvement Suggestions**:
    * **Original text:** [Quote text related to the license.]
    * **Improving comments:** [Provide your suggestions.]

**6. Readability Analysis**
 * **Flesch Reading Ease**: `{flesch_reading_ease}` (A higher score is better, with 60-70 being easily understood by most adults).
 * **Flesch-Kincaid Grade Level**: `{flesch_kincaid_grade}` (Represents the US school-grade level needed to understand the text).
 * **Gunning Fog Index**: `{gunning_fog_index}` (A score above 12 is generally considered too hard for most people).
 * **SMOG Index**: `{smog_index}` (Estimates the years of education needed to understand the text).
 * **Assessment**: Based on these scores, evaluate the overall readability and technical complexity of the language used.

**Final Answer**
 The final answer **must exactly match** the following format:
```
  * Project-Level README: Yes / No
  * **Score:** <number from 0 to 100>  
  * **Key Strengths**: <brief summary of the README's strongest points in 2-3 sentences> 
  * **Overall Improvement Suggestions:**
    - "Original text snippet 1" - Improving comment 1  
    - "Original text snippet 2" - Improving comment 2  
    - ...
```

  * **Project-Level README**: Indicate “Yes” if the README is project-level, otherwise “No.”
  * **Score**: Provide an overall quality score (100 = perfect).
  * **Key Strengths**: Provide the README's strongest points in 2-3 sentences
  * **Overall Improvement Suggestions**:
    * List each original text snippet that needs improvement, followed by your suggestion.

---

### **Folder-Level README Evaluation**

If the README is a **folder-level** file, use the following criteria instead.

For each criterion below, provide a brief assessment followed by specific, actionable comments for improvement.

**1. Folder Description**
 * **Assessment**: [Your evaluation of whether it Provides a clear **description** of what the folder contains (e.g., modules, scripts, data).]
 * **Improvement Suggestions**:
    * **Original text:** [Quote a specific line/section from the README.]
    * **Improving comments:** [Provide your suggestions to improve clarity.]

**2. Folder Purpose**
 * **Assessment**: [Your evaluation of whether it explains the **purpose** or **role** of the components inside this subfolder.]
 * **Improvement Suggestions**:
    * **Original text:** [Quote text related to purpose.]
    * **Improving comments:** [Provide your suggestions.]

**3. Usage**
 * **Assessment**: [Your evaluation of whether it includes **usage instructions** specific to this folder (e.g., commands, import paths, input/output files).]
 * **Improvement Suggestions**:
    * **Original text:** [Quote text related to usage.]
    * **Improving comments:** [Provide your suggestions.]

**4. Readability Analysis**
 * **Flesch Reading Ease**: `{flesch_reading_ease}` (A higher score is better, with 60-70 being easily understood by most adults).
 * **Flesch-Kincaid Grade Level**: `{flesch_kincaid_grade}` (Represents the US school-grade level needed to understand the text).
 * **Gunning Fog Index**: `{gunning_fog_index}` (A score above 12 is generally considered too hard for most people).
 * **SMOG Index**: `{smog_index}` (Estimates the years of education needed to understand the text).
 * **Assessment**: Based on these scores, evaluate the overall readability and technical complexity of the language used.

**Final Answer**
  The final answer **must exactly match** the following format:
 * Project-Level README: Yes / No
 * **Score:** <number from 0 to 100>  
  * **Key Strengths**: <brief summary of the README's strongest points in 2-3 sentences> 
  * **Overall Improvement Suggestions:**
    - "Original text snippet 1" - Improving comment 1  
    - "Original text snippet 2" - Improving comment 2  
    - ...
---

### **README path:**
{readme_path}

---

### **README Content:**
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

    def evaluate(self, files: list[str] | None = None) -> dict:
        self._enter_evaluation()
        evaluations, token_usage = self._evaluate(files)
        self._leave_evaluation(token_usage)
        return evaluations
    
    def _enter_evaluation(self):
        self.print_step(step_name=self.evaluation_name)

    def _leave_evaluation(self, token_usage):
        self.print_step(token_usage=token_usage)

    @abstractmethod
    def _evaluate(self, files: list[str]) -> tuple[dict, dict]:
        pass

class EvaluationREADMEResult(BaseModel):
    project_level: Optional[bool]=Field(description="A boolean value specifying if the README file is **project-level** README. TRUE: project-level, FALSE, folder-level")
    score: Optional[float]=Field(description="An overall score")
    key_strengths: Optional[str]=Field(description="A string specifying the key strengths of README file.")
    overall_improvement_suggestions: Optional[list[str]]=Field(description="A list of overall improvement suggestions")

EvaluationREADMEResultSchema = {
    "title": "EvaluationREADMEResult",
    "type": "object",
    "properties": {
        "project_level": {
            "anyOf": [{"type": "boolean"}, {"type": "null"}],
            "description": "A boolean value specifying if the README file is **project-level** README. TRUE: project-level, FALSE: folder-level.",
            "title": "Project Level"
        },
        "score": {
            "anyOf": [{"type": "number"}, {"type": "null"}],
            "description": "An overall score",
            "title": "Score"
        },
        "key_strengths": {
            "anyOf": [{"type": "string"}, {"type": "null"}],
            "description": "A string specifying the key strengths of README file.",
            "title": "Key Strengths",
        },
        "overall_improvement_suggestions": {
            "anyOf": [{"items": {"type": "string"}, "type": "array"}, {"type": "null"}],
            "description": "A list of improvement suggestions",
            "title": "Overall Improvement Suggestions"
        }
    },
    "required": ["project_level", "score", "key_strengths", "overall_improvement_suggestions"]
}

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
            
    def _evaluate(self, files: list[str]) -> tuple[dict, dict]:
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
            if len(readme_content.strip()) == 0:
                readme_evaluations[readme_file] = {
                    "evaluation": {
                        "project_level": "/" in readme_file,
                        "score": 0,
                        "key_strengths": f"{readme_file} is an empty file.",
                        "overall_improvement_suggestions": f"{readme_file} is an empty file.",
                    },
                    "reasoning_process": f"{readme_file} is an empty file.",
                }
                continue

            readability = PyphenReadability()
            flesch_reading_ease, flesch_kincaid_grade, gunning_fog_index, smog_index, \
                _, _, _, _, _ = readability.readability_metrics(readme_content)
            system_prompt = ChatPromptTemplate.from_template(
                EVALUATION_README_SYSTEM_PROMPT
            ).format(
                readme_content=readme_content,
                readme_path=readme_file,
                flesch_reading_ease=flesch_reading_ease,
                flesch_kincaid_grade=flesch_kincaid_grade,
                gunning_fog_index=gunning_fog_index,
                smog_index=smog_index,
            )
            # conversation = CommonConversation(llm=self.llm)
            agent = CommonAgentTwoChainSteps(llm=self.llm)
            response, _, token_usage, reasoning_process = agent.go(
                system_prompt=system_prompt,
                instruction_prompt="Before arriving at the conclusion, clearly explain your reasoning step by step. Now, let's begin the evaluation.",
                schema=EvaluationREADMEResultSchema,
            )
            response = EvaluationREADMEResult(**response)
            self.print_step(step_output=f"README: {readme_file}")
            self.print_step(step_output=reasoning_process)
            readme_evaluations[readme_file] = {
                "evaluation": {
                    "project_level": response.project_level,
                    "score": response.score,
                    "key_strengths": response.key_strengths,
                    "overall_improvement_suggestions": response.overall_improvement_suggestions,
                }, 
                "reasoning_process": reasoning_process
            }
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

    def _evaluate(self, files: list[str]) -> tuple[dict, dict]:
        if len(files) == 0:
            return {}, {**DEFAULT_TOKEN_USAGE}
        
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

