
import logging
from pathlib import Path
from typing import Callable, Optional
from langchain.prompts import ChatPromptTemplate
from langchain_openai.chat_models.base import BaseChatOpenAI
from pydantic import BaseModel, Field

from bioguider.agents.prompt_utils import EVALUATION_INSTRUCTION
from bioguider.utils.gitignore_checker import GitignoreChecker

from ..utils.pyphen_utils import PyphenReadability
from bioguider.agents.agent_utils import increase_token_usage, read_file, summarize_file
from bioguider.agents.common_agent_2step import CommonAgentTwoChainSteps
from bioguider.agents.evaluation_task import EvaluationTask
from bioguider.utils.constants import DEFAULT_TOKEN_USAGE, ProjectMetadata

logger = logging.getLogger(__name__)

STRUCTURED_EVALUATION_README_SYSTEM_PROMPT = """
You are an expert in evaluating the quality of README files in software repositories. 
Your task is to analyze the provided README file and generate a structured quality assessment based on the following criteria.
If a LICENSE file is present in the repository, its content will also be provided to support your evaluation of license-related criteria.
---

### **Evaluation Criteria**

1. **Available**: Is the README accessible and present?
   * Output: `Yes` or `No`

2. **Readability**: Evaluate based on readability metrics such as Flesch-Kincaid Grade Level, SMOG Index, etc.
   * Output: `Poor`, `Fair`, `Good`, or `Excellent`
   * Suggest specific improvements if necessary

3. **Project Purpose**: Is the project's goal or function clearly stated?
   * Output: `Yes` or `No`
   * Provide suggestions if unclear

4. **Hardware and Software Requirements**: Are hardware/software specs and compatibility details included?
   * Output: `Poor`, `Fair`, `Good`, or `Excellent`
   * Suggest how to improve the section if needed

5. **Dependencies**: Are all necessary software libraries and dependencies clearly listed?
   * Output: `Poor`, `Fair`, `Good`, or `Excellent`
   * Suggest improvements if applicable

6. **License Information**: Is license type clearly indicated?
   * Output: `Yes` or `No`
   * Suggest improvement if missing or unclear

7. **Author / Contributor Info**: Are contributor or maintainer details provided?
   * Output: `Yes` or `No`
   * Suggest improvement if missing

8. **Overall Score**: Give an overall quality rating of the README.
   * Output: `Poor`, `Fair`, `Good`, or `Excellent`

---

### **Readability Metrics**
 * **Flesch Reading Ease**: `{flesch_reading_ease}` (A higher score is better, with 60-70 being easily understood by most adults).
 * **Flesch-Kincaid Grade Level**: `{flesch_kincaid_grade}` (Represents the US school-grade level needed to understand the text).
 * **Gunning Fog Index**: `{gunning_fog_index}` (A score above 12 is generally considered too hard for most people).
 * **SMOG Index**: `{smog_index}` (Estimates the years of education needed to understand the text).

---

### **Final Report Ouput**
Your final report must **exactly match** the following format. Do not add or omit any sections.

**FinalAnswer**
**Available:** [Yes / No]
**Readability:** 
  * score: [Poor / Fair / Good / Excellent]
  * suggestions: <suggestions to improve README readability>
**Project Purpose:** 
  * score: [Yes / No]
  * suggestions: <suggestions to improve project purpose.>
**Hardware and software spec and compatibility description:**
  * score: [Poor / Fair / Good / Excellent]
  * suggestions: <suggestions to improve **hardware and software** description>
**Dependencies clearly stated:** 
  * score: [Poor / Fair / Good / Excellent]
  * suggestions: <suggestions to improve **Dependencies** description>
**License Information Included:** 
  * score: [Yes / No]
  * suggestions: <suggestions to improve **License Information**>
** Code contributor / Author information included
  * score: [Yes / No]
**Overall Score:** [Poor / Fair / Good / Excellent]

---

### **README Path**
{readme_path}

---

### **README content**
{readme_content}

---

### **LICENSE Path**
{license_path}

---

### **LICENSE Summarized Content**
{license_summarized_content}

"""

EVALUATION_README_SYSTEM_PROMPT = """
You are an expert in evaluating the quality of README files in software repositories. 
Your task is to analyze the provided README file and generate a comprehensive quality report.

---

### **Step 1:  Identify README type

First, determine whether the provided README is a **project-level README** (typically at the root of a repository) or a **folder-level README** (typically inside subdirectories).

---

### **Evaluation Criteria**

#### If the README is a **project-level** file, evaluate it using the following criteria.

For each criterion below, provide a brief assessment followed by specific, actionable comments for improvement.

**1. Project Clarity & Purpose**
 * **Assessment**: [Your evaluation of whether the project's purpose is clear.]
 * **Improvement Suggestions**:
    * **Original text:** [Quote a specific line/section from the README.]
    * **Improving comments:** [Provide your suggestions to improve clarity.]
    * **Original text:** [Quote a specific line/section from the README.]
    * **Improving comments:** [Provide your suggestions to improve clarity.]
    ...

**2. Installation Instructions**
 * **Assessment**: [Your evaluation of the installation instructions.]
 * **Improvement Suggestions**:
    * **Original text:** [Quote text related to installation.]
    * **Improving comments:** [Provide your suggestions.]
    * **Original text:** [Quote text related to installation.]
    * **Improving comments:** [Provide your suggestions.]
    ...

**3. Usage Instructions**
 * **Assessment**: [Your evaluation of the usage instructions.]
 * **Improvement Suggestions**:
    * **Original text:** [Quote text related to usage.]
    * **Improving comments:** [Provide your suggestions.]
    * **Original text:** [Quote text related to usage.]
    * **Improving comments:** [Provide your suggestions.]
    ...

**4. Contributing Guidelines**
 * **Assessment**: [Your evaluation of the contributing guidelines.]
 * **Improvement Suggestions**:
    * **Original text:** [Quote text related to contributions.]
    * **Improving comments:** [Provide your suggestions.]
    * **Original text:** [Quote text related to contributions.]
    * **Improving comments:** [Provide your suggestions.]
    ...

**5. License Information**
 * **Assessment**: [Your evaluation of the license information.]
 * **Improvement Suggestions**:
    * **Original text:** [Quote text related to the license.]
    * **Improving comments:** [Provide your suggestions.]
    * **Original text:** [Quote text related to the license.]
    * **Improving comments:** [Provide your suggestions.]
    ...

**6. Readability Analysis**
 * **Flesch Reading Ease**: `{flesch_reading_ease}` (A higher score is better, with 60-70 being easily understood by most adults).
 * **Flesch-Kincaid Grade Level**: `{flesch_kincaid_grade}` (Represents the US school-grade level needed to understand the text).
 * **Gunning Fog Index**: `{gunning_fog_index}` (A score above 12 is generally considered too hard for most people).
 * **SMOG Index**: `{smog_index}` (Estimates the years of education needed to understand the text).
 * **Assessment**: Based on these scores, evaluate the overall readability and technical complexity of the language used.

---

#### If if is a **folder-level** file, use the following criteria instead.

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

---

### Final Report Format

#### Your output **must exactly match**  the following template:

**FinalAnswer**

 * Project-Level README: Yes / No
 * **Score:** [Poor / Fair / Good / Excellent]
  * **Key Strengths**: <brief summary of the README's strongest points in 2-3 sentences> 
  * **Overall Improvement Suggestions:**
    - "Original text snippet 1" - Improving comment 1  
    - "Original text snippet 2" - Improving comment 2  
    - ...

#### Notes

* **Project-Level README**: "Yes" if root-level; "No" if folder-level.
* **Score**: Overall quality rating, could be Poor / Fair / Good / Excellent.
* **Key Strengths**: Briefly highlight the README's strongest aspects.
* **Improvement Suggestions**: Provide concrete snippets and suggested improvements.


---

### **README path:**
{readme_path}

---

### **README Content:**
{readme_content}
"""


class StructuredEvaluationREADMEResult(BaseModel):
    available_score: Optional[bool]=Field(description="A boolean value, Is the README accessible and present?")
    readability_score: Optional[str]=Field(description="A string value, could be `Poor`, `Fair`, `Good`, or `Excellent`")
    readability_suggestions: Optional[str]=Field(description="Suggestions to improve readability if necessary")
    project_purpose_score: Optional[bool]=Field(description="A boolean value. Is the project's goal or function clearly stated?")
    project_purpose_suggestions: Optional[str]=Field(description="Suggestions if not clear")
    hardware_and_software_spec_score: Optional[str]=Field(description="A string value, could be `Poor`, `Fair`, `Good`, or `Excellent`")
    hardware_and_software_spec_suggestions: Optional[str]=Field(description="Suggestions if not clear")
    dependency_score: Optional[str]=Field(description="A string value, could be `Poor`, `Fair`, `Good`, or `Excellent`")
    dependency_suggestions: Optional[str]=Field(description="Suggestions if dependencies are not clearly stated")
    license_score: Optional[bool]=Field(description="A boolean value, Are contributor or maintainer details provided?")
    license_suggestions: Optional[str]=Field(description="Suggestions to improve license information")
    contributor_author_score: Optional[bool]=Field(description="A boolean value. are contributors or author included?")
    overall_score: str=Field(description="A overall scroll for the README quality, could be `Poor`, `Fair`, `Good`, or `Excellent`")

class EvaluationREADMEResult(BaseModel):
    project_level: Optional[bool]=Field(description="A boolean value specifying if the README file is **project-level** README. TRUE: project-level, FALSE, folder-level")
    score: Optional[str]=Field(description="An overall score")
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
            "anyOf": [{"type": "string"}, {"type": "null"}],
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
        step_callback: Callable | None = None,
        summarized_files_db = None,
    ):
        super().__init__(llm, repo_path, gitignore_path, meta_data, step_callback, summarized_files_db)
        self.evaluation_name = "README Evaluation"

    def _structured_evaluate(self, free_readme_evaluations: dict[str, dict]):
        """ Evaluate README in structure:
        available: bool
        readability: score and suggestion
        project purpose: bool, suggestion
        hardware and software spec and compatibility description: score and suggestion
        dependencies clearly stated: score and suggestion
        license information included: bool and suggestion
        Code contributor / author information included: bool and suggestion
        overall score: 
        """
        total_token_usage = {**DEFAULT_TOKEN_USAGE}
        if free_readme_evaluations is None:
            return None, total_token_usage
        
        license_path = "LICENSE"
        license_content = read_file(Path(self.repo_path, license_path))
        license_summarized_content = summarize_file(
            llm=self.llm,
            name=license_path,
            content=license_content, 
            level=6,
            summary_instructions="What license is the repository using?",
        ) if license_content is not None else "N/A"
        license_path = license_path if license_content is not None else "N/A"
        structured_readme_evaluations = {}
        for readme_file in free_readme_evaluations.keys():
            evaluation = free_readme_evaluations[readme_file]["evaluation"]
            if not evaluation["project_level"]:
                continue
            full_path = Path(self.repo_path, readme_file)
            readme_content = read_file(full_path)
            if readme_content is None:
                logger.error(f"Error in reading file {readme_file}")
                continue
            if len(readme_content.strip()) == 0:
                structured_readme_evaluations[readme_file] = {
                    "structured_evaluation": StructuredEvaluationREADMEResult(
                        available_score=False,
                        readability_score="Poor",
                        readability_suggestions="No readability provided",
                        project_purpose_score=False,
                        project_purpose_suggestions="No project purpose provided",
                        hardware_and_software_spec_score="Poor",
                        hardware_and_software_spec_suggestions="No hardware and software spec provided",
                        dependency_score="Poor",
                        dependency_suggestions="No dependency provided",
                        license_score=False,
                        license_suggestions="No license information",
                        contributor_author_score=False,
                        overall_score="Poor",
                    ),
                    "structured_reasoning_process": f"{readme_file} is an empty file.",
                }
                continue
            readability = PyphenReadability()
            flesch_reading_ease, flesch_kincaid_grade, gunning_fog_index, smog_index, \
                _, _, _, _, _ = readability.readability_metrics(readme_content)
            system_prompt = ChatPromptTemplate.from_template(
                STRUCTURED_EVALUATION_README_SYSTEM_PROMPT
            ).format(
                readme_path=readme_file,
                readme_content=readme_content,
                license_path=license_path,
                license_summarized_content=license_summarized_content,
                flesch_reading_ease=flesch_reading_ease,
                flesch_kincaid_grade=flesch_kincaid_grade,
                gunning_fog_index=gunning_fog_index,
                smog_index=smog_index,
            )
            agent = CommonAgentTwoChainSteps(llm=self.llm)
            response, _, token_usage, reasoning_process = agent.go(
                system_prompt=system_prompt,
                instruction_prompt=EVALUATION_INSTRUCTION,
                schema=StructuredEvaluationREADMEResult,
            )
            self.print_step(step_output=f"README: {readme_file} structured evaluation")
            self.print_step(step_output=reasoning_process)
            structured_readme_evaluations[readme_file] = {
                "structured_evaluation": response,
                "structured_reasoning_process": reasoning_process,
            }
            total_token_usage = increase_token_usage(total_token_usage, token_usage)

        return structured_readme_evaluations, total_token_usage
        

    def _free_evaluate(self, files: list[str]):
        readme_files = files
        if readme_files is None or len(readme_files) == 0:
            return None, {**DEFAULT_TOKEN_USAGE}
        
        readme_evaluations = {}
        total_token_usage = {**DEFAULT_TOKEN_USAGE}
        for readme_file in readme_files:
            readme_path = Path(self.repo_path, readme_file)
            readme_content = read_file(readme_path)
            if readme_content is None:
                logger.error(f"Error in reading file {readme_file}")
                continue
            if len(readme_content.strip()) == 0:
                readme_evaluations[readme_file] = {
                    "evaluation": {
                        "project_level": not "/" in readme_file,
                        "score": "Poor",
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
                instruction_prompt=EVALUATION_INSTRUCTION,
                schema=EvaluationREADMEResultSchema,
            )
            response = EvaluationREADMEResult(**response)
            self.print_step(step_output=f"README: {readme_file} free evaluation")
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
            total_token_usage = increase_token_usage(total_token_usage, token_usage)
        return readme_evaluations, total_token_usage
            
    def _evaluate(self, files: list[str]) -> tuple[dict, dict, list[str]]:
        free_readme_evaluations, free_token_usage = self._free_evaluate(files)
        structured_readme_evaluations, structured_token_usage = self._structured_evaluate(free_readme_evaluations)

        # combine result
        combined_evaluations = {}
        for f in files:
            if not f in structured_readme_evaluations:
                combined_evaluations = {**free_readme_evaluations[f]}
            else:
                combined_evaluations[f] = {
                    **free_readme_evaluations[f],
                    **structured_readme_evaluations[f],
                }
        
        total_token_usage = increase_token_usage(free_token_usage, structured_token_usage)

        return combined_evaluations, total_token_usage, files
    
    def _collect_files(self):
        """
        Search for a README file in the repository directory.
        """
        possible_readme_files = [
            "readme.md",
            "readme.rst",
            "readme.txt",
            "readme",
        ]
        repo_path = self.repo_path
        gitignore_path = Path(repo_path, ".gitignore")
        gitignore_checker = GitignoreChecker(
            directory=repo_path, gitignore_path=gitignore_path
        )
        found_readme_files = gitignore_checker.check_files_and_folders(
            check_file_cb=lambda root_dir, relative_path: Path(relative_path).name.lower() in possible_readme_files,
        )
                
        return found_readme_files

