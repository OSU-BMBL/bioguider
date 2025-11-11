
import logging
from pathlib import Path
from typing import Callable
from langchain.prompts import ChatPromptTemplate
from langchain_openai.chat_models.base import BaseChatOpenAI

from bioguider.agents.prompt_utils import EVALUATION_INSTRUCTION
from bioguider.utils.gitignore_checker import GitignoreChecker

from ..utils.pyphen_utils import PyphenReadability
from bioguider.agents.agent_utils import ( 
    read_file, read_license_file, 
    summarize_file
)
from bioguider.agents.common_agent_2step import (
    CommonAgentTwoChainSteps, 
    CommonAgentTwoSteps,
)
from bioguider.agents.evaluation_task import EvaluationTask
from bioguider.utils.constants import (
    DEFAULT_TOKEN_USAGE, 
    ProjectMetadata,
    ProjectLevelEvaluationREADMEResult,
    StructuredEvaluationREADMEResult,
    FreeProjectLevelEvaluationREADMEResult,
    FreeFolderLevelEvaluationREADMEResult,
    EvaluationREADMEResult,
)
from bioguider.utils.utils import get_overall_score, increase_token_usage
from bioguider.rag.config import configs

logger = logging.getLogger(__name__)

README_PROJECT_LEVEL_SYSTEM_PROMPT = """
You are an expert in evaluating the quality of README files in software repositories. 
Your task is to analyze the provided README file and identify if it is a project-level README file or a folder-level README file.

---

### **Classification Guidelines**

 - **Project-level README**:
   - Located in the repository root (e.g., `/`, `.github/`, or `docs/`).
   - Contains project-wide content: overview, installation steps, global usage, badges, links to main docs or contributing guide.

 - **Folder-level README**:
   - Located inside a subfolder.
   - Describes the specific folder: its purpose, contents, how to use or run what's inside, local instructions.

---

### **Output Format**
Based solely on the file's **path**, **name**, and **content**, classify the README as either a **project-level** or **folder-level** README.
Output **exactly** the following format:

**FinalAnswer**
**Project-level:** [Yes / No]

---

### **README Path**
{readme_path}

---

### **README content**
{readme_content}

"""

STRUCTURED_EVALUATION_README_SYSTEM_PROMPT = """
You are an expert in evaluating the quality of README files in software repositories. 
Your task is to analyze the provided README file and generate a structured quality assessment based on the following criteria.
If a LICENSE file is present in the repository, its content will also be provided to support your evaluation of license-related criteria.
---

### **Evaluation Criteria**

1. **Available**: Is the README accessible and present?
   * Output: `Yes` or `No`

2. **Readability**: Evaluate based on readability metrics such as Flesch-Kincaid Grade Level, SMOG Index, etc.
   * Output: a number between 0 and 100 representing the overall quality rating.
   * Suggest specific improvements if necessary
   * **Grade Level**:
     - **85-100**: The README is exceptionally clear, polished, and engaging. It reads smoothly, with minimal effort required from the reader.
     - **65-84**: The README is clear and easy to understand, with a natural flow and minimal jargon.
     - **45-64**: The README is somewhat clear, but could benefit from more polish and consistency.
     - **0-44**: The README is difficult to understand, with unclear language, jargon, or overly complex sentences.

3. **Project Purpose**: Is the project's goal or function clearly stated?
   * Output: `Yes` or `No`
   * Provide suggestions if unclear

4. **Hardware and Software Requirements**: Are hardware/software specs and compatibility details included?
   * Output: a number between 0 and 100 representing the overall quality rating.
   * Suggest how to improve the section if needed
   * **Grade Level**:
     - **85-100**: The README provides a clear and comprehensive guide to the tutorial, with all necessary steps and information provided.
     - **65-84**: The README provides a clear and comprehensive guide to the tutorial, with most necessary steps and information provided.
     - **45-64**: The README provides a clear and comprehensive guide to the tutorial, with some necessary steps and information provided.
     - **0-44**: The README does not provide a clear and comprehensive guide to the tutorial, with no necessary steps and information provided.
     
5. **Dependencies**: Are all necessary software libraries and dependencies clearly listed?
   * Output: a number between 0 and 100 representing the overall quality rating.
   * Suggest improvements if applicable
   * **Grade Level**:
     - **85-100**: The README provides a clear and comprehensive guide to the tutorial, with all necessary steps and information provided.
     - **65-84**: The README provides a clear and comprehensive guide to the tutorial, with most necessary steps and information provided.
     - **45-64**: The README provides a clear and comprehensive guide to the tutorial, with some necessary steps and information provided.
     - **0-44**: The README does not provide a clear and comprehensive guide to the tutorial, with no necessary steps and information provided.

6. **License Information**: Is license type clearly indicated?
   * Output: `Yes` or `No`
   * Suggest improvement if missing or unclear

7. **Author / Contributor Info**: Are contributor or maintainer details provided?
   * Output: `Yes` or `No`
   * Suggest improvement if missing

8. **Overall Score**: Give an overall quality rating of the README.
   * Output: a number between 0 and 100 representing the overall quality rating.
   * **Grade Level**:
     - **85-100**: The README provides a clear and comprehensive guide to the tutorial, with all necessary steps and information provided.
     - **65-84**: The README provides a clear and comprehensive guide to the tutorial, with most necessary steps and information provided.
     - **45-64**: The README provides a clear and comprehensive guide to the tutorial, with some necessary steps and information provided.
     - **0-44**: The README does not provide a clear and comprehensive guide to the tutorial, with no necessary steps and information provided.

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
  * score: a number between 0 and 100 representing the overall quality rating.
  * suggestions: <suggestions to improve README readability>
**Project Purpose:** 
  * score: [Yes / No]
  * suggestions: <suggestions to improve project purpose.>
**Hardware and software spec and compatibility description:**
  * score: a number between 0 and 100 representing the overall quality rating.
  * suggestions: <suggestions to improve **hardware and software** description>
**Dependencies clearly stated:** 
  * score: a number between 0 and 100 representing the overall quality rating.
  * suggestions: <suggestions to improve **Dependencies** description>
**License Information Included:** 
  * score: [Yes / No]
  * suggestions: <suggestions to improve **License Information**>
** Code contributor / Author information included
  * score: [Yes / No]
**Overall Score:** a number between 0 and 100 representing the overall quality rating.

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

PROJECT_LEVEL_EVALUATION_README_SYSTEM_PROMPT = """
You are an expert in evaluating the quality of README files in software repositories. 
Your task is to analyze the provided project-level README file and generate a comprehensive quality report. 
You will be given:
1. A README file.
2. A structured evaluation of the README file and its reasoning process.

---

### **Output Format**
Your output must **exactly match** the following format. Do not add or omit any sections. 

**FinalAnswer**
**Available:**
  <Your assessment and suggestion here>
**Readability:** 
  <Your assessment and suggestion here>
**Project Purpose:** 
  <Your assessment and suggestion here>
**Hardware and software spec and compatibility description:**
  <Your assessment and suggestion here>
**Dependencies clearly stated:** 
  <Your assessment and suggestion here>
**License Information Included:** 
  <Your assessment and suggestion here>
**Code contributor / Author information included
  <Your assessment and suggestion here>
**Overall Score:**
  <Your assessment and suggestion here>
---

### **Instructions**
1. Based on the provided structured evaluation and its reasoning process, generate a free evaluation of the README file.
2. Focus on the explanation of assessment in structured evaluation and how to improve the README file based on the structured evaluation and its reasoning process.
   * For each suggestion to improve the README file, you **must provide some examples** of the original text snippet and the improving comments.
3. For each item in the structured evaluation, provide a detailed assessment followed by specific, actionable comments for improvement.
4. Your improvement suggestions must also include the original text snippet and the improving comments.
5. Your improvement suggestions must also include suggestions to improve readability.
6. In the **FinalAnswer** of output, in each section output, please first give a detailed explanation of the assessment, and then provide the detailed suggestion for improvement. If you think the it is good enough, you can say so.
  The following is an example of the output format:
  **FinalAnswer**
  **Available:**
    Detailed explanation of the assessment. Such as: The README file is present in the repository. The content of the file has been shared completely and is accessible. This confirms the availability of the README documentation for evaluation. There's no issue with availability.
    Detailed suggestion for improvement. Such as: Add a brief introductory section summarizing the project and its main purpose would help orient readers.
  **Readability:**
    Detailed explanation of the assessment. Such as: The README is relatively easy to read for someone around the sixth-grade level. While the technical details provided are moderately easy to understand for those familiar with programming and command-line tools, newbies or non-technical users might face challenges due to jargon and lack of introductory explanations.
    Detailed suggestion for improvement. Such as: 
    - Add a brief introductory section summarizing the project and its main purpose would help orient readers.
    - <original text snippet> - <improving comments>
    - <original text snippet> - <improving comments>
    - ...
    - Break down long instructions into smaller bullet points.
  **Project Purpose:**
    Detailed explanation of the assessment. Such as: The README indirectly describes project activities like benchmarking and assessing functionalities using LLMs and tools like Poetry. However, it lacks a direct statement that defines the overarching project goals or explains who the intended audience is.
    Detailed suggestion for improvement. Such as: 
    - Including a clear project purpose at the beginning, such as: "This project provides a framework for evaluating tabular data models using large language model (LLM)-based assessments. Developers and researchers interested in benchmarking data model performance will find this repository particularly useful."
    - <original text snippet> - <improving comments>
    - <original text snippet> - <improving comments>
    - ...
  **Hardware and software spec and compatibility description:**
    Detailed explanation of the assessment. Such as: The README provides partial information about software requirements, emphasizing tools like Poetry. Instructions regarding the setup of `.env` files and API keys are also provided. However, it doesn't specify hardware considerations like memory requirements or explain whether the software is compatible with particular operating systems. This omission can limit usability for certain users.
    Detailed suggestion for improvement. Such as: 
    - Adding a subsection titled "Hardware Requirements" to outline memory, processor, or other computational dependencies for running benchmarks effectively.
    - <original text snippet> - <improving comments>
    - <original text snippet> - <improving comments>
    - ...
  **Dependencies clearly stated:**
    Detailed explanation of the assessment. Such as: Dependencies are referenced sporadically throughout the README (e.g., using Poetry to install certain tools). However, there isn't a dedicated section that consolidates these into a simple and easy-to-follow format. This could hinder understanding, especially for users looking to quickly identify and install necessary dependencies.
    Detailed suggestion for improvement. Such as: 
    - The dependencies are listed in the README and requirements.txt file. No need to improve.
  **License Information Included:**
    Detailed explanation of the assessment. Such as: The README mentions the MIT license, which is known for its permissive nature and widespread acceptance. The license information is clear and understandable. No improvements are necessary here.
    Detailed suggestion for improvement. Such as: No need to improve.
  **Code contributor / Author information included:**
    Detailed explanation of the assessment. Such as: The README does not contain a section that credits contributors or maintains lines of communication for potential users or collaborators. This is an important omission, as it fails to acknowledge authors' efforts or encourage interaction.
    Detailed suggestion for improvement. Such as: 
    - Including a new "Contributors" section to credit the developers, provide contact information (e.g., email or GitHub profiles), or invite collaboration.
    - <original text snippet> - <improving comments>
  **Overall Score:**
    Detailed explanation of the assessment. Such as: The README is relatively easy to read for someone around the sixth-grade level. While the technical details provided are moderately easy to understand for those familiar with programming and command-line tools, newbies or non-technical users might face challenges due to jargon and lack of introductory explanations.
    Detailed suggestion for improvement. Such as: 
    - Add a brief introductory section summarizing the project and its main purpose would help orient readers.
    - <original text snippet> - <improving comments>
    - <original text snippet> - <improving comments>
    - ...
    - Break down long instructions into smaller bullet points.

---

### **Structured Evaluation and Reasoning Process**
{structured_evaluation}

---

### **README Path**
{readme_path}

---

### **README content**
{readme_content}

"""

FOLDER_LEVEL_EVALUATION_README_SYSTEM_PROMPT = """
You are an expert in evaluating the quality of README files in software repositories. 
Your task is to analyze the provided README file and generate a comprehensive quality report.

---

### **Evaluation Criteria**

The README file is a **folder-level** file, use the following criteria instead.

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
 * **Score:** [Poor / Fair / Good / Excellent]
  * **Key Strengths**: <brief summary of the README's strongest points in 2-3 sentences> 
  * **Overall Improvement Suggestions:**
    - "Original text snippet 1" - Improving comment 1  
    - "Original text snippet 2" - Improving comment 2  
    - ...

#### Notes

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

class EvaluationREADMETask(EvaluationTask):
    def __init__(
        self, 
        llm: BaseChatOpenAI, 
        repo_path: str, 
        gitignore_path: str,
        meta_data: ProjectMetadata | None = None,
        step_callback: Callable | None = None,
        summarized_files_db = None,
        collected_files: list[str] | None = None,
    ):
        super().__init__(llm, repo_path, gitignore_path, meta_data, step_callback, summarized_files_db)
        self.evaluation_name = "README Evaluation"
        self.collected_files = collected_files

    def _project_level_evaluate(self, readme_files: list[str]) -> tuple[dict, dict]:
        """
        Evaluate if the README files are a project-level README file.
        """
        total_token_usage = {**DEFAULT_TOKEN_USAGE}
        project_level_evaluations = {}
        for readme_file in readme_files:
            full_path = Path(self.repo_path, readme_file)
            readme_content = read_file(full_path)
            if readme_content is None or len(readme_content.strip()) == 0:
                logger.error(f"Error in reading file {readme_file}")
                project_level_evaluations[readme_file] = {
                    "project_level": "/" in readme_file,
                    "project_level_reasoning_process": f"Error in reading file {readme_file}" \
                        if readme_content is None else f"{readme_file} is an empty file.",
                }
                continue
            system_prompt = ChatPromptTemplate.from_template(
                README_PROJECT_LEVEL_SYSTEM_PROMPT
            ).format(
                readme_path=readme_file,
                readme_content=readme_content,
            )
            agent = CommonAgentTwoChainSteps(llm=self.llm)
            response, _, token_usage, reasoning_process = agent.go(
                system_prompt=system_prompt,
                instruction_prompt=EVALUATION_INSTRUCTION,
                schema=ProjectLevelEvaluationREADMEResult,
            )
            total_token_usage = increase_token_usage(total_token_usage, token_usage)
            self.print_step(step_output=f"README: {readme_file} project level README")
            project_level_evaluations[readme_file] = {
                "project_level": response.project_level,
                "project_level_reasoning_process": reasoning_process,
            }

        return project_level_evaluations, total_token_usage

    def _structured_evaluate(self, readme_project_level: dict[str, dict] | None = None):
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
        if readme_project_level is None:
            return None, total_token_usage
        
        license_content, license_path = read_license_file(self.repo_path)
        license_summarized_content = summarize_file(
            llm=self.llm,
            name=license_path,
            content=license_content, 
            level=6,
            summary_instructions="What license is the repository using?",
        ) if license_content is not None else "N/A"
        license_path = license_path if license_content is not None else "N/A"
        structured_readme_evaluations = {}
        for readme_file in readme_project_level.keys():
            project_level = readme_project_level[readme_file]["project_level"]
            if not project_level:
                continue
            full_path = Path(self.repo_path, readme_file)
            readme_content = read_file(full_path)
            if readme_content is None:
                logger.error(f"Error in reading file {readme_file}")
                continue
            if len(readme_content.strip()) == 0:
                structured_readme_evaluations[readme_file] = {
                    "evaluation": StructuredEvaluationREADMEResult(
                        available_score=False,
                        readability_score=0,
                        readability_suggestions="No readability provided",
                        project_purpose_score=False,
                        project_purpose_suggestions="No project purpose provided",
                        hardware_and_software_spec_score=0,
                        hardware_and_software_spec_suggestions="No hardware and software spec provided",
                        dependency_score=0,
                        dependency_suggestions="No dependency provided",
                        license_score=False,
                        license_suggestions="No license information",
                        contributor_author_score=False,
                        overall_score=0,
                    ),
                    "reasoning_process": f"{readme_file} is an empty file.",
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
            response.overall_score = get_overall_score(
                [
                    response.readability_score, 
                    response.project_purpose_score, 
                    response.hardware_and_software_spec_score, 
                    response.dependency_score, 
                    response.license_score, 
                    response.contributor_author_score,
                ],
                [3, 3, 1, 1, 1, 1],
            )
            self.print_step(step_output=f"README: {readme_file} structured evaluation")
            self.print_step(step_output=reasoning_process)
            structured_readme_evaluations[readme_file] = {
                "evaluation": response,
                "reasoning_process": reasoning_process,
            }
            total_token_usage = increase_token_usage(total_token_usage, token_usage)

        return structured_readme_evaluations, total_token_usage
        

    def _free_project_level_readme_evaluate(
        self, 
        readme_file: str,
        structured_reasoning_process: str,
    ) -> tuple[FreeProjectLevelEvaluationREADMEResult | None, dict, str]:
        readme_path = Path(self.repo_path, readme_file)
        readme_content = read_file(readme_path)
        if readme_content is None:
            logger.error(f"Error in reading file {readme_file}")
            return None, {**DEFAULT_TOKEN_USAGE}, f"Error in reading file {readme_file}"
        if readme_content.strip() == "":
            return FreeProjectLevelEvaluationREADMEResult(
                available=False,
                readability="Poor",
                project_purpose="Poor",
                hardware_and_software_spec="Poor",
                dependency="Poor",
            ), {**DEFAULT_TOKEN_USAGE}, f"{readme_file} is an empty file."
        
        system_prompt = ChatPromptTemplate.from_template(
            PROJECT_LEVEL_EVALUATION_README_SYSTEM_PROMPT
        ).format(
            readme_path=readme_file,
            readme_content=readme_content,
            structured_evaluation=structured_reasoning_process,
        )
        agent = CommonAgentTwoSteps(llm=self.llm)
        response, _, token_usage, reasoning_process = agent.go(
            system_prompt=system_prompt,
            instruction_prompt=EVALUATION_INSTRUCTION,
            schema=FreeProjectLevelEvaluationREADMEResult,
        )
        self.print_step(step_output=f"README: {readme_file} free project level README")
        self.print_step(step_output=reasoning_process)
        return response, token_usage, reasoning_process

    def _free_folder_level_readme_evaluate(
        self, 
        readme_file: str,
    ) -> tuple[FreeFolderLevelEvaluationREADMEResult | None, dict, str]:
        readme_path = Path(self.repo_path, readme_file)
        readme_content = read_file(readme_path)
        if readme_content is None:
            logger.error(f"Error in reading file {readme_file}")
            return None, {**DEFAULT_TOKEN_USAGE}, f"Error in reading file {readme_file}"
        if readme_content.strip() == "":
            return FreeFolderLevelEvaluationREADMEResult(
                score="Poor",
                key_strengths=f"{readme_file} is an empty file.",
                overall_improvement_suggestions=[f"{readme_file} is an empty file."],
            ), {**DEFAULT_TOKEN_USAGE}, f"{readme_file} is an empty file."
        
        readability = PyphenReadability()
        flesch_reading_ease, flesch_kincaid_grade, gunning_fog_index, smog_index, \
            _, _, _, _, _ = readability.readability_metrics(readme_content)
        system_prompt = ChatPromptTemplate.from_template(
            FOLDER_LEVEL_EVALUATION_README_SYSTEM_PROMPT
        ).format(
            readme_path=readme_file,
            readme_content=readme_content,
            flesch_reading_ease=flesch_reading_ease,
            flesch_kincaid_grade=flesch_kincaid_grade,
            gunning_fog_index=gunning_fog_index,
            smog_index=smog_index,
        )
        agent = CommonAgentTwoChainSteps(llm=self.llm)
        response, _, token_usage, reasoning_process = agent.go(
            system_prompt=system_prompt,
            instruction_prompt=EVALUATION_INSTRUCTION,
            schema=FreeFolderLevelEvaluationREADMEResult,
        )
        self.print_step(step_output=f"README: {readme_file} free folder level README")
        self.print_step(step_output=reasoning_process)
        return response, token_usage, reasoning_process

    def _free_evaluate(
        self, 
        readme_project_level: dict[str, dict], 
        structured_readme_evaluations: dict[str, dict]
    ):
        readme_files = readme_project_level.keys()
        if readme_files is None or len(readme_files) == 0:
            return None, {**DEFAULT_TOKEN_USAGE}
        
        free_readme_evaluations = {}
        total_token_usage = {**DEFAULT_TOKEN_USAGE}
        for readme_file in readme_files:
            readme_path = Path(self.repo_path, readme_file)
            project_level = readme_project_level[readme_file]["project_level"]
            readme_content = read_file(readme_path)
            if readme_content is None:
                logger.error(f"Error in reading file {readme_file}")
                continue
            if project_level:
                evaluation, token_usage, reasoning_process = self._free_project_level_readme_evaluate(
                    readme_file=readme_file,
                    structured_reasoning_process=structured_readme_evaluations[readme_file]["reasoning_process"],
                )
                if evaluation is None:
                    continue
                free_readme_evaluations[readme_file] = {
                    "evaluation": evaluation,
                    "reasoning_process": reasoning_process,
                }
                total_token_usage = increase_token_usage(total_token_usage, token_usage)
            
            else:
                evaluation, token_usage, reasoning_process = self._free_folder_level_readme_evaluate(
                    readme_file=readme_file,
                )
                if evaluation is None:
                    continue
                free_readme_evaluations[readme_file] = {
                    "evaluation": evaluation,
                    "reasoning_process": reasoning_process,
                }
                total_token_usage = increase_token_usage(total_token_usage, token_usage)

        return free_readme_evaluations, total_token_usage
            
    def _evaluate(self, files: list[str]) -> tuple[dict[str, EvaluationREADMEResult], dict, list[str]]:
        total_token_usage = {**DEFAULT_TOKEN_USAGE}
        project_level_evaluations, project_level_token_usage = self._project_level_evaluate(files)
        total_token_usage = increase_token_usage(total_token_usage, project_level_token_usage)
        structured_readme_evaluations, structured_token_usage = self._structured_evaluate(project_level_evaluations)
        total_token_usage = increase_token_usage(total_token_usage, structured_token_usage)
        free_readme_evaluations, free_token_usage = self._free_evaluate(project_level_evaluations, structured_readme_evaluations)
        total_token_usage = increase_token_usage(total_token_usage, free_token_usage)

        # combine result
        combined_evaluations = {}
        for f in files:
            if not f in free_readme_evaluations:
                continue
            project_level = project_level_evaluations[f]["project_level"]
            if project_level:
                combined_evaluations[f] = EvaluationREADMEResult(
                    project_level=project_level,
                    structured_evaluation=structured_readme_evaluations[f]["evaluation"],
                    free_evaluation=free_readme_evaluations[f]["evaluation"],
                    structured_reasoning_process=structured_readme_evaluations[f]["reasoning_process"],
                    free_reasoning_process=free_readme_evaluations[f]["reasoning_process"],
                )
            else:
                combined_evaluations[f] = EvaluationREADMEResult(
                    project_level=project_level,
                    structured_evaluation=None,
                    free_evaluation=free_readme_evaluations[f]["evaluation"],
                    structured_reasoning_process=None,
                    free_reasoning_process=free_readme_evaluations[f]["reasoning_process"],
                )
        
        return combined_evaluations, total_token_usage, files
    
    def _collect_files(self):
        """
        Search for a README file in the repository directory.
        """
        if self.collected_files is not None:
            return self.collected_files
        
        possible_readme_files = [
            "readme.md",
            "readme.rst",
            "readme.txt",
            "readme",
        ]
        repo_path = self.repo_path
        gitignore_path = Path(repo_path, ".gitignore")
        gitignore_checker = GitignoreChecker(
            directory=repo_path, gitignore_path=gitignore_path,
            exclude_dir_patterns=configs["file_filters"]["excluded_dirs"],
            exclude_file_patterns=configs["file_filters"]["excluded_files"],
        )
        found_readme_files = gitignore_checker.check_files_and_folders(
            check_file_cb=lambda root_dir, relative_path: Path(relative_path).name.lower() in possible_readme_files,
        )
                
        return found_readme_files

