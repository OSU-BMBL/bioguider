
import os
import logging
from typing import Callable, Optional, TypedDict
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from bioguider.agents.agent_tools import agent_tool
from bioguider.agents.agent_utils import read_file, write_file
from bioguider.agents.common_agent_2step import CommonAgentTwoSteps
from bioguider.utils.file_utils import extract_code_from_notebook

logger = logging.getLogger(__name__)

class DockerGenerationPlanResult(BaseModel):
    Dockerfile: str = Field(description="Dockerfile content")
    
class DockerGenerationWorkflowState(TypedDict):
    llm: Optional[BaseChatOpenAI]
    step_output_callback: Optional[Callable]
    provided_files: Optional[list[str]]
    intermediate_steps: Optional[str]
    step_dockerfile_content: Optional[str]
    step_output: Optional[str]
    step_thoughts: Optional[str]
    plan_thoughts: Optional[str]
    plan_actions: Optional[str]
    dockerfile: Optional[str]
    final_answer: Optional[str]

def extract_dockergeneration_related_content(filename: str):
    pass

DOCKERGENERATION_SYSTEM_PROMPT = ChatPromptTemplate.from_template("""
You are an expert in software containerization and reproducibility engineering.
Your task is to generate a **Dockerfile** that prepares the environment and runs a simple get-started example based on the provided files from a GitHub repository.
---
### Repository File Structure
Below is the 2-level file structure of the repository (`f` = file, `d` = directory, `l` - symlink, `u` - unknown):  
{repo_structure}

---
### **Input Files:**

You are given the contents of the following files extracted from the repository:

{extracted_files}
---
                                                                  
### **plan thoughts**
Here is the plan thoughts, you are in **generate_Dockerfile_tool** action:
{plan_thoughts}

---

### **Intermediate Output**
Here is the Dockerfile you generate before.
{step_dockerfile_content}

---

### **Intermediate Error**
Here is the error occurred in building or running the above generated Dockerfile:
{step_error}
                                                                  
### **Requirements:**
1. **Environment Setup**
   * When generating the Dockerfile, prioritize using the base image provided in the repository. If no base image is specified, select an appropriate one based on the project's context.
   * Use the relevant installation and configuration details from the input files (e.g., `requirements.txt`, `environment.yml`, `setup.py`, etc.).
   * Choose an appropriate base image (e.g., `python:3.10`, `r-base`, etc.) based on the language and setup instructions.
2. **Dependency Installation**
   * Include all commands necessary to install packages, tools, or dependencies as specified in the input files.
   * Make sure to always install common system utilities and development tools such as gcc, g++, build-essential, curl, wget, and similar essential packages.
3. **Running a Get-Started Example**
   * Identify a minimal executable script or command (e.g., `python example.py`, `Rscript demo.R`, `jupyter nbconvert --execute`) that demonstrates the basic functionality of the repository.
4. **Keep the Dockerfile Minimal and Reproducible**
   * Use best practices such as specifying exact versions where possible, minimizing layers, and using `COPY`, `WORKDIR`, and `CMD` appropriately.
5. The Dockerfile will be placed at the root of the repository.
   Therefore, in the Dockerfile, you can assume all repository files are accessible and can be copied as needed.
6. If the **Intermediate Output** and **Intermediate Error** are provided, you need to analyze them carefully, and try to fix them in the generated Dockerfile.
---
### **Output Format:**
Return only the Dockerfile content enclosed in triple backticks:
```dockerfile
# Dockerfile
<your generated Dockerfile content here>
```
Do not include any explanation, comments, or extra output outside the code block.
""")

class generate_Dockerfile_tool(agent_tool):
    """ Generate Dockerfile for provided repository
Args:
    output_path str: the output path to save Dockerfile
Returns:
    boolean: if Dockerfile is saved successfully
    """
    def __init__(
        self,
        llm: BaseChatOpenAI,
        repo_path: str,
        extracted_files: str,
        repo_structure: str,
        output_callback: Callable | None = None,
    ):
        super().__init__(llm, output_callback=output_callback)
        self.repo_path = repo_path
        self.repo_struture = repo_structure
        self.extracted_files = extracted_files
        self.plan_thoughts = None
        self.step_error: str = None
        self.step_dockerfile_content: str = None

    def set_intermediate_output(self, plan_thoughts: str, step_error: str, step_dockerfile_content: str):
        plan_thoughts = plan_thoughts.replace("{", "(").replace("}", ")")
        step_error = step_error.replace("{", "(").replace("}", ")")
        self.plan_thoughts = plan_thoughts
        self.step_error = step_error
        self.step_dockerfile_content = step_dockerfile_content

    def run(self, output_path: str):
        agent = CommonAgentTwoSteps(llm=self.llm)
        system_prompt = DOCKERGENERATION_SYSTEM_PROMPT.format(
            repo_structure = self.repo_struture,
            extracted_files = self.extracted_files,
            plan_thoughts=self.plan_thoughts,
            step_error=self.step_error,
            step_dockerfile_content=self.step_dockerfile_content
        )
        res, _, token_usage, reasoning = agent.go(
            system_prompt=system_prompt,
            instruction_prompt="Now, let's start to generate Dockerfile.",
            schema=DockerGenerationPlanResult,
        )
        res: DockerGenerationPlanResult = res
        self._print_step_output(step_output=reasoning) 
        self._print_token_usage(token_usage)
        if self.repo_path not in output_path:
            output_path = os.path.join(self.repo_path, output_path)
        content = res.Dockerfile
        if content.startswith("```dockerfile"):
            content = content[13:]
        content = content.strip().strip("```")
        write_file(output_path, content)

        return True

class write_file_tool():
    """write file tool
Args:
    file_name str: a string specifies file path that will be written to.
    file_content str: a string speifies file content.
Returns:
    bool, True if it is succeeded to write to file, otherwise False
    """
    def __init__(self, repo_path: str):
        self.repo_path = repo_path

    def run(self, file_name: str, file_content: str):
        if file_name is None or file_content is None:
            return False
        file_name = file_name
        content = file_content
        file_name = file_name.strip()
        if self.repo_path is not None and self.repo_path not in file_name:
            file_name = os.path.join(self.repo_path, file_name)
        try:
            with open(file_name, "w") as fobj:
                fobj.write(content)
                return True
        except Exception as e:
            logger.error(e)
            return False
        
class extract_python_file_from_notebook_tool:
    """extract code in a notebook to a python file
Args:
    notebook_path str: a string speicifies notebook path to extract.
    output_path str: a string specifies output python file path.
Returns:
    bool True if it is succeeded to extract to python file, otherwise False
    """
    def __init__(self, repo_path: str):
        self.repo_path = repo_path

    def run(self, notebook_path: str, output_path: str):
        # notebook_path = notebook_path_and_output_path[0]
        # output_path = notebook_path_and_output_path[1]
        if notebook_path is None or output_path is None:
            return False
        if self.repo_path not in notebook_path:
            notebook_path = os.path.join(self.repo_path, notebook_path)
        if self.repo_path not in output_path:
            output_path = os.path.join(self.repo_path, output_path)
        content = extract_code_from_notebook(notebook_path)
        try:
            with open(output_path, "w") as fobj:
                fobj.write(content)
                return True
        except FileNotFoundError as e:
            logger.error(str(e))
            return f"False, {output_path} does not exist."
        

def prepare_provided_files_string(repo_path: str, provided_files: list[str]):
    if provided_files is None or len(provided_files) == 0:
        return "N/A"
    str_provided_files = ""
    for fn in provided_files:
        file_path = os.path.join(repo_path, fn)
        if fn.endswith(".ipynb"): # python notebook
            content = extract_code_from_notebook(file_path)
        else:
            content = read_file(file_path)
        content = content.replace("{", "{{").replace("}", "}}")
        str_provided_files += f"""**{fn}**:\n{content}\n"""

    return str_provided_files

