
import os
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
from nanoid import generate

from bioguider.agents.agent_utils import (
    convert_plan_to_string, 
    get_tool_names_and_descriptions,
    PlanAgentResult,
    PlanAgentResultJsonSchema,
)
from bioguider.agents.peo_common_step import PEOCommonStep
from bioguider.agents.common_agent_2step import CommonAgentTwoChainSteps, CommonAgentTwoSteps
from bioguider.agents.dockergeneration_task_utils import (
    DockerGenerationWorkflowState, 
    prepare_provided_files_string,
)

DOCKERGENERATION_PLAN_SYSTEM_PROMPT = ChatPromptTemplate.from_template("""
You are an expert in software containerization and reproducibility engineering.
Your task is to generate a **Dockerfile** that prepares the environment and runs a simple get-started example based on the provided files from a GitHub repository.
---

### Repository File Structure
Below is the 2-level file structure of the repository (`f` = file, `d` = directory, `l` - symlink, `u` - unknown):  
{repo_structure}

### **Input Files:**

You are given the contents of the following files extracted from the repository:

{extracted_files}
---

### **Intermediate Dockerfile**
Here is the Dockerfile you generated before.
{intermediate_dockerfile_content}

---

### **Intermediate Error**
Here is the error when building or running the Dockerfile
{intermediate_error}

## ** Intermediate Thoughts **
Here is the thoughts you need to take into consideration.
{intermediate_thoughts}
---
                                                                       
### **Function Tools**
You have access to the following function tools:
{tools}
---

### Instructions:
1. We will iterate through multiple **Plan -> Execution -> Observation** loops as needed.
   - Plan stage(current stage) will make a plan based on provided **tools**, **intermediate output** and **repo structure**
   - Execution stage will execute the planned actions to generate Dockerfile
   - Observation stage will observe the Dockerfile that is generated in execution step and provide advice in **intermediate thoughts**
2. Your current task is to make a plan to achieve the goal.
   You can start by `write_file_tool` to prepare script files, then use `generate_Dockerfile_tool` to generate **Dockerfile**
3. When using `write_file_tool`, you must specify both the **file name** and **file content** as input.
   - Use `write_file_tool` to create new files, such as a minimal demo script.
   - You may also use it to **overwrite existing files** if **needed**.
   - If no update, **do not** use `write_file_tool` to overwrite existed file.
   - Always provide **complete and concrete file content**—do **not** include suggestions, placeholders, abstract descriptions, or part of content.
4. You can use `extract_python_file_from_notebook_tool` to extract python code from python notebook and save to a python file to avoid running python notebook with jupyter.
5. You may use the `python_repl` tool to execute Python code, but this should **also be avoided in the first step**.
6. The Dockerfile will be placed at the root of the repository.
   Therefore, in the Dockerfile, you can assume all repository files are accessible and can be copied as needed.
7. If you are given **Intermediate Error** and **Intermediate Dockerfile**, you need to analyze them carefully, and try to fix them with new generated Dockerfile.
   You need to provide concrete resolution in your reasoning process.
8. When using `generate_Dockerfile_tool` to generate a Dockerfile, please use `demo-bioguider-{docker_id}.Dockerfile` as file name.
9. Always use `generate_Dockerfile_tool` as the **final action step** in your plan to ensure the Dockerfile is generated at the end of the process.
---

### **Output Format**  
Your plan should be returned as a sequence of step actions in the following format:

Step: <tool name>   # Tool name must be one of {tool_names}  
Step Input: <file or directory name>

Step: <tool name>  
Step Input: <file or directory name>
...
""")

class DockerGenerationPlanStep(PEOCommonStep):
    def __init__(
         self, 
         llm: BaseChatOpenAI,
         repo_path: str,
         repo_structure: str,
         gitignore_path: str,
         custom_tools: list[BaseTool] | None = None,
      ):
        super().__init__(llm)
        self.step_name = "Dockerfile Generation Plan Step"
        self.repo_path = repo_path
        self.repo_structure = repo_structure
        self.gitignore_path = gitignore_path
        self.custom_tools = custom_tools

    def _prepare_intermediate_steps(self, state: DockerGenerationWorkflowState):
        _, intermediate_thoughts = super()._build_intermediate_analysis_and_thoughts(state)
        intermediate_dockerfile_content = state["step_dockerfile_content"] if "step_dockerfile_content" in state else "N/A"
        intermediate_error = state["step_output"] if "step_output" in state else "N/A"
        intermediate_error = intermediate_error.replace("{", "(").replace("}", ")")

        return intermediate_dockerfile_content, intermediate_error, intermediate_thoughts

    def _prepare_system_prompt(self, state: DockerGenerationWorkflowState) -> str:
        docker_id = generate('1234567890abcdefhijklmnopqrstuvwxyz', size=10)
        tool_names, tools_desc = get_tool_names_and_descriptions(self.custom_tools)
        provided_files = state["provided_files"]
        str_provided_files = prepare_provided_files_string(self.repo_path, provided_files)
         
        intermediate_dockerfile_content, intermediate_error, intermediate_thoughts = self._prepare_intermediate_steps(state)
        system_prompt = DOCKERGENERATION_PLAN_SYSTEM_PROMPT.format(
            repo_structure=self.repo_structure,
            tools=tools_desc,
            tool_names=tool_names,
            extracted_files=str_provided_files,
            intermediate_dockerfile_content=intermediate_dockerfile_content,
            intermediate_error=intermediate_error,
            intermediate_thoughts=intermediate_thoughts,
            docker_id=docker_id,
        )
        self._print_step(
            state,
            step_output="**Intermediate Step Output**\n" + intermediate_error
        )
        self._print_step(
            state,
            step_output="**Intermediate Step Thoughts**\n" + intermediate_thoughts
        )
        return system_prompt         

    def _execute_directly(self, state: DockerGenerationWorkflowState):
        system_prompt = self._prepare_system_prompt(state)
        agent = CommonAgentTwoChainSteps(llm=self.llm)
        res, _, token_usage, reasoning = agent.go(
            system_prompt=system_prompt,
            instruction_prompt="Now, let's begin to make a plan",
            schema=PlanAgentResultJsonSchema,
        )
        res = PlanAgentResult(**res)
        self._print_step(state, step_output=f"**Reasoning Process**\n{reasoning}")
        self._print_step(state, step_output=f"**Plan**\n{str(res.actions)}")
        state["plan_thoughts"] = reasoning
        state["plan_actions"] = convert_plan_to_string(res)

        return state, token_usage
        


