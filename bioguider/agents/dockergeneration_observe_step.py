
import os
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from bioguider.utils.constants import DEFAULT_TOKEN_USAGE
from bioguider.agents.agent_utils import read_file
from bioguider.utils.utils import run_command
from bioguider.agents.dockergeneration_task_utils import DockerGenerationWorkflowState
from bioguider.agents.common_agent_2step import CommonAgentTwoChainSteps, CommonAgentTwoSteps
from bioguider.agents.peo_common_step import PEOCommonStep

DOCKERGENERATION_OBSERVE_SYSTEM_PROMPT = """You are an expert in software containerization and reproducibility engineering.
We have a generated **Dockerfile**, here is its content:
{dockerfile_content}

Here is the output of docker image building with command "docker build":
```{docker_build_output}```

Here is the output of running docker image with command "docker run":
```{docker_run_output}```

### **Instructions**
1. Carefully review the **Dockerfile**, the `docker build` output, and the `docker run` output.
2. Diagnose the root cause of any failure and describe a concrete fix that the next round's
   plan can apply (e.g. a missing apt package, a wrong ENTRYPOINT, a bad working directory).
3. Be precise and support your reasoning with evidence from the input.

### **Notes**
- We are generating the Dockerfile over multiple rounds. Your thoughts will be persisted and
  fed back into the next planning round, so focus on **actionable** advice.

### **Output Format (STRICT — read carefully)**
You MUST return a **single raw JSON object** — nothing else. No prose before or after,
no `<think>` blocks, no markdown, no ```json fences, no commentary.

The JSON object has exactly one field:

  - `thoughts`: string. A concrete, actionable diagnosis and fix suggestion for the next
                round. Reference specific lines from the Dockerfile or specific error
                messages from the build/run output as evidence.

**Hard rules — any violation means your answer is wrong:**
  1. Return raw JSON only. No code fences, no leading/trailing text.
  2. The `thoughts` value MUST be a plain string — never a nested object or array.
  3. Keep it focused: one problem, one fix, per sentence.

**Concrete example (copy this shape exactly):**
{{"thoughts": "The build failed at step 5 with 'E: Unable to locate package libglib2.0-0'. The base image `python:3.11-slim` does not ship glib. Next round, add `apt-get update && apt-get install -y libglib2.0-0` before the pip install step."}}
"""

class DockerGenerationObserveResult(BaseModel):
    thoughts: str = Field(description="thoughts on input")

MAX_TIMEOUT = 900 # 15 mins
MAX_ERROR_OUTPTU_LENGTH = 2048 # 2k
class DockerGenerationObserveStep(PEOCommonStep):
    def __init__(self, llm, repo_path: str):
        super().__init__(llm)
        self.step_name = "Docker Generation Observe"
        self.repo_path = repo_path

    def _build_system_prompt(
        self, 
        state: DockerGenerationWorkflowState,
        build_error: str,
        run_error: str,
    ):
        dockerfile=state["dockerfile"]
        dockerfile_path = os.path.join(self.repo_path, dockerfile)
        dockerfile_content = read_file(dockerfile_path)
        return ChatPromptTemplate.from_template(DOCKERGENERATION_OBSERVE_SYSTEM_PROMPT).format(
            dockerfile_content=dockerfile_content,
            docker_build_output=build_error,
            docker_run_output=run_error,
        )
    
    @staticmethod
    def _extract_error_message(output: str):
        if isinstance(output, bytes):
            output = output.decode('utf-8')
        extracted_msg = ""
        output_lower = output.lower()
        if "error:" in output_lower:
            ix = output_lower.find("error:")
            extracted_msg = output[ix:]
        elif "error" in output_lower:
            ix = output_lower.find("error")
            extracted_msg = output[ix:]
        else:
            extracted_msg = output
        if len(extracted_msg) > MAX_ERROR_OUTPTU_LENGTH:
            extracted_msg = extracted_msg[((-1) * MAX_ERROR_OUTPTU_LENGTH):]
        return extracted_msg

    def _execute_directly(self, state: DockerGenerationWorkflowState):
        token_usage = {**DEFAULT_TOKEN_USAGE}
        if "dockerfile" in state and len(state["dockerfile"]) > 0:
            dockerfile=state["dockerfile"]
            dockerfile_path = os.path.join(self.repo_path, dockerfile)
            docker_image_name: str = os.path.splitext(dockerfile)[0]
            docker_image_name = docker_image_name.lower()
            
            out, error, code = run_command([
                "docker", "build", 
                "-t", docker_image_name, 
                "-f", dockerfile_path,
                self.repo_path
            ], timeout=MAX_TIMEOUT)
            if code != 0:
                error_msg = DockerGenerationObserveStep._extract_error_message(error)
                system_prompt = self._build_system_prompt(state, error_msg, "N/A")
                agent = CommonAgentTwoChainSteps(llm=self.llm)
                res, _, token_usage, reasoning = agent.go(
                    system_prompt=system_prompt,
                    instruction_prompt="Now, let's begin observing.",
                    schema=DockerGenerationObserveResult,
                )
                state["step_dockerfile_content"] = read_file(dockerfile_path)
                state["step_output"] = error_msg
                state["step_thoughts"] = res.thoughts
                self._print_step(
                    state,
                    step_output=f"**Observation Reasoning Process**\n{reasoning}"
                )
                return state, token_usage
            out, error, code = run_command([
                "docker", "run",
                "--name", "bioguider_demo",
                docker_image_name
            ], timeout=MAX_TIMEOUT)
            run_command([
                "docker", "rm", "-f",
                "bioguider_demo"
            ], timeout=MAX_TIMEOUT)
            run_command([
                "docker", "rmi", docker_image_name
            ], timeout=MAX_TIMEOUT)
            if code != 0:
                system_prompt = self._build_system_prompt(
                    state, 
                    "docker build successfully.",
                    error,
                )
                agent = CommonAgentTwoChainSteps(llm=self.llm)
                res, _, token_usage, reasoning = agent.go(
                    system_prompt=system_prompt,
                    instruction_prompt="Now, let's begin observing.",
                    schema=DockerGenerationObserveResult,
                )
                state["step_dockerfile_content"] = read_file(dockerfile_path)
                state["step_output"] = error
                state["step_thoughts"] = res.thoughts
                self._print_step(
                    state,
                    step_output=f"**Observation Reasoning Process**\n{reasoning}",
                )
                return state, token_usage
            
            state["final_answer"] = read_file(dockerfile_path)
            return state, token_usage
        
        state["step_thoughts"] = "No Dockerfile is generated."
        return state, token_usage


        
        


