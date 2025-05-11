
import os
from langchain.prompts import ChatPromptTemplate
from langchain_openai.chat_models.base import BaseChatOpenAI
from pydantic import BaseModel, Field

from bioguider.agents.agent_utils import DEFAULT_TOKEN_USAGE, ObservationResult, run_command, read_file
from bioguider.agents.dockergeneration_task_utils import DockerGenerationWorkflowState
from bioguider.agents.common_agent_2step import CommonAgentTwoSteps
from bioguider.agents.peo_common_step import PEOCommonStep

DOCKERGENERATION_OBSERVE_SYSTEM_PROMPT = """You are an expert in software containerization and reproducibility engineering.
We have a generated **Dockerfile**, here is its content:
{dockerfile_content}

Here is the output of docker image building with command "docker build":
```{docker_build_output}```

Here is the output of running docker image with command "docker run":
```{docker_run_output}```

### **Instructions**
1. Carefully review **Dockerfile**, output of building docker image and output of running docker image, give your
thoughts and advice as the following format:
```
**Thoughts**: you thoughts here
``` 
2. Be precise and support your reasoning with evidence from the input.

### **Notes**
- We are generating Dockerfile over multiple rounds, your thoughts and the output of this step will be persisted, 
we'll continue with the next round accordingly
"""

class DockerGenerationObserveResult(BaseModel):
    thoughts: str = Field(description="thoughts on input")

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
    
    def _execute_directly(self, state: DockerGenerationWorkflowState):
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
            ], timeout=600)
            if code != 0:
                system_prompt = self._build_system_prompt(state, error, "N/A")
                agent = CommonAgentTwoSteps(llm=self.llm)
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
                    step_output=f"**Observation Reasoning Process**\n{reasoning}"
                )
                return state, token_usage
            out, error, code = run_command([
                "docker", "run",
                "--name", "bioguider_demo",
                docker_image_name
            ], timeout=600)
            run_command([
                "docker", "rm", "-f",
                "bioguider_demo"
            ], timeout=600)
            run_command([
                "docker", "rmi", docker_image_name
            ], timeout=600)
            if code != 0:
                system_prompt = self._build_system_prompt(
                    state, 
                    "docker build successfully.",
                    error,
                )
                agent = CommonAgentTwoSteps(llm=self.llm)
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
        token_usage = {**DEFAULT_TOKEN_USAGE}
        return state, token_usage


        
        


