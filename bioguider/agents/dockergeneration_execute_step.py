
import logging
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain.tools import BaseTool
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.callbacks.openai_info import OpenAICallbackHandler

from bioguider.utils.constants import DEFAULT_TOKEN_USAGE
from bioguider.agents.agent_utils import (
    CustomPromptTemplate,
    CustomOutputParser,
)
from bioguider.agents.peo_common_step import PEOCommonStep
from bioguider.agents.dockergeneration_task_utils import (
    DockerGenerationWorkflowState, 
    generate_Dockerfile_tool,
)

logger = logging.getLogger(__name__)

DOCKERGENERATION_EXECUTION_SYSTEM_PROMPT = """You are an expert in software containerization and reproducibility engineering.
You are given a **plan** and must complete it strictly using Python code and the available tools.

---
### **Available Tools**
{tools}

---
### **Your Task**  
Follow the given plan step by step using the exact format below:

```
Thought: Describe what you are thinking or planning to do next.  
Action: The tool you are going to use (must be one of: {tool_names})  
Action Input: The input to the selected action  
Observation: The result returned by the action
```

You may repeat the **Thought → Action → Action Input → Observation** loop as needed.  

Once all steps in the plan have been executed, end the loop and output all the results and generated Dockerfile using this format:

```
Thought: I have completed the plan.
Final Answer:
Action: {{tool_name}}
Action Input: {{file_name1}}
Action Observation: {{Observation1}}
---
Action: {{tool_name}}
Action Input: {{file_name2}}
Action Observation: {{Observation2}}
---
**Dockerfile file name**: {{docker file path}}
...
```

---

### **Important Notes**

- You must strictly follow the provided plan.  
- **Do not take any additional or alternative actions**, even if:  
  - No relevant result is found  
  - The file content is missing, empty, or irrelevant  
- If no information is found in a step, simply proceed to the next action in the plan without improvising.  
- Only use the tools specified in the plan actions. No independent decisions or extra steps are allowed.
---

### **Plan**  
{plan_actions}

### **Plan Thoughts**
{plan_thoughts}

### **Actions Already Taken**  
{agent_scratchpad}

---

{input}

---
"""

class DockerGenerationExecuteStep(PEOCommonStep):
    def __init__(
        self,
        llm: BaseChatOpenAI,
        repo_path: str,
        repo_structure: str,
        gitignore_path: str,
        custom_tools: list[BaseTool] | None = None,
    ):
        super().__init__(llm)
        self.step_name = "Docker Generation Execute Step"
        self.repo_path = repo_path
        self.repo_structure = repo_structure
        self.gitignore_path = gitignore_path
        self.custom_tools = custom_tools if custom_tools is not None else []
        self.generate_tool: generate_Dockerfile_tool | None = None
    
    def set_generate_Dockerfile_tool(self, tool: generate_Dockerfile_tool):
        self.generate_tool = tool

    def _execute_directly(self, state: DockerGenerationWorkflowState):
        plan_actions = state["plan_actions"]
        plan_thoughts = state["plan_thoughts"]
        step_output = state["step_output"] if "step_output" in state and \
            state["step_output"] is not None else "N/A"
        step_dockerfile_content = state["step_dockerfile_content"] if "step_dockerfile_content" in state and \
            state["step_dockerfile_content"] is not None else "N/A"
        self.generate_tool.set_intermediate_output(
            plan_thoughts=plan_thoughts,
            step_error=step_output,
            step_dockerfile_content=step_dockerfile_content,
        )
        prompt = CustomPromptTemplate(
            template=DOCKERGENERATION_EXECUTION_SYSTEM_PROMPT,
            tools=self.custom_tools,
            plan_actions=plan_actions,
            input_variables=[
                "tools", "tool_names", "agent_scratchpad",
                "intermediate_steps", "plan_actions", "plan_thoughts",
            ],
        )
        output_parser = CustomOutputParser()
        agent = create_react_agent(
            llm = self.llm,
            tools = self.custom_tools,
            prompt = prompt,
            output_parser=output_parser,
            stop_sequence=["\nObservation:"],
        )
        callback_handler = OpenAICallbackHandler()
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.custom_tools,
            max_iterations=10,
        )
        response = agent_executor.invoke(
            input={
                "plan_actions": plan_actions, 
                "plan_thoughts": plan_thoughts,
                "input": "Now, let's begin."
            },
            config={
                "callbacks": [callback_handler],
                "recursion_limit": 20,
            }
        )
        if "output" in response:
            output = response["output"]
            self._print_step(state, step_output=f"**Execute Output:** \n{output}")
            if "**Final Answer**" in output:
                final_answer = output.split("**Final Answer:**")[-1].strip().strip(":")
                step_output = final_answer
            elif "Final Answer" in output:
                final_answer = output.split("Final Answer")[-1].strip().strip(":")
                step_output = final_answer
            else:
                step_output = output
            self._print_step(state, step_output=step_output)
            state["step_output"] = step_output
            if "**Dockerfile file name**" in step_output:
                dockerfile: str = step_output.split("**Dockerfile file name**")[-1]
                dockerfile = dockerfile.strip().strip(":")
                dockerfile = dockerfile.strip("```").strip()
                state["dockerfile"] = dockerfile
            else:
                state["dockerfile"] = None
            # state["dockerfile"] = f"demo-bioguider-{docker_id}.Dockerfile"
        else:
            logger.error("No output found in the response.")
            self._print_step(
                state,
                step_output="Error: No output found in the response.",
            )
            state["step_output"] = "Error: No output found in the response."
        
        
        token_usage = vars(callback_handler)
        token_usage = {**DEFAULT_TOKEN_USAGE, **token_usage}
            
        return state, token_usage

