import logging
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate, StringPromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.callbacks.openai_info import OpenAICallbackHandler

from bioguider.utils.constants import DEFAULT_TOKEN_USAGE
from bioguider.agents.agent_utils import (
    CustomPromptTemplate, 
    CustomOutputParser,
)
from bioguider.agents.common_agent_2step import CommonAgentTwoSteps
from bioguider.agents.peo_common_step import PEOCommonStep, PEOWorkflowState
from bioguider.agents.collection_task_utils import CollectionWorkflowState

logger = logging.getLogger(__name__)

COLLECTION_EXECUTION_SYSTEM_PROMPT = """---

You are an expert Python developer.  
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

Once all steps in the plan have been executed, output all the results using this format:

```
Thought: I have completed the plan.
Final Answer:
Action: {{tool_name}}
Action Input: {{input1}}
Action Observation: {{Observation1}}
---
Action: {{tool_name}}
Action Input: {{input2}}
Action Observation: {{Observation2}}
---
...
```

---

### **Example**
```
Action: summarize_file_tool  
Action Input: README.md  
Action Input: "Please extract license information in summarized file content."
Observation: # BioGuider\nBioGuider is a Python package for bioinformatics.\n...
...
Final Answer:
Action: summarize_file_tool
Action Input: README.md
Action Input: "N/A"
Action Observation: # BioGuider\nBioGuider is a Python package for bioinformatics.\n...
---
Action: check_file_related_tool
Action Input: pyproject.toml
Action Observation: Yes, the file is related to the project.
---
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

### **Actions Already Taken**  
{agent_scratchpad}

---

{input}

---
"""

class CollectionExecuteStep(PEOCommonStep):
    def __init__(
        self,
        llm: BaseChatOpenAI,
        repo_path: str,
        repo_structure: str,
        gitignore_path: str,
        custom_tools: list[BaseTool] | None = None,
    ):
        super().__init__(llm)
        self.step_name = "Collection Execution Step"
        self.repo_path = repo_path
        self.repo_structure = repo_structure
        self.gitignore_path = gitignore_path
        self.custom_tools = custom_tools if custom_tools is not None else []
        

    def _execute_directly(self, state: PEOWorkflowState):
        plan_actions = state["plan_actions"]
        prompt = CustomPromptTemplate(
            template=COLLECTION_EXECUTION_SYSTEM_PROMPT,
            tools=self.custom_tools,
            plan_actions=plan_actions,
            input_variables=[
                "tools", "tool_names", "agent_scratchpad", 
                "intermediate_steps", "plan_actions",
            ],
        )
        output_parser = CustomOutputParser()
        agent = create_react_agent(
            llm=self.llm,
            tools=self.custom_tools,
            prompt=prompt,
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
            input={"plan_actions": plan_actions, "input": "Now, let's begin."},
            config={
                "callbacks": [callback_handler],
                "recursion_limit": 20,
            },
        )

        # parse the response
        if "output" in response:
            output = response["output"]
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