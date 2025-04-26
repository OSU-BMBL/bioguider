
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate, StringPromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.callbacks.openai_info import OpenAICallbackHandler

from bioguider.agents.agent_utils import (
    DEFAULT_TOKEN_USAGE,
    CustomPromptTemplate, 
    CustomOutputParser,
    get_tool_names_and_descriptions,
)
from bioguider.agents.common_agent_2step import CommonAgentTwoSteps
from bioguider.agents.peo_common_step import PEOCommonStep, PEOWorkflowState, PlanAgentResult, PlanAgentResultJsonSchema
from bioguider.agents.collection_task_utils import CollectionWorkflowState

COLLECTION_EXECUTION_SYSTEM_PROMPT = ("""
You are an expert Python developer.
You are given a **plan** and are expected to complete it using Python code and the available tools.

---
### **Available Tools**
{tools}
---
### **Your Task**
Execute the plan step by step using the format below:

```
Thought: Describe what you are thinking or planning to do next.
Action: The tool you are going to use (must be one of: {tool_names})
Action Input: The input to the selected action
Observation: The result returned by the action
```
You may repeat the **Thought → Action → Action Input → Observation** loop as many times as needed.
Once the plan is fully completed, output the result in the following format:
```
Thought: I have completed the plan.
Final Answer: Summarize all files related to the goal.:
{{file_name1}}
{{file_name2}}
...
```
---
### **Example**
```
Action: read_file
Action Input: README.md
Observation: No
...
Final Answer:
**README.md**
**Dockerfile**
**requirements.txt**
...
```
---
### **Notes**
Please follow the plan exactly as speicified. **Do not take any actions** outside of the outlined steps.

### **Plan**
{plan_actions}
### **Actions Already Taken**
{agent_scratchpad}
---

{input}
""")

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
        

    def _execute_direct(self, state: PEOWorkflowState):
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
            max_iterations=5,
        )
        response = agent_executor.invoke(
            input={"plan_actions": plan_actions, "input": "Now, let's begin."},
            config={"callbacks": [callback_handler]},
        )

        # parse the response
        if "output" in response:
            output = response["output"]
            if "**Final Answer**" in output:
                final_answer = output.split("**Final Answer:**")[-1].strip().strip(":")
                step_output = final_answer
                self._print_step(state, step_output=step_output)
                state["step_output"] = step_output
            elif "Final Answer" in output:
                final_answer = output.split("Final Answer")[-1].strip().strip(":")
                step_output = final_answer
                self._print_step(state, step_output=step_output)
                state["step_output"] = step_output
            else:
                state["step_output"] = output
        
        token_usage = vars(callback_handler)
        token_usage = {**DEFAULT_TOKEN_USAGE, **token_usage}
            
        return state, token_usage