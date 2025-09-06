import logging

from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain.tools import BaseTool, StructuredTool
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.callbacks.openai_info import OpenAICallbackHandler

from bioguider.agents.consistency_collection_task_utils import ConsistencyCollectionWorkflowState
from bioguider.agents.consistency_collection_task_utils import ConsistencyCollectionWorkflowState
from bioguider.database.code_structure_db import CodeStructureDb
from bioguider.utils.constants import DEFAULT_TOKEN_USAGE
from bioguider.agents.agent_utils import CustomOutputParser, CustomPromptTemplate
from bioguider.agents.peo_common_step import ( 
    PEOCommonStep,
)

logger = logging.getLogger(__name__)

CONSISTENCY_EVAL_EXECUTION_SYSTEM_PROMPT = """You are an expert developer specializing in the biomedical domain.

You are given a **plan** and are expected to complete it using the available tools.

---

### **Available Tools**
{tools}

---

### **Your Task**

Your job is to **execute the given plan step by step**, using the tools available to you.

---

### **Output Format (Strict Order Required)**

For **each step**, follow the **exact format** below and **do not change the order of the fields** under any circumstances:

```
Thought: Describe what you are thinking or planning to do next.  
Action: The tool you are going to use (must be one of: {tool_names})  
Action Input: The input provided to the selected action  
Observation: The result returned by the action
```

---

### **Important Instructions**
1. You may repeat the **Thought → Action → Action Input → Observation** loop as needed.  
2. Once all steps in the plan have been executed, output all the results using this format:
3. For each step, **only execute one tool**.

```
Thought: I have completed the plan.
Final Answer:
Action: <tool_name>
Action Input: <input>
Action Observation: <Observation1>
---
Action: <tool_name>
Action Input: <input>
Action Observation: <Observation2>
---
...
```

---

### **Plan**
{plan_actions}

### **Actions Already Taken**
{agent_scratchpad}

---

{input}
"""

class ConsistencyCollectionExecuteStep(PEOCommonStep):
    def __init__(
        self, 
        llm: BaseChatOpenAI, 
        code_structure_db: CodeStructureDb,
        custom_tools: list[BaseTool] | None = None,
    ):
        self.llm = llm
        self.code_structure_db = code_structure_db
        self.step_name = "Consistency Collection Execute Step"
        self.custom_tools = custom_tools if custom_tools is not None else []

    def _execute_directly(self, state: ConsistencyCollectionWorkflowState):
        plan_actions = state["plan_actions"]
        prompt = CustomPromptTemplate(
            template=CONSISTENCY_EVAL_EXECUTION_SYSTEM_PROMPT,
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
            max_iterations=30,
        )
        response = agent_executor.invoke(
            input={
                "plan_actions": plan_actions,
                "input": "Now, let's begin."
            },
            config={
                "callbacks": [callback_handler],
                "recursion_limit": 20,
            },
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