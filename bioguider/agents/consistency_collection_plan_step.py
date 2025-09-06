from langchain_openai.chat_models.base import BaseChatOpenAI

from langchain.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate, StringPromptTemplate
from bioguider.agents.agent_utils import (
    convert_plan_to_string, 
    get_tool_names_and_descriptions,
    PlanAgentResultJsonSchema,
    PlanAgentResult,
)
from bioguider.agents.common_agent_2step import CommonAgentTwoChainSteps, CommonAgentTwoSteps
from bioguider.agents.consistency_collection_task_utils import ConsistencyCollectionWorkflowState
from bioguider.agents.peo_common_step import PEOCommonStep

CONSISTANCE_EVAL_PLAN_SYSTEM_PROMPT = ChatPromptTemplate.from_template("""### **Goal**  
You are an expert developer specializing in the biomedical domain. 
Your task is to collect the function, class, and method definitions and docstrings for a given user guide/API documentation.

---

### **Function Tools**  
You have access to the following function tools:  
{tools}

---

### **Intermediate Steps**  
Here are the results from previous steps:  
{intermediate_steps}

---

### **Intermediate Thoughts**  
- **Analysis**: {intermediate_analysis}  
- **Thoughts**: {intermediate_thoughts}

---

### **Instructions**
1. We will iterate through multiple **Plan -> Execution -> Observation** loops as needed.  
   - All variables and tool outputs are **persisted across rounds**, so you can build on prior results.  
   - Develop your plan **incrementally**, and reflect on intermediate observations before proceeding.  
   - Limit each step to **one or two actions** — avoid trying to complete everything in a single step.

2. Your task is to evaluate the consistency of the user guide/API documentation.

3. You may use 
   - the `retrieve_function_definition_and_docstring` tool to get the function definition and docstring or,
   - the `retrieve_class_definition_and_docstring` to get the class definition and docstring or,
   - the `retrieve_class_and_method_definition_and_docstring` to get the class and method definition and docstring.

4. Your plan can only use the above tools, **do not** make up any tools not in the above tools list.

5. If no function, class, or method is found in the given user guide/API documentation, you should return "N/A" as an empty plan.
   Our tools can only retrieve the **function**, **class**, **method** definition and docstring, **do not** make up any function, class, or method name.
    
    
### **Input User Guide/API Documentation**
{user_guide_api_documentation}

### **Output Format**
Your plan **must exactly match** a sequence of steps in the following format, **do not** make up anything:

Step: <tool name>   # Tool name **must be one** of {tool_names}  
Step Input: <function/class/method name>
Step Input: <file path, if not sure, just put "N/A">

Step: <tool name>  # Tool name **must be one** of {tool_names}  
Step Input: <function/class/method name>
Step Input: <file path, if not sure, just put "N/A">
...

...
""")

class ConsistencyCollectionPlanStep(PEOCommonStep):
    """
    ConsistencyCollectionPlanStep is a step in the consistency collection plan process.
    It is responsible for initializing the tools and compiling the step.
    """
    
    def __init__(
        self, 
        llm: BaseChatOpenAI,
        custom_tools: list[BaseTool] | None = None,
    ):
        super().__init__(llm)
        self.step_name = "Consistency Collection Plan Step"
        self.custom_tools = custom_tools if custom_tools is not None else []
    
    def _prepare_system_prompt(self, state: ConsistencyCollectionWorkflowState) -> str:
        user_guide_api_documentation = state["user_guide_api_documentation"]
        intermediate_steps = self._build_intermediate_steps(state)
        step_analysis, step_thoughts = self._build_intermediate_analysis_and_thoughts(state)
        tool_names, tools_desc = get_tool_names_and_descriptions(self.custom_tools)
        system_prompt = CONSISTANCE_EVAL_PLAN_SYSTEM_PROMPT.format(
            tools=tools_desc,
            intermediate_steps=intermediate_steps,
            intermediate_analysis=step_analysis,
            intermediate_thoughts=step_thoughts,
            tool_names=tool_names,
            user_guide_api_documentation=user_guide_api_documentation,
        )
        self._print_step(
            state,
            step_output="**Intermediate Step Output**\n" + intermediate_steps
        )
        self._print_step(
            state,
            step_output="**Intermediate Step Analysis**\n{step_analysis}\n**Intermediate Step Thoughts**\n{step_thoughts}",
        )
        return system_prompt

    def _execute_directly(self, state: ConsistencyCollectionWorkflowState):
        system_prompt = self._prepare_system_prompt(state)
        agent = CommonAgentTwoSteps(llm=self.llm)
        res, _, token_usage, reasoning_process = agent.go(
            system_prompt=system_prompt,
            instruction_prompt="Now, let's begin the consistency collection plan step.",
            schema=PlanAgentResultJsonSchema,
        )
        PEOCommonStep._reset_step_state(state)
        res = PlanAgentResult(**res)
        self._print_step(state, step_output=f"**Reasoning Process**\n{reasoning_process}")
        self._print_step(state, step_output=f"**Plan**\n{str(res.actions)}")
        state["plan_actions"] = convert_plan_to_string(res)

        return state, token_usage