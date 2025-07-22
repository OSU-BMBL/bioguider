
from langchain.prompts import ChatPromptTemplate
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from bioguider.agents.agent_utils import get_tool_names_and_descriptions
from bioguider.agents.common_agent_2step import CommonAgentTwoChainSteps, CommonAgentTwoSteps
from bioguider.agents.identification_task_utils import IdentificationWorkflowState
from bioguider.agents.peo_common_step import PEOCommonStep

## plan system prompt
IDENTIFICATION_PLAN_SYSTEM_PROMPT = ChatPromptTemplate.from_template("""### **Goal**
You are an expert developer in the field of biomedical domain. Your goal is:
{goal}

### **Repository File Structure**
Here is the 2-level file structure of the repository (f - file, d - directory, l - symlink, u - unknown):
{repo_structure}

### **Function Tools**
You are provided the following function tools:
{tools}

### Intermediate Steps
Hers are the intermediate steps results:
{intermediate_steps}

### Intermediate Thoughts
Analysis: {intermediate_analysis}
Thoughts: {intermediate_thoughts}

### **Instruction**
We will repeat **Plan - Execution - Observation** loops as many times as needed. All the results in each round will be persisted, 
meaning that states and variables will persisted through multiple rounds of plan execution. Be sure to take advantage of this by 
developing your collection plan incrementally and reflect on the intermediate observations at each round, instead of coding up 
everything in one go. Be sure to take only one or two actions in each step.

### **Important Instructions**
{important_instructions}

### **Output**
You plan should follow this format:
Step: tool name, should be one of {tool_names}
Step Input: file name or directory name
Step: tool name, should be one of {tool_names}
Step Input: file name or directory name
""")

class IdentificationPlanResult(BaseModel):
    """ Identification Plan Result """
    actions: list[dict] = Field(description="a list of action dictionary, e.g. [{'name': 'read_file', 'input': 'README.md'}, ...]")

IdentificationPlanResultJsonSchema = {
    "title": "identification_plan_result",
    "description": "plan result",
    "type": "object",
    "properties": {
        "actions": {
            "type": "array",
            "description": """a list of action dictionary, e.g. [{'name': 'read_file', 'input': 'README.md'}, ...]""",
            "title": "Actions",
            "items": {"type": "object"}
        },
    },
    "required": ["actions"],
}

class IdentificationPlanStep(PEOCommonStep):
    def __init__(
        self,
        llm: BaseChatOpenAI,
        repo_path: str,
        repo_structure: str,
        gitignore_path: str,
        custom_tools: list[BaseTool] | None = None,
    ): 
        super().__init__(llm)
        self.step_name = "Identification Plan Step"
        self.repo_path = repo_path
        self.repo_structure = repo_structure
        self.gitignore_path = gitignore_path
        self.custom_tools = custom_tools if custom_tools is not None else []

    def _prepare_system_prompt(self, state: IdentificationWorkflowState) -> str:
        goal = state["goal"]
        important_instructions = "N/A" if not "plan_instructions" in state else state["plan_instructions"]
        repo_structure = self.repo_structure
        intermdediate_steps = self._build_intermediate_steps(state)
        step_analysis, step_thoughts = self._build_intermediate_analysis_and_thoughts(state)
        self._print_step(
            state,
            step_output="**Intermediate Step Output**\n" + intermdediate_steps
        )
        self._print_step(
            state,
            step_output="**Intermediate Step Analysis**\n{step_analysis}\n**Intermediate Step Thoughts**\n{step_thoughts}",
        )
        tool_names, tools_desc = get_tool_names_and_descriptions(self.custom_tools)
        return IDENTIFICATION_PLAN_SYSTEM_PROMPT.format(
            goal=goal,
            repo_structure=repo_structure,
            tools=tools_desc,
            intermediate_steps=intermdediate_steps,
            intermediate_analysis=step_analysis,
            intermediate_thoughts=step_thoughts,
            tool_names=tool_names,
            important_instructions=important_instructions,
        )

    def _convert_to_plan_actions_text(self, actions: list[dict]) -> str:
        plan_str = ""
        for action in actions:
            action_str = f"Step: {action['name']}\n"
            action_str += f"Step Input: {action['input']}\n"
            plan_str += action_str
        return plan_str

    def _execute_directly(self, state: IdentificationWorkflowState):
        system_prompt = self._prepare_system_prompt(state)
        agent = CommonAgentTwoSteps(llm=self.llm)
        res, _, token_usage, reasoning_process = agent.go(
            system_prompt=system_prompt,
            instruction_prompt="Now, let's begin.",
            schema=IdentificationPlanResultJsonSchema,
        )
        PEOCommonStep._reset_step_state(state)
        res = IdentificationPlanResult(**res)
        self._print_step(
            state,
            step_output="**Reasoning Process**\n" + reasoning_process,
        )
        self._print_step(
            state,
            step_output=f"**Plan**\n{res.actions}"
        )
        state["plan_actions"] = self._convert_to_plan_actions_text(res.actions)

        return state, token_usage

